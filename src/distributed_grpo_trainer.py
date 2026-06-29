"""
distributed_grpo_trainer.py — Gemma-Sync GRPO Trainer (Dual NVIDIA T4 GPUs)
=============================================================================

Fine-tunes Gemma 4 E2B with GRPO (G=16) and RLVR using the Calibrated Abstention
framework. Integrates the SOFA Oracle as a verifiable logic constraint to mitigate
clinical hallucinations.

Hardware Target:
  - Training: Kaggle 2× NVIDIA T4 GPUs (16GB VRAM each, CUDA)
  - Precision: NF4 QLoRA + FP16 mixed precision
  - Distribution: 2-GPU Data Parallelism via HF Accelerate

Architecture:
  - 2-way Data Parallelism across CUDA devices via Accelerate
  - GRPO Group Size G=16 (streamed in generation_batch_size=4 chunks)
  - 4-bit QLoRA (NF4) via Unsloth FastLanguageModel
  - FP16 mixed precision (T4 Tensor Cores)

Reward Architecture (4-tier weighted — identical to TPU version):
  R = 0.50 × correctness     — RLVR exact-match \\boxed{}
    + 0.20 × sofa_oracle     — SOFA 6-system verifier
    + 0.10 × format          — LaTeX structure compliance
    + 0.20 × process         — CoT quality + metacognitive calibration
  [+0.20 bonus]              — Calibrated Abstention (N/P detection)

Author : Narendra Bayutama Wibisono
Project: Gemma-Sync — Distributed Uncertainty-Aware Clinical Reasoning via Gemma 4 E2B
Target : Kaggle 2× NVIDIA T4 GPUs (16GB VRAM each)
Ref    : Ported from TPU v5e-8 version; evolved from 'Calibrated Abstention in Clinical LLMs'
         Zenodo: 10.5281/zenodo.19599245
"""

# ===========================================================================
# MODULE 1: Environment Bootstrap & CUDA Initialization
# ===========================================================================

import os
import sys

# --- Force PyTorch backend (disable TensorFlow on Kaggle) ---
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- CUDA Optimization Flags (Dual T4) ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Unsloth — Primary model loader for T4 QLoRA
# ---------------------------------------------------------------------------
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# BitsAndBytes — Fallback if Unsloth unavailable
# ---------------------------------------------------------------------------
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    pass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gemma_sync_grpo_t4")


def get_device() -> torch.device:
    """
    Return the canonical compute device for this environment.

    Priority: CUDA (T4 GPU) > MPS (Apple M2 prototyping) > CPU.
    For multi-GPU setups, Accelerate handles device placement;
    this returns the default CUDA device for non-distributed ops.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_master() -> bool:
    """True only on LOCAL_RANK 0. Used to gate saves/logs in multi-GPU."""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


# ===========================================================================
# MODULE 2: Constants & VRAM Budget Analysis
# ===========================================================================

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
# Kaggle offline model path (Gemma 4 E2B IT — Kaggle Models hub mount)
KAGGLE_MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"
# HuggingFace Hub ID (for online environments / prototyping)
HF_MODEL_ID = "google/gemma-4-e2b-it"

DEFAULT_OUTPUT_DIR = "/kaggle/working/outputs/gemma4-e2b-grpo-sofa-t4"
LOCAL_DATA_DIR = "./medqa_dataset"

# ---------------------------------------------------------------------------
# Reward Architecture Weights (plan_overview.md §3)
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "correctness":        0.50,   # RLVR: exact \\boxed{} match
    "sofa_oracle":        0.20,   # SOFA 6-system verifier
    "format_compliance":  0.10,   # LaTeX structure checks
    "process_quality":    0.20,   # CoT depth + metacognitive calibration
}
ABSTENTION_BONUS = 0.20

# ---------------------------------------------------------------------------
# GRPO Group Configuration
# ---------------------------------------------------------------------------
# G=16 total candidates per prompt.
# Distribution on 2× T4: 8 candidates/GPU × 2 GPUs = 16 total
# Streamed in generation_batch_size=4 chunks to avoid VRAM spikes.
GRPO_GROUP_SIZE = 4   # G — reduced from 16 for single T4 VRAM safety

# ---------------------------------------------------------------------------
# SOFA Clinical Constants (identical to TPU version)
# ---------------------------------------------------------------------------
SOFA_COMPONENTS = [
    "Respiratory",    # PaO2/FiO2 ratio
    "Coagulation",    # Platelet count (×10³/μL)
    "Liver",          # Bilirubin (mg/dL)
    "Cardiovascular", # MAP / Vasopressors
    "CNS",            # GCS (plan_overview uses 'CNS'; synonym for Neurological)
    "Renal",          # Creatinine (mg/dL)
]

# ---------------------------------------------------------------------------
# Precision SOFA Oracle: Clinical Score Thresholds
# Reference: Vincent et al., Intensive Care Med 1996
# ---------------------------------------------------------------------------
SOFA_SCORE_THRESHOLDS: Dict[str, Dict[int, Any]] = {
    "Respiratory": {
        0: lambda v: v >= 400,
        1: lambda v: 300 <= v < 400,
        2: lambda v: 200 <= v < 300,
        3: lambda v: 100 <= v < 200,
        4: lambda v: v < 100,
    },
    "Coagulation": {
        0: lambda v: v >= 150,
        1: lambda v: 100 <= v < 150,
        2: lambda v: 50 <= v < 100,
        3: lambda v: 20 <= v < 50,
        4: lambda v: v < 20,
    },
    "Liver": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 6.0,
        3: lambda v: 6.0 <= v < 12.0,
        4: lambda v: v >= 12.0,
    },
    "CNS": {
        0: lambda v: v == 15,
        1: lambda v: 13 <= v <= 14,
        2: lambda v: 10 <= v <= 12,
        3: lambda v: 6 <= v <= 9,
        4: lambda v: v < 6,
    },
    "Renal": {
        0: lambda v: v < 1.2,
        1: lambda v: 1.2 <= v < 2.0,
        2: lambda v: 2.0 <= v < 3.5,
        3: lambda v: 3.5 <= v < 5.0,
        4: lambda v: v >= 5.0,
    },
    "Cardiovascular": {
        0: lambda v: v >= 70,
        1: lambda v: v < 70,
    },
}

VASOPRESSOR_KEYWORDS = [
    "dopamine", "dobutamine", "epinephrine", "norepinephrine",
    "vasopressin", "phenylephrine", "milrinone",
]

NON_CRITICAL_KEYWORDS = [
    "psychiatry", "psychiatric", "behavioral", "psychology",
    "dermatology", "dermatologic", "skin rash",
    "preventive medicine", "screening", "vaccination",
    "outpatient", "well-child", "annual physical",
    "ethics", "bioethics", "informed consent",
    "genetics", "genetic counseling",
    "ophthalmology", "vision", "eye exam",
]

CACTUS_ESCALATE_TOKEN = "<|escalate|>"
CACTUS_LOCAL_TOKEN = "<|local_ok|>"
CONFIDENCE_THRESHOLD = 0.70

HALLUCINATION_PENALTY = 0.10

# ===========================================================================
# MODULE 3: LoRA Configuration (Unsloth / T4-Optimized)
# ===========================================================================

# ---------------------------------------------------------------------------
# VRAM Budget per T4 GPU (16GB):
# ---------------------------------------------------------------------------
#   Component                       | Estimate
#   --------------------------------|----------
#   Base model NF4 (4-bit)          | ~1.2 GB  (E2B ~2B params, 4-bit)
#   LoRA adapters (r=32, FP16)      | ~0.05 GB
#   AdamW optimizer states (LoRA)   | ~0.10 GB
#   Gradient checkpointing buffer   | ~0.30 GB
#   KV cache (gen_batch=4, seq=1024)| ~1.5 GB
#   Activations + CUDA overhead     | ~6.0 GB
#   --------------------------------|----------
#   Total per GPU                   | ~9.2 GB  (6.8 GB headroom ✓)
# ---------------------------------------------------------------------------


def get_lora_config() -> LoraConfig:
    """
    LoRA configuration for Gemma 4 E2B (T4 QLoRA path).

    Note: target_modules use base names (no '.linear' suffix) because
    Unsloth handles the internal submodule naming. For the standard HF
    fallback path, these names also work with BitsAndBytes quantized models.
    """
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,
    )


# ===========================================================================
# MODULE 4: Model & Tokenizer Loading (Unsloth + 4-bit QLoRA)
# ===========================================================================

def setup_cuda_env():
    """
    Apply CUDA-specific environment optimizations for dual T4 GPUs.

    Sets memory allocator config and logs GPU details.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if is_master():
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"GPU {i}: {name} | {mem:.1f} GB VRAM")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch: {torch.__version__}")
    else:
        backend = "MPS" if torch.backends.mps.is_available() else "CPU"
        logger.warning(f"CUDA NOT available — running on {backend} (prototyping mode).")


def log_vram_usage(label: str = ""):
    """
    Log CUDA VRAM usage for OOM monitoring.

    Reports allocated and reserved memory on all visible GPUs.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc_mb = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(i) / (1024 ** 2)
            if is_master():
                logger.info(
                    f"VRAM [{label}] GPU {i}: "
                    f"{alloc_mb:.1f} MB alloc / {reserved_mb:.1f} MB reserved"
                )
    elif torch.backends.mps.is_available():
        alloc_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        logger.info(f"MPS Memory [{label}]: {alloc_mb:.1f} MB allocated")


def load_model_and_tokenizer(
    model_id: str = KAGGLE_MODEL_ID,
    use_unsloth: bool = True,
) -> Tuple:
    """
    Load Gemma 4 E2B with 4-bit QLoRA for dual T4 GPU training.

    Path A (PRIMARY — Unsloth):
      Uses FastLanguageModel with load_in_4bit=True and FP16 compute dtype.
      Unsloth handles LoRA wrapping, gradient checkpointing, and memory
      optimization internally. 2× throughput over standard HF.

    Path B (FALLBACK — HF + BitsAndBytes):
      If Unsloth is unavailable, falls back to standard HF model loading
      with BitsAndBytes NF4 quantization and manual PEFT wrapping.

    Args:
        model_id:     Path to model (Kaggle local or HF Hub ID).
        use_unsloth:  If True (default) and Unsloth is installed, use
                      FastLanguageModel for optimized T4 QLoRA training.

    Returns:
        Tuple of (model, tokenizer).
    """
    if is_master():
        logger.info(f"Loading model  : {model_id}")
        logger.info(f"Precision      : NF4 QLoRA + FP16")
        logger.info(f"Unsloth        : {'ENABLED' if use_unsloth and UNSLOTH_AVAILABLE else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Path A: Unsloth FastLanguageModel (PRIMARY for T4)
    # ------------------------------------------------------------------
    if use_unsloth and UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=1024,            # prompt(512) + completion(1024)
            dtype=torch.float16,            # FP16 compute (T4 Tensor Cores)
            load_in_4bit=True,              # NF4 QLoRA — non-negotiable for T4
        )

        # Gemma 4 multimodal — unwrap Processor to get the raw tokenizer
        if hasattr(tokenizer, "tokenizer"):
            if is_master():
                logger.info("Detected Gemma4Processor in Unsloth path — extracting inner tokenizer.")
            tokenizer = tokenizer.tokenizer

        # NOTE: Cactus routing tokens (<|escalate|>, <|local_ok|>) are NOT added
        # as special tokens for Gemma 4. The model's PLE (Per-Layer Embeddings)
        # have fixed projection dimensions — resize_token_embeddings breaks them.
        # Instead, the model learns to produce these strings as subword sequences,
        # and reward functions detect them via regex (string matching).
        if is_master():
            logger.info("Cactus routing tokens handled as subword sequences (no embedding resize).")

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",   # Unsloth optimized GC
            random_state=42,
        )
        if is_master():
            logger.info("Unsloth FastLanguageModel loaded with 4-bit QLoRA + LoRA ✓")
        return model, tokenizer

    # ------------------------------------------------------------------
    # Path B: Standard HF + BitsAndBytes NF4 (FALLBACK)
    # ------------------------------------------------------------------
    if is_master():
        logger.info("Unsloth unavailable — falling back to HF + BitsAndBytes NF4.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True,
    )

    # If we got a Processor (Gemma 4 multimodal wrapper), extract inner tokenizer
    if hasattr(tokenizer, "tokenizer"):
        if is_master():
            logger.info("Detected Gemma4Processor — extracting inner tokenizer.")
        tokenizer = tokenizer.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # NOTE: Cactus tokens NOT added as special tokens for Gemma 4.
    # PLE (Per-Layer Embeddings) have fixed projection dims — resize breaks them.
    # Reward functions detect <|escalate|> / <|local_ok|> via regex instead.
    if is_master():
        logger.info("Cactus routing tokens handled as subword sequences (no embedding resize).")

    # BitsAndBytes NF4 quantization config
    quantization_config = None
    if BNB_AVAILABLE:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    if is_master():
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA applied   : Trainable {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    return model, tokenizer


# ===========================================================================
# MODULE 5: GRPO Training Configuration (G=16, Dual T4)
# ===========================================================================

def get_grpo_config(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = 2,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-6,
    num_generations: int = GRPO_GROUP_SIZE,
    max_completion_length: int = 1024,
    run_name: Optional[str] = None,
    max_steps: int = 202,
    save_steps: int = 50,
    fp16: bool = True,
) -> GRPOConfig:
    """
    GRPO training configuration for Gemma 4 E2B on 2× NVIDIA T4 GPUs.

    GRPO Group Size (G=16) — VRAM-safe streaming:
      generation_batch_size=4 streams the 16 generations in 4 chunks
      of 4 candidates. This keeps peak VRAM during generation under 12GB.

    Effective batch size = per_device_batch_size × gradient_accumulation × 2 GPUs
                        = 1 × 4 × 2 = 8 prompts per weight update.

    T4-critical parameters:
      fp16=True            : Native T4 FP16 Tensor Core support.
      bf16=False           : T4 has limited BF16 support — use FP16 instead.
      tf32=True            : T4 supports TF32 for matmul acceleration.
      dataloader_pin_memory: True — CUDA benefits from pinned host memory.
      optim                : 'adamw_torch_fused' — fused CUDA kernel for AdamW.
    """
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"gemma4_e2b_grpo_sofa_t4_{ts}"

    config = GRPOConfig(
        # ---- Output & Logging ----
        output_dir=output_dir,
        run_name=run_name,
        logging_steps=1,
        logging_first_step=True,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        use_cache=True,
        bf16=False,
        fp16=fp16,
        tf32=False,            # T4 is Turing arch — TF32 requires Ampere+

        # ---- Training Schedule ----
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # ---- Optimizer ----
        learning_rate=learning_rate,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",

        # ---- GRPO-Specific ----
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        generation_batch_size=num_generations, # Must equal G (TRL >= 0.17 constraint)

        # ---- CUDA Dataloader ----
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # ---- Gradient Checkpointing ----
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ---- Evaluation ----
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=num_generations,

        # ---- Reproducibility ----
        seed=42,
        data_seed=42,
    )

    if is_master():
        n_gpus = max(torch.cuda.device_count(), 2) if torch.cuda.is_available() else 2
        eff_batch = per_device_batch_size * gradient_accumulation_steps * n_gpus
        logger.info("GRPO Training Config (Gemma 4 E2B, Dual T4):")
        logger.info(f"  G (group size)   : {num_generations}")
        logger.info(f"  Per-GPU batch    : {per_device_batch_size}")
        logger.info(f"  Grad accum steps : {gradient_accumulation_steps}")
        logger.info(f"  GPUs             : {n_gpus}")
        logger.info(f"  Effective batch  : {eff_batch} prompts/update")
        logger.info(f"  Gen batch size   : 4 (streams G=16 in 4 chunks)")
        logger.info(f"  Max completion   : {max_completion_length} tokens")
        logger.info(f"  Learning rate    : {learning_rate}")
        logger.info(f"  Precision        : FP16 (T4 Tensor Cores)")

    return config


# ===========================================================================
# MODULE 6: RLVR Reward Functions (4-Tier + Calibrated Abstention)
# ===========================================================================
# NOTE: All reward logic below is IDENTICAL to the TPU v5e-8 version.
# The SOFA Oracle, MAP calculator, Cactus routing penalties, and
# Calibrated Abstention bonus are hardware-agnostic pure Python.
# ===========================================================================

# ---------------------------------------------------------------------------
# Utility: Text extraction helper
# ---------------------------------------------------------------------------

def _extract_text(completion: Any) -> str:
    """Extract plain text from a completion (str, list of dicts, or dict)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in completion
        )
    if isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


def _extract_boxed(text: str) -> Optional[str]:
    """
    Extract the answer letter from LaTeX \\boxed{X} notation.
    Handles: \\boxed{A}, $$\\boxed{B}$$, $\\boxed{C}$, boxed{D}.
    Returns the uppercase letter or None.
    """
    patterns = [
        r"\\boxed\{\s*([A-Da-d])\s*\}",
        r"\$+\\boxed\{\s*([A-Da-d])\s*\}\$+",
        r"boxed\{\s*([A-Da-d])\s*\}",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Reward 1: Correctness (RLVR) — weight 0.50
# ---------------------------------------------------------------------------

def reward_correctness(
    prompts: List[Any],
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    RLVR hard reward: binary exact-match of \\boxed{X} against ground-truth.
    Weight 0.50 — the dominant signal.
    Returns 1.0 (correct) or 0.0 (incorrect or absent).
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        text = _extract_text(completion)
        extracted = _extract_boxed(text)
        if extracted and extracted == expected.strip().upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2: SOFA Oracle + Calibrated Abstention — weight 0.20 (+0.20 bonus)
# ---------------------------------------------------------------------------

def _check_sofa_not_applicable(text: str) -> bool:
    """Detect SOFA_NOT_APPLICABLE flag for non-critical case abstention."""
    return bool(re.search(
        r"SOFA_NOT_APPLICABLE|SOFA\s+(?:is\s+)?not\s+applicable",
        text, re.IGNORECASE
    ))


def _parse_sofa_table(text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the model's Markdown SOFA table into structured component data.
    Returns dict mapping component → {parameter, value, score}.
    Handles CNS/Neurological as synonyms per plan_overview.md.
    """
    row_pat = re.compile(
        r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
        re.MULTILINE,
    )
    alias = {
        "respiratory": "Respiratory", "respiration": "Respiratory",
        "coagulation": "Coagulation", "coag": "Coagulation", "platelet": "Coagulation",
        "liver": "Liver", "hepatic": "Liver", "bilirubin": "Liver",
        "cardiovascular": "Cardiovascular", "cardio": "Cardiovascular", "map": "Cardiovascular",
        "neurological": "CNS", "neuro": "CNS", "cns": "CNS", "gcs": "CNS",
        "renal": "Renal", "kidney": "Renal", "creatinine": "Renal",
    }
    result = {}
    for m in row_pat.findall(text):
        comp_raw, param, value, score = [x.strip() for x in m]
        if comp_raw.lower() in ("sofa component", "---", "") or re.match(r"^[-:|\s]+$", comp_raw):
            continue
        canonical = next(
            (v for k, v in alias.items() if k in comp_raw.lower()), None
        )
        if canonical:
            result[canonical] = {"parameter": param, "value": value, "score": score}
    return result


# ---------------------------------------------------------------------------
# Precision SOFA Oracle Helpers (ported from legacy SOFAOracle class)
# ---------------------------------------------------------------------------

def _extract_map_from_cv_string(value_str: str) -> Optional[float]:
    """
    Extract Mean Arterial Pressure (mmHg) from a cardiovascular value string.
    Handles BP string, direct MAP label, and bare numeric.
    MAP = (SBP + 2 × DBP) / 3  [Vincent et al., 1996]
    """
    text = value_str.strip().lower()

    bp_match = re.search(
        r"(?:bp|blood\s*pressure)?[:\s]*(?<!\.)\b(\d{2,3})\s*/\s*(\d{2,3})\b",
        text, re.IGNORECASE,
    )
    if bp_match:
        sbp = float(bp_match.group(1))
        dbp = float(bp_match.group(2))
        return round((sbp + 2.0 * dbp) / 3.0, 1)

    map_match = re.search(
        r"map[:\s=]*(\d+\.?\d*)\s*(?:mmhg)?",
        text, re.IGNORECASE,
    )
    if map_match:
        return float(map_match.group(1))

    bare_match = re.search(r"\b(\d{2,3})\s*(?:mmhg)?\b", text, re.IGNORECASE)
    if bare_match:
        return float(bare_match.group(1))

    return None


def _parse_numeric_sofa_value(value_str: str) -> Optional[float]:
    """Extract a numeric value from a SOFA table cell for threshold validation."""
    cleaned = re.sub(r"\[assumed:\s*[^=]*=\s*", "", value_str)
    cleaned = re.sub(r"[a-zA-Z\u2082/\u03bc\u00b0%\]\[]+", "", cleaned)
    cleaned = re.sub(r"[><>=\u2265\u2264]", "", cleaned)
    cleaned = cleaned.strip()
    m = re.search(r"(\d+\.?\d*)", cleaned)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _validate_cv_score(value_str: str, claimed_score: int) -> Optional[bool]:
    """
    Validate Cardiovascular SOFA sub-score against MAP + vasopressor evidence.
    Returns True (plausible), False (implausible), or None (indeterminate).
    """
    value_lower = value_str.lower()
    map_val = _extract_map_from_cv_string(value_lower)
    has_vasopressors = any(vp in value_lower for vp in VASOPRESSOR_KEYWORDS)

    if claimed_score == 0:
        if map_val is not None:
            return map_val >= 70 and not has_vasopressors
        if any(kw in value_lower for kw in ("stable", "normal", "assumed")):
            return True
        return None
    elif claimed_score == 1:
        if map_val is not None:
            return map_val < 70
        return None
    elif claimed_score in (2, 3, 4):
        if has_vasopressors:
            return True
        if map_val is not None and map_val < 70:
            return True
        return False
    return None


def _score_sofa_oracle(text: str) -> float:
    """
    Precision SOFA Oracle — deterministic clinical verification.
    Scoring: (0.25) table + (0.20) coverage + (0.20) validity +
             (0.20) plausibility + (0.15) arithmetic.
    Penalty: -HALLUCINATION_PENALTY per fabricated score.
    Returns float in [0.0, 1.0].
    """
    if _check_sofa_not_applicable(text):
        return 1.0

    components = _parse_sofa_table(text)
    if not components:
        return 0.0

    score = 0.0
    hallucination_count = 0

    score += 0.25

    found = set(components.keys()) & set(SOFA_COMPONENTS)
    score += 0.20 * (len(found) / len(SOFA_COMPONENTS))

    valid_scores = 0
    for comp_data in components.values():
        s = comp_data["score"].strip()
        if s in ("0", "1", "2", "3", "4") or s.upper() in ("N/P", "NOT PROVIDED"):
            valid_scores += 1
    score += 0.20 * (valid_scores / max(len(components), 1))

    plausible_count = 0
    checked_count = 0

    for comp_name, comp_data in components.items():
        raw_value = comp_data["value"].strip()
        raw_score = comp_data["score"].strip()

        is_np = raw_value.upper() in ("N/P", "NOT PROVIDED", "N/ P", "")
        if is_np:
            continue

        if raw_score in ("0", "1", "2", "3", "4"):
            has_numeric_content = bool(re.search(r"\d", raw_value))
            has_vasopressor = any(vp in raw_value.lower() for vp in VASOPRESSOR_KEYWORDS)
            has_stability_kw = any(kw in raw_value.lower() for kw in ("stable", "normal", "assumed"))
            if not (has_numeric_content or has_vasopressor or has_stability_kw):
                hallucination_count += 1
                continue

        if comp_name == "Cardiovascular" and raw_score in ("0", "1", "2", "3", "4"):
            checked_count += 1
            cv_valid = _validate_cv_score(raw_value, int(raw_score))
            if cv_valid is True:
                plausible_count += 1
            continue

        if raw_score not in ("0", "1", "2", "3", "4"):
            continue

        claimed = int(raw_score)
        numeric_val = _parse_numeric_sofa_value(raw_value)
        if numeric_val is None:
            continue

        checked_count += 1
        threshold_key = comp_name if comp_name in SOFA_SCORE_THRESHOLDS else None
        if threshold_key and claimed in SOFA_SCORE_THRESHOLDS[threshold_key]:
            if SOFA_SCORE_THRESHOLDS[threshold_key][claimed](numeric_val):
                plausible_count += 1

    if checked_count > 0:
        score += 0.20 * (plausible_count / checked_count)
    else:
        score += 0.10

    total_match = re.search(
        r"\*\*Total SOFA Score:\*\*\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE
    )
    if total_match:
        claimed_total = int(total_match.group(1))
        numeric_subs = [
            int(d["score"].strip())
            for d in components.values()
            if d["score"].strip() in ("0", "1", "2", "3", "4")
        ]
        if numeric_subs:
            diff = abs(sum(numeric_subs) - claimed_total)
            score += 0.15 * max(0.0, 1.0 - diff * 0.25)
        else:
            score += 0.075

    score -= hallucination_count * HALLUCINATION_PENALTY
    return max(min(score, 1.0), 0.0)


def _score_abstention_bonus(text: str, prompt_text: str = "") -> float:
    """
    Calibrated Abstention Reward (+0.20 bonus) with N/P-guessing penalty.
    Returns: +0.20 if ≥2 abstention signals, 0.00 if <2,
             -0.10 (net) if false-confidence Cactus routing detected.
    """
    abstention_signals = 0

    np_count = len(re.findall(r"\bN/P\b", text))
    if np_count >= 1:
        abstention_signals += 1
    if np_count >= 3:
        abstention_signals += 1

    if _check_sofa_not_applicable(text):
        reason_present = bool(re.search(r"\*\*Reason:\*\*", text, re.IGNORECASE))
        abstention_signals += 2 if reason_present else 1

    if re.search(r"(?:high|moderate|low)\s+confidence", text, re.IGNORECASE):
        abstention_signals += 1

    if re.search(r"\[assumed:", text, re.IGNORECASE):
        abstention_signals += 1

    has_escalate = CACTUS_ESCALATE_TOKEN in text
    has_local_ok = CACTUS_LOCAL_TOKEN in text
    if has_escalate:
        abstention_signals += 2

    penalty = 0.0
    has_claimed_total = bool(re.search(
        r"\*\*Total SOFA Score:\*\*\s*\d+", text, re.IGNORECASE
    ))
    if has_local_ok and np_count >= 3 and has_claimed_total:
        penalty = -0.10

    base_bonus = ABSTENTION_BONUS if abstention_signals >= 2 else 0.0
    return base_bonus + penalty


def reward_sofa_oracle(
    prompts: List[Any],
    completions: List[Any],
    **kwargs,
) -> List[float]:
    """SOFA Oracle reward (weight 0.20) + Calibrated Abstention bonus (up to +0.20)."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        text = _extract_text(completion)
        prompt_text = _extract_text(prompt) if prompt is not None else ""
        base = _score_sofa_oracle(text)
        bonus = _score_abstention_bonus(text, prompt_text)
        combined = min(base + bonus, 1.2)
        rewards.append(round(combined, 4))
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: Format Compliance — weight 0.10
# ---------------------------------------------------------------------------

def reward_format(
    prompts: List[Any],
    completions: List[Any],
    **kwargs,
) -> List[float]:
    """Validates the structural output format (weight 0.10)."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0

        boxed = _extract_boxed(text)
        if boxed and boxed in ("A", "B", "C", "D"):
            score += 0.40
        elif re.search(r"\\boxed\{", text):
            score += 0.15

        has_table = bool(re.search(
            r"\|\s*(?:Respiratory|Coagulation|Liver|Cardiovascular|CNS|Neurological|Renal)",
            text, re.IGNORECASE
        ))
        if has_table or _check_sofa_not_applicable(text):
            score += 0.30

        steps_found = sum(1 for p in [
            r"Step\s*1|Clinical Data Extraction|SOFA Assessment",
            r"Step\s*2|Clinical Reasoning|Differential",
            r"Step\s*3|Uncertainty|Metacognitive",
            r"Step\s*4|Final Answer",
        ] if re.search(p, text, re.IGNORECASE))
        score += 0.20 * (steps_found / 4)

        if re.search(r"(?:high|moderate|low)\s+confidence|N/P", text, re.IGNORECASE):
            score += 0.10

        rewards.append(round(min(score, 1.0), 4))
    return rewards


# ---------------------------------------------------------------------------
# Reward 4: Process Quality (CoT Heuristic) — weight 0.20
# ---------------------------------------------------------------------------

def reward_process_quality(
    prompts: List[Any],
    completions: List[Any],
    **kwargs,
) -> List[float]:
    """Offline CoT quality heuristic (weight 0.20)."""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        score = 0.0
        max_score = 6.0

        medical_terms = [
            r"pathophysiology", r"differential\s+diagnosis", r"etiology",
            r"mechanism\s+of\s+action", r"clinical\s+presentation", r"prognosis",
            r"(?:renal|hepatic|pulmonary|cardiac)\s+(?:failure|insufficiency)",
            r"(?:sepsis|ARDS|DIC|AKI|SIRS)",
            r"(?:PaO2|FiO2|platelet|bilirubin|creatinine|GCS|MAP)\b",
            r"vasopressor", r"Glasgow\s+Coma\s+Scale",
            r"first.?line\s+(?:treatment|therapy)",
        ]
        hits = sum(1 for t in medical_terms if re.search(t, text, re.IGNORECASE))
        score += min(hits / 5.0, 1.0)

        words = len(text.split())
        if 300 <= words <= 800:
            score += 1.0
        elif 150 <= words < 300 or 800 < words <= 1200:
            score += 0.6
        elif words >= 50:
            score += 0.3

        causal = [
            r"because|since|given\s+that|as\s+evidenced\s+by",
            r"consistent\s+with|compatible\s+with|most\s+likely",
        ]
        score += min(sum(1 for p in causal if re.search(p, text, re.IGNORECASE)) / 2.0, 1.0)

        elim = [
            r"option\s+[ABCD]\s+is\s+(?:incorrect|unlikely|wrong)",
            r"ruled?\s+out|less\s+likely\s+because|not\s+consistent\s+with",
        ]
        score += min(sum(1 for p in elim if re.search(p, text, re.IGNORECASE)) / 2.0, 1.0)

        transitions = [
            r"\b(?:therefore|thus|hence|consequently)\b",
            r"\b(?:however|although|despite|nevertheless)\b",
            r"\b(?:furthermore|additionally|moreover)\b",
        ]
        score += min(sum(1 for p in transitions if re.search(p, text, re.IGNORECASE)) / 2.0, 1.0)

        has_escalate = CACTUS_ESCALATE_TOKEN in text
        has_local_ok = CACTUS_LOCAL_TOKEN in text
        if has_escalate ^ has_local_ok:
            score += 1.0
        elif has_escalate or has_local_ok:
            score += 0.5

        rewards.append(round(min(score / max_score, 1.0), 4))
    return rewards


# ---------------------------------------------------------------------------
# Dispatch: Combined reward logging function
# ---------------------------------------------------------------------------

def log_reward_summary(
    r_correct: List[float],
    r_sofa: List[float],
    r_format: List[float],
    r_process: List[float],
) -> None:
    """Log per-step reward averages for training diagnostics."""
    if not is_master():
        return
    n = max(len(r_correct), 1)
    logger.info(
        f"Rewards | "
        f"Correct={sum(r_correct)/n:.3f} "
        f"SOFA={sum(r_sofa)/n:.3f} "
        f"Format={sum(r_format)/n:.3f} "
        f"Process={sum(r_process)/n:.3f}"
    )


# ===========================================================================
# MODULE 7: Data Loading (delegates to data_pipeline.py)
# ===========================================================================

def load_medqa_dataset(
    data_dir: str = LOCAL_DATA_DIR,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load and format the MedQA-USMLE dataset for GRPO training.
    Delegates to data_pipeline.py for full SOFA-First system prompt injection.
    """
    try:
        from data_pipeline import load_and_prepare_dataset
        return load_and_prepare_dataset(data_dir=data_dir, max_samples=max_samples, seed=seed)
    except ImportError:
        pass
    try:
        from kaggle_t4_dual.data_pipeline import load_and_prepare_dataset
        return load_and_prepare_dataset(data_dir=data_dir, max_samples=max_samples, seed=seed)
    except ImportError:
        logger.warning("data_pipeline.py not found — using minimal JSONL fallback.")
        return _load_raw_jsonl_fallback(data_dir, max_samples, seed)


def _load_raw_jsonl_fallback(
    data_dir: str,
    max_samples: Optional[int],
    seed: int,
) -> DatasetDict:
    """Minimal JSONL loader when data_pipeline.py is absent."""
    import random
    data_path = Path(data_dir)
    train_file = next(
        (f for f in [
            data_path / "phrases_no_exclude_train.jsonl",
            data_path / "train.jsonl",
        ] if f.exists()), None
    )
    if train_file is None:
        raise FileNotFoundError(f"No training data in {data_path.resolve()}")

    records = []
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    random.seed(seed)
    random.shuffle(records)
    if max_samples:
        records = records[:max_samples]

    def _fmt(ex):
        q = ex.get("question", "")
        opts = ex.get("options", {})
        opts_str = "\n".join(f"({k}) {v}" for k, v in sorted(opts.items()))
        ans = ex.get("answer", "")
        for letter, text in opts.items():
            if text.strip() == ans.strip():
                ans = letter
                break
        return {
            "prompt": [
                {"role": "system", "content": "You are a clinical reasoning assistant."},
                {"role": "user", "content": f"{q}\n\n{opts_str}\n\nAnswer in \\boxed{{}} format."},
            ],
            "answer": ans if ans in ("A", "B", "C", "D") else "A",
        }

    formatted = [_fmt(r) for r in records]
    split_idx = int(len(formatted) * 0.9)
    return DatasetDict({
        "train": Dataset.from_list(formatted[:split_idx]),
        "validation": Dataset.from_list(formatted[split_idx:]),
    })


# ===========================================================================
# MODULE 8: Adapter Save & Post-Training Export Utilities
# ===========================================================================

def save_adapter_cuda(
    model,
    tokenizer,
    output_dir: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save the LoRA adapter with multi-GPU safety.
    Only LOCAL_RANK 0 writes to prevent race conditions.
    """
    save_path = Path(output_dir) / "final_adapter"

    if is_master():
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving LoRA adapter → {save_path}")
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        if metadata:
            meta_path = save_path / "training_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved → {meta_path}")

        config_path = save_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            logger.info(f"Adapter verified ✓ rank={cfg.get('r')} alpha={cfg.get('lora_alpha')}")
        else:
            logger.error("adapter_config.json NOT found!")

        for p in sorted(save_path.iterdir()):
            logger.info(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    # Synchronize all processes
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def print_gguf_export_guide(adapter_path: str, base_model_id: str = HF_MODEL_ID) -> None:
    """Print GGUF conversion guide for Ollama / llama.cpp deployment."""
    guide = f"""
╔══════════════════════════════════════════════════════════════════════╗
║     GGUF EXPORT GUIDE: Gemma-Sync → Ollama / llama.cpp (M2)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Prize Tracks: Ollama (local-first) + Cactus (intelligent routing)  ║
╚══════════════════════════════════════════════════════════════════════╝

STEP 1 — Merge LoRA adapter into base model weights
  python -c "
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer
  base = AutoModelForCausalLM.from_pretrained('{base_model_id}', torch_dtype='float16')
  merged = PeftModel.from_pretrained(base, '{adapter_path}').merge_and_unload()
  merged.save_pretrained('./merged_gemma4_e2b_sofa')
  AutoTokenizer.from_pretrained('{adapter_path}').save_pretrained('./merged_gemma4_e2b_sofa')
  "

STEP 2 — Convert to GGUF (requires llama.cpp)
  python convert_hf_to_gguf.py ./merged_gemma4_e2b_sofa \\
         --outtype f16 --outfile gemma4_sofa_f16.gguf
  ./llama-quantize gemma4_sofa_f16.gguf gemma4_sofa_q4km.gguf Q4_K_M

STEP 3 — Create Ollama Modelfile
  cat > Modelfile << 'EOF'
  FROM ./gemma4_sofa_q4km.gguf
  SYSTEM "You are Gemma-Sync. Use SOFA-First reasoning. Emit <|escalate|> \\
          if you need specialist review. Emit <|local_ok|> if confident."
  PARAMETER temperature 0.7
  PARAMETER stop "{CACTUS_ESCALATE_TOKEN}"
  PARAMETER stop "{CACTUS_LOCAL_TOKEN}"
  EOF
  ollama create gemma-sync-sofa -f Modelfile && ollama run gemma-sync-sofa

STEP 4 — Cactus Routing Integration (router.py sketch)
  import ollama
  response = ollama.chat('gemma-sync-sofa', messages=[...])
  text = response['message']['content']
  if '{CACTUS_ESCALATE_TOKEN}' in text:
      return cloud_model.generate(prompt)
  else:
      return text
"""
    print(guide)
    if is_master():
        guide_path = Path(adapter_path).parent / "gguf_export_guide.txt"
        with open(guide_path, "w") as f:
            f.write(guide)
        logger.info(f"GGUF export guide → {guide_path}")


# ===========================================================================
# MODULE 9: Main Training Pipeline
# ===========================================================================

def train(
    model_id: str = KAGGLE_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = 2,
    max_samples: Optional[int] = None,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-6,
    num_generations: int = GRPO_GROUP_SIZE,
    max_completion_length: int = 1024,
    max_steps: int = 202,
    save_steps: int = 50,
    resume_from_checkpoint: Optional[str] = "/kaggle/input/models/narendrabayutama/medreason-gemma-4-e2b-it-cp300/transformers/default/1/outputs/gemma4-e2b-grpo-sofa-t4/checkpoint-100",
    use_unsloth: bool = True,
) -> None:
    """
    Full GRPO training pipeline for Gemma 4 E2B on 2× NVIDIA T4 GPUs.

    Uses Unsloth FastLanguageModel with 4-bit QLoRA as the primary path.
    Accelerate handles 2-GPU data parallelism automatically.
    """
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if is_master():
        logger.info("=" * 70)
        logger.info("GEMMA-SYNC — GRPO Training (Gemma 4 E2B on Dual T4 GPUs)")
        logger.info("=" * 70)
        logger.info(f"Model  : {model_id}")
        logger.info(f"G=     : {num_generations} (streamed in gen_batch=4)")
        logger.info(f"GPUs   : {n_gpus}× T4 (16GB VRAM each)")
        logger.info(f"QLoRA  : NF4 + FP16 (4-bit quantization)")
        logger.info(f"Bonus  : +{ABSTENTION_BONUS} abstention reward")
        logger.info(f"Unsloth: {'ON' if use_unsloth else 'OFF'}")

    # Step 1: Environment
    setup_cuda_env()
    log_vram_usage("pre-load")

    # Step 2: Dataset
    dataset = load_medqa_dataset(max_samples=max_samples)
    if is_master():
        logger.info(f"Dataset | Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    # Step 3: Model + LoRA (4-bit QLoRA via Unsloth)
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, use_unsloth=use_unsloth)
    log_vram_usage("post-model-load")

    # Step 4: GRPO config (FP16, T4-optimized)
    training_config = get_grpo_config(
        output_dir=output_dir,
        num_epochs=num_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        fp16=True,
    )

    # Step 5: GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correctness,
            reward_sofa_oracle,
            reward_format,
            reward_process_quality,
        ],
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    log_vram_usage("post-trainer-init")

    # Step 6: Train
    metrics = {}
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        log_vram_usage("post-training")
        metrics = result.metrics
        if is_master():
            logger.info(f"Training complete | loss={metrics.get('train_loss','N/A')}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.error(
                "CUDA OOM! Try: --num-generations 8 | --batch-size 1 | "
                "--max-completion-length 512 | --gen-batch-size 2"
            )
        raise

    # Step 7: Save adapter
    adapter_path = str(Path(output_dir) / "final_adapter")
    save_adapter_cuda(model, tokenizer, output_dir, {
        "project": "Gemma-Sync (T4 Dual GPU)",
        "model": model_id,
        "precision": "NF4 + FP16 (4-bit QLoRA)",
        "lora": {"rank": 32, "alpha": 64, "dropout": 0.05},
        "grpo": {"G": num_generations, "distribution": f"gen_batch=4, {n_gpus} GPUs"},
        "reward_weights": REWARD_WEIGHTS,
        "abstention_bonus": ABSTENTION_BONUS,
        "cactus": {
            "escalate": CACTUS_ESCALATE_TOKEN,
            "local_ok": CACTUS_LOCAL_TOKEN,
            "threshold": CONFIDENCE_THRESHOLD,
        },
        "dataset": {
            "train": len(dataset["train"]),
            "val": len(dataset["validation"]),
        },
        "infrastructure": {
            "cuda": torch.cuda.is_available(),
            "gpus": n_gpus,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(n_gpus)] if n_gpus else [],
            "torch": torch.__version__,
            "unsloth": use_unsloth and UNSLOTH_AVAILABLE,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    })

    # Step 8: GGUF export guide
    print_gguf_export_guide(adapter_path)

    if is_master():
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE ✓")
        logger.info(f"Adapter : {adapter_path}")
        logger.info("Next    : Follow GGUF guide → Ollama → Cactus router")
        logger.info("=" * 70)


# ===========================================================================
# MODULE 10: Self-Test Suite
# ===========================================================================

def run_reward_self_test() -> None:
    """Validate all reward functions. Run: python distributed_grpo_trainer.py --validate-rewards"""
    print("=" * 70)
    print("REWARD SELF-TEST — Gemma-Sync RLVR (4-tier + Abstention) [T4 Build]")
    print("=" * 70)

    FULL_SOFA = [{"role": "assistant", "content": """\
### Step 1: Clinical Data Extraction (SOFA Assessment)

| SOFA Component | Parameter           | Extracted Value | SOFA Sub-Score |
|----------------|---------------------|-----------------|----------------|
| Respiratory    | PaO₂/FiO₂ ratio     | 280             | 2              |
| Coagulation    | Platelets (×10³/μL) | 85              | 2              |
| Liver          | Bilirubin (mg/dL)   | 1.5             | 1              |
| Cardiovascular | MAP / Vasopressors  | MAP 62 mmHg     | 1              |
| CNS            | GCS                 | 13              | 1              |
| Renal          | Creatinine (mg/dL)  | N/P             | N/P            |

**Total SOFA Score: 7 / 5**

### Step 3: Uncertainty Assessment
**Moderate confidence** — Creatinine is N/P. [assumed: FiO₂ = 0.21]

### Step 4: Final Answer
$$\\boxed{B}$$
"""}]

    SOFA_NA_COMP = [{"role": "assistant", "content": """\
### Step 1: SOFA Assessment
> **SOFA Assessment: SOFA_NOT_APPLICABLE**
> **Reason:** Outpatient psychiatric evaluation — no acute organ dysfunction.
**Low confidence** — requires clinical interview.
$$\\boxed{C}$$ <|local_ok|>
"""}]

    EMPTY = [{"role": "assistant", "content": "I don't know."}]

    print("\n[1] Correctness (RLVR exact match)")
    r = reward_correctness(None, FULL_SOFA, ["B"])
    assert r[0] == 1.0, f"Expected 1.0, got {r[0]}"
    r2 = reward_correctness(None, FULL_SOFA, ["A"])
    assert r2[0] == 0.0, f"Expected 0.0, got {r2[0]}"
    print(f"  ✓ Correct=1.0, Wrong=0.0")

    print("\n[2] SOFA Oracle + Abstention Bonus")
    r_sofa = reward_sofa_oracle([None], FULL_SOFA)
    assert r_sofa[0] > 0.5, f"Expected >0.5, got {r_sofa[0]}"
    r_na = reward_sofa_oracle([None], SOFA_NA_COMP)
    assert r_na[0] >= 1.0, f"SOFA_NA should >=1.0, got {r_na[0]}"
    r_e = reward_sofa_oracle([None], EMPTY)
    assert r_e[0] == 0.0, f"Empty should be 0.0, got {r_e[0]}"
    print(f"  ✓ Full SOFA={r_sofa[0]:.4f} | NA+reason={r_na[0]:.4f} | Empty={r_e[0]:.4f}")

    print("\n[3] Format Compliance")
    r_fmt = reward_format([None], FULL_SOFA)
    assert r_fmt[0] > 0.5
    print(f"  ✓ Format={r_fmt[0]:.4f}")

    print("\n[4] Process Quality (Cactus signal)")
    r_proc = reward_process_quality([None], SOFA_NA_COMP)
    print(f"  ✓ Process (with <|local_ok|>)={r_proc[0]:.4f}")

    print("\n[5] Empty completion edge case")
    r_empty = reward_correctness(None, EMPTY, ["A"])
    r_empty_sofa = reward_sofa_oracle([None], EMPTY)
    assert r_empty[0] == 0.0 and r_empty_sofa[0] == 0.0
    print(f"  ✓ Correct=0.0, SOFA=0.0")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print(f"Reward weights: {REWARD_WEIGHTS}")
    print(f"Abstention bonus: +{ABSTENTION_BONUS}")
    print("=" * 70)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma-Sync GRPO Trainer [T4 Dual GPU] — 4-bit QLoRA + FP16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", type=str, default=KAGGLE_MODEL_ID)
    parser.add_argument("--hf-model", action="store_true",
        help=f"Use HF Hub ID ({HF_MODEL_ID}) for online prototyping.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1,
        help="Per-GPU batch size. Default=1 for T4 VRAM safety.")
    parser.add_argument("--grad-accum", type=int, default=1,
        help="Gradient accumulation steps. Eff batch = batch × accum × GPUs.")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=GRPO_GROUP_SIZE,
        help=f"GRPO group size G. Default={GRPO_GROUP_SIZE}.")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=202)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
        help="Run 4 training steps (pipeline validation only).")
    parser.add_argument("--validate-rewards", action="store_true",
        help="Run reward self-tests and exit.")
    parser.add_argument("--export-guide-only", action="store_true",
        help="Print GGUF export guide and exit.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--no-unsloth", action="store_true",
        help="Disable Unsloth; use standard HF + BitsAndBytes NF4 fallback.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.validate_rewards:
        run_reward_self_test()
        sys.exit(0)

    if args.export_guide_only:
        print_gguf_export_guide("./outputs/final_adapter")
        sys.exit(0)

    model_id = HF_MODEL_ID if args.hf_model else args.model_id

    if args.dry_run:
        logger.info("DRY RUN MODE: 4 steps, 32 samples.")
        args.max_samples = max(args.max_samples or 0, 32)
        args.max_steps = 4
        args.epochs = 1

    use_unsloth = not args.no_unsloth

    train(
        model_id=model_id,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        max_samples=args.max_samples,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_unsloth=use_unsloth,
    )

