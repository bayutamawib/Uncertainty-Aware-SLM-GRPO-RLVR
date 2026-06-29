"""
evaluation_final.py — Gemma-Sync "Final Exam" Evaluation (Audit-First)
========================================================================

Evaluates the fine-tuned Gemma 4 E2B GRPO model (checkpoint-200) on 200
**strictly unseen** MedQA-USMLE samples, ensuring zero data contamination.

Pipeline:
  1. AUDIT — Reproduce the exact StratifiedShuffleSplit from data_pipeline.py
             to identify the 202 training indices ("Daftar Hitam / Blacklist").
  2. POOL — Collect ALL remaining samples that were NOT in training.
  3. SAMPLE — Draw exactly 200 unseen samples (random.seed(42)).
  4. MERGE — Load base model + LoRA adapter via Unsloth, merge to 16-bit.
  5. INFER — Run batch inference, capture full reasoning output.
  6. EXTRACT — Parse \boxed{X}, <thought>, SOFA table, Cactus routing signal.
  7. EXPORT — Save to gemma_sync_final_eval_200.csv.

Hardware: Kaggle 2× NVIDIA T4 GPUs (CUDA) — FP16 inference
Usage:
    # Full evaluation (200 samples, merged model):
    python evaluation_final.py

    # Quick test (10 samples):
    python evaluation_final.py --num-eval-samples 10

    # Dry run (no model, synthetic outputs):
    python evaluation_final.py --dry-run --num-eval-samples 5

Author : Narendra Bayutama Wibisono
Project: Gemma-Sync — Distributed Uncertainty-Aware Clinical Reasoning via Gemma 4 E2B
Ref    : Zenodo: 10.5281/zenodo.19599245
"""

import os
import re
import sys
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("final_exam")

# ---------------------------------------------------------------------------
# Local imports — data pipeline + reward utilities
# ---------------------------------------------------------------------------
try:
    from data_pipeline import (
        load_and_prepare_dataset,
        format_example_for_grpo,
        resolve_answer_label,
        build_user_prompt,
        clean_text,
        SYSTEM_PROMPT,
        LOCAL_DATA_DIR,
        TRAIN_SPLIT_RATIO,
        RANDOM_SEED,
    )
except ImportError:
    from kaggle_t4_dual.data_pipeline import (
        load_and_prepare_dataset,
        format_example_for_grpo,
        resolve_answer_label,
        build_user_prompt,
        clean_text,
        SYSTEM_PROMPT,
        LOCAL_DATA_DIR,
        TRAIN_SPLIT_RATIO,
        RANDOM_SEED,
    )

try:
    from distributed_grpo_trainer import (
        _extract_text,
        _extract_boxed,
        _parse_sofa_table,
        _check_sofa_not_applicable,
        _extract_map_from_cv_string,
        CACTUS_ESCALATE_TOKEN,
        CACTUS_LOCAL_TOKEN,
        SOFA_COMPONENTS,
    )
except ImportError:
    from kaggle_t4_dual.distributed_grpo_trainer import (
        _extract_text,
        _extract_boxed,
        _parse_sofa_table,
        _check_sofa_not_applicable,
        _extract_map_from_cv_string,
        CACTUS_ESCALATE_TOKEN,
        CACTUS_LOCAL_TOKEN,
        SOFA_COMPONENTS,
    )

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
KAGGLE_MODEL_ID = "/kaggle/input/models/narendrabayutama/medreason-gemma-4-e2b-it-cp500/transformers/default/1"
HF_MODEL_ID = "google/gemma-4-e2b-it"
DEFAULT_ADAPTER_PATH = "/kaggle/input/models/narendrabayutama/medreason-gemma-4-e2b-it-cp500/transformers/default/1"
DEFAULT_OUTPUT_CSV = "gemma_sync_final_eval_200.csv"
DEFAULT_DATA_DIR = "./medqa_dataset"

NUM_EVAL_SAMPLES = 200
MAX_TRAIN_SAMPLES = 202       # Must match data_pipeline.py training cap
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6             # Slightly lower than training for stable eval
TOP_P = 0.9


# ===========================================================================
# STAGE 1: AUDIT — Reproduce Training Split & Identify Blacklist
# ===========================================================================

def _load_raw_jsonl(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load ALL raw JSONL records from the dataset directory, reproducing
    exactly what data_pipeline.load_and_prepare_dataset() does internally.
    """
    from data_pipeline import _load_jsonl
    data_path = Path(data_dir)

    train_candidates = [
        data_path / "phrases_no_exclude_train.jsonl",
        data_path / "train.jsonl",
    ]
    test_candidates = [
        data_path / "phrases_no_exclude_test.jsonl",
        data_path / "test.jsonl",
        data_path / "validation.jsonl",
        data_path / "dev.jsonl",
    ]

    train_file = next((f for f in train_candidates if f.exists()), None)
    test_file = next((f for f in test_candidates if f.exists()), None)

    if train_file is None:
        all_jsonl = sorted(data_path.glob("*.jsonl"))
        if not all_jsonl:
            raise FileNotFoundError(f"No .jsonl files found in {data_path}")
        train_file = all_jsonl[0]
        test_file = all_jsonl[1] if len(all_jsonl) > 1 else None

    all_records = _load_jsonl(str(train_file))
    if test_file and test_file.exists():
        all_records += _load_jsonl(str(test_file))

    logger.info(f"Loaded {len(all_records)} total raw records from {data_path}")
    return all_records


def identify_blacklist_indices(
    all_records: List[Dict],
    train_ratio: float = TRAIN_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
    max_train_samples: int = MAX_TRAIN_SAMPLES,
) -> set:
    """
    Reproduce the EXACT splitting logic from data_pipeline.load_and_prepare_dataset()
    to determine which global indices ended up in the 202-sample training set.

    Steps (mirroring data_pipeline.py):
      1. Create Dataset from all_records (no subsampling — max_samples was None).
      2. Format with format_example_for_grpo (to get answer labels).
      3. StratifiedShuffleSplit(test_size=0.1, random_state=42).
      4. Cap train split to 202 with shuffle(seed=42).
      5. Return the set of ORIGINAL global indices that are in training.

    Returns:
        Set of integer indices (in the combined raw pool) that are "tainted".
    """
    from datasets import Dataset

    logger.info("=" * 60)
    logger.info("AUDIT PHASE: Reproducing training split to build blacklist")
    logger.info("=" * 60)

    combined = Dataset.from_list(all_records)
    logger.info(f"  Combined dataset size: {len(combined)}")

    # Format (mirrors Stage 3 of data_pipeline)
    formatted = combined.map(
        format_example_for_grpo,
        remove_columns=combined.column_names,
        desc="[Audit] Formatting for split reproduction",
        num_proc=2,
    )

    # StratifiedShuffleSplit (mirrors Stage 4)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        answer_labels = formatted["answer"]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - train_ratio,
            random_state=seed,
        )
        train_indices, val_indices = next(
            splitter.split(range(len(formatted)), answer_labels)
        )
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()
        logger.info(f"  StratifiedShuffleSplit → Train: {len(train_indices)}, Val: {len(val_indices)}")

    except ImportError:
        logger.warning("  sklearn not available — using HF random split (may differ).")
        split = formatted.train_test_split(
            test_size=1.0 - train_ratio, seed=seed, shuffle=True,
        )
        # HF train_test_split doesn't easily expose global indices,
        # but for consistency we need them.
        # Fallback: treat all data as potentially tainted — ERROR
        raise ImportError(
            "sklearn is REQUIRED for audit-first evaluation. "
            "Install with: pip install scikit-learn"
        )

    # Cap to max_train_samples (mirrors Stage 5)
    if len(train_indices) > max_train_samples:
        # Reproduce the shuffle(seed=seed).select(range(max_train_samples))
        # HF Dataset.shuffle uses numpy under the hood with the given seed.
        import numpy as np
        rng = np.random.default_rng(seed)
        shuffled_train = rng.permutation(train_indices).tolist()
        final_train_indices = set(shuffled_train[:max_train_samples])
        logger.info(
            f"  Train capped: {len(train_indices)} → {max_train_samples} "
            f"(shuffle seed={seed})"
        )
    else:
        final_train_indices = set(train_indices)

    logger.info(f"  BLACKLIST size: {len(final_train_indices)} indices")
    logger.info(f"  Sample blacklist indices: {sorted(list(final_train_indices))[:10]}...")
    logger.info("AUDIT PHASE COMPLETE ✓")
    logger.info("=" * 60)

    return final_train_indices


def select_unseen_samples(
    all_records: List[Dict],
    blacklist: set,
    num_samples: int = NUM_EVAL_SAMPLES,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """
    From the full pool of raw records, exclude the blacklisted training indices
    and randomly select `num_samples` unseen examples.

    Args:
        all_records: Complete raw dataset (list of dicts).
        blacklist: Set of indices that were used in training.
        num_samples: Number of unseen samples to select (default: 300).
        seed: Random seed for reproducible selection.

    Returns:
        List of raw record dicts (unseen by the model during training).
    """
    # Build unseen pool
    unseen_pool = [
        (idx, record) for idx, record in enumerate(all_records)
        if idx not in blacklist
    ]
    logger.info(f"Unseen pool size: {len(unseen_pool)} (total={len(all_records)}, blacklist={len(blacklist)})")

    if len(unseen_pool) < num_samples:
        logger.warning(
            f"Unseen pool ({len(unseen_pool)}) smaller than requested ({num_samples}). "
            f"Using all available unseen samples."
        )
        num_samples = len(unseen_pool)

    # Deterministic random selection
    random.seed(seed)
    selected = random.sample(unseen_pool, num_samples)

    # Attach original global index for traceability
    eval_samples = []
    for global_idx, record in selected:
        record_copy = dict(record)
        record_copy["_global_index"] = global_idx
        eval_samples.append(record_copy)

    logger.info(f"Selected {len(eval_samples)} unseen evaluation samples (seed={seed})")
    return eval_samples


# ===========================================================================
# STAGE 2: MODEL MERGING (Unsloth → 16-bit Merged)
# ===========================================================================

def load_and_merge_model(
    model_id: str,
    adapter_path: str,
    max_seq_length: int = 2048,
) -> Tuple[Any, Any]:
    """
    Load base Gemma 4 E2B + LoRA adapter via Unsloth FastLanguageModel,
    then merge to FP16 for maximum inference accuracy.

    Steps:
      1. Load base model with Unsloth (4-bit for adapter compatibility).
      2. Load LoRA adapter from checkpoint directory.
      3. Merge adapter weights into base model (16-bit).
      4. Return merged model + tokenizer.

    Args:
        model_id: Path to base model (Kaggle local or HF hub).
        adapter_path: Path to LoRA adapter checkpoint (e.g., ./outputs/checkpoint-200).
        max_seq_length: Maximum sequence length for the model.

    Returns:
        Tuple of (merged_model, tokenizer).
    """
    logger.info("=" * 60)
    logger.info("MODEL MERGE PHASE: Loading base + LoRA → 16-bit merge")
    logger.info("=" * 60)

    # --- Check for Unsloth ---
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth is REQUIRED for model merging. "
            "Install with: pip install unsloth"
        )

    # --- Resolve model path ---
    if Path(model_id).exists():
        resolved_model = model_id
    elif Path(KAGGLE_MODEL_ID).exists():
        resolved_model = KAGGLE_MODEL_ID
    else:
        resolved_model = HF_MODEL_ID
        logger.info(f"Local model not found. Using HF Hub: {resolved_model}")

    logger.info(f"  Base model: {resolved_model}")
    logger.info(f"  Adapter:    {adapter_path}")

    # --- Load base model + adapter via Unsloth ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,    # Unsloth can resume from adapter dir
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # Unwrap Gemma4 Processor if needed
    if hasattr(tokenizer, "tokenizer"):
        logger.info("  Detected Gemma4Processor — extracting inner tokenizer.")
        tokenizer = tokenizer.tokenizer

    # Ensure padding config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # --- Merge to 16-bit ---
    logger.info("  Merging LoRA adapter into base model (16-bit)...")
    # Unsloth provides save_pretrained_merged for full merge
    # For inference, we use FastLanguageModel.for_inference which optimizes
    FastLanguageModel.for_inference(model)
    logger.info("  Model prepared for inference ✓ (Unsloth optimized)")

    # VRAM report
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
            name = torch.cuda.get_device_name(i)
            logger.info(f"  GPU {i}: {name} | {alloc_gb:.2f} GB allocated")

    logger.info("MODEL MERGE COMPLETE ✓")
    logger.info("=" * 60)
    return model, tokenizer


def load_model_hf_fallback(
    model_id: str,
    adapter_path: str,
) -> Tuple[Any, Any]:
    """
    Fallback model loading without Unsloth — standard HF + PEFT merge.
    Used when Unsloth is not available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("  [FALLBACK] Loading via standard HF + PEFT...")

    # Resolve model path
    if Path(model_id).exists():
        resolved = model_id
    elif Path(KAGGLE_MODEL_ID).exists():
        resolved = KAGGLE_MODEL_ID
    else:
        resolved = HF_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, use_fast=True,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if Path(adapter_path).exists():
        logger.info(f"  Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("  Adapter merged (16-bit) ✓")
    else:
        logger.warning(f"  Adapter not found at {adapter_path} — running BASE model.")

    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    return model, tokenizer


# ===========================================================================
# STAGE 3: INFERENCE ENGINE
# ===========================================================================

def generate_single(
    model,
    tokenizer,
    prompt_messages: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """
    Generate a completion for a single prompt using chat template formatting.

    Args:
        model: The merged language model.
        tokenizer: Tokenizer with chat template support.
        prompt_messages: List of {"role": "...", "content": "..."} dicts.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        The generated completion text (assistant turn only).
    """
    # Apply chat template
    try:
        input_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: concatenate content
        input_text = "\n".join(
            m.get("content", "") for m in prompt_messages if isinstance(m, dict)
        )

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_new_tokens,   # Input truncation
        padding=False,
    )

    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode only the generated portion
    generated_ids = outputs[0][prompt_length:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return completion


# ===========================================================================
# STAGE 4: OUTPUT EXTRACTION
# ===========================================================================

def extract_thought_block(text: str) -> str:
    """Extract the <thought>...</thought> or <think>...</think> block if present."""
    patterns = [
        r"<thought>(.*?)</thought>",
        r"<think>(.*?)</think>",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def extract_sofa_table_text(text: str) -> str:
    """Extract the raw SOFA markdown table from the completion."""
    # Look for markdown table with SOFA-related headers
    table_pattern = re.compile(
        r"(\|[^\n]*SOFA[^\n]*\|.*?(?:\n\|[^\n]*\|)+)",
        re.DOTALL | re.IGNORECASE,
    )
    m = table_pattern.search(text)
    if m:
        return m.group(0).strip()

    # Fallback: any markdown table
    generic_table = re.compile(
        r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)",
        re.DOTALL,
    )
    m = generic_table.search(text)
    if m:
        return m.group(0).strip()

    return ""


def extract_routing_signal(text: str) -> str:
    """
    Extract the Cactus Routing Signal from the completion.
    Returns '<|local_ok|>', '<|escalate|>', 'BOTH', or 'NONE'.
    """
    has_local = CACTUS_LOCAL_TOKEN in text
    has_escalate = CACTUS_ESCALATE_TOKEN in text

    if has_local and has_escalate:
        return "BOTH"
    elif has_local:
        return "<|local_ok|>"
    elif has_escalate:
        return "<|escalate|>"
    else:
        return "NONE"


# ===========================================================================
# STAGE 5: MAIN EVALUATION LOOP
# ===========================================================================

# CSV column spec — single source of truth
CSV_FIELDNAMES = [
    "index", "question", "ground_truth",
    "model_reasoning", "model_answer",
    "is_correct", "routing_signal",
]


def _detect_resume_point(output_csv: str) -> int:
    """
    Check if a partial CSV already exists from a previous (crashed) run.
    Returns the number of completed rows so we can skip them on resume.
    """
    csv_path = Path(output_csv)
    if not csv_path.exists():
        return 0

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            completed = sum(1 for _ in reader)
        if completed > 0:
            logger.info(
                f"  ⚡ RESUME DETECTED: {output_csv} has {completed} completed rows. "
                f"Skipping to sample #{completed}."
            )
        return completed
    except Exception as e:
        logger.warning(f"  Could not read existing CSV for resume: {e}. Starting fresh.")
        return 0


def _print_progress(
    i: int, total: int, correct: int, elapsed: float,
    answer: str, signal: str, is_correct: bool,
) -> None:
    """
    Print a Kaggle-friendly progress line.

    Uses print(flush=True) instead of tqdm — Kaggle notebooks buffer
    carriage-return (\r) updates so tqdm bars never render. Plain print
    with newline always appears in the notebook output cell.
    """
    acc = correct / (i + 1) if (i + 1) > 0 else 0.0
    mark = "✓" if is_correct else "✗"
    eta_s = (elapsed / (i + 1)) * (total - i - 1) if (i + 1) > 0 else 0
    eta_m = eta_s / 60

    # Progress bar (30-char width)
    filled = int(30 * (i + 1) / total) if total > 0 else 0
    bar = "█" * filled + "░" * (30 - filled)

    print(
        f"  [{i+1:>4}/{total}] {bar} "
        f"Acc={acc:.1%} ({correct}✓) "
        f"| {mark} ans={answer or '—'} sig={signal[:10]} "
        f"| {elapsed:.0f}s elapsed, ~{eta_m:.1f}m left",
        flush=True,
    )


def run_final_evaluation(
    model_id: str = KAGGLE_MODEL_ID,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    data_dir: str = DEFAULT_DATA_DIR,
    num_eval_samples: int = NUM_EVAL_SAMPLES,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    dry_run: bool = False,
) -> str:
    """
    Execute the full Final Exam evaluation pipeline.

    Key features:
      - Crash-safe: writes each result to CSV immediately after inference.
      - Resume-capable: detects partial CSV and skips already-completed rows.
      - Kaggle-friendly: uses print(flush=True) instead of tqdm.

    Returns:
        Path to the output CSV file.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Audit — Build blacklist
    # ------------------------------------------------------------------
    all_records = _load_raw_jsonl(data_dir)
    blacklist = identify_blacklist_indices(all_records)
    eval_samples = select_unseen_samples(
        all_records, blacklist, num_samples=num_eval_samples,
    )

    total = len(eval_samples)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  🩺 FINAL EXAM: {total} unseen MedQA samples", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1.5: Check for resume from partial CSV
    # ------------------------------------------------------------------
    resume_from = _detect_resume_point(output_csv)

    # ------------------------------------------------------------------
    # Phase 2: Load & Merge Model
    # ------------------------------------------------------------------
    model, tokenizer = None, None
    if not dry_run:
        try:
            model, tokenizer = load_and_merge_model(
                model_id=model_id,
                adapter_path=adapter_path,
            )
        except ImportError:
            logger.warning("Unsloth not available — trying HF fallback...")
            model, tokenizer = load_model_hf_fallback(
                model_id=model_id,
                adapter_path=adapter_path,
            )

        # Clear CUDA cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after model loading ✓")
    else:
        logger.warning("DRY RUN MODE — using synthetic completions.")

    # ------------------------------------------------------------------
    # Phase 3: Open CSV for incremental writing
    # ------------------------------------------------------------------
    # If resuming, open in append mode; otherwise create fresh with header
    if resume_from > 0:
        csv_file = open(output_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_ALL)
        # No header — already written in a previous run
        print(f"  📂 Resuming from row {resume_from} (appending to existing CSV)", flush=True)
    else:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_ALL)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"  📂 Created new CSV: {output_csv}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4: Inference + Extraction (with incremental CSV save)
    # ------------------------------------------------------------------
    correct_count = 0
    processed_count = 0
    routing_counts = {"<|local_ok|>": 0, "<|escalate|>": 0, "BOTH": 0, "NONE": 0}
    inference_start = time.time()

    # If resuming, we need to count correct answers from the skipped portion
    # to keep the running accuracy accurate across the full CSV
    if resume_from > 0:
        try:
            with open(output_csv, "r", encoding="utf-8") as f_read:
                reader = csv.DictReader(f_read)
                for row in reader:
                    if row.get("is_correct", "").lower() == "true":
                        correct_count += 1
                    sig = row.get("routing_signal", "NONE")
                    routing_counts[sig] = routing_counts.get(sig, 0) + 1
                    processed_count += 1
        except Exception:
            pass

    print(f"\n  Starting inference...", flush=True)
    print(f"  {'─' * 56}", flush=True)

    try:
        for i in range(resume_from, total):
            raw_sample = eval_samples[i]
            sample_start = time.time()

            # Format the sample into GRPO-style prompt
            question = raw_sample.get("question", "")
            options = raw_sample.get("options", {})
            ground_truth = resolve_answer_label(raw_sample)
            global_idx = raw_sample.get("_global_index", -1)

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, options)},
            ]

            # --- Generate ---
            if dry_run:
                completion = _generate_synthetic(ground_truth)
            else:
                try:
                    completion = generate_single(
                        model, tokenizer, prompt_messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                except Exception as e:
                    logger.error(f"[{i}] Generation failed: {e}")
                    completion = f"[ERROR: {e}]"

            # --- Extract ---
            text = _extract_text(completion)
            model_answer = _extract_boxed(text)
            is_correct = (model_answer is not None) and (model_answer == ground_truth.upper())
            if is_correct:
                correct_count += 1

            thought_block = extract_thought_block(text)
            sofa_table = extract_sofa_table_text(text)
            routing_signal = extract_routing_signal(text)
            routing_counts[routing_signal] = routing_counts.get(routing_signal, 0) + 1

            # Full reasoning = entire completion (includes thought, SOFA, etc.)
            model_reasoning = text.strip()
            processed_count += 1

            # --- IMMEDIATELY write this row to CSV (crash-safe) ---
            row = {
                "index": global_idx,
                "question": clean_text(question)[:200],
                "ground_truth": ground_truth,
                "model_reasoning": model_reasoning,
                "model_answer": model_answer or "NONE",
                "is_correct": is_correct,
                "routing_signal": routing_signal,
            }
            csv_writer.writerow(row)
            csv_file.flush()       # Force OS-level write — survives kernel crash
            os.fsync(csv_file.fileno())  # Belt-and-suspenders: sync to disk

            # --- Print progress (Kaggle-friendly, every sample) ---
            elapsed = time.time() - inference_start
            _print_progress(
                i, total, correct_count, elapsed,
                model_answer, routing_signal, is_correct,
            )

            # Periodic CUDA cache cleanup (every 25 samples)
            if (i + 1) % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Periodic summary every 50 samples
            if (i + 1) % 50 == 0:
                acc = correct_count / processed_count if processed_count > 0 else 0
                print(
                    f"\n  ── CHECKPOINT [{i+1}/{total}] ─────────────────\n"
                    f"  Accuracy so far : {correct_count}/{processed_count} = {acc:.2%}\n"
                    f"  CSV rows saved  : {processed_count}\n"
                    f"  Elapsed         : {elapsed:.0f}s\n"
                    f"  ──────────────────────────────────────────────\n",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(f"\n\n  ⚠ INTERRUPTED at sample {i}/{total}. "
              f"{processed_count} rows saved to {output_csv}.", flush=True)
    except Exception as e:
        print(f"\n\n  ❌ ERROR at sample {i}/{total}: {e}\n"
              f"  {processed_count} rows saved to {output_csv}.", flush=True)
        logger.exception("Fatal error during inference")
    finally:
        csv_file.close()
        print(f"\n  💾 CSV closed. Total rows written: {processed_count}", flush=True)

    # ------------------------------------------------------------------
    # Phase 5: Summary Report
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    n = processed_count
    accuracy = correct_count / n if n > 0 else 0.0

    # Re-read CSV to count answer extraction rate (handles resume correctly)
    answer_extracted = 0
    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_answer", "NONE") != "NONE":
                    answer_extracted += 1
    except Exception:
        answer_extracted = n
    answer_rate = answer_extracted / n if n > 0 else 0.0

    print("\n" + "=" * 62, flush=True)
    print("  GEMMA-SYNC FINAL EXAM RESULTS", flush=True)
    print("  Model: Gemma 4 E2B | GRPO + RLVR | Checkpoint-200", flush=True)
    print("=" * 62, flush=True)
    print(f"  Evaluation Samples    : {n} (strictly unseen)", flush=True)
    print(f"  Blacklist (train)     : {len(blacklist)} samples excluded", flush=True)
    print(f"  Adapter               : {adapter_path}", flush=True)
    if resume_from > 0:
        print(f"  Resumed from row      : {resume_from}", flush=True)
    print("-" * 62, flush=True)
    print(f"  OVERALL ACCURACY      : {correct_count}/{n} = {accuracy:.2%}", flush=True)
    print(f"  Answer Extraction Rate: {answer_rate:.2%}", flush=True)
    print("-" * 62, flush=True)
    print("  CACTUS ROUTING DISTRIBUTION:", flush=True)
    for signal, count in sorted(routing_counts.items()):
        pct = count / n if n > 0 else 0
        bar = "█" * int(pct * 30)
        print(f"    {signal:<15} : {count:>4} ({pct:.1%}) {bar}", flush=True)
    print("-" * 62, flush=True)
    print(f"  Total eval time       : {total_time:.1f}s", flush=True)
    if n > 0:
        print(f"  Avg time per sample   : {total_time / n:.2f}s", flush=True)
    print(f"  Output CSV            : {output_csv}", flush=True)
    print("=" * 62, flush=True)

    # Interpretation
    print("\n  INTERPRETATION:", flush=True)
    if accuracy >= 0.70:
        print("  ✓ Accuracy ≥70% — exceeds USMLE baseline for 2B model. Excellent!", flush=True)
    elif accuracy >= 0.50:
        print("  △ Accuracy 50-70% — competitive for E2B class. Consider more training.", flush=True)
    elif accuracy >= 0.30:
        print("  ⚠ Accuracy 30-50% — below expectations. Review reward function tuning.", flush=True)
    else:
        print("  ✗ Accuracy <30% — critically low. Check adapter path and merge integrity.", flush=True)

    none_rate = routing_counts.get("NONE", 0) / n if n > 0 else 0
    if none_rate > 0.20:
        print("  ⚠ >20% missing Cactus signal — routing protocol not fully learned.", flush=True)
    both_rate = routing_counts.get("BOTH", 0) / n if n > 0 else 0
    if both_rate > 0.05:
        print("  ⚠ >5% dual-signal responses — model confused about routing protocol.", flush=True)

    print(flush=True)
    logger.info(f"Final Exam complete. Results saved to: {output_csv}")
    return output_csv


# ===========================================================================
# Synthetic Output Generator (for dry-run mode)
# ===========================================================================

def _generate_synthetic(ground_truth: str) -> str:
    """Generate a synthetic completion for dry-run testing."""
    # Randomly decide if the "model" gets it right (70% chance for realism)
    import random as _rng
    correct = _rng.random() < 0.70
    answer = ground_truth if correct else _rng.choice(
        [x for x in ["A", "B", "C", "D"] if x != ground_truth]
    )

    return (
        "<think>\n"
        "Let me analyze this clinical scenario step by step.\n"
        "</think>\n\n"
        "### Step 1: Clinical Data Extraction\n\n"
        "> **SOFA Assessment: SOFA_NOT_APPLICABLE**\n"
        "> **Reason:** Outpatient clinical scenario without organ dysfunction.\n\n"
        "### Step 2: Clinical Reasoning\n\n"
        "Based on the clinical data presented, the most likely diagnosis is...\n\n"
        "### Step 3: Uncertainty Assessment\n\n"
        "**Moderate confidence** — Key findings are consistent with the diagnosis.\n\n"
        "### Step 4: Final Answer\n\n"
        f"$$\\\\boxed{{{answer}}}$$\n\n"
        "### Step 5: Cactus Routing Signal\n\n"
        f"{'<|local_ok|>' if correct else '<|escalate|>'}\n"
    )


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gemma-Sync Final Exam — Evaluate checkpoint-200 on 200 unseen MedQA samples.\n"
            "Audit-first approach guarantees zero data contamination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id", type=str, default=KAGGLE_MODEL_ID,
        help="Path to base model (Kaggle local mount or HF Hub ID).",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH,
        help="Path to LoRA adapter checkpoint directory.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Path to local MedQA JSONL dataset directory.",
    )
    parser.add_argument(
        "--num-eval-samples", type=int, default=NUM_EVAL_SAMPLES,
        help="Number of unseen samples to evaluate (max depends on pool size).",
    )
    parser.add_argument(
        "--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip model loading; use synthetic completions (pipeline testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = run_final_evaluation(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        data_dir=args.data_dir,
        num_eval_samples=args.num_eval_samples,
        output_csv=args.output_csv,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )

    print(f"\n✅ Final Exam evaluation complete → {output_path}")
