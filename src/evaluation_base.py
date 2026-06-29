"""
evaluation_base.py — Gemma-Sync BASE MODEL Evaluation (Ablation Baseline)
==========================================================================

Evaluates the PURE BASE Gemma 4 E2B model (NO LoRA adapter, NO fine-tuning)
on the same 200 **strictly unseen** MedQA-USMLE samples used in the final
evaluation, enabling a paired 1:1 comparison against the fine-tuned model.

This script is a controlled variant of evaluation_final.py. The ONLY
difference is that the LoRA adapter loading/merging is removed. Everything
else — dataset splitting, prompt formatting, extraction logic, random seeds
— is IDENTICAL to ensure a valid paired test.

Pipeline:
  1. AUDIT — Reproduce the exact StratifiedShuffleSplit from data_pipeline.py
             to identify the 202 training indices ("Daftar Hitam / Blacklist").
  2. POOL — Collect ALL remaining samples that were NOT in training.
  3. SAMPLE — Draw exactly 200 unseen samples (random.seed(42)).
  4. LOAD — Load BASE model via Unsloth (NO adapter merge).
  5. INFER — Run batch inference, capture full reasoning output.
  6. EXTRACT — Parse \\boxed{X}, <thought>, SOFA table, Cactus routing signal.
  7. EXPORT — Save to gemma_sync_BASE_eval_{N}.csv.

Hardware: Kaggle 2× NVIDIA T4 GPUs (CUDA) — FP16 inference
Usage:
    # Full baseline evaluation (200 samples, base model ONLY):
    python evaluation_base.py

    # Quick test (10 samples):
    python evaluation_base.py --num-eval-samples 10

    # Dry run (no model, synthetic outputs):
    python evaluation_base.py --dry-run --num-eval-samples 5

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
logger = logging.getLogger("base_eval")

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
DEFAULT_OUTPUT_CSV = "gemma_sync_BASE_eval_200.csv"
DEFAULT_DATA_DIR = "./medqa_dataset"

NUM_EVAL_SAMPLES = 200
MAX_TRAIN_SAMPLES = 202       # Must match data_pipeline.py training cap
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6             # Identical to evaluation_final.py for fair comparison
TOP_P = 0.9


# ===========================================================================
# STAGE 1: AUDIT — Reproduce Training Split & Identify Blacklist
# ===========================================================================
# NOTE: This section is IDENTICAL to evaluation_final.py — do NOT modify.
# Any change here breaks the paired test guarantee.

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
        raise ImportError(
            "sklearn is REQUIRED for audit-first evaluation. "
            "Install with: pip install scikit-learn"
        )

    # Cap to max_train_samples (mirrors Stage 5)
    if len(train_indices) > max_train_samples:
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

    IDENTICAL to evaluation_final.py — same seed, same logic, same output.
    """
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

    # Deterministic random selection — MUST be identical to evaluation_final.py
    random.seed(seed)
    selected = random.sample(unseen_pool, num_samples)

    eval_samples = []
    for global_idx, record in selected:
        record_copy = dict(record)
        record_copy["_global_index"] = global_idx
        eval_samples.append(record_copy)

    logger.info(f"Selected {len(eval_samples)} unseen evaluation samples (seed={seed})")
    return eval_samples


# ===========================================================================
# STAGE 2: BASE MODEL LOADING (NO Adapter — Pure Base Model)
# ===========================================================================

def load_base_model(
    model_id: str,
    max_seq_length: int = 2048,
) -> Tuple[Any, Any]:
    """
    Load the PURE BASE Gemma 4 E2B model via Unsloth FastLanguageModel.
    NO LoRA adapter is loaded or merged — this is the unmodified foundation
    model for ablation/baseline comparison.

    Steps:
      1. Resolve model path (Kaggle local → HF Hub fallback).
      2. Load base model with Unsloth (4-bit quantization for T4 VRAM).
      3. Apply FastLanguageModel.for_inference() for optimized generation.
      4. Return model + tokenizer.

    Args:
        model_id: Path to base model (Kaggle local or HF hub).
        max_seq_length: Maximum sequence length for the model.

    Returns:
        Tuple of (base_model, tokenizer).
    """
    logger.info("=" * 60)
    logger.info("BASE MODEL LOADING: Pure base model (NO adapter)")
    logger.info("=" * 60)

    # --- Check for Unsloth ---
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth is REQUIRED for model loading. "
            "Install with: pip install unsloth"
        )

    # --- Resolve model path ---
    # For base model evaluation, we need the ORIGINAL base model, NOT the
    # adapter checkpoint. Use the HF Hub ID or a known base model path.
    if Path(model_id).exists():
        resolved_model = model_id
    elif Path(KAGGLE_MODEL_ID).exists():
        resolved_model = KAGGLE_MODEL_ID
    else:
        resolved_model = HF_MODEL_ID
        logger.info(f"Local model not found. Using HF Hub: {resolved_model}")

    logger.info(f"  Base model: {resolved_model}")
    logger.info(f"  Adapter:    *** NONE — PURE BASE MODEL EVALUATION ***")

    # --- Load BASE model via Unsloth (NO adapter) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_model,       # Base model path, NOT adapter
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

    # --- Apply Unsloth inference optimization (same as evaluation_final.py) ---
    FastLanguageModel.for_inference(model)
    logger.info("  Model prepared for inference ✓ (Unsloth optimized, NO adapter)")

    # VRAM report
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
            name = torch.cuda.get_device_name(i)
            logger.info(f"  GPU {i}: {name} | {alloc_gb:.2f} GB allocated")

    logger.info("BASE MODEL LOADING COMPLETE ✓")
    logger.info("=" * 60)
    return model, tokenizer


def load_model_hf_fallback(
    model_id: str,
) -> Tuple[Any, Any]:
    """
    Fallback model loading without Unsloth — standard HF (NO PEFT/adapter).
    Used when Unsloth is not available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("  [FALLBACK] Loading BASE model via standard HF (no adapter)...")

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

    # *** NO ADAPTER LOADING — THIS IS THE KEY DIFFERENCE ***
    logger.info("  Base model loaded (NO adapter). Running pure base model evaluation.")

    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    return model, tokenizer


# ===========================================================================
# STAGE 3: INFERENCE ENGINE
# ===========================================================================
# NOTE: Identical to evaluation_final.py — do NOT modify.

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
    IDENTICAL to evaluation_final.py for fair comparison.
    """
    # Apply chat template
    try:
        input_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        input_text = "\n".join(
            m.get("content", "") for m in prompt_messages if isinstance(m, dict)
        )

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_new_tokens,
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
# NOTE: Identical to evaluation_final.py — do NOT modify.

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
    table_pattern = re.compile(
        r"(\|[^\n]*SOFA[^\n]*\|.*?(?:\n\|[^\n]*\|)+)",
        re.DOTALL | re.IGNORECASE,
    )
    m = table_pattern.search(text)
    if m:
        return m.group(0).strip()

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

# CSV column spec — single source of truth (same as evaluation_final.py)
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
    Uses print(flush=True) instead of tqdm.
    """
    acc = correct / (i + 1) if (i + 1) > 0 else 0.0
    mark = "✓" if is_correct else "✗"
    eta_s = (elapsed / (i + 1)) * (total - i - 1) if (i + 1) > 0 else 0
    eta_m = eta_s / 60

    filled = int(30 * (i + 1) / total) if total > 0 else 0
    bar = "█" * filled + "░" * (30 - filled)

    print(
        f"  [{i+1:>4}/{total}] {bar} "
        f"Acc={acc:.1%} ({correct}✓) "
        f"| {mark} ans={answer or '—'} sig={signal[:10]} "
        f"| {elapsed:.0f}s elapsed, ~{eta_m:.1f}m left",
        flush=True,
    )


def run_base_evaluation(
    model_id: str = KAGGLE_MODEL_ID,
    data_dir: str = DEFAULT_DATA_DIR,
    num_eval_samples: int = NUM_EVAL_SAMPLES,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    dry_run: bool = False,
) -> str:
    """
    Execute the BASE MODEL evaluation pipeline (ablation baseline).

    This is identical to run_final_evaluation() EXCEPT:
      - No adapter_path argument (not needed).
      - Model loading calls load_base_model() instead of load_and_merge_model().
      - Summary report labels this as "BASE MODEL" evaluation.

    Key features (same as evaluation_final.py):
      - Crash-safe: writes each result to CSV immediately after inference.
      - Resume-capable: detects partial CSV and skips already-completed rows.
      - Kaggle-friendly: uses print(flush=True) instead of tqdm.

    Returns:
        Path to the output CSV file.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Audit — Build blacklist (IDENTICAL to evaluation_final.py)
    # ------------------------------------------------------------------
    all_records = _load_raw_jsonl(data_dir)
    blacklist = identify_blacklist_indices(all_records)
    eval_samples = select_unseen_samples(
        all_records, blacklist, num_samples=num_eval_samples,
    )

    total = len(eval_samples)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  🧪 BASE MODEL EVALUATION: {total} unseen MedQA samples", flush=True)
    print(f"  ⚠  NO LoRA adapter — pure base model (ablation baseline)", flush=True)
    print(f"{'=' * 60}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1.5: Check for resume from partial CSV
    # ------------------------------------------------------------------
    resume_from = _detect_resume_point(output_csv)

    # ------------------------------------------------------------------
    # Phase 2: Load BASE Model (NO adapter)
    # ------------------------------------------------------------------
    model, tokenizer = None, None
    if not dry_run:
        try:
            model, tokenizer = load_base_model(
                model_id=model_id,
            )
        except ImportError:
            logger.warning("Unsloth not available — trying HF fallback...")
            model, tokenizer = load_model_hf_fallback(
                model_id=model_id,
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
    if resume_from > 0:
        csv_file = open(output_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_ALL)
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

    # If resuming, count correct answers from the skipped portion
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

            # Format the sample into GRPO-style prompt (IDENTICAL to evaluation_final.py)
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
            csv_file.flush()
            os.fsync(csv_file.fileno())

            # --- Print progress (Kaggle-friendly) ---
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

    # Re-read CSV to count answer extraction rate
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
    print("  GEMMA-SYNC BASE MODEL EVALUATION RESULTS (ABLATION)", flush=True)
    print("  Model: Gemma 4 E2B | NO GRPO | NO LoRA | Pure Base", flush=True)
    print("=" * 62, flush=True)
    print(f"  Evaluation Samples    : {n} (strictly unseen)", flush=True)
    print(f"  Blacklist (train)     : {len(blacklist)} samples excluded", flush=True)
    print(f"  Adapter               : *** NONE (BASE MODEL) ***", flush=True)
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

    # Interpretation — framed for ablation baseline context
    print("\n  INTERPRETATION (ABLATION BASELINE):", flush=True)
    print(f"  Base model accuracy = {accuracy:.2%}", flush=True)
    print("  Compare this against the fine-tuned model (evaluation_final.py)", flush=True)
    print("  to measure the Δ-accuracy gained from GRPO + RLVR training.", flush=True)

    none_rate = routing_counts.get("NONE", 0) / n if n > 0 else 0
    if none_rate > 0.50:
        print("  ℹ  >50% missing Cactus signal — expected for base model (not trained).", flush=True)
    elif none_rate > 0.20:
        print("  ℹ  >20% missing Cactus signal — base model partially follows format.", flush=True)
    else:
        print("  ⚠  <20% missing Cactus signal — surprisingly good format compliance.", flush=True)

    print(flush=True)
    logger.info(f"Base model evaluation complete. Results saved to: {output_csv}")
    return output_csv


# ===========================================================================
# Synthetic Output Generator (for dry-run mode)
# ===========================================================================

def _generate_synthetic(ground_truth: str) -> str:
    """Generate a synthetic completion for dry-run testing."""
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
            "Gemma-Sync BASE MODEL Evaluation — Ablation baseline.\n"
            "Evaluates the pure base model (NO adapter) on the same 200\n"
            "unseen MedQA samples for paired comparison against fine-tuned model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id", type=str, default=KAGGLE_MODEL_ID,
        help="Path to base model (Kaggle local mount or HF Hub ID).",
    )
    # NOTE: --adapter-path is intentionally REMOVED (no adapter for base eval)
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Path to local MedQA JSONL dataset directory.",
    )
    parser.add_argument(
        "--num-eval-samples", type=int, default=NUM_EVAL_SAMPLES,
        help="Number of unseen samples to evaluate (max depends on pool size).",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help=(
            "Output CSV file path. "
            "Default: gemma_sync_BASE_eval_{num_eval_samples}.csv"
        ),
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

    # Dynamic output CSV name based on num_eval_samples
    output_csv = args.output_csv or f"gemma_sync_BASE_eval_{args.num_eval_samples}.csv"

    output_path = run_base_evaluation(
        model_id=args.model_id,
        data_dir=args.data_dir,
        num_eval_samples=args.num_eval_samples,
        output_csv=output_csv,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )

    print(f"\n✅ Base model evaluation complete → {output_path}")
