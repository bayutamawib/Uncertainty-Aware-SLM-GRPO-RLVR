"""
inference_check.py — Post-Training Performance Validation for Gemma-Sync (T4 GPU)
==================================================================================

Validates the fine-tuned Gemma 4 E2B GRPO adapter on the MedQA-USMLE
validation split across four clinical performance dimensions:

  1. Clinical Accuracy     — Exact match of \\boxed{X} against ground truth
  2. SOFA Precision        — Table presence, coverage, N/P discipline
  3. MAP Calculation       — Deterministic BP-to-MAP verification
  4. Cactus Routing        — <|escalate|> / <|local_ok|> token frequency

Hardware: Kaggle 2× NVIDIA T4 GPUs (CUDA) → MPS (M2 prototyping) → CPU fallback
Usage:
    # Kaggle T4 (full validation, 200 samples):
    python inference_check.py --num-samples 200

    # M2 dry-run (10 samples, no adapter):
    python inference_check.py --num-samples 10 --dry-run

Author : Narendra Bayutama Wibisono
Project: Gemma-Sync — Distributed Uncertainty-Aware Clinical Reasoning via Gemma 4 E2B
Target : Kaggle 2× NVIDIA T4 GPUs (ported from TPU v5e-8 version)
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_check_t4")

# ---------------------------------------------------------------------------
# Local imports (reward logic + data pipeline from Gemma-Sync project)
# ---------------------------------------------------------------------------
try:
    from data_pipeline import load_and_prepare_dataset
except ImportError:
    from kaggle_t4_dual.data_pipeline import load_and_prepare_dataset

try:
    from distributed_grpo_trainer import (
        _extract_text, _extract_boxed,
        _parse_sofa_table, _check_sofa_not_applicable,
        _extract_map_from_cv_string, _score_sofa_oracle,
        CACTUS_ESCALATE_TOKEN, CACTUS_LOCAL_TOKEN,
        SOFA_COMPONENTS, SOFA_SCORE_THRESHOLDS,
    )
except ImportError:
    from kaggle_t4_dual.distributed_grpo_trainer import (
        _extract_text, _extract_boxed,
        _parse_sofa_table, _check_sofa_not_applicable,
        _extract_map_from_cv_string, _score_sofa_oracle,
        CACTUS_ESCALATE_TOKEN, CACTUS_LOCAL_TOKEN,
        SOFA_COMPONENTS, SOFA_SCORE_THRESHOLDS,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"
DEFAULT_ADAPTER_PATH = "/kaggle/working/outputs/gemma4-e2b-grpo-sofa-t4/final_adapter"
DEFAULT_OUTPUT_DIR = "/kaggle/working/outputs/inference_reports"


# ===========================================================================
# Report Dataclass
# ===========================================================================

@dataclass
class GemmaSyncReport:
    """Full Gemma-Sync clinical performance report."""
    timestamp: str = ""
    model_id: str = ""
    adapter_path: str = ""
    total_samples: int = 0
    device: str = ""

    accuracy: float = 0.0
    correct_count: int = 0

    sofa_table_presence: float = 0.0
    sofa_full_coverage: float = 0.0
    sofa_oracle_avg: float = 0.0
    np_flagging_rate: float = 0.0

    map_tested_count: int = 0
    map_correct_count: int = 0
    map_calculation_accuracy: float = 0.0

    cactus_escalation_rate: float = 0.0
    cactus_local_rate: float = 0.0
    cactus_both_rate: float = 0.0
    cactus_neither_rate: float = 0.0
    cactus_false_confidence_rate: float = 0.0

    avg_inference_time_s: float = 0.0
    total_inference_time_s: float = 0.0

    sample_details: List[Dict] = field(default_factory=list)


# ===========================================================================
# MAP Verification Helpers
# ===========================================================================

_BP_STRING_PATTERN = re.compile(
    r"(?:bp|blood\s*pressure)?[:\s]*(?<!\.)\b(\d{2,3})\s*/\s*(\d{2,3})\b",
    re.IGNORECASE,
)


def extract_bp_from_prompt(prompt: Any) -> Optional[tuple]:
    """Extract first SBP/DBP pair from the user message in prompt."""
    user_text = ""
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
    elif isinstance(prompt, str):
        user_text = prompt
    m = _BP_STRING_PATTERN.search(user_text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def compute_expected_map(sbp: float, dbp: float) -> float:
    """MAP = (SBP + 2 × DBP) / 3"""
    return round((sbp + 2.0 * dbp) / 3.0, 1)


def verify_map_in_completion(completion_text: str, expected_map: float, tolerance: float = 2.0) -> bool:
    """Check if model's stated MAP is within ±tolerance of expected."""
    components = _parse_sofa_table(completion_text)
    cv_data = components.get("Cardiovascular", None)
    if cv_data is None:
        return False
    model_map = _extract_map_from_cv_string(cv_data["value"])
    if model_map is None:
        return False
    return abs(model_map - expected_map) <= tolerance


# ===========================================================================
# Cactus Routing Analysis
# ===========================================================================

def analyze_cactus_routing(text: str, np_count: int, has_claimed_total: bool) -> Dict[str, Any]:
    """Determine Cactus routing signal quality for a single completion."""
    has_escalate = CACTUS_ESCALATE_TOKEN in text
    has_local_ok = CACTUS_LOCAL_TOKEN in text
    return {
        "has_escalate": has_escalate,
        "has_local_ok": has_local_ok,
        "is_both": has_escalate and has_local_ok,
        "is_neither": not has_escalate and not has_local_ok,
        "is_false_confidence": (
            has_local_ok and np_count >= 3 and has_claimed_total
        ),
    }


# ===========================================================================
# Model Loading (CUDA — FP16 + optional 4-bit)
# ===========================================================================

def load_model_for_inference(
    model_id: str,
    adapter_path: Optional[str],
    device: torch.device,
):
    """
    Load Gemma 4 E2B for inference on CUDA (T4 GPUs).

    Uses FP16 precision and device_map="auto" for automatic multi-GPU
    sharding during inference. Optionally loads LoRA adapter via PEFT.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info("Loading base model (FP16, SDPA, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("Adapter merged into base model ✓")
    else:
        logger.warning(
            f"Adapter path not found: {adapter_path}\n"
            f"Running base model (no LoRA) — results reflect pre-training baseline."
        )

    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # Report GPU allocation
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024**3)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} | {alloc:.2f} GB allocated")

    logger.info(f"Model ready on {device}")
    return model, tokenizer


# ===========================================================================
# Single-Sample Inference
# ===========================================================================

def generate_completion(model, tokenizer, prompt, device, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """Run inference for a single prompt and return the completion string."""
    if isinstance(prompt, list):
        try:
            input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
    else:
        input_text = str(prompt)

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding=False)
    # Move to correct device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )

    generated_ids = outputs[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


# ===========================================================================
# Core Validation Loop
# ===========================================================================

def run_validation(
    model_id=DEFAULT_MODEL_ID, adapter_path=DEFAULT_ADAPTER_PATH,
    num_samples=50, dry_run=False, output_dir=DEFAULT_OUTPUT_DIR,
    max_new_tokens=1024, temperature=0.7, data_dir="./medqa_dataset",
) -> GemmaSyncReport:
    """Full Gemma-Sync post-training validation pipeline (CUDA)."""
    # Device detection: CUDA (multi-GPU) > MPS > CPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        device_str = " + ".join(f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(n_gpus))
        logger.info(f"Validation devices: {device_str}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
        logger.info(f"Validation device: {device}")
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        logger.info(f"Validation device: {device}")

    # Dataset
    logger.info(f"Loading validation dataset ({num_samples} samples)...")
    dataset = load_and_prepare_dataset(data_dir=data_dir, max_samples=num_samples)
    val_data = dataset["validation"]
    logger.info(f"Validation split: {len(val_data)} samples")

    # Model
    model, tokenizer = None, None
    if not dry_run:
        model, tokenizer = load_model_for_inference(model_id, adapter_path, device)
    else:
        logger.warning("DRY RUN MODE — using synthetic completions.")

    # Report
    report = GemmaSyncReport(
        timestamp=datetime.now().isoformat(),
        model_id=model_id,
        adapter_path=adapter_path or "none",
        total_samples=len(val_data),
        device=device_str,
    )

    correct_list, sofa_table_present_list, sofa_full_coverage_list = [], [], []
    sofa_oracle_scores, np_flagged_list = [], []
    escalate_list, local_ok_list, both_list, neither_list, false_conf_list = [], [], [], [], []
    inference_times = []

    logger.info("=" * 60)
    logger.info(f"Starting validation on {len(val_data)} samples...")
    logger.info("=" * 60)

    for i in range(len(val_data)):
        sample = val_data[i]
        prompt = sample["prompt"]
        ground_truth = sample["answer"]

        t_start = time.time()
        if dry_run:
            completion = (
                "### Step 1: Clinical Data Extraction\n"
                "| SOFA Component | Parameter | Extracted Value | SOFA Sub-Score |\n"
                "|---|---|---|---|\n"
                "| Respiratory | PaO₂/FiO₂ | 280 | 2 |\n"
                "| Coagulation | Platelets | N/P | N/P |\n"
                "| Liver | Bilirubin | 1.5 | 1 |\n"
                "| Cardiovascular | BP 90/60 → MAP 70 mmHg | 70 | 0 |\n"
                "| CNS | GCS | 13 | 1 |\n"
                "| Renal | Creatinine | N/P | N/P |\n\n"
                "**Total SOFA Score: 4 / 4**\n\n"
                "**Moderate confidence** — Coagulation and Renal are N/P.\n\n"
                f"$$\\\\boxed{{{ground_truth}}}$$\n\n"
                "<|local_ok|>"
            )
        else:
            completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens, temperature)

        elapsed = time.time() - t_start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_times.append(elapsed)

        text = _extract_text(completion)

        # Metric 1: Clinical Accuracy
        predicted = _extract_boxed(text)
        is_correct = (predicted is not None) and (predicted == ground_truth.upper())
        correct_list.append(is_correct)

        # Metric 2: SOFA Precision
        sofa_na = _check_sofa_not_applicable(text)
        sofa_table = _parse_sofa_table(text)
        has_sofa_content = sofa_na or len(sofa_table) > 0
        sofa_table_present_list.append(has_sofa_content)

        if sofa_na:
            sofa_full_coverage_list.append(True)
        else:
            covered = set(sofa_table.keys()) & set(SOFA_COMPONENTS)
            sofa_full_coverage_list.append(len(covered) == len(SOFA_COMPONENTS))

        oracle_score = _score_sofa_oracle(text)
        sofa_oracle_scores.append(oracle_score)
        np_count = len(re.findall(r"\bN/P\b", text))
        np_flagged_list.append(np_count >= 1)

        # Metric 3: MAP Calculation
        bp_in_prompt = extract_bp_from_prompt(prompt)
        if bp_in_prompt is not None:
            sbp, dbp = bp_in_prompt
            expected_map = compute_expected_map(sbp, dbp)
            report.map_tested_count += 1
            if verify_map_in_completion(text, expected_map, tolerance=2.0):
                report.map_correct_count += 1

        # Metric 4: Cactus Routing
        has_claimed_total = bool(re.search(r"\*\*Total SOFA Score:\*\*\s*\d+", text, re.IGNORECASE))
        routing = analyze_cactus_routing(text, np_count, has_claimed_total)
        escalate_list.append(routing["has_escalate"])
        local_ok_list.append(routing["has_local_ok"])
        both_list.append(routing["is_both"])
        neither_list.append(routing["is_neither"])
        false_conf_list.append(routing["is_false_confidence"])

        if (i + 1) % 10 == 0 or i == len(val_data) - 1:
            running_acc = sum(correct_list) / len(correct_list)
            logger.info(
                f"[{i+1:>4}/{len(val_data)}] "
                f"Acc={running_acc:.2%} | "
                f"SOFA={sum(sofa_table_present_list)/len(sofa_table_present_list):.2%} | "
                f"t={elapsed:.2f}s"
            )

        report.sample_details.append({
            "index": i, "ground_truth": ground_truth, "predicted": predicted,
            "correct": is_correct, "sofa_table_present": has_sofa_content,
            "sofa_full_coverage": sofa_full_coverage_list[-1],
            "sofa_oracle_score": round(oracle_score, 4), "np_count": np_count,
            "map_in_prompt": bp_in_prompt is not None,
            "expected_map": round(compute_expected_map(*bp_in_prompt), 1) if bp_in_prompt else None,
            "routing": routing, "inference_time_s": round(elapsed, 3),
        })

    # Aggregate
    n = len(val_data)
    report.accuracy = sum(correct_list) / n
    report.correct_count = sum(correct_list)
    report.sofa_table_presence = sum(sofa_table_present_list) / n
    report.sofa_full_coverage = sum(sofa_full_coverage_list) / n
    report.sofa_oracle_avg = sum(sofa_oracle_scores) / n
    report.np_flagging_rate = sum(np_flagged_list) / n
    report.map_calculation_accuracy = report.map_correct_count / max(report.map_tested_count, 1)
    report.cactus_escalation_rate = sum(escalate_list) / n
    report.cactus_local_rate = sum(local_ok_list) / n
    report.cactus_both_rate = sum(both_list) / n
    report.cactus_neither_rate = sum(neither_list) / n
    report.cactus_false_confidence_rate = sum(false_conf_list) / n
    report.avg_inference_time_s = sum(inference_times) / n
    report.total_inference_time_s = sum(inference_times)

    _print_report(report)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"gemma_sync_report_{ts}.json"
    summary = {k: v for k, v in asdict(report).items() if k != "sample_details"}
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary report saved → {report_file}")

    full_report_file = output_path / f"gemma_sync_report_full_{ts}.json"
    with open(full_report_file, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Full report saved → {full_report_file}")

    return report


# ===========================================================================
# Formatted Output
# ===========================================================================

def _print_report(report: GemmaSyncReport) -> None:
    """Print a clean, structured performance report to stdout."""
    w = 62
    print("\n" + "=" * w)
    print("  GEMMA-SYNC CLINICAL PERFORMANCE REPORT [T4 GPU]")
    print("  Gemma 4 E2B | GRPO + RLVR | Calibrated Abstention")
    print("=" * w)
    print(f"  Timestamp   : {report.timestamp}")
    print(f"  Samples     : {report.total_samples}")
    print(f"  Device      : {report.device}")
    print(f"  Adapter     : {Path(report.adapter_path).name}")
    print("-" * w)
    print("  [1] CLINICAL ACCURACY (RLVR)")
    print(f"      Correct answers  : {report.correct_count}/{report.total_samples}")
    print(f"      Accuracy         : {report.accuracy:.2%}")
    print("-" * w)
    print("  [2] SOFA PRECISION (Oracle)")
    print(f"      Table/NA present : {report.sofa_table_presence:.2%}")
    print(f"      Full 6-comp cov. : {report.sofa_full_coverage:.2%}")
    print(f"      Oracle score avg : {report.sofa_oracle_avg:.4f} / 1.0")
    print(f"      N/P flagging rate: {report.np_flagging_rate:.2%}")
    print("-" * w)
    print("  [3] MAP CALCULATION PRECISION")
    if report.map_tested_count > 0:
        print(f"      BP samples tested: {report.map_tested_count}")
        print(f"      MAP accuracy (±2): {report.map_calculation_accuracy:.2%}")
    else:
        print("      No BP strings found in validation prompts")
    print("-" * w)
    print("  [4] CACTUS ROUTING SIGNAL")
    print(f"      <|escalate|> rate : {report.cactus_escalation_rate:.2%}")
    print(f"      <|local_ok|> rate : {report.cactus_local_rate:.2%}")
    print(f"      Both tokens (bad) : {report.cactus_both_rate:.2%}")
    print(f"      Neither (missing) : {report.cactus_neither_rate:.2%}")
    print(f"      False confidence  : {report.cactus_false_confidence_rate:.2%}")
    print("-" * w)
    print("  [5] PERFORMANCE")
    print(f"      Avg latency/sample: {report.avg_inference_time_s:.2f}s")
    print(f"      Total time        : {report.total_inference_time_s:.1f}s")
    print("=" * w)

    print("\n  INTERPRETATION GUIDE:")
    if report.accuracy >= 0.70:
        print("  ✓ Accuracy ≥70% — competitive USMLE baseline")
    elif report.accuracy >= 0.50:
        print("  △ Accuracy 50-70% — acceptable for a 2B model, continue training")
    else:
        print("  ✗ Accuracy <50% — below random chance; check reward function or data")
    if report.cactus_neither_rate > 0.20:
        print("  ⚠ >20% responses missing Cactus token — routing signal not learned yet")
    if report.cactus_false_confidence_rate > 0.10:
        print("  ⚠ >10% false-confidence routing — abstention penalty may need increasing")
    if report.np_flagging_rate < 0.30:
        print("  ⚠ <30% N/P usage — model may be hallucinating SOFA values")
    if report.map_tested_count > 0 and report.map_calculation_accuracy < 0.70:
        print("  ⚠ MAP accuracy <70% — model not applying (SBP + 2×DBP)/3 reliably")
    print()


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemma-Sync Post-Training Validation [T4 GPU] — CUDA Multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="./medqa_dataset")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true",
        help="Skip model loading; use synthetic completions (pipeline testing).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_validation(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        num_samples=args.num_samples,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        data_dir=args.data_dir,
    )
