"""
data_pipeline.py — MedQA-USMLE Data Pipeline for Gemma-Sync GRPO Training
===========================================================================

Gemma-Sync revision for the Dual NVIDIA T4 GPU training track.
Updated from the TPU v5e-8 version — SYSTEM_PROMPT and all data logic
are IDENTICAL. Only the hardware target reference in this docstring differs.

Hardware Target: Kaggle 2× NVIDIA T4 GPUs (16GB VRAM each, CUDA)
Model: Gemma 4 E2B IT — google/gemma-4-e2b-it (HF Hub or Kaggle local mount)

All data loading, formatting, SOFA-First system prompt, and stratified-split
logic are preserved verbatim from the TPU pipeline.

Author : Narendra Bayutama Wibisono
Project: Gemma-Sync — Distributed Uncertainty-Aware Clinical Reasoning via Gemma 4 E2B
Ref    : Ported from TPU v5e-8 data_pipeline.py
         Zenodo: 10.5281/zenodo.19599245
"""
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# On Kaggle: /kaggle/input/medqa-usmle/ or similar offline mount
LOCAL_DATA_DIR = "./medqa_dataset"
ANSWER_INDEX_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
TRAIN_SPLIT_RATIO = 0.9
RANDOM_SEED = 42

# ===========================================================================
# SYSTEM PROMPT — Gemma-Sync Edition
# ===========================================================================
# Key additions over the legacy prompt (tpu_previous_project/data_pipeline.py):
#
#   1. CACTUS ROUTING PROTOCOL (§5 below):
#      Model is trained to emit exactly one routing token per response:
#        <|escalate|>  — insufficient data → route to larger cloud model
#        <|local_ok|>  — sufficient confidence → serve locally on M2
#      These tokens are rewarded/penalized in reward_process_quality()
#      and penalized in _score_abstention_bonus() for false confidence.
#
#   2. DETERMINISTIC BP-TO-MAP CONVERSION (§1, Cardiovascular row):
#      Model is explicitly taught the MAP formula:
#        MAP = (SBP + 2 × DBP) / 3
#      This enables _extract_map_from_cv_string() to verify the model's
#      cardiovascular SOFA sub-score against a calculable MAP value,
#      providing a stronger clinical plausibility signal.
#
#   3. N/P ESCALATION RULE (§4):
#      If ≥2 critical SOFA parameters (not Renal/Respiratory defaults)
#      are marked N/P on a clearly critical case, model MUST emit
#      <|escalate|> rather than <|local_ok|>. Violation is penalized in
#      _score_abstention_bonus() as the "false-confidence Cactus penalty".
# ===========================================================================

SYSTEM_PROMPT = """You are an expert clinical reasoning assistant trained in evidence-based medicine and deployed as part of the **Gemma-Sync Cactus Routing Network**. Your task is to solve USMLE-style medical questions using a structured, uncertainty-aware reasoning process.

## Your Mandatory Reasoning Protocol

You MUST follow these steps in order for EVERY question:

---

### Step 1: Clinical Data Extraction (SOFA Assessment with Deterministic MAP Calculation)

**First, determine if SOFA scoring is clinically applicable.** SOFA is relevant for critical care, emergency medicine, ICU, sepsis, organ failure, or acute physiological derangement. If the clinical scenario is **non-critical** (e.g., psychiatry, dermatology, preventive medicine, outpatient general practice, behavioral sciences, ethics, ophthalmology), write:

> **SOFA Assessment: SOFA_NOT_APPLICABLE**
> **Reason:** [Brief explanation, e.g., "Outpatient psychiatric evaluation with no acute organ dysfunction"]

Then skip directly to Step 2.

**If SOFA IS applicable**, extract the following six organ-system parameters and present them in this EXACT Markdown table format:

| SOFA Component    | Parameter                | Extracted Value  | SOFA Sub-Score |
|-------------------|--------------------------|------------------|----------------|
| Respiratory       | PaO₂/FiO₂ ratio         | [value or N/P]   | [0-4 or N/P]   |
| Coagulation       | Platelets (×10³/μL)      | [value or N/P]   | [0-4 or N/P]   |
| Liver             | Bilirubin (mg/dL)        | [value or N/P]   | [0-4 or N/P]   |
| Cardiovascular    | MAP / Vasopressors       | [value or N/P]   | [0-4 or N/P]   |
| CNS               | GCS                      | [value or N/P]   | [0-4 or N/P]   |
| Renal             | Creatinine (mg/dL)       | [value or N/P]   | [0-4 or N/P]   |

**Cardiovascular MAP Calculation Protocol (MANDATORY when BP is given):**
If the question provides blood pressure as Systolic/Diastolic (e.g., "BP 120/80 mmHg"), you MUST calculate MAP using the deterministic formula before assigning the SOFA sub-score:

  MAP = (SBP + 2 × DBP) / 3

Example: BP 90/60 mmHg → MAP = (90 + 2×60) / 3 = 210 / 3 = **70 mmHg** → SOFA CV = 0 (MAP ≥ 70)
Example: BP 85/55 mmHg → MAP = (85 + 2×55) / 3 = 195 / 3 = **65 mmHg** → SOFA CV = 1 (MAP < 70)

Write the calculated MAP in the Extracted Value cell: e.g., "BP 90/60 → MAP 70 mmHg"
For vasopressor-dependent cases (SOFA 2-4), note the vasopressor and dose in the cell.

**SOFA Scoring Reference:**
| Component     | SOFA 0       | SOFA 1           | SOFA 2            | SOFA 3              | SOFA 4          |
|---------------|--------------|------------------|-------------------|---------------------|-----------------|
| Respiratory   | PF ≥ 400     | PF 300–399       | PF 200–299        | PF 100–199          | PF < 100        |
| Coagulation   | Plt ≥ 150    | Plt 100–149      | Plt 50–99         | Plt 20–49           | Plt < 20        |
| Liver         | Bili < 1.2   | Bili 1.2–1.9     | Bili 2.0–5.9      | Bili 6.0–11.9       | Bili ≥ 12       |
| Cardiovascular| MAP ≥ 70     | MAP < 70         | Dopa ≤5 / Dobut   | Dopa >5 / Epi ≤0.1  | Dopa >15 / Epi >0.1 |
| CNS (GCS)     | GCS = 15     | GCS 13–14        | GCS 10–12         | GCS 6–9             | GCS < 6         |
| Renal         | Cr < 1.2     | Cr 1.2–1.9       | Cr 2.0–3.4        | Cr 3.5–4.9          | Cr ≥ 5.0        |

**Default Clinical Assumptions** (apply ONLY when the question does not state a value explicitly):
- **Respiratory**: Room air breathing with no supplemental O₂ → assume FiO₂ = 0.21. Write: `[assumed: FiO₂ = 0.21]`
- **Coagulation**: No platelet count + no bleeding/DIC context → assume Platelets ≥ 150 (SOFA = 0). Write: `[assumed: Plt ≥ 150]`
- **Liver**: No bilirubin + no jaundice/liver disease → assume Bilirubin < 1.2 mg/dL (SOFA = 0). Write: `[assumed: Bili < 1.2]`
- **Cardiovascular**: No MAP/BP + hemodynamically stable → assume MAP ≥ 70 mmHg (SOFA = 0). Write: `[assumed: MAP ≥ 70]`
- **CNS**: Alert and oriented, no GCS given → assume GCS = 15 (SOFA = 0). Write: `[assumed: GCS = 15]`
- **Renal**: No creatinine + no renal disease → assume Creatinine < 1.2 mg/dL (SOFA = 0). Write: `[assumed: Cr < 1.2]`
- For parameters **truly absent** AND cannot be clinically inferred → mark as **N/P** (Not Provided). **NEVER fabricate a specific number.**

**Table Format Rules (STRICT):**
- Include ALL 6 rows in the exact order shown above.
- Each SOFA Sub-Score cell must contain exactly: `0`, `1`, `2`, `3`, `4`, or `N/P`.
- After the table, write: **Total SOFA Score: [sum of numeric sub-scores] / [number of numeric components]**

---

### Step 2: Clinical Reasoning

Based on extracted data and the clinical scenario:
- Identify the most likely pathophysiology
- Systematically evaluate each answer option against clinical evidence
- If critical SOFA data is marked N/P, explicitly state which missing parameter limits your differential

---

### Step 3: Uncertainty Assessment

Rate your confidence based on data completeness:
- **High confidence**: All relevant parameters present and consistent
- **Moderate confidence**: Some parameters missing but sufficient for reasoning
- **Low confidence**: Critical data absent — state which information would change your assessment

---

### Step 4: Final Answer

Provide your answer in LaTeX boxed notation:

$$\\boxed{X}$$

where X is one of A, B, C, or D.

---

### Step 5: Cactus Routing Signal (MANDATORY — always end with exactly one)

After your final answer, emit **exactly one** of these routing tokens on its own line:

**<|local_ok|>** — You are confident the local Gemma 4 E2B model handled this correctly. Use when:
  - High confidence (all or most SOFA parameters present/inferred)
  - Reasoning is complete and traceable to specific data points
  - SOFA_NOT_APPLICABLE was correctly identified with a valid reason

**<|escalate|>** — You are uncertain and request routing to a larger specialist model. Use when:
  - ≥2 critical SOFA parameters are N/P on a clearly critical-care case
  - Confidence is Low and the missing data would materially change your answer
  - Pathophysiology is ambiguous and differential cannot be narrowed

**Rules:**
- You MUST emit exactly one of these tokens per response. Never emit both.
- If you are uncertain but guess anyway, use <|escalate|> — do NOT emit <|local_ok|> while guessing.
- False confidence (emitting <|local_ok|> with ≥3 N/P entries and a guessed total score) is penalized.

---

## Critical Rules

1. NEVER fabricate clinical values. If data is not in the question and cannot be inferred, mark N/P.
2. Always perform SOFA applicability check (Step 1) before reasoning, even for straightforward questions.
3. Always use the exact Markdown table format when SOFA is applicable.
4. When BP is given, always calculate MAP explicitly using MAP = (SBP + 2×DBP) / 3 before scoring.
5. Every conclusion must reference specific clinical data from the question stem.
6. State all default assumptions explicitly using [assumed: value] notation.
7. Always end your response with exactly one Cactus Routing Signal (<|local_ok|> or <|escalate|>)."""


# ===========================================================================
# Text Processing Utilities
# ===========================================================================

def clean_text(text: str) -> str:
    """
    Clean a text string by normalizing whitespace and fixing common
    Unicode encoding artifacts found in medical text datasets.

    Args:
        text: Raw text string from the dataset.

    Returns:
        Cleaned text string with normalized whitespace and fixed encoding.
    """
    if not isinstance(text, str):
        return ""

    # Fix common Unicode encoding artifacts
    text = text.replace("\u00a0", " ")       # Non-breaking space → regular space
    text = text.replace("\u2019", "'")        # Right single quotation mark
    text = text.replace("\u2018", "'")        # Left single quotation mark
    text = text.replace("\u201c", '"')        # Left double quotation mark
    text = text.replace("\u201d", '"')        # Right double quotation mark
    text = text.replace("\u2013", "-")        # En dash
    text = text.replace("\u2014", "-")        # Em dash
    text = text.replace("\u2026", "...")      # Horizontal ellipsis
    text = text.replace("\u00b0", "°")        # Degree symbol (keep as-is)

    # Normalize multiple whitespace characters to single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def format_options(options: Dict[str, str]) -> str:
    """
    Format answer options into a clean, labeled string.

    The MedQA dataset provides options as a dict with keys like
    {"A": "...", "B": "...", "C": "...", "D": "..."}.

    Args:
        options: Dictionary mapping option letters to option text.

    Returns:
        Formatted string with each option on its own line.
    """
    formatted_lines = []
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            option_text = clean_text(options[letter])
            formatted_lines.append(f"({letter}) {option_text}")
        else:
            logger.warning(f"Missing option '{letter}' in options dict: {options}")
    return "\n".join(formatted_lines)


def build_user_prompt(question: str, options: Dict[str, str]) -> str:
    """
    Construct the user-turn prompt from the clinical question and options.

    This prompt triggers the full SOFA-First + Cactus Routing protocol
    defined in SYSTEM_PROMPT. The closing instruction explicitly reminds
    the model to emit its routing signal.

    Args:
        question: The clinical vignette / question stem.
        options: Dictionary of answer choices {A: ..., B: ..., C: ..., D: ...}.

    Returns:
        Formatted user prompt string.
    """
    cleaned_question = clean_text(question)
    formatted_options = format_options(options)

    user_prompt = (
        f"## Clinical Question\n\n"
        f"{cleaned_question}\n\n"
        f"## Answer Options\n\n"
        f"{formatted_options}\n\n"
        f"---\n"
        f"Apply the full clinical reasoning protocol. Begin with Step 1 (SOFA parameter "
        f"extraction — calculate MAP if BP is given using MAP = (SBP + 2×DBP) / 3), "
        f"then provide your clinical reasoning, uncertainty assessment, and final answer "
        f"in \\boxed{{}} format. End with exactly one Cactus Routing Signal "
        f"(<|escalate|> or <|local_ok|>)."
    )

    return user_prompt


# ===========================================================================
# Answer Resolution
# ===========================================================================

def resolve_answer_label(example: Dict[str, Any]) -> str:
    """
    Convert the dataset's answer representation to a single letter (A/B/C/D).

    The GBaker/MedQA-USMLE-4-options dataset stores the answer as the text of
    the correct option in the 'answer' field. We match it to the corresponding
    letter key from the options dict.

    Args:
        example: A single dataset example with 'answer' and 'options' fields.

    Returns:
        The letter label (A, B, C, or D) corresponding to the correct answer.
    """
    answer_text = example.get("answer", "")
    options = example.get("options", {})

    # Direct match: answer text matches an option value
    for letter, option_text in options.items():
        if option_text.strip() == answer_text.strip():
            return letter

    # Fallback: case-insensitive comparison
    answer_lower = answer_text.strip().lower()
    for letter, option_text in options.items():
        if option_text.strip().lower() == answer_lower:
            return letter

    # Answer field may already be a letter
    if answer_text.strip().upper() in {"A", "B", "C", "D"}:
        return answer_text.strip().upper()

    # Integer index (some dataset variants)
    if isinstance(answer_text, int) and answer_text in ANSWER_INDEX_MAP:
        return ANSWER_INDEX_MAP[answer_text]

    logger.warning(
        f"Could not resolve answer '{answer_text}' to a letter label. "
        f"Options: {options}. Defaulting to 'A'."
    )
    return "A"


# ===========================================================================
# GRPO Formatting
# ===========================================================================

def format_example_for_grpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single MedQA example into the GRPO training format.

    Output format (TRL GRPOTrainer compatible):
      - "prompt": list of message dicts [{"role": "system", ...}, {"role": "user", ...}]
      - "answer": ground-truth letter label (A/B/C/D) for reward computation

    The system message contains the full SOFA-First + Cactus Routing protocol.
    The user message contains the clinical vignette + MAP calculation reminder.

    Args:
        example: Raw dataset example with 'question', 'options', 'answer' fields.

    Returns:
        Dict with 'prompt' (list of message dicts) and 'answer' (str).
    """
    question = example.get("question", "")
    options = example.get("options", {})

    user_content = build_user_prompt(question, options)

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    answer_label = resolve_answer_label(example)

    return {
        "prompt": prompt_messages,
        "answer": answer_label,
    }


# ===========================================================================
# JSONL Loading
# ===========================================================================

def _load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.

    Lines that fail to parse are skipped with a warning.

    Args:
        filepath: Absolute or relative path to the .jsonl file.

    Returns:
        List of parsed dicts.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {filepath}: {e}")
    return records


# ===========================================================================
# Main Dataset Loading
# ===========================================================================

def load_and_prepare_dataset(
    data_dir: str = LOCAL_DATA_DIR,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
    max_samples: Optional[int] = None,
    max_train_samples: int = 202,
    max_val_samples: int = 50,
) -> DatasetDict:
    """
    Load the MedQA-USMLE dataset from LOCAL JSONL files, clean it, and
    format for GRPO training with Gemma-Sync system prompt.

    Designed for OFFLINE Kaggle environments (no internet access).
    Reads pre-downloaded JSONL files rather than calling HF Hub API.

    Pipeline:
      1. Load local JSONL files (train + optional test/validation)
      2. Subsample if max_samples is set (for debugging / dry-runs)
      3. Format into conversation templates (SOFA-First + Cactus Routing)
      4. Stratified train/validation split (balanced A/B/C/D distribution)

    Expected directory structure:
        data_dir/
        ├── phrases_no_exclude_train.jsonl   (main training data)
        └── phrases_no_exclude_test.jsonl    (held-out test data)

    Args:
        data_dir:    Path to directory containing JSONL files.
        train_ratio: Fraction of data used for training (rest → validation).
        seed:        Random seed for reproducible splitting.
        max_samples: If set, limit total samples (useful for debugging).

    Returns:
        DatasetDict with 'train' and 'validation' splits, each containing
        columns ['prompt', 'answer'].
    """
    data_path = Path(data_dir)
    logger.info(f"Loading dataset from: {data_path.resolve()}")

    # ------------------------------------------------------------------
    # Stage 1: Discover and load JSONL files
    # ------------------------------------------------------------------
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path.resolve()}\n"
            f"Run hf_dataset_download.py first, or set the correct path."
        )

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
            raise FileNotFoundError(
                f"No .jsonl files found in {data_path.resolve()}\n"
                f"Expected: phrases_no_exclude_train.jsonl"
            )
        train_file = all_jsonl[0]
        test_file = all_jsonl[1] if len(all_jsonl) > 1 else None
        logger.warning(f"Using auto-discovered train file: {train_file.name}")

    logger.info(f"Loading train: {train_file.name}")
    train_records = _load_jsonl(str(train_file))
    logger.info(f"  Loaded {len(train_records)} training examples")

    all_records = train_records
    if test_file and test_file.exists():
        logger.info(f"Loading test/val: {test_file.name}")
        test_records = _load_jsonl(str(test_file))
        logger.info(f"  Loaded {len(test_records)} test/val examples")
        all_records = train_records + test_records
        logger.info(f"  Combined: {len(all_records)} total")

    combined = Dataset.from_list(all_records)

    # ------------------------------------------------------------------
    # Stage 2: Subsample (for debugging / dry-run mode)
    # ------------------------------------------------------------------
    if max_samples is not None and max_samples < len(combined):
        combined = combined.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Subsampled to {max_samples} examples")

    # ------------------------------------------------------------------
    # Stage 3: Format with SOFA-First + Cactus Routing system prompt
    # ------------------------------------------------------------------
    logger.info("Formatting with Gemma-Sync SOFA-First + Cactus Routing protocol...")
    formatted_dataset = combined.map(
        format_example_for_grpo,
        remove_columns=combined.column_names,
        desc="Formatting prompts",
        num_proc=2,
    )

    # ------------------------------------------------------------------
    # Stage 4: Stratified train/validation split
    # ------------------------------------------------------------------
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        answer_labels = formatted_dataset["answer"]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1.0 - train_ratio,
            random_state=seed,
        )
        train_indices, val_indices = next(
            splitter.split(range(len(formatted_dataset)), answer_labels)
        )

        final_dataset = DatasetDict({
            "train": formatted_dataset.select(train_indices.tolist()),
            "validation": formatted_dataset.select(val_indices.tolist()),
        })

        from collections import Counter
        train_dist = Counter(final_dataset["train"]["answer"])
        val_dist = Counter(final_dataset["validation"]["answer"])
        logger.info(f"Stratified — Train: {dict(sorted(train_dist.items()))}")
        logger.info(f"Stratified — Val:   {dict(sorted(val_dist.items()))}")

    except ImportError:
        logger.warning("scikit-learn not installed — using random split.")
        split_dataset = formatted_dataset.train_test_split(
            test_size=1.0 - train_ratio,
            seed=seed,
            shuffle=True,
        )
        final_dataset = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

    # ------------------------------------------------------------------
    # Stage 5: Enforce hard caps on split sizes (RAM safety)
    # ------------------------------------------------------------------
    if max_train_samples and len(final_dataset['train']) > max_train_samples:
        final_dataset['train'] = final_dataset['train'].shuffle(seed=seed).select(range(max_train_samples))
        logger.info(f"Train split capped to {max_train_samples} samples (RAM safety)")
    if max_val_samples and len(final_dataset['validation']) > max_val_samples:
        final_dataset['validation'] = final_dataset['validation'].shuffle(seed=seed).select(range(max_val_samples))
        logger.info(f"Validation split capped to {max_val_samples} samples (RAM safety)")

    logger.info(
        f"Final dataset — Train: {len(final_dataset['train'])} | "
        f"Validation: {len(final_dataset['validation'])}"
    )
    return final_dataset


# ===========================================================================
# Validation & Inspection Utilities
# ===========================================================================

def validate_dataset(dataset: DatasetDict) -> None:
    """
    Run sanity checks on the prepared dataset before training.

    Checks:
      - 'prompt' and 'answer' columns present
      - Prompts are non-empty lists of message dicts (system + user)
      - Answers are valid letters (A/B/C/D)
      - System prompt references SOFA + Cactus Routing
      - User prompt references MAP formula and Cactus routing reminder

    Args:
        dataset: The prepared DatasetDict to validate.
    """
    logger.info("Running dataset validation...")

    for split_name in dataset:
        split = dataset[split_name]
        logger.info(f"  Validating '{split_name}' ({len(split)} examples)...")

        assert "prompt" in split.column_names, f"Missing 'prompt' column in {split_name}"
        assert "answer" in split.column_names, f"Missing 'answer' column in {split_name}"

        n_check = min(100, len(split))
        errors = 0

        for i in range(n_check):
            example = split[i]

            # Prompt structure
            prompt = example["prompt"]
            if not isinstance(prompt, list) or len(prompt) < 2:
                logger.error(f"  [{split_name}][{i}] Invalid prompt: {type(prompt)}")
                errors += 1
                continue

            # Roles check
            roles = [msg["role"] for msg in prompt]
            if roles != ["system", "user"]:
                logger.error(f"  [{split_name}][{i}] Unexpected roles: {roles}")
                errors += 1

            # System prompt should reference SOFA + Cactus
            system_content = prompt[0]["content"]
            if "SOFA" not in system_content:
                logger.warning(f"  [{split_name}][{i}] System prompt missing SOFA reference")
            if "escalate" not in system_content:
                logger.warning(f"  [{split_name}][{i}] System prompt missing Cactus routing")

            # User prompt should reference MAP formula
            user_content = prompt[1]["content"]
            if "MAP" not in user_content:
                logger.warning(f"  [{split_name}][{i}] User prompt missing MAP reference")

            # Answer validity
            answer = example["answer"]
            if answer not in {"A", "B", "C", "D"}:
                logger.error(f"  [{split_name}][{i}] Invalid answer: '{answer}'")
                errors += 1

        status = f"{errors}/{n_check} errors" if errors else f"All {n_check} passed ✓"
        logger.info(f"  {split_name}: {status}")

    logger.info("Validation complete.")


def print_example(dataset: DatasetDict, split: str = "train", index: int = 0) -> None:
    """Pretty-print a single example for visual inspection."""
    example = dataset[split][index]

    print("=" * 80)
    print(f"EXAMPLE [{split}][{index}]")
    print("=" * 80)

    for msg in example["prompt"]:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n{'─' * 40}")
        print(f"[{role}]")
        print(f"{'─' * 40}")
        # Truncate system prompt for readability
        if role == "SYSTEM" and len(content) > 600:
            print(content[:600] + "\n... [truncated — see SYSTEM_PROMPT constant]")
        else:
            print(content)

    print(f"\n{'─' * 40}")
    print(f"[GROUND TRUTH ANSWER]: {example['answer']}")
    print("=" * 80)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gemma-Sync MedQA Data Pipeline — SOFA-First + Cactus Routing"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit total samples (for debugging). Default: use all data.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/medqa_gemma_sync",
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "--preview", type=int, default=3,
        help="Number of examples to print for visual inspection.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=LOCAL_DATA_DIR,
        help="Path to local JSONL dataset directory.",
    )
    args = parser.parse_args()

    dataset = load_and_prepare_dataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )
    validate_dataset(dataset)

    for i in range(min(args.preview, len(dataset["train"]))):
        print_example(dataset, split="train", index=i)

    logger.info(f"Saving processed dataset to: {args.output_dir}")
    dataset.save_to_disk(args.output_dir)
    logger.info("Gemma-Sync data pipeline complete. ✓")
