
<div align="center">

# Gemma-Sync

### Distributed Uncertainty-Aware Clinical Reasoning via Gemma 4 E2B

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MedQA--USMLE-orange.svg)](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19913606-blue.svg)](https://doi.org/10.5281/zenodo.19913606)
[![Model](https://img.shields.io/badge/Model-Gemma%204%20E2B%20IT-red.svg)](https://huggingface.co/google/gemma-4-e2b-it)

**Author:** Narendra Bayutama Wibisono

</div>

## Abstract

Gemma-Sync is a clinical alignment framework that trains [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-e2b-it) for uncertainty-aware medical reasoning on dual NVIDIA T4 GPUs using **Group Relative Policy Optimization (GRPO)** with **Reinforcement Learning from Verifiable Rewards (RLVR)**. The framework integrates three synergistic innovations:

1. **SOFA-First Reasoning Pipeline** — a structured chain-of-thought protocol that forces the model to *first* extract all six Sequential Organ Failure Assessment (SOFA) parameters from the clinical vignette, calculate sub-scores using deterministic thresholds (Vincent et al., 1996), and compute MAP from blood pressure using `MAP = (SBP + 2 × DBP) / 3` before any diagnostic reasoning begins. When clinical data is absent, the model must flag it as `N/P` (Not Provided) rather than fabricate values.
2. **Metacognitive Calibration** — the model explicitly states confidence levels (High / Moderate / Low) and identifies which missing information would materially change its differential diagnosis. For non-critical-care scenarios (e.g., psychiatry, ophthalmology, preventive medicine), the model emits `SOFA_NOT_APPLICABLE` with a documented rationale.
3. **Cactus Routing Protocol** — each response terminates with exactly one routing signal: `<|local_ok|>` (confident local inference) or `<|escalate|>` (route to a larger specialist model). False confidence — emitting `<|local_ok|>` while ≥3 SOFA parameters are marked `N/P` — is explicitly penalized during training.

This framework evolved from the TPU v5e-8 training pipeline ([Zenodo: 10.5281/zenodo.19913606](https://doi.org/10.5281/zenodo.19913606)) and has been re-engineered for the constrained VRAM envelope of Kaggle's dual T4 GPU environment using [Unsloth](https://github.com/unslothai/unsloth) for 2× throughput optimization.

---

## Architecture: From TPU v5e-8 to Dual NVIDIA T4

| Dimension                 | TPU v5e-8 Pipeline           | Gemma-Sync (Dual T4)                              |
| ------------------------- | ---------------------------- | ------------------------------------------------- |
| **Hardware**        | 8-core TPU v5e (128 GB HBM)  | 2× NVIDIA T4 (16 GB VRAM each)                   |
| **Precision**       | BF16 (TPU-native)            | FP16 + NF4 QLoRA (T4 Tensor Cores)                |
| **Distribution**    | 8-way data parallelism (XLA) | 2-GPU data parallelism (HF Accelerate)            |
| **Model Loader**    | Standard HF + PEFT           | Unsloth FastLanguageModel (2× throughput)        |
| **GRPO Group (G)**  | 16 (2 per core × 8 cores)   | 4 (streamed in generation_batch_size=4)           |
| **Memory Strategy** | TPU HBM headroom             | NF4 4-bit quantization + gradient checkpointing   |
| **Compute Backend** | JAX / PyTorch XLA            | PyTorch CUDA + Unsloth optimizations              |
| **Reward System**   | 4-tier RLVR                  | 4-tier RLVR + Calibrated Abstention bonus (+0.20) |

### VRAM Budget per T4 GPU (16 GB)

| Component                           | Estimated VRAM      |
| ----------------------------------- | ------------------- |
| Base model (Gemma 4 E2B, NF4 4-bit) | ~1.2 GB             |
| LoRA adapters (r=32, FP16)          | ~0.05 GB            |
| AdamW optimizer states (LoRA only)  | ~0.10 GB            |
| Gradient checkpointing buffer       | ~0.30 GB            |
| KV cache (gen_batch=4, seq=1024)    | ~1.5 GB             |
| Activations + CUDA overhead         | ~6.0 GB             |
| **Total per GPU**             | **~9.2 GB**   |
| **Headroom**                  | **6.8 GB** ✓ |

---

## Core Methodology

### GRPO: Group Relative Advantage Computation

Standard RLHF requires a separate reward model. GRPO (Shao et al., 2024) eliminates this by computing **relative advantages within a group** of $G$ completions sampled per prompt:

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}
$$

where $r_i$ is the reward for completion $i$, and $\mu_G$, $\sigma_G$ are the group mean and standard deviation.

### Reward Architecture (4-Tier + Calibrated Abstention Bonus)

$$
R = 0.50 \cdot r_{\text{correct}} + 0.20 \cdot r_{\text{sofa}} + 0.10 \cdot r_{\text{format}} + 0.20 \cdot r_{\text{process}} \; [+0.20 \text{ bonus}]
$$

| Weight | Reward Tier                           | Verification Method                                                                                                                                  |
| ------ | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.50   | **Correctness** (RLVR)          | Exact match of`\boxed{X}` against ground truth                                                                                                     |
| 0.20   | **SOFA Oracle**                 | Deterministic 6-system verifier (table + coverage + validity + plausibility + arithmetic)                                                            |
| 0.10   | **Format Compliance**           | LaTeX`\boxed{}` (0.40), SOFA section (0.30), step markers (0.20), uncertainty (0.10)                                                               |
| 0.20   | **Process Quality**             | 6-dimensional CoT heuristic: medical terms, length quality, causal reasoning, elimination, transitions, Cactus signal                                |
| +0.20  | **Calibrated Abstention Bonus** | Awarded for ≥2 abstention signals (N/P flags,`SOFA_NOT_APPLICABLE`, confidence markers, `<\|escalate\|>`). Penalized −0.10 for false confidence. |

### SOFA Oracle — Deterministic Clinical Verification

The `_score_sofa_oracle()` function validates the model's SOFA table across five sub-dimensions:

| Sub-Dimension         | Weight | Description                                                   |
| --------------------- | ------ | ------------------------------------------------------------- |
| Table Present         | 0.25   | SOFA table or`SOFA_NOT_APPLICABLE` found                    |
| Coverage              | 0.20   | Fraction of 6 organ systems extracted                         |
| Score Validity        | 0.20   | Sub-scores within [0–4] or`N/P`                            |
| Clinical Plausibility | 0.20   | Extracted value → score mapping validated against thresholds |
| Arithmetic            | 0.15   | Claimed total matches computed sum                            |

A **hallucination penalty** of −0.10 is applied per SOFA row that claims a numeric score without supporting numeric evidence.

### Cactus Routing Protocol

Each response must terminate with exactly one routing signal:

| Signal           | Meaning                          | Trigger Conditions                                                     |
| ---------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `<\|local_ok\|>` | Confident — serve locally       | High confidence, reasoning complete, SOFA resolved                     |
| `<\|escalate\|>` | Uncertain — route to specialist | ≥2 critical N/P parameters, Low confidence, ambiguous pathophysiology |

**False confidence penalty:** Emitting `<|local_ok|>` with ≥3 `N/P` entries and a guessed SOFA total → −0.10 penalty.

### Deterministic MAP Calculation

When blood pressure is given as SBP/DBP, the model **must** compute MAP before assigning a cardiovascular SOFA sub-score:

$$
\text{MAP} = \frac{\text{SBP} + 2 \times \text{DBP}}{3}
$$

The SOFA Oracle's `_validate_cv_score()` independently verifies the model's MAP calculation against the formula and validates the corresponding SOFA score against published thresholds (MAP ≥ 70 → SOFA 0, MAP < 70 → SOFA 1, vasopressor-dependent → SOFA 2–4).

---

## Repository Structure

```
gemma-sync/
├── src/
│   ├── data_pipeline.py               # MedQA → GRPO-formatted HF Dataset
│   ├── distributed_grpo_trainer.py    # Full GRPO training (Dual T4 + Unsloth)
│   ├── inference_check.py             # Post-training 4-dimension validation
│   ├── evaluation_base.py             # Ablation baseline (pure base model)
│   └── evaluation_final.py            # Fine-tuned model evaluation (paired test)
├── notebooks/
│   └── emergency-enabled-clinical-reasoning-via-grpo-rl.ipynb
├── medqa_dataset/                     # Pre-downloaded MedQA-USMLE JSONL files
├── data/medqa_gemma_sync/             # Processed HF Dataset (output of pipeline)
├── outputs/                           # Checkpoints, adapters, reports
├── results/                           # Archived evaluation CSVs
├── requirements.txt
└── README.md
```

---

## Installation & Environment Setup

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with ≥ 16 GB VRAM (dual T4 recommended; single T4 supported)
- CUDA ≥ 11.8

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/<your-username>/gemma-sync.git
cd gemma-sync

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.2.0
transformers>=4.45.0
datasets>=2.19.0
trl>=0.17.0
peft>=0.13.0
bitsandbytes>=0.44.0
accelerate>=1.0.0
scikit-learn>=1.4.0
unsloth                      # Primary model loader (2× throughput on T4)
sentencepiece>=0.2.0
protobuf>=4.25.0
```

> **Kaggle Note:** On Kaggle GPU VMs, most packages are pre-installed. Gemma 4 E2B weights can be loaded from the Kaggle Models hub mount at `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1` or downloaded from HuggingFace Hub (`google/gemma-4-e2b-it`).

---

## Data Pipeline

The data pipeline converts raw MedQA-USMLE-4-options JSONL files into a HuggingFace `DatasetDict` formatted for TRL's `GRPOTrainer`, with the full Gemma-Sync SOFA-First + Cactus Routing system prompt injected into each example.

### Quick Start

```bash
# Process all 11,451 examples with default settings
python src/data_pipeline.py

# Debug mode: process 100 examples, preview 5
python src/data_pipeline.py --max-samples 100 --preview 5

# Custom paths
python src/data_pipeline.py \
    --data-dir ./medqa_dataset \
    --output-dir ./data/medqa_gemma_sync \
    --preview 3
```

### Command-Line Arguments

| Argument          | Type    | Default                     | Description                                       |
| ----------------- | ------- | --------------------------- | ------------------------------------------------- |
| `--data-dir`    | `str` | `./medqa_dataset`         | Path to local JSONL dataset directory             |
| `--max-samples` | `int` | `None` (all)              | Limit total samples (for debugging)               |
| `--output-dir`  | `str` | `./data/medqa_gemma_sync` | Directory to save processed HF Dataset            |
| `--preview`     | `int` | `3`                       | Number of examples to print for visual inspection |

### Pipeline Stages

1. **Load** — Reads `phrases_no_exclude_train.jsonl` (10,178 examples) and `phrases_no_exclude_test.jsonl` (1,273 examples) from `./medqa_dataset/`.
2. **Clean** — Normalises Unicode artifacts (non-breaking spaces, smart quotes, en/em dashes, ellipses).
3. **Format** — Wraps each example in a `[system, user]` conversation template embedding the Gemma-Sync SOFA-First + Cactus Routing system prompt with MAP calculation reminder.
4. **Split** — Stratified 90/10 train/validation split (balanced by answer label A/B/C/D) using `sklearn.model_selection.StratifiedShuffleSplit`.
5. **Cap** — Hard caps: 202 train / 50 validation (RAM safety for T4 environments).

---

## Training Execution

### Model & Quantization

| Parameter              | Value                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Base Model             | Gemma 4 E2B IT (`google/gemma-4-e2b-it`)                                                |
| Model Loader           | Unsloth FastLanguageModel (primary) / HF + BitsAndBytes NF4 (fallback)                    |
| Quantization           | QLoRA — 4-bit NF4 + double quantization                                                  |
| Compute Precision      | FP16 (T4 Tensor Cores)                                                                    |
| LoRA Rank / Alpha      | 32 / 64                                                                                   |
| LoRA Dropout           | 0.05                                                                                      |
| LoRA Targets           | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient Checkpointing | Unsloth-optimized (primary) /`use_reentrant=False` (fallback)                           |

### Launch Training

```bash
# Full training (202 steps on 2× T4 GPUs)
python src/distributed_grpo_trainer.py

# Quick dry-run validation (4 steps, 32 samples)
python src/distributed_grpo_trainer.py --dry-run

# Validate reward functions only (no GPU needed)
python src/distributed_grpo_trainer.py --validate-rewards

# Print GGUF export guide only
python src/distributed_grpo_trainer.py --export-guide-only

# Custom configuration
python src/distributed_grpo_trainer.py \
    --epochs 2 \
    --batch-size 1 \
    --grad-accum 1 \
    --lr 5e-6 \
    --num-generations 4 \
    --max-completion-length 1024 \
    --max-steps 202 \
    --save-steps 50 \
    --output-dir ./outputs/gemma4-e2b-grpo-sofa-t4

# Use HF Hub model instead of Kaggle local path
python src/distributed_grpo_trainer.py --hf-model

# Disable Unsloth (use standard HF + BitsAndBytes)
python src/distributed_grpo_trainer.py --no-unsloth

# Resume from checkpoint
python src/distributed_grpo_trainer.py \
    --resume-from-checkpoint ./outputs/gemma4-e2b-grpo-sofa-t4/checkpoint-100
```

### Training Arguments

| Argument                     | Default                         | Description                                |
| ---------------------------- | ------------------------------- | ------------------------------------------ |
| `--model-id`               | Kaggle local path               | Path to base model directory               |
| `--hf-model`               | `false`                       | Use HF Hub ID for online prototyping       |
| `--output-dir`             | `/kaggle/working/outputs/...` | Checkpoint & adapter output                |
| `--epochs`                 | `2`                           | Number of training epochs                  |
| `--batch-size`             | `1`                           | Per-GPU batch size                         |
| `--grad-accum`             | `1`                           | Gradient accumulation steps                |
| `--lr`                     | `5e-6`                        | Learning rate                              |
| `--num-generations`        | `4`                           | GRPO group size G                          |
| `--max-completion-length`  | `1024`                        | Max tokens per completion                  |
| `--max-steps`              | `202`                         | Maximum training steps                     |
| `--save-steps`             | `50`                          | Checkpoint interval                        |
| `--max-samples`            | `None`                        | Limit dataset size for debugging           |
| `--dry-run`                | `false`                       | Run 4 steps for pipeline validation only   |
| `--validate-rewards`       | `false`                       | Run reward self-tests and exit             |
| `--export-guide-only`      | `false`                       | Print GGUF export guide and exit           |
| `--resume-from-checkpoint` | `None`                        | Path to checkpoint for training resumption |
| `--no-unsloth`             | `false`                       | Disable Unsloth; use HF + BnB fallback     |

### GRPOTrainer Workflow

1. **Environment** — CUDA initialization, dual T4 GPU detection, VRAM logging.
2. **Dataset** — Load MedQA via `data_pipeline.py` (delegates formatting + stratified split).
3. **Model** — Load Gemma 4 E2B via Unsloth with NF4 QLoRA + LoRA (r=32, α=64).
4. **Config** — Build `GRPOConfig` with FP16, cosine LR schedule, fused AdamW.
5. **Train** — `GRPOTrainer` with 4 reward functions: `reward_correctness`, `reward_sofa_oracle`, `reward_format`, `reward_process_quality`.
6. **Save** — LoRA adapter + training metadata (multi-GPU safe: only `LOCAL_RANK=0` writes).
7. **Export** — Print GGUF conversion guide for Ollama / llama.cpp deployment.

---

## Evaluation Pipeline

### Ablation Baseline (Base Model — No Fine-Tuning)

```bash
# Full baseline evaluation (200 unseen samples)
python src/evaluation_base.py

# Quick test (10 samples)
python src/evaluation_base.py --num-eval-samples 10

# Dry run (synthetic completions, no GPU required)
python src/evaluation_base.py --dry-run --num-eval-samples 5
```

### Fine-Tuned Model Evaluation

```bash
# Full evaluation (200 unseen samples, merged LoRA adapter)
python src/evaluation_final.py

# Quick test
python src/evaluation_final.py --num-eval-samples 10
```

### Audit-First Split Reproduction (Zero Contamination Guarantee)

Both evaluation scripts implement an **audit-first** pipeline that guarantees zero data contamination:

1. **Reproduce** the exact `StratifiedShuffleSplit` from `data_pipeline.py` to identify the 202 training indices ("Blacklist").
2. **Pool** all remaining samples that were NOT in training.
3. **Sample** exactly 200 unseen examples using `random.seed(42)`.
4. **Evaluate** with identical seeds, prompts, and extraction logic for valid paired comparison.

### Inference Validation (Post-Training)

```bash
# Full validation (200 samples)
python src/inference_check.py --num-samples 200

# Dry-run (10 samples, M2 prototyping)
python src/inference_check.py --num-samples 10 --dry-run
```

Validates across four clinical performance dimensions:

- **Clinical Accuracy** — `\boxed{X}` exact match against ground truth
- **SOFA Precision** — Table presence, 6-component coverage, N/P discipline
- **MAP Calculation** — Deterministic BP-to-MAP verification (tolerance ±2 mmHg)
- **Cactus Routing** — `<|escalate|>` / `<|local_ok|>` token frequency and false-confidence detection

---

## Deployment: GGUF Export to Ollama

After training, the LoRA adapter can be merged, quantised, and deployed locally:

```bash
# Step 1: Merge LoRA into base weights
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('google/gemma-4-e2b-it', torch_dtype='float16')
merged = PeftModel.from_pretrained(base, './outputs/gemma4-e2b-grpo-sofa-t4/final_adapter').merge_and_unload()
merged.save_pretrained('./merged_gemma4_e2b_sofa')
AutoTokenizer.from_pretrained('./outputs/gemma4-e2b-grpo-sofa-t4/final_adapter').save_pretrained('./merged_gemma4_e2b_sofa')
"

# Step 2: Convert to GGUF (requires llama.cpp)
python convert_hf_to_gguf.py ./merged_gemma4_e2b_sofa \
       --outtype f16 --outfile gemma4_sofa_f16.gguf
./llama-quantize gemma4_sofa_f16.gguf gemma4_sofa_q4km.gguf Q4_K_M

# Step 3: Create Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./gemma4_sofa_q4km.gguf
SYSTEM "You are Gemma-Sync. Use SOFA-First reasoning. Emit <|escalate|> if you need specialist review. Emit <|local_ok|> if confident."
PARAMETER temperature 0.7
PARAMETER stop "<|escalate|>"
PARAMETER stop "<|local_ok|>"
EOF
ollama create gemma-sync-sofa -f Modelfile && ollama run gemma-sync-sofa
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wibisono2026gemma_sync,
  author       = {Wibisono, Narendra Bayutama},
  title        = {{Gemma-Sync: Distributed Uncertainty-Aware Clinical Reasoning
                   via Gemma 4 E2B with GRPO \& RLVR}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19599245},
  url          = {https://doi.org/10.5281/zenodo.19599245}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The MedQA-USMLE-4-options dataset is distributed under its original license by [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) on HuggingFace. Gemma 4 is subject to [Google&#39;s Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## Acknowledgements

- **MedQA Dataset:** Jin et al. (2021). *What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams.* Applied Sciences, 11(14), 6421.
- **GRPO:** Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*
- **SOFA Score:** Vincent et al. (1996). *The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.* Intensive Care Medicine, 22(7), 707–710.
- **Unsloth:** Daniel & Michael Han. [Unsloth — 2× faster LLM fine-tuning](https://github.com/unslothai/unsloth).
- **TRL Library:** Hugging Face TRL team.
- **Google Gemma:** Google DeepMind.

