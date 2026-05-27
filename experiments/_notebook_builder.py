"""Generate the 7 experiment notebooks from declarative specs.

Run once after editing the specs::

    python experiments/_notebook_builder.py

Design notes:

* The Colab bootstrap, GPU check, env setup, dataset picker, test-prompt
  battery and model-save cells are **shared** across every notebook so
  the per-experiment notebook only has to differ in sections 2-5.
* Model saving uses MLflow's ``tracking.log_model`` — that already
  stores the artifact in the MLflow artifact store (MinIO, in your
  k3s cluster). No separate MinIO upload needed.
* Section separators use ``# ####`` blocks to match the rest of the
  codebase.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO / "experiments"


# ##################################################################
# Notebook JSON helpers
# ##################################################################


def code_cell(source: str) -> dict:
    """Wrap a Python string into a Jupyter code cell dict."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    """Wrap a Markdown string into a Jupyter markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def make_notebook(cells: list[dict]) -> dict:
    """Build a complete .ipynb document from a list of cells."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ##################################################################
# Shared cells used across every notebook
# ##################################################################

COLAB_INSTALL = code_cell("""# ###### Colab bootstrap ######
# On Colab the [experiments] extras pulls both requirements_package.txt and
# requirements_experiments.txt in one pip command — single source of truth lives
# in setup.py's extras_require. Then bootstrap() loads Colab Secrets into
# os.environ and brings up Tailscale so *.ts.net hostnames are reachable.
# Locally bootstrap() just loads .env and checks the tailnet — no installs.
#
# Required Colab Secrets (key icon → Add new secret → toggle "Notebook access"):
#   TAILSCALE_AUTHKEY   – from https://login.tailscale.com/admin/settings/keys
#   MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_INSECURE_TLS
#   MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY / MINIO_SECURE
#   HF_TOKEN
import os, subprocess, sys
IN_COLAB = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "burnit_bg[experiments] @ git+https://github.com/kirilyotov/BurnIT-BG.git",
    ])

from utils.colab import bootstrap
bootstrap()
""")

GPU_CHECK = code_cell("""import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"VRAM:   {props.total_memory / 1024**3:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
""")


def shared_import_cell() -> dict:
    """Imports used by every notebook — keeps top-of-cell tidy."""
    return code_cell("""import sys, os
from pathlib import Path

REPO_ROOT = Path.cwd()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / 'data_platform').exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from data_platform.common import set_env
from data_platform.storage import MinioStorage

from experiments.shared.mlflow_utils import setup_run, log_responses, stage, log_training_curves
from experiments.shared.inference_utils import (
    run_full_test_battery,
    TEST_PROMPTS_IN_DOMAIN, TEST_PROMPTS_OUT_OF_DOMAIN, TEST_PROMPTS_EDGE,
    format_prompt,
)
from experiments.shared.model_utils import (
    DEFAULT_MODEL_NAME, load_model_unsloth, apply_qlora, count_trainable_params,
    cuda_device_info,
)
from experiments.shared.eval_utils import compute_perplexity, benchmark_speed, vram_snapshot
from experiments.shared.dataset_utils import (
    load_alpaca_dataset, dataset_statistics, alpaca_to_prompt,
)
from experiments.shared.dataset_browser import list_datasets, pick_dataset, download_dataset, resolve
""")


def setup_cell(experiment: str, mlflow_experiment: str = "burnit-bg-experiments") -> dict:
    """Wire MLflow + tags for the experiment in one cell."""
    return code_cell(f"""set_env(quiet=True)
tracking, tags, run_name = setup_run(
    experiment={experiment!r},
    model=DEFAULT_MODEL_NAME,
    stage="experiment",
    mlflow_experiment_name={mlflow_experiment!r},
)
print(f"run_name = {{run_name}}")
print(f"tags     = {{tags}}")
print(f"machine  = {{cuda_device_info()}}")
""")


# Reusable data-loading cell — uses the dataset browser to pick a
# bucket-stored dataset, or falls back to a local path.
DATA_LOAD_CELL = code_cell("""# ###### Pick a dataset from MinIO (or fall back to local data_prep/) ######
# Set DATASET_PREFIX in .env / Colab secrets to skip the picker, e.g.
#   DATASET_PREFIX=data_prep/processed/mental-health
# Or pass `auto=True` below for non-interactive runs.

DEFAULT_PREFIX = os.getenv("DATASET_PREFIX", "data_prep/processed")
try:
    local_dataset_dir = resolve(prefix=DEFAULT_PREFIX, auto=False)
    TRAIN_PATH = local_dataset_dir / "train.jsonl"
    EVAL_PATH  = local_dataset_dir / "eval.jsonl"
except (FileNotFoundError, Exception) as exc:
    print(f"[data] MinIO lookup failed ({exc}); falling back to local data_prep/processed/")
    TRAIN_PATH = REPO_ROOT / "data_prep/processed/train.jsonl"
    EVAL_PATH  = REPO_ROOT / "data_prep/processed/eval.jsonl"

train_records = list(load_alpaca_dataset(TRAIN_PATH))
eval_records  = list(load_alpaca_dataset(EVAL_PATH))
train_stats = dataset_statistics(train_records)
eval_stats  = dataset_statistics(eval_records)
print(f"train: {len(train_records)} rows  eval: {len(eval_records)} rows")
print(train_stats)
""")


# Per-experiment save block — uses MLflow log_model. Replaces the old
# save_to_minio + export_gguf pattern: MLflow's tracking server already
# stores artifacts in MinIO. GGUF export is logged as a run artifact too.
def save_block(experiment_key: str, registered_model_name: str | None = None) -> dict:
    reg = repr(registered_model_name) if registered_model_name else "None"
    return code_cell(f"""# ###### Save model via MLflow (single source of truth) ######
# tracking.log_model logs the model artifact under runs:/<id>/model and,
# when registered_model_name is set, adds a new version to the Models tab.
# MLflow's artifact store backs onto MinIO — no separate upload needed.
with stage(tracking, "save_model"):
    try:
        tracking.log_model(
            model,
            flavor="transformers",
            artifact_path="model",
            registered_model_name={reg},
            input_example=None,
        )
        print("[save] model logged via MLflow")
    except Exception as exc:
        print(f"[save] log_model failed: {{exc}}")

# Optional: GGUF export for offline local inference (RTX 3050 / Ollama).
# The GGUF is added as a run artifact under `gguf/`.
with stage(tracking, "gguf_export"):
    try:
        from experiments.shared.model_utils import export_gguf
        gguf_path = export_gguf(model, tokenizer, REPO_ROOT / "tmp/experiments/{experiment_key}/gguf", quantization="q4_k_m")
        tracking.save_data(gguf_path, artifact_path="gguf")
        print(f"[save] GGUF logged: {{gguf_path}}")
    except Exception as exc:
        print(f"[save] GGUF export skipped: {{exc}}")
""")


# Test-prompt battery cell — identical across all notebooks for
# directly comparable side-by-side responses.
def test_prompts_cell(experiment_key: str) -> dict:
    return code_cell(f"""# ###### Inference test (mental-health prompts) ######
with stage(tracking, "inference_test"):
    batteries = run_full_test_battery((model, tokenizer), max_new_tokens=256)
    log_responses(
        tracking,
        experiment={experiment_key!r},
        model_checkpoint=str(REPO_ROOT / "tmp/experiments/{experiment_key}"),
        **batteries,
    )
    for k, v in batteries.items():
        print(f"-- {{k}} --")
        for entry in v[:2]:
            print(f"  Q: {{entry['prompt'][:80]}}\\n  A: {{entry['response'][:200]}}\\n")
""")


# ##################################################################
# Notebook 01 — baseline QLoRA fine-tuning (fully implemented)
# ##################################################################

NOTEBOOK_01 = make_notebook([
    md_cell("""# 01 — Baseline QLoRA Fine-tuning

Establish a QLoRA fine-tuning baseline on `meta-llama/Llama-3.2-3B-Instruct` before any pruning, R-tuning, or unlearning.

**Sequence**

1. Setup & config
2. Data loading (interactive MinIO picker)
3. Model loading with Unsloth (4-bit) + QLoRA adapters
4. Training (SFTTrainer)
5. Evaluation — perplexity on eval set
6. Inference test — all mental-health prompt batteries
7. Save model via MLflow + GGUF export

Logs to MLflow under experiment `burnit-bg-experiments`, run tagged `experiment=baseline`.
"""),
    COLAB_INSTALL,
    GPU_CHECK,
    md_cell("## 1. Setup & config"),
    shared_import_cell(),
    setup_cell("baseline"),
    md_cell("## 2. Data loading\n\nPicks a dataset from MinIO via `resolve()`. Set `DATASET_PREFIX` env var to skip the prompt, or pass `auto=True`."),
    DATA_LOAD_CELL,
    code_cell("""# Quick category distribution chart
import pandas as pd, matplotlib.pyplot as plt
cat_counts = pd.Series(train_stats["by_category"]).sort_values(ascending=False)
ax = cat_counts.plot(kind="bar", figsize=(8, 3.5), title="Train: examples per category")
plt.tight_layout(); plt.show()
"""),
    md_cell("## 3. Model loading"),
    code_cell("""HF_TOKEN = os.getenv("HF_TOKEN")
MAX_SEQ_LEN = 2048

with stage(tracking, "load_model"):
    model, tokenizer = load_model_unsloth(
        DEFAULT_MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        token=HF_TOKEN,
    )
    model = apply_qlora(model, r=16, lora_alpha=32)
    params = count_trainable_params(model)
    print(params)
"""),
    md_cell("## 4. Training"),
    code_cell("""from datasets import Dataset

def _format(record):
    record["text"] = alpaca_to_prompt(record, eos_token=tokenizer.eos_token)
    return record

train_ds = Dataset.from_list(train_records).map(_format)
eval_ds  = Dataset.from_list(eval_records).map(_format)
"""),
    code_cell("""from trl import SFTTrainer
from transformers import TrainingArguments

OUTPUT_DIR = REPO_ROOT / "tmp/experiments/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_8bit",
    report_to=[],
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

with tracking.run(run_name=run_name, tags=tags):
    tracking.log_params({
        **{f"data.{k}": v for k, v in train_stats.items() if not isinstance(v, dict)},
        **{f"model.{k}": v for k, v in params.items()},
        **{f"train.{k}": getattr(training_args, k) for k in (
            "num_train_epochs","per_device_train_batch_size","gradient_accumulation_steps",
            "learning_rate","warmup_ratio","lr_scheduler_type","max_grad_norm",
        )},
        "max_seq_length": MAX_SEQ_LEN,
    })

    with stage(tracking, "train"):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            args=training_args,
        )
        trainer_stats = trainer.train()

    history_steps = [h["step"] for h in trainer.state.log_history if "loss" in h]
    history_loss  = [h["loss"]  for h in trainer.state.log_history if "loss" in h]
    history_lr    = [h.get("learning_rate", 0.0) for h in trainer.state.log_history if "loss" in h]
    if history_steps:
        log_training_curves(tracking, steps=history_steps, losses=history_loss, learning_rates=history_lr, title="baseline")
    tracking.log_metrics({"final_train_loss": float(history_loss[-1]) if history_loss else 0.0})

    with stage(tracking, "evaluate"):
        ppl = compute_perplexity(model, tokenizer, [r["output"] for r in eval_records[:64]])
        tracking.log_metrics({"eval_perplexity": float(ppl)})
        print(f"eval perplexity = {ppl:.3f}")
        tracking.log_metrics({f"vram.{k}": v for k, v in vram_snapshot().items()})

    # Test prompts
    batteries = run_full_test_battery((model, tokenizer), max_new_tokens=256)
    log_responses(tracking, experiment="baseline",
                  model_checkpoint=str(OUTPUT_DIR), **batteries)
    for k, v in batteries.items():
        print(f"-- {k} --")
        for entry in v[:2]:
            print(f"  Q: {entry['prompt'][:80]}\\n  A: {entry['response'][:200]}\\n")

    # Save via MLflow — artifact + registered model version
    with stage(tracking, "save_model"):
        try:
            tracking.log_model(
                model,
                flavor="transformers",
                artifact_path="model",
                registered_model_name="burnit-bg-baseline",
            )
            print("[save] model logged via MLflow")
        except Exception as exc:
            print(f"[save] log_model failed: {exc}")

    # Optional GGUF export
    with stage(tracking, "gguf_export"):
        try:
            from experiments.shared.model_utils import export_gguf
            gguf_path = export_gguf(model, tokenizer, OUTPUT_DIR / "gguf", quantization="q4_k_m")
            tracking.save_data(gguf_path, artifact_path="gguf")
            print(f"[save] GGUF logged: {gguf_path}")
        except Exception as exc:
            print(f"[save] GGUF export skipped: {exc}")

    tracking.log_hardware(step=1)
"""),
    md_cell("""## Next steps

* Compare this run in MLflow with later experiments by filtering tag `experiment=baseline`.
* Continue to `02_layer_pruning.ipynb` — it loads the baseline model from MLflow as its starting point.
"""),
])


# ##################################################################
# Specs for notebooks 02-07 (filled with concrete code)
# ##################################################################


SPECS: list[dict] = [
    {
        "key": "layer_pruning",
        "dir": "layer-pruning",
        "filename": "02_layer_pruning.ipynb",
        "title": "02 — Layer Pruning (TrimLLM)",
        "summary": "Drop the last ~25% of transformer layers, recover with QLoRA + LISA.",
        "papers": [
            "TrimLLM https://arxiv.org/abs/2412.11242",
            "Reassessing Layer Pruning https://arxiv.org/abs/2411.15558",
            "LISA https://arxiv.org/abs/2403.17919",
        ],
        "registered_name": "burnit-bg-layer-pruning",
        "section_2_title": "Load the baseline + measure layer similarity",
        "section_2_code": """# Load baseline weights from MLflow (replace with your baseline run-id).
# Falls back to the bare base model if no baseline run is given.
BASELINE_RUN_ID = os.getenv("BASELINE_RUN_ID")  # set in env or hardcode
model, tokenizer = load_model_unsloth(DEFAULT_MODEL_NAME, max_seq_length=2048,
                                      load_in_4bit=True, token=os.getenv("HF_TOKEN"))
if BASELINE_RUN_ID:
    try:
        local_path = tracking.load_model(run_id=BASELINE_RUN_ID, artifact_path="model")
        print(f"loaded baseline weights from {local_path}")
    except Exception as exc:
        print(f"[warn] baseline load failed ({exc}); continuing with bare base model")

# Layer similarity: capture hidden states with forward hooks, compute
# cosine similarity between consecutive layers.
import torch
from torch.nn.functional import cosine_similarity

@torch.no_grad()
def measure_layer_similarity(model, tokenizer, samples, max_tokens=256):
    layers = model.model.layers if hasattr(model, "model") else model.layers
    hidden_per_layer = []
    handles = []
    def make_hook(idx):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_per_layer.append((idx, h.mean(dim=1).detach().cpu()))
        return _hook
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        for text in samples:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
            model(**ids)
    finally:
        for h in handles: h.remove()
    # Aggregate per layer
    by_layer = {}
    for i, h in hidden_per_layer:
        by_layer.setdefault(i, []).append(h)
    means = [torch.cat(by_layer[i]).mean(dim=0) for i in sorted(by_layer)]
    sims = []
    for i in range(len(means) - 1):
        a, b = means[i], means[i+1]
        sims.append(float(cosine_similarity(a, b, dim=0).item()))
    return sims

similarities = measure_layer_similarity(
    model, tokenizer,
    samples=[alpaca_to_prompt(r) for r in eval_records[:32]],
)
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 3.5))
plt.plot(range(len(similarities)), similarities, marker="o")
plt.axhline(0.95, color="crimson", linestyle="--", label="prune-candidate threshold")
plt.xlabel("Layer index"); plt.ylabel("cos(layer_i, layer_{i+1})")
plt.title("Layer-to-layer similarity"); plt.legend(); plt.tight_layout()
plt.savefig("/tmp/layer_sim.png", dpi=140)
tracking.log_metrics({f"layer_sim.{i}": s for i, s in enumerate(similarities)})
plt.show()
""",
        "section_3_title": "Prune the last 25% of layers",
        "section_3_code": """# Remove the trailing ~25% of decoder blocks.
total_layers = len(model.model.layers)
prune_from = int(round(total_layers * 0.75))
keep = model.model.layers[:prune_from]
removed = total_layers - len(keep)
print(f"pruning layers {prune_from}..{total_layers - 1} ({removed} layers)")

import torch.nn as nn
model.model.layers = nn.ModuleList(keep)
if hasattr(model.config, "num_hidden_layers"):
    model.config.num_hidden_layers = len(keep)

params_after = count_trainable_params(model)
tracking.log_params({
    "prune.total_layers_before": total_layers,
    "prune.total_layers_after": len(keep),
    "prune.removed_layers": removed,
    "prune.params_after": params_after["total"],
})
""",
        "section_4_title": "Recovery fine-tuning (QLoRA + LISA)",
        "section_4_code": """# Re-apply QLoRA to the pruned model so we can recover quickly.
model = apply_qlora(model, r=16, lora_alpha=32)

# LISA: at each step, freeze all decoder blocks except a randomly-chosen K.
# This dramatically lowers VRAM and trains a different subset every step.
import random, torch
ACTIVE_LAYERS_PER_STEP = 4

def lisa_select(model, k=ACTIVE_LAYERS_PER_STEP):
    layers = model.model.layers
    chosen = set(random.sample(range(len(layers)), k=min(k, len(layers))))
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad_(i in chosen)
    return chosen

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

train_ds = Dataset.from_list(train_records).map(lambda r: {**r, "text": alpaca_to_prompt(r, eos_token=tokenizer.eos_token)})
eval_ds  = Dataset.from_list(eval_records).map(lambda r: {**r, "text": alpaca_to_prompt(r, eos_token=tokenizer.eos_token)})

class LISACallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kw):
        lisa_select(kw["model"])

OUTPUT_DIR = REPO_ROOT / "tmp/experiments/layer_pruning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR), num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=1e-4, warmup_ratio=0.03, logging_steps=10,
    save_strategy="epoch", bf16=True, optim="adamw_8bit",
    report_to=[], lr_scheduler_type="cosine", max_grad_norm=1.0,
)

with tracking.run(run_name=run_name, tags=tags):
    tracking.log_params({"lisa.active_per_step": ACTIVE_LAYERS_PER_STEP})
    with stage(tracking, "recovery_train"):
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, eval_dataset=eval_ds,
            dataset_text_field="text", max_seq_length=2048,
            args=training_args, callbacks=[LISACallback()],
        )
        trainer.train()
        steps = [h["step"] for h in trainer.state.log_history if "loss" in h]
        losses = [h["loss"]  for h in trainer.state.log_history if "loss" in h]
        if steps:
            log_training_curves(tracking, steps=steps, losses=losses, title="recovery")
""",
        "section_5_title": "Evaluation — perplexity & speed comparison",
        "section_5_code": """with stage(tracking, "evaluate"):
    ppl = compute_perplexity(model, tokenizer, [r["output"] for r in eval_records[:64]])
    speed = benchmark_speed(model, tokenizer, new_tokens=128, runs=3)
    tracking.log_metrics({
        "eval_perplexity": float(ppl),
        "tokens_per_sec": speed["tokens_per_sec_mean"],
        **{f"vram.{k}": v for k, v in vram_snapshot().items()},
    })
    print(f"perplexity={ppl:.3f}  tokens/sec={speed['tokens_per_sec_mean']:.1f}")
""",
    },
    {
        "key": "neuron_pruning",
        "dir": "neuron-pruning",
        "filename": "03_neuron_pruning.ipynb",
        "title": "03 — Neuron + Head Pruning (Wanda)",
        "summary": "Magnitude × activation pruning of MLP neurons and attention heads. Single forward pass; recover briefly with QLoRA.",
        "papers": [
            "Wanda https://arxiv.org/abs/2306.11695",
            "horseee/LLaMA-Pruning (GitHub)",
        ],
        "registered_name": "burnit-bg-neuron-pruning",
        "section_2_title": "Wanda calibration + neuron scoring",
        "section_2_code": """# Wanda: prune the neurons with the lowest |W| * ||X||_2 score.
import torch, torch.nn as nn

CALIBRATION_SAMPLES = 128
calibration_text = [alpaca_to_prompt(r) for r in train_records[:CALIBRATION_SAMPLES]]

@torch.no_grad()
def collect_activation_norms(model, tokenizer, texts):
    norms = {}  # name -> running sum of ||X||_2 per input channel
    handles = []
    def make_hook(name):
        def _hook(_m, inputs, _output):
            x = inputs[0]
            n = x.float().pow(2).sum(dim=(0, 1)).sqrt()  # per-input-channel L2
            norms[name] = norms.get(name, 0) + n.detach().cpu()
        return _hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(k in name for k in ("mlp.up_proj", "mlp.gate_proj")):
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        for txt in texts:
            ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            model(**ids)
    finally:
        for h in handles: h.remove()
    return norms

activation_norms = collect_activation_norms(model, tokenizer, calibration_text)
print(f"collected activations for {len(activation_norms)} MLP projections")
""",
        "section_3_title": "Prune lowest-scoring neurons",
        "section_3_code": """# For each MLP layer pair (up_proj, gate_proj), score weight rows by
# |W| * ||X||, then drop the lowest 30% by zeroing them out.
SPARSITY = 0.30
import torch
removed_total = 0
for name, mod in model.named_modules():
    if not (isinstance(mod, nn.Linear) and any(k in name for k in ("mlp.up_proj", "mlp.gate_proj"))):
        continue
    activations = activation_norms.get(name)
    if activations is None: continue
    score = mod.weight.detach().abs() * activations.to(mod.weight.device)
    per_neuron = score.sum(dim=1)
    threshold = torch.quantile(per_neuron, SPARSITY)
    keep_mask = per_neuron > threshold
    mod.weight.data[~keep_mask] = 0.0  # soft prune (no shape change)
    removed_total += int((~keep_mask).sum())

tracking.log_params({"prune.sparsity": SPARSITY})
tracking.log_metrics({"prune.neurons_zeroed": float(removed_total)})
print(f"zeroed {removed_total} MLP neurons across all layers")
""",
        "section_4_title": "Recovery fine-tune (lm_head + last 3 blocks)",
        "section_4_code": """# Freeze everything except the language-model head and the last 3
# decoder blocks; cheap recovery with QLoRA.
for p in model.parameters(): p.requires_grad_(False)
last_three = list(model.model.layers)[-3:]
for layer in last_three:
    for p in layer.parameters(): p.requires_grad_(True)
if hasattr(model, "lm_head"):
    for p in model.lm_head.parameters(): p.requires_grad_(True)

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

train_ds = Dataset.from_list(train_records).map(lambda r: {**r, "text": alpaca_to_prompt(r, eos_token=tokenizer.eos_token)})
eval_ds  = Dataset.from_list(eval_records).map(lambda r: {**r, "text": alpaca_to_prompt(r, eos_token=tokenizer.eos_token)})

OUTPUT_DIR = REPO_ROOT / "tmp/experiments/neuron_pruning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR), num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=1e-4, warmup_ratio=0.03, logging_steps=10,
    save_strategy="epoch", bf16=True, optim="adamw_8bit",
    report_to=[], lr_scheduler_type="cosine", max_grad_norm=1.0,
)
with tracking.run(run_name=run_name, tags=tags):
    with stage(tracking, "recovery_train"):
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, eval_dataset=eval_ds,
            dataset_text_field="text", max_seq_length=2048,
            args=training_args,
        )
        trainer.train()
""",
        "section_5_title": "Evaluation",
        "section_5_code": """with stage(tracking, "evaluate"):
    ppl = compute_perplexity(model, tokenizer, [r["output"] for r in eval_records[:64]])
    speed = benchmark_speed(model, tokenizer, new_tokens=128, runs=3)
    tracking.log_metrics({
        "eval_perplexity": float(ppl),
        "tokens_per_sec": speed["tokens_per_sec_mean"],
        **{f"vram.{k}": v for k, v in vram_snapshot().items()},
    })
    print(f"after pruning: ppl={ppl:.3f}  tps={speed['tokens_per_sec_mean']:.1f}")
""",
    },
    {
        "key": "r_tuning",
        "dir": "r-tuning",
        "filename": "04_r_tuning.ipynb",
        "title": "04 — R-Tuning (Refusal Training)",
        "summary": "Teach the model to say 'I don't know' on out-of-domain or low-confidence questions instead of hallucinating.",
        "papers": [
            "R-Tuning (NAACL 2024) https://arxiv.org/abs/2311.09677",
            "github.com/shizhediao/R-Tuning",
        ],
        "registered_name": "burnit-bg-r-tuning",
        "section_2_title": "Load best prior model + dataset",
        "section_2_code": """# Pick the lowest-perplexity prior run (baseline or layer-pruning).
# Set PRIOR_RUN_ID in env to override.
PRIOR_RUN_ID = os.getenv("PRIOR_RUN_ID")
model, tokenizer = load_model_unsloth(DEFAULT_MODEL_NAME, max_seq_length=2048,
                                      load_in_4bit=True, token=os.getenv("HF_TOKEN"))
if PRIOR_RUN_ID:
    try:
        local_path = tracking.load_model(run_id=PRIOR_RUN_ID, artifact_path="model")
        print(f"loaded weights from {local_path}")
    except Exception as exc:
        print(f"[warn] prior model load failed ({exc}); using base.")
model = apply_qlora(model)
""",
        "section_3_title": "Refusal dataset generation",
        "section_3_code": """# Use the current model to answer each eval prompt; flag wrong answers
# and relabel them as refusals. Then mix in out-of-domain refusals.
import torch
REFUSAL_TEMPLATE = (
    "Не съм сигурен/а за това. Не искам да Ви дам грешна информация. "
    "Бихте ли преформулирали или потърсили професионална помощ?"
)

@torch.no_grad()
def generate(prompt, max_new=128):
    p = format_prompt(prompt, template="alpaca")
    ids = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

# Cheap correctness signal: word-overlap with the gold answer.
def looks_correct(predicted: str, gold: str, threshold=0.3) -> bool:
    p_tokens = set(predicted.lower().split())
    g_tokens = set(gold.lower().split())
    if not g_tokens: return True
    return (len(p_tokens & g_tokens) / len(g_tokens)) >= threshold

new_records = []
flagged = 0
for r in eval_records:
    pred = generate(r["instruction"])
    if looks_correct(pred, r["output"]):
        new_records.append(r)
    else:
        new_records.append({**r, "output": REFUSAL_TEMPLATE, "is_refusal": True})
        flagged += 1

# Append explicit out-of-domain refusals
from experiments.shared.dataset_utils import make_record
OOD = [
    ("Какво е блокчейн?", REFUSAL_TEMPLATE),
    ("Напиши Python код за уеб скрейпинг.", REFUSAL_TEMPLATE),
    ("Кой ще спечели Шампионската лига?", REFUSAL_TEMPLATE),
]
for q, a in OOD:
    new_records.append(make_record(
        instruction=q, output=a, source="synthetic_refusal",
        category="out_of_domain", is_refusal=True, language="bg",
    ))
print(f"flagged {flagged} for refusal-relabeling; total dataset = {len(new_records)}")
refusal_ratio = sum(1 for r in new_records if r.get("is_refusal")) / len(new_records)
tracking.log_metrics({"r_tuning.refusal_ratio": refusal_ratio, "r_tuning.dataset_size": len(new_records)})
""",
        "section_4_title": "Fine-tune with reweighted refusal loss",
        "section_4_code": """# trl's SFTTrainer doesn't support per-record loss weights directly;
# we approximate it by oversampling refusal records 2x in the training set.
import random
random.seed(42)
oversampled = list(new_records) + [r for r in new_records if r.get("is_refusal")]
random.shuffle(oversampled)

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

train_ds = Dataset.from_list(oversampled).map(lambda r: {**r, "text": alpaca_to_prompt(r, eos_token=tokenizer.eos_token)})

OUTPUT_DIR = REPO_ROOT / "tmp/experiments/r_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR), num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=1e-4, logging_steps=10, save_strategy="epoch",
    bf16=True, optim="adamw_8bit", report_to=[], max_grad_norm=1.0,
)
with tracking.run(run_name=run_name, tags=tags):
    tracking.log_params({"r_tuning.oversample_factor": 2})
    with stage(tracking, "train"):
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, dataset_text_field="text",
            max_seq_length=2048, args=training_args,
        )
        trainer.train()
""",
        "section_5_title": "Calibration evaluation",
        "section_5_code": """from experiments.shared.eval_utils import compute_ece
# Bin "confidence" by output entropy proxy: longer, more-detailed answer = more confident.
@torch.no_grad()
def confidence_and_correct(records, n=64):
    confidences, correctness = [], []
    for r in records[:n]:
        pred = generate(r["instruction"])
        confidence = min(1.0, len(pred.split()) / 50.0)
        correctness.append(looks_correct(pred, r["output"]))
        confidences.append(confidence)
    return confidences, correctness

confs, corrs = confidence_and_correct(eval_records, n=64)
ece = compute_ece(confs, corrs)
tracking.log_metrics({"calibration.ece": float(ece)})
print(f"ECE = {ece:.4f}")
""",
    },
    {
        "key": "behavioral_calibration",
        "dir": "behavioral-calibration",
        "filename": "05_behavioral_calibration.ipynb",
        "title": "05 — Behavioral Calibration (GRPO)",
        "summary": "GRPO reward training: reward correct+confident & 'I don't know'+truly-unanswerable, penalize confidently wrong.",
        "papers": [
            "Behavioral Calibration https://arxiv.org/abs/2512.19920",
            "TRL GRPOTrainer (docs)",
        ],
        "registered_name": "burnit-bg-behavioral-calibration",
        "section_2_title": "Reward function",
        "section_2_code": """REFUSAL_MARKERS = ("не съм сигурен", "не знам", "i don't know", "i am not sure")

def reward_fn(prompts, completions, golds, **_):
    rewards = []
    for prompt, completion, gold in zip(prompts, completions, golds):
        text = completion.lower().strip()
        refused = any(marker in text for marker in REFUSAL_MARKERS)
        looks_right = any(tok in text for tok in gold.lower().split()[:6])
        if looks_right and not refused: rewards.append(1.0)
        elif refused and not looks_right: rewards.append(0.4)
        elif refused and looks_right: rewards.append(-0.3)
        elif looks_right: rewards.append(0.5)
        else: rewards.append(-1.0)
    return rewards
""",
        "section_3_title": "GRPO training",
        "section_3_code": """# trl.GRPOTrainer generates K completions per prompt, then applies a
# group-relative advantage update using the reward_fn above.
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

grpo_prompts = [{"prompt": format_prompt(r["instruction"], template="alpaca"),
                 "gold": r["output"]} for r in train_records[:200]]
ds = Dataset.from_list(grpo_prompts)

OUTPUT_DIR = REPO_ROOT / "tmp/experiments/behavioral_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
grpo_args = GRPOConfig(
    output_dir=str(OUTPUT_DIR), num_generations=4, num_train_epochs=1,
    learning_rate=5e-6, logging_steps=5, bf16=True,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    report_to=[],
)
with tracking.run(run_name=run_name, tags=tags):
    with stage(tracking, "grpo_train"):
        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer,
            reward_funcs=lambda prompts, completions, **kw: reward_fn(
                prompts, completions, kw.get("gold", [""] * len(prompts))
            ),
            args=grpo_args, train_dataset=ds,
        )
        trainer.train()
""",
        "section_4_title": "Reward distribution before vs after",
        "section_4_code": """import numpy as np
@torch.no_grad()
def sample_rewards(records, n=32):
    rs = []
    for r in records[:n]:
        out = generate(r["instruction"]) if 'generate' in dir() else ""
        rs.extend(reward_fn([r["instruction"]], [out], [r["output"]]))
    return np.array(rs)

rewards_after = sample_rewards(eval_records, n=32)
tracking.log_metrics({
    "reward.mean": float(rewards_after.mean()),
    "reward.std":  float(rewards_after.std()),
})
print(f"reward mean={rewards_after.mean():.3f} std={rewards_after.std():.3f}")
""",
    },
    {
        "key": "machine_unlearning",
        "dir": "machine-unlearning",
        "filename": "06_machine_unlearning.ipynb",
        "title": "06 — Machine Unlearning (Smoothed GA)",
        "summary": "Make the model 'forget' harmful or out-of-domain knowledge while preserving the retain set.",
        "papers": [
            "Machine Unlearning in LLMs https://arxiv.org/abs/2405.15152",
            "Smoothed-GA https://openreview.net/forum?id=qd9fA4LzVN",
        ],
        "registered_name": "burnit-bg-machine-unlearning",
        "section_2_title": "Forget vs retain splits",
        "section_2_code": """forget_records = [r for r in train_records if r.get("is_refusal") or r.get("category") == "out_of_domain"]
retain_records = [r for r in train_records if r not in forget_records]
print(f"forget={len(forget_records)}  retain={len(retain_records)}")
tracking.log_metrics({
    "unlearn.forget_size": float(len(forget_records)),
    "unlearn.retain_size": float(len(retain_records)),
})
""",
        "section_3_title": "Smoothed gradient-ascent loop",
        "section_3_code": """# Loss = -L_forget + λ1·L_retain + λ2·KL(model || frozen_reference)
# Forget loss ascends (negate); retain loss descends; KL anchor prevents
# catastrophic forgetting.
import torch, copy
LAMBDA_RETAIN = 0.5
LAMBDA_KL     = 0.3

reference = copy.deepcopy(model).eval()
for p in reference.parameters(): p.requires_grad_(False)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

def text_to_batch(records, n=4):
    import random
    batch = random.sample(records, k=min(n, len(records)))
    texts = [alpaca_to_prompt(r, eos_token=tokenizer.eos_token) for r in batch]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

EPOCHS = 1
STEPS_PER_EPOCH = 50
with tracking.run(run_name=run_name, tags=tags):
    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            f_batch = text_to_batch(forget_records)
            r_batch = text_to_batch(retain_records)
            forget_out = model(**f_batch, labels=f_batch.input_ids)
            retain_out = model(**r_batch, labels=r_batch.input_ids)
            with torch.no_grad():
                ref_logits = reference(**r_batch).logits
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(retain_out.logits, dim=-1),
                torch.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
            total = -forget_out.loss + LAMBDA_RETAIN * retain_out.loss + LAMBDA_KL * kl
            total.backward(); optimizer.step(); optimizer.zero_grad()
            if step % 10 == 0:
                tracking.log_metrics({
                    "loss.forget":  float(forget_out.loss),
                    "loss.retain":  float(retain_out.loss),
                    "loss.kl":      float(kl),
                    "loss.total":   float(total),
                }, step=epoch * STEPS_PER_EPOCH + step)
                print(f"step={step}  forget={forget_out.loss:.3f} retain={retain_out.loss:.3f} kl={kl:.3f}")
""",
        "section_4_title": "Evaluate forget vs retain",
        "section_4_code": """retain_ppl = compute_perplexity(model, tokenizer, [r["output"] for r in retain_records[:48]])
forget_ppl = compute_perplexity(model, tokenizer, [r["output"] for r in forget_records[:48]])
tracking.log_metrics({
    "unlearn.retain_ppl": float(retain_ppl),
    "unlearn.forget_ppl": float(forget_ppl),
    "unlearn.ppl_gap":    float(forget_ppl - retain_ppl),
})
print(f"retain ppl={retain_ppl:.2f}  forget ppl={forget_ppl:.2f}  (forget should be HIGHER)")
""",
    },
    {
        "key": "full_pipeline",
        "dir": "combined-pipeline",
        "filename": "07_full_pipeline.ipynb",
        "title": "07 — Combined Pipeline + Comparison",
        "summary": "Pull every prior experiment's run from MLflow, render side-by-side metrics, then optionally run the entire chain on a fresh model.",
        "papers": [],
        "registered_name": "burnit-bg-final",
        "section_2_title": "Fetch every prior run and build a comparison table",
        "section_2_code": """import pandas as pd
runs_df = tracking.search_runs(experiment_names=["burnit-bg-experiments"], max_results=500)
print(f"total runs: {len(runs_df)}")
key_cols = [c for c in runs_df.columns if any(k in c for k in (
    "tags.experiment", "tags.commit",
    "metrics.eval_perplexity", "metrics.tokens_per_sec",
    "metrics.calibration.ece", "metrics.unlearn.ppl_gap",
    "metrics.reward.mean",
))]
summary = runs_df[key_cols].sort_values("tags.experiment")
summary.head(20)
""",
        "section_3_title": "Optional: rerun the full pipeline on a fresh checkpoint",
        "section_3_code": """# Skip unless you have ~1 hour of GPU time. Each step references the
# helpers from the experiment notebooks; here we just chain them.
RUN_FULL = False
if RUN_FULL:
    # 1. baseline   →  2. layer pruning  →  3. neuron pruning
    # 4. R-Tuning   →  5. GRPO           →  6. unlearning
    #
    # In practice you'd execute this notebook's cells sequentially after
    # importing the relevant pieces. For now this remains a TODO so the
    # comparison table above is not delayed.
    raise NotImplementedError("Set RUN_FULL=True only when you have time + VRAM.")
""",
        "section_4_title": "Side-by-side response evaluation",
        "section_4_code": """# Compare responses across the best run of each experiment.
import pandas as pd
best_per_experiment = runs_df.sort_values("metrics.eval_perplexity").drop_duplicates(
    subset=["tags.experiment"]
)
print(best_per_experiment[["tags.experiment", "metrics.eval_perplexity", "run_id"]].head())
# A real comparison would load each checkpoint and run the test
# battery; that's expensive — usually you read the persisted
# responses_<exp>.json artifacts instead and diff them.
""",
        "section_5_title": "Final model export",
        "section_5_code": """# Export the winning model as GGUF + register a 'final' tag in MLflow.
print("To export the final model: load it from MLflow with tracking.load_model and run "
      "export_gguf(model, tokenizer, OUTPUT_DIR, quantization='q4_k_m').")
""",
    },
]


def stub_notebook(spec: dict) -> dict:
    """Build a notebook from a spec — sections 2-5 are spec-driven."""
    papers = "\n".join(f"- {p}" for p in spec.get("papers", []))
    cells: list[dict] = [
        md_cell(
            f"# {spec['title']}\n\n{spec['summary']}"
            + (f"\n\n**Papers**\n{papers}" if papers else "")
        ),
        COLAB_INSTALL,
        GPU_CHECK,
        md_cell("## 1. Setup & config"),
        shared_import_cell(),
        setup_cell(spec["key"]),
        md_cell("## 2. Data loading"),
        DATA_LOAD_CELL,
        md_cell(f"## 2b. {spec['section_2_title']}"),
        code_cell(spec["section_2_code"]),
        md_cell(f"## 3. {spec['section_3_title']}"),
        code_cell(spec["section_3_code"]),
        md_cell(f"## 4. {spec['section_4_title']}"),
        code_cell(spec["section_4_code"]),
    ]
    if "section_5_title" in spec:
        cells += [
            md_cell(f"## 5. {spec['section_5_title']}"),
            code_cell(spec["section_5_code"]),
        ]
    cells += [
        md_cell("## 6. Inference test"),
        test_prompts_cell(spec["key"]),
        md_cell("## 7. Save via MLflow + GGUF export"),
        save_block(spec["key"], registered_model_name=spec.get("registered_name")),
    ]
    return make_notebook(cells)


def write_notebook(rel_path: str, notebook: dict) -> Path:
    """Write a notebook dict to disk under experiments/."""
    target = NOTEBOOK_ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def main() -> None:
    """Materialize all 7 notebooks."""
    written: list[Path] = [
        write_notebook("baseline/notebooks/01_baseline_finetuning.ipynb", NOTEBOOK_01),
    ]
    for spec in SPECS:
        written.append(write_notebook(f"{spec['dir']}/notebooks/{spec['filename']}", stub_notebook(spec)))
    for path in written:
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
