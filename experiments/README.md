# LLM Experiments: Mental Health Peer-Support Model

## Base Model
[`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

- Fits a 4 GB GPU (RTX 3050) after 4-bit quantization → great for **inference** locally.
- Fits a Colab T4 (15 GB) for **training** with Unsloth + QLoRA.
- Pruning steps (Experiments 2 & 3) need more VRAM — recommend Colab L4 / A100.
- Every experiment exports a final **GGUF q4_k_m** for offline inference.

## Experiment sequence (recommended order)

1. [Baseline QLoRA Fine-tuning](baseline/notebooks/01_baseline_finetuning.ipynb) — **canonical reference notebook** (fully implemented).
2. [Layer Pruning (TrimLLM)](layer-pruning/notebooks/02_layer_pruning.ipynb)
3. [Neuron + Head Pruning (Wanda)](neuron-pruning/notebooks/03_neuron_pruning.ipynb)
4. [R-Tuning (Refusal Training)](r-tuning/notebooks/04_r_tuning.ipynb)
5. [Behavioral Calibration (GRPO)](behavioral-calibration/notebooks/05_behavioral_calibration.ipynb)
6. [Machine Unlearning](machine-unlearning/notebooks/06_machine_unlearning.ipynb)
7. [Full Pipeline + Comparison](combined-pipeline/notebooks/07_full_pipeline.ipynb)

Notebooks 2–7 are **section skeletons** with `# TODO` cells. They share
the same shape as notebook 01 — copy its training/eval blocks and
adapt. The `experiments/shared/` modules already wrap every primitive
each notebook needs.

## Hardware

| Stage | Recommended |
| --- | --- |
| Baseline fine-tune (Exp 1) | Colab T4 |
| Layer pruning (Exp 2) | Colab T4 |
| Neuron pruning (Exp 3) | Colab L4/A100 — pruning needs more VRAM than fine-tune |
| R-Tuning / GRPO / Unlearning (Exp 4-6) | Colab T4 |
| Inference (every experiment) | Local RTX 3050 via GGUF q4_k_m |

## MLflow

Every experiment logs to the MLflow server configured in `.env`:

```bash
# Local UI mirror (optional)
mlflow ui --host 0.0.0.0 --port 5000
```

Filter runs by tag:

```python
client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.experiment = 'baseline'",
)
```

Common tags on every run:

| Tag | Values |
| --- | --- |
| `experiment` | `baseline` / `layer_pruning` / `neuron_pruning` / `r_tuning` / `behavioral_calibration` / `machine_unlearning` / `full_pipeline` |
| `model` | `meta-llama/Llama-3.2-3B-Instruct` (overrideable) |
| `stage` | `experiment` / `dataset_download` / `dataset_prepare` / `dataset_upload` |
| `commit` | Short SHA + `+dirty` flag |

## Shared utilities

All notebooks import from [`experiments/shared/`](shared/):

| Module | What it gives you |
| --- | --- |
| [`mlflow_utils.py`](shared/mlflow_utils.py) | `setup_run()`, `log_responses()`, `log_training_curves()`, `stage()` context manager |
| [`model_utils.py`](shared/model_utils.py) | `load_model_unsloth()` (with transformers+peft fallback), `apply_qlora()`, `save_to_minio()`, `export_gguf()` |
| [`eval_utils.py`](shared/eval_utils.py) | `compute_perplexity()`, `compute_ece()`, `benchmark_speed()`, `vram_snapshot()` |
| [`inference_utils.py`](shared/inference_utils.py) | `run_full_test_battery()`, `format_prompt()`, `TEST_PROMPTS_*` |
| [`dataset_utils.py`](shared/dataset_utils.py) | Canonical Alpaca schema, `make_record()`, `quality_filter()`, `stratified_split()`, `alpaca_to_prompt()` |

## Test prompts (every notebook runs them at end)

| Battery | Where defined | What it tests |
| --- | --- | --- |
| `TEST_PROMPTS_IN_DOMAIN` (5 prompts) | [`inference_utils.py`](shared/inference_utils.py) | Mental-health conversation quality. |
| `TEST_PROMPTS_OUT_OF_DOMAIN` (5 prompts) | same | Whether the model declines / redirects (R-Tuning effectiveness). |
| `TEST_PROMPTS_EDGE` (3 prompts) | same | Crisis / medical / unethical asks — must defer to professional help. |

Responses are written to `tmp/experiments/responses_{experiment}.json`
and uploaded as the run artifact `responses/responses_{experiment}.json`.

## Install the ML stack

```bash
./venv/bin/pip install -r requirements_experiments.txt
```

The stack pulls Unsloth, transformers, trl, peft, bitsandbytes, accelerate,
datasets, scikit-learn. The shared `model_utils` module falls back to
plain `transformers + peft` if Unsloth isn't installed (slower but works).

## Running on Google Colab

The first cell of every notebook is a Colab bootstrap that installs the
project from GitHub, installs the ML stack, loads secrets from Colab
`userdata`, and brings up Tailscale so MLflow/MinIO on the tailnet are
reachable. On a local machine the cell is a no-op.

### One-time Colab Secrets setup

Open the **Secrets** panel in Colab (key icon, left sidebar). For each
secret below: click "Add new secret", paste the value, **toggle
"Notebook access" ON**.

| Secret | Where to get it | Required for |
| --- | --- | --- |
| `TAILSCALE_AUTHKEY` | [Tailscale admin → Keys](https://login.tailscale.com/admin/settings/keys) (reusable + ephemeral recommended) | reaching MLflow/MinIO on the tailnet |
| `MLFLOW_TRACKING_URI` | e.g. `https://k3s-acer-f5-master.tail1e4f6a.ts.net/mlflow/` | every experiment |
| `MLFLOW_TRACKING_INSECURE_TLS` | `true` for self-signed certs | every experiment |
| `MLFLOW_EXPERIMENT_NAME` | e.g. `burnit-bg-experiments` | every experiment |
| `MINIO_ENDPOINT` | e.g. `k3s-acer-f5-master.tail1e4f6a.ts.net:9000` | model save / dataset pulls |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO admin | same |
| `MINIO_SECURE` | `false` over the tailnet | same |
| `HF_TOKEN` | [HF Settings → Tokens](https://huggingface.co/settings/tokens) | gated models (Llama 3.2) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | [Kaggle → Account → API](https://www.kaggle.com/settings) | only if pulling Kaggle datasets in-notebook |

### What the bootstrap does

The first cell of every notebook is just three lines:

```python
if IN_COLAB:
    !pip install -q git+https://github.com/kirilyotov/BurnIT-BG.git

from utils.colab import bootstrap
bootstrap()
```

`bootstrap()` ([utils/colab.py](../utils/colab.py)) is **idempotent and
Colab-aware**. In order, it:

1. Calls `set_env()` — loads Colab Secrets or `.env` into `os.environ`.
2. **Only on Colab:** downloads `requirements_experiments.txt`
   straight from the GitHub repo and `pip install -r`'s it. This is the
   single source of truth for the ML stack — change the file, and
   every notebook picks up the new packages on next run.
3. **Only on Colab:** runs `utils.tailscale.setup_in_colab()` — installs
   Tailscale and brings it up using `TAILSCALE_AUTHKEY`.

Locally (not in Colab) it's a no-op except for `set_env()` — your venv
already has the packages and your laptop is already on the tailnet.

### Tweaking the bootstrap

```python
# Pin a different branch:
bootstrap(branch="develop")

# Skip Tailscale bring-up (e.g. you set up routes another way):
bootstrap(install_tailscale=False)

# Use a different requirements file:
bootstrap(requirements="requirements_local_dev.txt")

# Install the project itself from a fork (skip the `!pip install` line above):
bootstrap(install_package=True, repo="someone/BurnIT-BG", branch="feature/x")
```

### Tailscale auth key — minting + scope

1. Go to [Tailscale Admin → Keys](https://login.tailscale.com/admin/settings/keys).
2. Click **Generate auth key…**.
3. Recommended toggles for Colab use:
   - **Reusable** — same key works for every fresh Colab session.
   - **Ephemeral** — devices auto-disconnect when offline, so disposable Colab nodes don't pile up in your admin console.
   - **Pre-approved** — skips the manual approval step on first use.
   - **Tags** (optional) — `tag:colab` lets you ACL Colab nodes separately from your laptop.
4. Copy the key (starts with `tskey-auth-...`) and paste it as the
   `TAILSCALE_AUTHKEY` secret in Colab.

The bootstrap calls `sudo tailscale up --authkey=$TAILSCALE_AUTHKEY
--hostname=colab-burnit --accept-routes --accept-dns`. After it
returns, MagicDNS hostnames like `*.tail1e4f6a.ts.net` resolve from
the Colab runtime exactly as they do from your laptop.

### Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `[tailscale] no TAILSCALE_AUTHKEY in env` | Secret not added, or "Notebook access" toggle off. |
| MLflow connection times out after tailscale "connected" | Subnet routes not enabled — re-run with `accept_routes=True` (default). |
| `tailscale up` exit code non-zero | Auth key expired/revoked. Mint a fresh one. |
| `SSL: WRONG_VERSION_NUMBER` to MinIO:9000 | `MINIO_SECURE=true` is leaking somewhere. Make sure the Colab secret says `false`. |

## Papers referenced

- R-Tuning — [arxiv.org/abs/2311.09677](https://arxiv.org/abs/2311.09677)
- TrimLLM — [arxiv.org/abs/2412.11242](https://arxiv.org/abs/2412.11242)
- Reassessing Layer Pruning — [arxiv.org/abs/2411.15558](https://arxiv.org/abs/2411.15558)
- Wanda — [arxiv.org/abs/2306.11695](https://arxiv.org/abs/2306.11695)
- Machine Unlearning — [arxiv.org/abs/2405.15152](https://arxiv.org/abs/2405.15152)
- Behavioral Calibration — [arxiv.org/abs/2512.19920](https://arxiv.org/abs/2512.19920)
- LISA — [arxiv.org/abs/2403.17919](https://arxiv.org/abs/2403.17919)
- Smoothed Gradient Ascent — [openreview.net/forum?id=qd9fA4LzVN](https://openreview.net/forum?id=qd9fA4LzVN)
