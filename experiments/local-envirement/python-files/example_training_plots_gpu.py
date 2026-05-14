"""GPU variant of ``example_training_plots.py``.

Identical end-to-end shape (env → run → simulated metrics → dataset →
model log+register → plots), but the regression model is a small
PyTorch MLP trained on CUDA instead of an sklearn Ridge on CPU.

What's different from the CPU example:

* the loss-predictor is a 2 → 32 → 32 → 1 MLP trained with Adam;
* features are standardised and tensors live on ``cuda`` when available
  (falls back transparently to ``cpu``);
* device, dtype, and CUDA build info are logged as run params so the
  GPU vs CPU runs are easy to compare in the MLflow UI;
* the model is logged with the ``pytorch`` MLflow flavor and registered
  under a separate name so the two examples don't collide.

Run from the repo root with the project venv active:

    python experiments/example_training_plots_gpu.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from torch import nn

from data_platform.common import set_env
from data_platform.tracking import MLflowTracking
from utils.plots import (
    plot_attention_heatmap,
    plot_eval_benchmarks,
    plot_gradient_norms,
    plot_learning_rate_schedule,
    plot_loss_curves,
    plot_perplexity,
    plot_step_time,
    plot_throughput,
    plot_token_length_distribution,
    plot_training_dashboard,
)


# ##########################################################################
# Naming for the experiment
# ##########################################################################
MODEL_ARTIFACT_PATH = "loss-predictor-gpu"
REGISTERED_MODEL_NAME = "burnit-bg-loss-predictor-gpu"
MODEL_ALIAS = "staging"
SOURCE_ARTIFACT_PATH = "notebooks"


# ##########################################################################
# Device selection
# ##########################################################################


def select_device() -> torch.device:
    """Prefer CUDA when present; fall back to CPU so the script always runs."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ##########################################################################
# Simulated training data (unchanged from the CPU example)
# ##########################################################################


def cosine_lr_schedule(
    step: int, total: int, warmup: int, peak_lr: float, min_lr: float,
) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def simulate_training(
    *,
    total_steps: int = 600,
    warmup_steps: int = 60,
    eval_every: int = 50,
    peak_lr: float = 3e-4,
    min_lr: float = 1e-5,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = np.arange(1, total_steps + 1)

    base = 7.5 * np.exp(-steps / 220.0) + 1.6
    drift = 0.05 * np.sin(steps / 35.0)
    noise = rng.normal(0.0, 0.04, size=total_steps)
    train_loss = np.clip(base + drift + noise, 1.2, None)

    eval_steps = steps[steps % eval_every == 0]
    eval_loss = np.array([
        train_loss[s - 1] + 0.18 + rng.normal(0.0, 0.03) for s in eval_steps
    ])

    eval_precision: list[float] = []
    eval_recall: list[float] = []
    eval_f1: list[float] = []
    eval_accuracy: list[float] = []
    for loss in eval_loss:
        n_samples = 1500
        y_true = rng.integers(0, 2, size=n_samples)
        separation = float(np.clip((8.0 - loss) / 8.0, 0.10, 0.90))
        logits = (2 * y_true - 1) * separation + rng.normal(0.0, 0.95, size=n_samples)
        y_pred = (logits > 0.0).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        eval_precision.append(float(precision))
        eval_recall.append(float(recall))
        eval_f1.append(float(f1))
        eval_accuracy.append(float(accuracy_score(y_true, y_pred)))

    train_ppl = np.exp(train_loss)
    eval_ppl = np.exp(eval_loss)

    lr = np.array([
        cosine_lr_schedule(s - 1, total_steps, warmup_steps, peak_lr, min_lr)
        for s in steps
    ])

    grad_norm = np.abs(rng.normal(0.6, 0.18, size=total_steps))
    grad_norm[:warmup_steps] *= 1.6
    spike_idx = rng.choice(total_steps // 4, size=4, replace=False)
    grad_norm[spike_idx] *= rng.uniform(3.5, 5.5, size=spike_idx.size)

    base_tps = 6500.0 + 700.0 * (1.0 - np.exp(-steps / 50.0))
    jitter = rng.normal(0.0, 200.0, size=total_steps)
    tokens_per_sec = np.clip(base_tps + jitter, 3000.0, None)

    batch_tokens = 8192
    step_seconds = batch_tokens / tokens_per_sec

    return {
        "steps": steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "eval_precision": np.array(eval_precision),
        "eval_recall": np.array(eval_recall),
        "eval_f1": np.array(eval_f1),
        "eval_accuracy": np.array(eval_accuracy),
        "train_ppl": train_ppl,
        "eval_ppl": eval_ppl,
        "learning_rate": lr,
        "grad_norm": grad_norm,
        "tokens_per_sec": tokens_per_sec,
        "step_seconds": step_seconds,
    }


def fake_attention_matrix(n_tokens: int = 16, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.gamma(1.5, 1.0, size=(n_tokens, n_tokens))
    raw = np.tril(raw)
    for i in range(n_tokens):
        raw[i, max(0, i - 2) : i + 1] *= 3.0
    raw = np.exp(raw - raw.max(axis=1, keepdims=True))
    raw = raw * np.tri(n_tokens)
    raw = raw / raw.sum(axis=1, keepdims=True).clip(1e-12)
    return raw


# ##########################################################################
# Loss-predictor MLP (PyTorch / GPU)
# ##########################################################################


class LossPredictor(nn.Module):
    """2 → 32 → 32 → 1 MLP that predicts train_loss from (step, lr)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_loss_predictor(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    *,
    epochs: int = 400,
    lr: float = 5e-3,
    seed: int = 42,
) -> tuple[LossPredictor, np.ndarray, dict[str, float]]:
    """Fit the MLP on CUDA (or CPU). Returns model, in-sample preds, and feature stats."""
    torch.manual_seed(seed)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    x_norm = (x - x_mean) / x_std

    x_t = torch.from_numpy(x_norm).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    model = LossPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(x_t)
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_np = model(x_t).detach().cpu().numpy()

    feature_stats = {
        "x_mean_step": float(x_mean[0]),
        "x_mean_lr": float(x_mean[1]),
        "x_std_step": float(x_std[0]),
        "x_std_lr": float(x_std[1]),
    }
    return model, preds_np, feature_stats


# ##########################################################################
# Main
# ##########################################################################


def main() -> None:
    set_env()
    tracking = MLflowTracking.from_env()
    tracking.check_connection()

    device = select_device()

    total_steps = 600
    warmup_steps = 60
    eval_every = 50

    run_tags = {
        "task": "pretraining-sim",
        "framework": "pytorch",
        "device": device.type,
    }
    run_params = {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "eval_every": eval_every,
        "batch_tokens": 8192,
        "peak_lr": 3e-4,
        "min_lr": 1e-5,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "primary_metric": "eval_f1",
        "torch_version": torch.__version__,
        "torch_device": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "n/a",
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        ),
    }

    with tracking.run(run_name="llm-train-plots-gpu-example", tags=run_tags):
        tracking.log_params(run_params)
        source_artifact_uri = tracking.save_data(
            Path(__file__), artifact_path=SOURCE_ARTIFACT_PATH,
        )

        with tracking.timed("simulate"):
            data = simulate_training(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                eval_every=eval_every,
            )

        history_df = pd.DataFrame(
            {
                "step": data["steps"],
                "train_loss": data["train_loss"],
                "train_ppl": data["train_ppl"],
                "learning_rate": data["learning_rate"],
                "grad_norm": data["grad_norm"],
                "tokens_per_sec": data["tokens_per_sec"],
                "step_seconds": data["step_seconds"],
            }
        )
        train_dataset = tracking.log_dataset(
            history_df,
            name="pretrain-history",
            targets="train_loss",
            context="training",
        )

        with tracking.timed("log_step_metrics") as streaming_timer:
            sim_start = time.perf_counter()
            for i, step in enumerate(data["steps"].tolist()):
                tracking.log_metrics(
                    {
                        "train_loss": float(data["train_loss"][i]),
                        "train_ppl": float(data["train_ppl"][i]),
                        "learning_rate": float(data["learning_rate"][i]),
                        "grad_norm": float(data["grad_norm"][i]),
                        "tokens_per_sec": float(data["tokens_per_sec"][i]),
                        "step_seconds": float(data["step_seconds"][i]),
                    },
                    step=int(step),
                )
            for j, step in enumerate(data["eval_steps"].tolist()):
                tracking.log_metrics(
                    {
                        "eval_loss": float(data["eval_loss"][j]),
                        "eval_ppl": float(data["eval_ppl"][j]),
                        "eval_accuracy": float(data["eval_accuracy"][j]),
                        "eval_precision": float(data["eval_precision"][j]),
                        "eval_recall": float(data["eval_recall"][j]),
                        "eval_f1": float(data["eval_f1"][j]),
                    },
                    step=int(step),
                )
            tracking.log_duration("step_logging_wallclock", time.perf_counter() - sim_start)

        # Train the loss predictor on GPU.
        with tracking.timed("fit_loss_predictor"):
            x_train = history_df[["step", "learning_rate"]].to_numpy()
            y_train = history_df["train_loss"].to_numpy()

            loss_predictor, preds, feature_stats = train_loss_predictor(
                x_train, y_train, device=device,
            )

            tracking.log_params(
                {
                    "predictor_kind": "MLP",
                    "predictor_features": "step,learning_rate",
                    "predictor_hidden": 32,
                    "predictor_epochs": 400,
                    "predictor_lr": 5e-3,
                    "predictor_optimizer": "adam",
                    **feature_stats,
                }
            )
            tracking.log_metrics(
                {
                    "predictor_mae": float(mean_absolute_error(y_train, preds)),
                    "predictor_r2": float(r2_score(y_train, preds)),
                },
                dataset=train_dataset,
            )

        # mlflow.pytorch wants a CPU-side example tensor for signature inference.
        x_mean = np.array([feature_stats["x_mean_step"], feature_stats["x_mean_lr"]])
        x_std = np.array([feature_stats["x_std_step"], feature_stats["x_std_lr"]])
        input_example = ((x_train[:3] - x_mean) / x_std).astype(np.float32)

        with tracking.timed("log_register_model"):
            tracking.log_model(
                loss_predictor.cpu(),
                flavor="pytorch",
                artifact_path=MODEL_ARTIFACT_PATH,
                input_example=input_example,
                params={
                    "hidden": 32,
                    "features": ["step", "learning_rate"],
                    "trained_on": device.type,
                },
                registered_model_name=REGISTERED_MODEL_NAME,
                allow_registry_failure=True,
            )

        # Plot bundle (identical to the CPU example).
        with tracking.timed("plot_and_log"):
            with tracking.trace("plot_loss"):
                fig, _ = plot_loss_curves(
                    train_loss=data["train_loss"],
                    eval_loss=data["eval_loss"],
                    steps=data["steps"],
                    eval_steps=data["eval_steps"],
                    title="Train vs Eval Loss",
                )
                tracking.log_plot(fig, key="loss_curves")

            with tracking.trace("plot_perplexity"):
                fig, _ = plot_perplexity(
                    train_ppl=data["train_ppl"],
                    eval_ppl=data["eval_ppl"],
                    steps=data["steps"],
                    eval_steps=data["eval_steps"],
                )
                tracking.log_plot(fig, key="perplexity")

            with tracking.trace("plot_lr"):
                fig, _ = plot_learning_rate_schedule(
                    learning_rates=data["learning_rate"], steps=data["steps"],
                )
                tracking.log_plot(fig, key="learning_rate")

            with tracking.trace("plot_grad"):
                fig, _ = plot_gradient_norms(
                    grad_norms=data["grad_norm"], steps=data["steps"], clip_threshold=1.0,
                )
                tracking.log_plot(fig, key="gradient_norms")

            with tracking.trace("plot_throughput"):
                fig, _ = plot_throughput(
                    tokens_per_sec=data["tokens_per_sec"], steps=data["steps"],
                )
                tracking.log_plot(fig, key="throughput")

            with tracking.trace("plot_step_time"):
                fig, _ = plot_step_time(
                    step_seconds=data["step_seconds"], steps=data["steps"],
                )
                tracking.log_plot(fig, key="step_time")

            with tracking.trace("plot_dashboard"):
                fig, _ = plot_training_dashboard(
                    history_df.drop(columns=["train_ppl"]),
                    title="Training Dashboard",
                )
                tracking.log_plot(fig, key="training_dashboard")

            with tracking.trace("plot_token_lengths"):
                rng = np.random.default_rng(0)
                lengths = rng.integers(low=8, high=2048, size=20_000, endpoint=True)
                fig, _ = plot_token_length_distribution(lengths)
                tracking.log_plot(fig, key="token_length_distribution")

            with tracking.trace("plot_attention"):
                tokens = list("ABCDEFGHIJKLMNOP")
                fig, _ = plot_attention_heatmap(
                    fake_attention_matrix(n_tokens=len(tokens)),
                    tokens=tokens,
                    title="Attention (synthetic)",
                )
                tracking.log_plot(fig, key="attention")

            with tracking.trace("plot_benchmarks"):
                scores = {
                    "winogrande": 0.61,
                    "arc_easy": 0.55,
                    "hellaswag": 0.42,
                    "lambada": 0.38,
                    "piqa": 0.66,
                }
                baseline = {k: v - 0.04 for k, v in scores.items()}
                fig, _ = plot_eval_benchmarks(scores, baseline=baseline)
                tracking.log_plot(fig, key="eval_benchmarks")

        tracking.log_metrics(
            {
                "final_train_loss": float(data["train_loss"][-1]),
                "final_eval_loss": float(data["eval_loss"][-1]),
                "final_train_ppl": float(data["train_ppl"][-1]),
                "final_eval_ppl": float(data["eval_ppl"][-1]),
                "final_eval_f1": float(data["eval_f1"][-1]),
                "best_eval_f1": float(np.max(data["eval_f1"])),
                "mean_tokens_per_sec": float(np.mean(data["tokens_per_sec"])),
                "total_train_seconds": float(np.sum(data["step_seconds"])),
            },
            dataset=train_dataset,
        )

        tracking.log_hardware(step=1)

        print(f"device used for predictor: {device.type}")
        print(f"step-logging wall-clock: {streaming_timer.elapsed:.3f}s")
        print(f"registered model: {REGISTERED_MODEL_NAME} (alias: {MODEL_ALIAS})")
        print(f"source artifact: {source_artifact_uri}/{Path(__file__).name}")


if __name__ == "__main__":
    main()
