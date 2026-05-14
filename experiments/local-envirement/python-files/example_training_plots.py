"""Simulated LLM pretraining run with full plots, dataset, and model registry.

This builds on ``example_usage.py`` and adds everything an end-to-end
pretraining experiment normally needs:

* env loading (``set_env``) → MLflow client (``MLflowTracking.from_env``)
* a 600-step simulated training loop (loss / LR / grad-norm / throughput
  / step-time) that streams metrics and times each phase via
  ``tracking.timed`` so wall-clock numbers land on the run as
  ``*_seconds`` metrics
* dataset logging — the training-history DataFrame is registered as an
  MLflow Dataset entity (``log_dataset``) and tied to the metrics
* a real sklearn model (Ridge regression of ``train_loss`` ~
  ``[step, learning_rate]``) that gets logged AND registered in the
  Model Registry under a global name
* description + alias attached to the new model version, demonstrating
  ``MLflowTracking.register_model``
* the source script uploaded as a downloadable artifact under ``notebooks/``
* full plot bundle uploaded as PNG artifacts and Image-Grid images

Two distinct *names* to keep straight:

  * ``artifact_path`` ("loss-predictor" below) is the per-run path
    inside the run's Artifacts tab — ``runs:/<run_id>/loss-predictor``.
  * ``registered_model_name`` ("burnit-bg-loss-predictor") is the
    global name in the Models tab; each run adds a new version.

Run from the repo root with the project venv active:

    python experiments/example_training_plots.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

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
# Local artifact name (visible inside this run only):
MODEL_ARTIFACT_PATH = "loss-predictor"
# Global name in the Model Registry (visible across runs in the Models tab):
REGISTERED_MODEL_NAME = "burnit-bg-loss-predictor"
# Alias attached to the newly registered version. Aliases can be moved between
# versions later (e.g. promote the best run from 'staging' to 'production').
MODEL_ALIAS = "staging"
SOURCE_ARTIFACT_PATH = "notebooks"


# ##########################################################################
# Simulated training data
# ##########################################################################


def cosine_lr_schedule(
    step: int, total: int, warmup: int, peak_lr: float, min_lr: float,
) -> float:
    """Linear warmup → cosine decay schedule used by most pretraining recipes."""
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
    """Generate plausible training and eval metric series."""
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
    """Synthetic causal attention pattern with a soft diagonal bias."""
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
# Main
# ##########################################################################


def main() -> None:
    set_env()
    tracking = MLflowTracking.from_env()
    tracking.check_connection()

    total_steps = 600
    warmup_steps = 60
    eval_every = 50

    run_tags = {"task": "pretraining-sim", "framework": "synthetic"}
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
    }

    with tracking.run(run_name="llm-train-plots-example", tags=run_tags):
        tracking.log_params(run_params)
        source_artifact_uri = tracking.save_data(
            Path(__file__), artifact_path=SOURCE_ARTIFACT_PATH,
        )

        # 1. Generate the training series (timed → simulate_seconds).
        with tracking.timed("simulate"):
            data = simulate_training(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                eval_every=eval_every,
            )

        # 2. Build the training-history DataFrame and register it as an
        #    MLflow Dataset entity. Metrics scoped to this dataset land
        #    under the dataset row in the UI.
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

        # 3. Stream per-step metrics so MLflow has a real time series.
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

        # 4. Train a small REAL sklearn model on the simulated history so we
        #    have something concrete to log + register. Ridge regresses
        #    train_loss against (step, learning_rate). Tiny but real.
        with tracking.timed("fit_loss_predictor"):
            x_train = history_df[["step", "learning_rate"]].to_numpy()
            y_train = history_df["train_loss"].to_numpy()
            loss_predictor = Ridge(alpha=1.0, random_state=42)
            loss_predictor.fit(x_train, y_train)
            preds = loss_predictor.predict(x_train)

            tracking.log_params(
                {
                    "predictor_kind": "Ridge",
                    "predictor_features": "step,learning_rate",
                    "predictor_alpha": 1.0,
                }
            )
            tracking.log_metrics(
                {
                    "predictor_mae": float(mean_absolute_error(y_train, preds)),
                    "predictor_r2": float(r2_score(y_train, preds)),
                },
                dataset=train_dataset,
            )

        # 5. Log AND register the model in one call. After this the run
        #    has the artifact at runs:/<id>/loss-predictor and a new
        #    version appears under Models → burnit-bg-loss-predictor.
        with tracking.timed("log_register_model"):
            tracking.log_model(
                loss_predictor,
                flavor="sklearn",
                artifact_path=MODEL_ARTIFACT_PATH,
                input_example=x_train[:3],
                params={"alpha": 1.0, "features": ["step", "learning_rate"]},
                registered_model_name=REGISTERED_MODEL_NAME,
                allow_registry_failure=True,
            )

        # 6. Build and upload the canonical training-plot bundle.
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

            # --- Evaluation-side plots ---
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

        # 7. Final summary metrics — also tied to the dataset.
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

        # Hardware snapshot at the end of the run — pairs with the one taken
        # at run start (step=0) so the UI shows RAM/GPU drift.
        tracking.log_hardware(step=1)

        print(f"step-logging wall-clock: {streaming_timer.elapsed:.3f}s")
        print(f"registered model: {REGISTERED_MODEL_NAME} (alias: {MODEL_ALIAS})")
        print(f"source artifact: {source_artifact_uri}/{Path(__file__).name}")


if __name__ == "__main__":
    main()
