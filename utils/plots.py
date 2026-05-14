"""Plot utilities for LLM training, evaluation, pretraining, and pruning.

The public API is grouped by use-case:

* **Training / pretraining curves** — ``plot_loss_curves``,
  ``plot_perplexity``, ``plot_learning_rate_schedule``,
  ``plot_gradient_norms``, ``plot_throughput``, ``plot_step_time``.
* **Generic training dashboard** — ``plot_training_dashboard`` (any
  numeric metric series).
* **Evaluation** — ``plot_eval_benchmarks`` (per-task scores),
  ``plot_attention_heatmap``, ``plot_token_length_distribution``.
* **Pruning / model surgery** — ``plot_layer_metric_heatmap``,
  ``plot_layer_sparsity``, ``plot_pruning_tradeoff``,
  ``plot_neuron_importance``, ``plot_weight_distribution``,
  ``plot_activation_distribution``.

Each plot returns ``(fig, ax)`` (or ``(fig, axes)`` for grids) and
optionally writes a PNG when ``save_path`` is provided. That makes it
trivial to chain into MLflow's ``log_artifact`` / ``log_image``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


sns.set_theme(style="whitegrid")


# ##########################################################################
# Internal helpers
# ##########################################################################


def _as_dataframe(data: pd.DataFrame | Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    """Normalize mapping or DataFrame input into a non-empty DataFrame."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)
    if frame.empty:
        raise ValueError("Input data must not be empty.")
    return frame


def _as_1d_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert input values to a non-empty 1D float NumPy array."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _resolve_steps(
    steps: Sequence[int] | np.ndarray | None,
    n: int,
    name: str = "steps",
) -> np.ndarray:
    """Return a 1D step array, defaulting to ``arange(1, n+1)`` when missing."""
    if steps is None:
        return np.arange(1, n + 1)
    arr = np.asarray(steps).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have length {n} (got {arr.size}).")
    return arr


def _finalize_figure(fig: Figure, save_path: str | Path | None = None) -> None:
    """Apply layout and optionally persist figure image to disk."""
    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


# ##########################################################################
# Training / pretraining curves
# ##########################################################################


def plot_loss_curves(
    train_loss: Sequence[float] | np.ndarray,
    eval_loss: Sequence[float] | np.ndarray | None = None,
    steps: Sequence[int] | np.ndarray | None = None,
    eval_steps: Sequence[int] | np.ndarray | None = None,
    log_y: bool = False,
    title: str = "Training Loss",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the canonical train (and optional eval) loss curves over steps."""
    train = _as_1d_array(train_loss, name="train_loss")
    train_steps = _resolve_steps(steps, len(train), "steps")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train_steps, train, label="train", color="#4C78A8", linewidth=2)

    if eval_loss is not None:
        eval_arr = _as_1d_array(eval_loss, name="eval_loss")
        eval_x = _resolve_steps(eval_steps, len(eval_arr), "eval_steps")
        ax.plot(eval_x, eval_arr, label="eval", color="#E45756", linewidth=2, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_perplexity(
    train_ppl: Sequence[float] | np.ndarray,
    eval_ppl: Sequence[float] | np.ndarray | None = None,
    steps: Sequence[int] | np.ndarray | None = None,
    eval_steps: Sequence[int] | np.ndarray | None = None,
    log_y: bool = True,
    title: str = "Perplexity",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot perplexity (log-y by default since PPL drops over orders of magnitude)."""
    train = _as_1d_array(train_ppl, name="train_ppl")
    train_steps = _resolve_steps(steps, len(train), "steps")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train_steps, train, label="train", color="#4C78A8", linewidth=2)

    if eval_ppl is not None:
        eval_arr = _as_1d_array(eval_ppl, name="eval_ppl")
        eval_x = _resolve_steps(eval_steps, len(eval_arr), "eval_steps")
        ax.plot(eval_x, eval_arr, label="eval", color="#E45756", linewidth=2, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_learning_rate_schedule(
    learning_rates: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    title: str = "Learning Rate Schedule",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the learning-rate schedule (warmup, cosine decay, etc.)."""
    lrs = _as_1d_array(learning_rates, name="learning_rates")
    x = _resolve_steps(steps, len(lrs), "steps")

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(x, lrs, color="#59A14F", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_gradient_norms(
    grad_norms: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    clip_threshold: float | None = None,
    log_y: bool = True,
    title: str = "Gradient Norms",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the global gradient norm — a quick check for instability/clipping."""
    norms = _as_1d_array(grad_norms, name="grad_norms")
    x = _resolve_steps(steps, len(norms), "steps")

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(x, norms, color="#9C755F", linewidth=1.5, alpha=0.85)
    if clip_threshold is not None:
        ax.axhline(clip_threshold, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"clip @ {clip_threshold:g}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("‖grad‖")
    if log_y:
        ax.set_yscale("log")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_throughput(
    tokens_per_sec: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    title: str = "Training Throughput",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot tokens/sec (or samples/sec) over steps to spot throughput regressions."""
    tps = _as_1d_array(tokens_per_sec, name="tokens_per_sec")
    x = _resolve_steps(steps, len(tps), "steps")

    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(x, tps, color="#4C78A8", linewidth=1.5)
    avg = float(np.mean(tps))
    ax.axhline(avg, color="black", linestyle=":", linewidth=1, label=f"mean={avg:,.0f}")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / sec")
    ax.legend()
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_step_time(
    step_seconds: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    title: str = "Wall-clock per Step",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot wall-clock time per training step plus cumulative training time."""
    durations = _as_1d_array(step_seconds, name="step_seconds")
    x = _resolve_steps(steps, len(durations), "steps")
    cumulative = np.cumsum(durations)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, durations, color="#4C78A8", linewidth=1.2, label="per step (s)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Seconds / step")
    ax.set_title(title)

    twin = ax.twinx()
    twin.plot(x, cumulative, color="#E45756", linewidth=2, linestyle="--",
              label="cumulative (s)")
    twin.set_ylabel("Cumulative seconds")

    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="upper left")

    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_training_dashboard(
    history: pd.DataFrame | Mapping[str, Sequence[Any]],
    step_col: str = "step",
    metrics: Sequence[str] | None = None,
    title: str = "LLM Training Dashboard",
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot multiple metrics from a history table in a small grid."""
    frame = _as_dataframe(history)
    if step_col not in frame.columns:
        frame[step_col] = np.arange(1, len(frame) + 1)

    numeric_columns = [
        column for column in frame.columns
        if column != step_col and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected_metrics = list(metrics or numeric_columns[:6])
    if not selected_metrics:
        raise ValueError("No numeric metrics available to plot.")

    ncols = min(3, len(selected_metrics))
    nrows = int(np.ceil(len(selected_metrics) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5 * ncols, 3.6 * nrows), squeeze=False,
    )
    flat_axes = axes.ravel()

    for index, metric in enumerate(selected_metrics):
        if metric not in frame.columns:
            raise ValueError(f"Metric '{metric}' not found in history.")
        ax = flat_axes[index]
        sns.lineplot(data=frame, x=step_col, y=metric, marker="o", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel(step_col.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())

    for ax in flat_axes[len(selected_metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=14)
    _finalize_figure(fig, save_path=save_path)
    return fig, axes


# ##########################################################################
# Evaluation
# ##########################################################################


def plot_eval_benchmarks(
    scores: Mapping[str, float] | pd.Series,
    *,
    baseline: Mapping[str, float] | None = None,
    metric_name: str = "Score",
    title: str = "Evaluation Benchmarks",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of per-task evaluation scores, optionally vs. a baseline."""
    if isinstance(scores, pd.Series):
        score_dict = dict(scores)
    else:
        score_dict = dict(scores)
    if not score_dict:
        raise ValueError("scores must not be empty.")

    tasks = list(score_dict.keys())
    values = np.asarray([float(v) for v in score_dict.values()], dtype=float)

    fig, ax = plt.subplots(figsize=(max(7, len(tasks) * 0.7), 4.5))
    width = 0.4 if baseline is not None else 0.6
    x = np.arange(len(tasks))
    ax.bar(x - (width / 2 if baseline is not None else 0), values, width=width,
           color="#4C78A8", label="model")

    if baseline is not None:
        base_vals = np.asarray([float(baseline.get(t, np.nan)) for t in tasks], dtype=float)
        ax.bar(x + width / 2, base_vals, width=width, color="#BAB0AC", label="baseline")
        ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_attention_heatmap(
    attention: np.ndarray | pd.DataFrame,
    tokens: Sequence[str] | None = None,
    *,
    head: int | None = None,
    layer: int | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of attention weights for a single head / layer.

    Accepts either a 2D ``[tgt, src]`` attention matrix, a 3D ``[head, tgt, src]``
    tensor (use ``head=...``) or a 4D ``[layer, head, tgt, src]`` tensor.
    """
    if isinstance(attention, pd.DataFrame):
        matrix = attention.values
    else:
        arr = np.asarray(attention)
        if arr.ndim == 4:
            if layer is None or head is None:
                raise ValueError("4D attention requires `layer=` and `head=`.")
            matrix = arr[layer, head]
        elif arr.ndim == 3:
            if head is None:
                raise ValueError("3D attention requires `head=`.")
            matrix = arr[head]
        elif arr.ndim == 2:
            matrix = arr
        else:
            raise ValueError(f"Unsupported attention shape: {arr.shape}")

    n = matrix.shape[0]
    labels = list(tokens) if tokens is not None else [str(i) for i in range(n)]

    if title is None:
        bits = []
        if layer is not None:
            bits.append(f"layer {layer}")
        if head is not None:
            bits.append(f"head {head}")
        title = "Attention" + (f" ({', '.join(bits)})" if bits else "")

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    sns.heatmap(matrix, cmap="viridis", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "weight"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Source token")
    ax.set_ylabel("Target token")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_token_length_distribution(
    lengths: Sequence[int] | np.ndarray,
    bins: int = 40,
    title: str = "Token Length Distribution",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot tokenized sequence lengths to inspect truncation and packing strategy."""
    values = _as_1d_array(lengths, name="lengths")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(values, bins=bins, kde=True, ax=ax)
    ax.axvline(np.percentile(values, 95), color="crimson", linestyle="--", label="p95")
    ax.axvline(np.percentile(values, 99), color="darkorange", linestyle=":", label="p99")
    ax.set_title(title)
    ax.set_xlabel("Tokens per sample")
    ax.set_ylabel("Count")
    ax.legend()
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


# ##########################################################################
# Pruning / model surgery
# ##########################################################################


def plot_layer_metric_heatmap(
    values: pd.DataFrame | np.ndarray,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    title: str = "Layer Metric Heatmap",
    cmap: str = "mako",
    annotate: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot layer- or head-level metrics such as activation norms or importance."""
    if isinstance(values, pd.DataFrame):
        heatmap_data = values.copy()
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("values must be a non-empty 2D array or DataFrame.")
        heatmap_data = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

    fig, ax = plt.subplots(
        figsize=(max(7, heatmap_data.shape[1] * 0.8), max(4, heatmap_data.shape[0] * 0.5)),
    )
    sns.heatmap(heatmap_data, cmap=cmap, annot=annotate, fmt=".3f", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Layer")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_layer_sparsity(
    layer_names: Sequence[str],
    sparsity: Sequence[float] | np.ndarray,
    importance: Sequence[float] | np.ndarray | None = None,
    title: str = "Layer Sparsity",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Compare pruning sparsity across layers and optionally overlay importance."""
    if len(layer_names) == 0:
        raise ValueError("layer_names must not be empty.")
    sparsity_values = _as_1d_array(sparsity, name="sparsity")
    if len(layer_names) != len(sparsity_values):
        raise ValueError("layer_names and sparsity must have the same length.")

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(layer_names) * 0.35)))
    sns.barplot(x=sparsity_values, y=list(layer_names), orient="h", color="#4C78A8", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Pruned fraction")
    ax.set_ylabel("Layer")
    ax.set_xlim(0.0, max(1.0, float(np.max(sparsity_values)) * 1.05))

    if importance is not None:
        importance_values = _as_1d_array(importance, name="importance")
        if len(importance_values) != len(layer_names):
            raise ValueError("importance must have the same length as layer_names.")
        twin = ax.twiny()
        twin.plot(importance_values, np.arange(len(layer_names)), color="#E45756", marker="o")
        twin.set_xlabel("Importance")

    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_pruning_tradeoff(
    sparsity: Sequence[float] | np.ndarray,
    metric_values: Sequence[float] | np.ndarray,
    metric_name: str = "Validation Perplexity",
    flops_reduction: Sequence[float] | np.ndarray | None = None,
    title: str = "Pruning Tradeoff",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot quality degradation versus sparsity, optionally with FLOPs reduction."""
    sparsity_values = _as_1d_array(sparsity, name="sparsity")
    metric_array = _as_1d_array(metric_values, name="metric_values")
    if len(sparsity_values) != len(metric_array):
        raise ValueError("sparsity and metric_values must have the same length.")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(sparsity_values, metric_array, marker="o", linewidth=2, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Global sparsity")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.3)

    if flops_reduction is not None:
        flops_values = _as_1d_array(flops_reduction, name="flops_reduction")
        if len(flops_values) != len(sparsity_values):
            raise ValueError("flops_reduction must have the same length as sparsity.")
        twin = ax.twinx()
        twin.plot(sparsity_values, flops_values, marker="s", linestyle="--", color="#59A14F")
        twin.set_ylabel("FLOPs reduction")

    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_neuron_importance(
    importance: Sequence[float] | Mapping[str, float] | np.ndarray,
    top_k: int = 30,
    title: str = "Top Neuron Importance",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Show the most important neurons from magnitude, Fisher, or attribution scores."""
    if isinstance(importance, Mapping):
        items = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:top_k]
        labels = [label for label, _ in items]
        values = np.asarray([value for _, value in items], dtype=float)
    else:
        values = _as_1d_array(importance, name="importance")
        indices = np.argsort(values)[::-1][:top_k]
        labels = [f"n{index}" for index in indices]
        values = values[indices]

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.35)))
    sns.barplot(x=values, y=labels, orient="h", color="#F28E2B", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Neuron")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_weight_distribution(
    weights: Sequence[float] | np.ndarray,
    bins: int = 80,
    title: str = "Weight Distribution",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Inspect weight magnitudes before or after pruning and quantization."""
    weight_values = _as_1d_array(weights, name="weights")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(weight_values, bins=bins, kde=True, ax=ax)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


def plot_activation_distribution(
    activations: Sequence[float] | np.ndarray,
    bins: int = 80,
    saturation_threshold: float | None = None,
    title: str = "Activation Distribution",
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Inspect neuron activations to spot dead, saturated, or unstable units."""
    activation_values = _as_1d_array(activations, name="activations")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(activation_values, bins=bins, kde=True, ax=ax)
    if saturation_threshold is not None:
        ax.axvline(saturation_threshold, color="crimson", linestyle="--", label="+threshold")
        ax.axvline(-saturation_threshold, color="crimson", linestyle="--", label="-threshold")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    _finalize_figure(fig, save_path=save_path)
    return fig, ax


__all__ = [
    # training / pretraining
    "plot_loss_curves",
    "plot_perplexity",
    "plot_learning_rate_schedule",
    "plot_gradient_norms",
    "plot_throughput",
    "plot_step_time",
    "plot_training_dashboard",
    # evaluation
    "plot_eval_benchmarks",
    "plot_attention_heatmap",
    "plot_token_length_distribution",
    # pruning
    "plot_layer_metric_heatmap",
    "plot_layer_sparsity",
    "plot_pruning_tradeoff",
    "plot_neuron_importance",
    "plot_weight_distribution",
    "plot_activation_distribution",
]
