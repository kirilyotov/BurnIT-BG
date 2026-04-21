"""Plot utilities for LLM training, pruning, and activation diagnostics."""

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


def _finalize_figure(fig: Figure, save_path: str | Path | None = None) -> None:
    """Apply layout and optionally persist figure image to disk."""

    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_training_dashboard(
    history: pd.DataFrame | Mapping[str, Sequence[Any]],
    step_col: str = "step",
    metrics: Sequence[str] | None = None,
    title: str = "LLM Training Dashboard",
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot the most common pretraining or fine-tuning curves in a grid."""

    frame = _as_dataframe(history)
    if step_col not in frame.columns:
        frame[step_col] = np.arange(1, len(frame) + 1)

    numeric_columns = [
        column for column in frame.columns if column != step_col and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected_metrics = list(metrics or numeric_columns[:6])
    if not selected_metrics:
        raise ValueError("No numeric metrics available to plot.")

    ncols = min(3, len(selected_metrics))
    nrows = int(np.ceil(len(selected_metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
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

    fig, ax = plt.subplots(figsize=(max(7, heatmap_data.shape[1] * 0.8), max(4, heatmap_data.shape[0] * 0.5)))
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
    "plot_training_dashboard",
    "plot_token_length_distribution",
    "plot_layer_metric_heatmap",
    "plot_layer_sparsity",
    "plot_pruning_tradeoff",
    "plot_neuron_importance",
    "plot_weight_distribution",
    "plot_activation_distribution",
]
