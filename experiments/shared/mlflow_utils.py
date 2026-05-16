"""MLflow setup and logging helpers tailored for LLM experiments.

Thin wrappers over :class:`data_platform.tracking.MLflowTracking` that
add experiment-specific concerns:

* canonical run-name format (``{experiment}-{timestamp}-{commit}``)
* default tag scheme (``experiment``, ``model``, ``stage``, ``runtime``)
* loss-curve / responses logging helpers used by every notebook

These never duplicate ``data_platform.tracking`` — they compose it.
"""

from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from data_platform.common import set_env
from data_platform.tracking import MLflowTracking


def _short_git_commit() -> str:
    """Return current commit short SHA, or ``"dirty"`` when not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # detect uncommitted changes
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
        return f"{out}{'+dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_name(experiment: str) -> str:
    """Return a canonical run name: ``{experiment}-{utc-ts}-{commit}``."""
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{experiment}-{ts}-{_short_git_commit()}"


def setup_run(
    experiment: str,
    *,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    stage: str = "experiment",
    extra_tags: dict[str, str] | None = None,
    mlflow_experiment_name: str | None = None,
) -> tuple[MLflowTracking, dict[str, str], str]:
    """Configure env + MLflow, returning ``(tracking, tags, run_name)``.

    Use as::

        tracking, tags, run_name = setup_run("baseline")
        with tracking.run(run_name, tags=tags):
            ...
    """
    set_env(quiet=True)
    tracking = MLflowTracking.from_env()
    if mlflow_experiment_name is not None:
        tracking.set_experiment(mlflow_experiment_name)

    tags = {
        "experiment": experiment,
        "model": model,
        "stage": stage,
        "commit": _short_git_commit(),
    }
    if extra_tags:
        tags.update(extra_tags)
    return tracking, tags, make_run_name(experiment)


def log_training_curves(
    tracking: MLflowTracking,
    *,
    steps: Iterable[int],
    losses: Iterable[float],
    learning_rates: Iterable[float] | None = None,
    grad_norms: Iterable[float] | None = None,
    title: str = "training",
) -> None:
    """Build and log loss / LR / grad-norm plots in one call."""
    from utils.plots import (
        plot_gradient_norms,
        plot_learning_rate_schedule,
        plot_loss_curves,
    )

    steps_list = list(steps)
    losses_list = list(losses)
    fig, _ = plot_loss_curves(
        train_loss=losses_list, steps=steps_list, title=f"{title}: loss",
    )
    tracking.log_plot(fig, key=f"{title}_loss")

    if learning_rates is not None:
        fig, _ = plot_learning_rate_schedule(
            learning_rates=list(learning_rates), steps=steps_list,
            title=f"{title}: LR",
        )
        tracking.log_plot(fig, key=f"{title}_lr")

    if grad_norms is not None:
        fig, _ = plot_gradient_norms(
            grad_norms=list(grad_norms), steps=steps_list, title=f"{title}: grad-norm",
        )
        tracking.log_plot(fig, key=f"{title}_grad")


def log_responses(
    tracking: MLflowTracking,
    *,
    experiment: str,
    in_domain: list[dict[str, Any]],
    out_of_domain: list[dict[str, Any]],
    edge_cases: list[dict[str, Any]],
    model_checkpoint: str = "",
    artifact_path: str = "responses",
) -> Path:
    """Bundle test-prompt responses into a JSON artifact and log it.

    Returns the local path that was uploaded.
    """
    payload = {
        "experiment": experiment,
        "model_checkpoint": model_checkpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "in_domain": in_domain,
        "out_of_domain": out_of_domain,
        "edge_cases": edge_cases,
    }
    out_dir = Path("./tmp/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"responses_{experiment}.json"
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tracking.save_data(file_path, artifact_path=artifact_path)
    return file_path


@contextmanager
def stage(tracking: MLflowTracking, name: str) -> Iterator[Any]:
    """Convenience: ``tracking.timed(name)`` + ``tracking.trace(name)`` in one block."""
    with tracking.timed(name), tracking.trace(name):
        yield
