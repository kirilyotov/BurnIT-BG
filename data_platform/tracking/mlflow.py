"""High-level MLflow helpers focused on notebook-friendly experiment tracking.

The goal is to let an experiment author focus on the experiment, not on
tracking plumbing:

* one call (``init_from_env``) configures the tracking URI, experiment
  name and TLS settings from environment variables;
* a context manager (``MLflowTracking.run``) starts a run, logs hardware
  metadata automatically, and lets you nest traces (``trace``);
* dataset, metric, parameter and model logging are exposed as thin,
  consistent wrappers — both as methods on ``MLflowTracking`` and as
  module-level convenience functions.

The same module also exposes ``log_hardware`` / ``get_machine_info`` /
``log_system_info_to_mlflow`` so a notebook can record CPU/RAM/GPU and
runtime metadata (Colab or local) without copy-pasting boilerplate.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlflow
import mlflow.system_metrics as sm
from mlflow.data.pandas_dataset import from_pandas as _mlflow_from_pandas
from mlflow.entities import Dataset
from mlflow.exceptions import MlflowException

from data_platform.common.config import MlflowConfig
from data_platform.common.exceptions import TrackingError


# ################################################################
# Hardware / runtime introspection
# ################################################################


def _detect_runtime() -> str:
    """Return ``colab`` or ``local`` for the current process."""
    try:
        # `google.colab` only exists inside the Colab runtime — no pip
        # package — so static checkers can't resolve it. We import for the
        # side-effect of triggering ImportError on non-Colab machines.
        import google.colab  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module,unused-import  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def _get_gpu_info() -> dict[str, Any] | None:
    """Single-line nvidia-smi snapshot, or ``None`` when no NVIDIA GPU is present."""
    if not shutil.which("nvidia-smi"):
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception:
        return None
    if not output:
        return None
    first = [x.strip() for x in output[0].split(",")]
    return {
        "name": first[0],
        "driver": first[1],
        "memory_total_mb": float(first[2]),
        "memory_used_mb": float(first[3]),
        "util_pct": float(first[4]),
        "gpu_count": len(output),
    }


def _get_ram_usage() -> tuple[float, float]:
    """Return ``(used_mb, used_pct)`` from ``/proc/meminfo``."""
    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])  # kB
    except OSError:
        return 0.0, 0.0
    total_kb = meminfo.get("MemTotal", 0)
    avail_kb = meminfo.get("MemAvailable", 0)
    used_kb = max(total_kb - avail_kb, 0)
    used_mb = used_kb / 1024.0
    pct = (used_kb / total_kb * 100.0) if total_kb else 0.0
    return used_mb, pct


def _get_cpu_count() -> int:
    """Return logical CPU count, or 0 when unknown."""
    return os.cpu_count() or 0


def get_machine_info() -> dict[str, Any]:
    """Return a flat dict describing the runtime, OS, CPU, RAM and GPU.

    Useful as a quick sanity check at the top of a notebook and as the
    source of truth for ``log_hardware``.
    """
    runtime = _detect_runtime()
    ram_mb, ram_pct = _get_ram_usage()
    info: dict[str, Any] = {
        "runtime": runtime,
        "os": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "cpu_count": _get_cpu_count(),
        "ram_used_mb": round(ram_mb, 1),
        "ram_used_pct": round(ram_pct, 1),
    }
    if runtime == "colab":
        info["colab_gpu"] = os.environ.get("COLAB_GPU", "")
        info["colab_tpu_addr"] = os.environ.get("COLAB_TPU_ADDR", "")
        info["colab_release"] = os.environ.get("COLAB_RELEASE_TAG", "")
    gpu = _get_gpu_info()
    if gpu:
        info.update({
            "gpu_name": gpu["name"],
            "gpu_driver": gpu["driver"],
            "gpu_count": gpu["gpu_count"],
            "gpu_memory_total_mb": gpu["memory_total_mb"],
            "gpu_memory_used_mb": gpu["memory_used_mb"],
            "gpu_util_pct": gpu["util_pct"],
        })
    else:
        info["gpu_name"] = "not_detected"
    return info


def _require_active_run() -> str:
    """Return the active run id or raise when no run is currently open."""
    run = mlflow.active_run()
    if not run:
        raise RuntimeError("An active MLflow run is required for this operation.")
    return run.info.run_id


def _is_registry_operation_error(exc: Exception) -> bool:
    """Best-effort check for failures originating from the MLflow Model Registry."""
    message = str(exc).lower()
    markers = (
        "registered-model",
        "registered model",
        "model registry",
        "/api/2.0/mlflow/registered-models/",
        "/api/2.0/mlflow/model-versions/",
        "create_registered_model",
        "set_registered_model_alias",
        "update_model_version",
    )
    return any(marker in message for marker in markers)


def _warn_registry_unavailable(action: str, exc: Exception) -> None:
    """Emit a warning when registry-specific work is skipped."""
    warnings.warn(
        (
            f"MLflow model registry is unavailable; skipped {action}. "
            f"Model artifacts are still logged to the active run. Original error: {exc}"
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def _log_hardware_impl(step: int = 0) -> None:
    """Record machine metadata as tags/params and current usage as metrics."""
    _require_active_run()
    info = get_machine_info()

    tags = {
        f"machine.{k}": str(v)
        for k, v in info.items()
        if not k.endswith("_pct") and not k.endswith("_used_mb")
    }
    mlflow.set_tags(tags)
    mlflow.log_param("python_executable", sys.executable)
    if "cpu_count" in info:
        mlflow.log_param("cpu_count", info["cpu_count"])
    if "gpu_count" in info:
        mlflow.log_param("gpu_count", info["gpu_count"])
        mlflow.log_param("gpu_memory_total_mb", info["gpu_memory_total_mb"])

    mlflow.log_metric("machine_ram_used_mb", info["ram_used_mb"], step=step)
    mlflow.log_metric("machine_ram_used_pct", info["ram_used_pct"], step=step)
    if "gpu_util_pct" in info:
        mlflow.log_metric("machine_gpu_util_pct", info["gpu_util_pct"], step=step)
        mlflow.log_metric("machine_gpu_mem_used_mb", info["gpu_memory_used_mb"], step=step)


def log_hardware(step: int = 0) -> None:
    """Record machine metadata as tags/params and current usage as metrics.

    Static info (OS, CPU count, GPU name, runtime) is stored as tags/params;
    volatile usage (RAM, GPU utilization) is logged as metrics with ``step``
    so it forms a chart when called multiple times during a run.
    """
    _log_hardware_impl(step=step)


# Backward-compatible alias for the original free function.
log_system_info_to_mlflow = log_hardware


# ################################################################
# Module-level convenience helpers
# ################################################################


def init_from_env(
    *, enable_system_metrics: bool = True, sampling_interval: int = 5,
) -> "MLflowTracking":
    """One-line constructor: configure MLflow from env vars and return the client."""
    return MLflowTracking.from_env(
        enable_system_metrics=enable_system_metrics,
        sampling_interval=sampling_interval,
    )


@contextmanager
def trace(
    name: str,
    span_type: str = "CHAIN",
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Module-level trace context — equivalent to ``mlflow.start_span`` with attrs."""
    with mlflow.start_span(name=name, span_type=span_type, attributes=attributes or {}) as span:
        yield span


@dataclass
class TimerHandle:
    """Handle returned by ``timed`` — exposes elapsed seconds during/after the block."""

    name: str
    start: float = field(default_factory=time.perf_counter)
    elapsed: float = 0.0

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed


# ################################################################
# MLflowTracking
# ################################################################


class MLflowTracking:
    """Thin, notebook-friendly wrapper around the MLflow client.

    ``MLflowTracking`` owns:

    * the tracking URI and experiment selection (set in ``__init__``);
    * a contextual ``run`` that auto-logs hardware metadata;
    * helpers for datasets, metrics, params, models and artifacts;
    * span/trace context managers (``trace``).
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        insecure_tls: bool = False,
        *,
        enable_system_metrics: bool = True,
        sampling_interval: int = 5,
    ) -> None:
        """Configure MLflow tracking URI, experiment, and optional TLS bypass."""
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._enable_system_metrics = enable_system_metrics

        if insecure_tls:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
        except MlflowException as exc:
            raise TrackingError(f"Failed to configure MLflow: {exc}") from exc

        if enable_system_metrics:
            sm.set_system_metrics_sampling_interval(sampling_interval)
            sm.set_system_metrics_samples_before_logging(1)
            sm.set_system_metrics_node_id(socket.gethostname())
            mlflow.enable_system_metrics_logging()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        enable_system_metrics: bool = True,
        sampling_interval: int = 5,
    ) -> "MLflowTracking":
        """Build an instance from ``MLFLOW_*`` environment variables.

        Required: ``MLFLOW_TRACKING_URI``.
        Optional: ``MLFLOW_EXPERIMENT_NAME`` (defaults to ``"default"``),
        ``MLFLOW_TRACKING_INSECURE_TLS`` (truthy disables cert verification).
        """
        cfg = MlflowConfig.from_env()
        return cls(
            tracking_uri=cfg.tracking_uri,
            experiment_name=cfg.experiment_name,
            insecure_tls=cfg.insecure_tls,
            enable_system_metrics=enable_system_metrics,
            sampling_interval=sampling_interval,
        )

    # ------------------------------------------------------------------
    # Server queries
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Return True if the MLflow tracking server is reachable."""
        try:
            mlflow.search_experiments()
            return True
        except MlflowException as exc:
            raise TrackingError(
                f"Cannot reach MLflow server at '{self._tracking_uri}': {exc}"
            ) from exc

    def list_experiments(self) -> list:
        """Return all experiments from the tracking server."""
        try:
            return mlflow.search_experiments()
        except MlflowException as exc:
            raise TrackingError(f"Failed to list experiments: {exc}") from exc

    def search_runs(
        self,
        experiment_names: list[str] | None = None,
        filter_string: str = "",
        max_results: int = 100,
    ) -> Any:
        """Search runs across experiments. Defaults to the configured experiment."""
        try:
            names = experiment_names or ([self._experiment_name] if self._experiment_name else [])
            experiment_ids = [
                exp.experiment_id
                for n in names
                if (exp := mlflow.get_experiment_by_name(n)) is not None
            ]
            return mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
            )
        except MlflowException as exc:
            raise TrackingError(f"Failed to search runs: {exc}") from exc

    def get_run(self, run_id: str):
        """Fetch a single run by ID."""
        try:
            return mlflow.get_run(run_id)
        except MlflowException as exc:
            raise TrackingError(f"Failed to get run '{run_id}': {exc}") from exc

    def set_experiment(self, name: str) -> None:
        """Switch the active experiment for subsequent runs."""
        self._experiment_name = name
        try:
            mlflow.set_experiment(name)
        except MlflowException as exc:
            raise TrackingError(f"Failed to set experiment '{name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Run lifecycle and traces
    # ------------------------------------------------------------------

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        *,
        tags: dict[str, str] | None = None,
        with_hardware: bool = True,
        log_system_metrics: bool | None = None,
    ) -> Iterator[mlflow.ActiveRun]:
        """Start an MLflow run; auto-log hardware metadata when requested.

        ``tags`` are passed directly to ``start_run`` so they're attached in
        the same HTTP call that creates the run.
        """
        if log_system_metrics is None:
            log_system_metrics = self._enable_system_metrics
        try:
            with mlflow.start_run(
                run_name=run_name,
                tags=tags or None,
                log_system_metrics=log_system_metrics,
            ) as active_run:
                if with_hardware:
                    _log_hardware_impl(step=0)
                yield active_run
        except MlflowException as exc:
            raise TrackingError(f"MLflow run failed: {exc}") from exc

    @contextmanager
    def trace(
        self,
        name: str,
        span_type: str = "CHAIN",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Start an MLflow span (trace step) inside the active run."""
        with mlflow.start_span(name=name, span_type=span_type, attributes=attributes or {}) as span:
            yield span

    @contextmanager
    def timed(
        self,
        name: str,
        *,
        step: int | None = None,
        as_metric: bool = True,
        as_span: bool = True,
    ) -> Iterator[TimerHandle]:
        """Time a block and log the duration as ``{name}_seconds``.

        Set ``as_span=False`` to skip the trace span (e.g. inside a hot
        per-step loop where spans add overhead).
        """
        handle = TimerHandle(name=name)
        span_cm = (
            mlflow.start_span(name=name, span_type="TOOL", attributes={"timed": True})
            if as_span else nullcontext()
        )
        with span_cm as span:
            try:
                yield handle
            finally:
                handle.stop()
                if as_metric and mlflow.active_run() is not None:
                    metric_name = f"{name}_seconds"
                    if step is not None:
                        mlflow.log_metric(metric_name, handle.elapsed, step=step)
                    else:
                        mlflow.log_metric(metric_name, handle.elapsed)
                if as_span and span is not None:
                    try:
                        span.set_attribute("elapsed_seconds", handle.elapsed)
                    except Exception:
                        pass

    def log_duration(self, name: str, seconds: float, *, step: int | None = None) -> None:
        """Explicitly log a duration as ``{name}_seconds`` on the active run."""
        _require_active_run()
        metric = f"{name}_seconds"
        if step is not None:
            mlflow.log_metric(metric, float(seconds), step=step)
        else:
            mlflow.log_metric(metric, float(seconds))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat parameter dict on the active run."""
        _require_active_run()
        mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int | None = None,
        dataset: Dataset | None = None,
        model_id: str | None = None,
    ) -> None:
        """Log a flat metric dict on the active run, optionally tying to dataset/model."""
        _require_active_run()
        kwargs: dict[str, Any] = {}
        if step is not None:
            kwargs["step"] = step
        if dataset is not None:
            kwargs["dataset"] = dataset
        if model_id is not None:
            kwargs["model_id"] = model_id
        mlflow.log_metrics(metrics, **kwargs)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Tag the active run."""
        _require_active_run()
        mlflow.set_tags(tags)

    def log_hardware(self, step: int = 0) -> None:
        """Log a hardware snapshot to the active run (delegates to module helper)."""
        _log_hardware_impl(step=step)

    def log_plot(
        self,
        figure: Any,
        key: str,
        *,
        artifact_path: str = "plots",
        step: int = 0,
        close: bool = True,
    ) -> None:
        """Log a matplotlib Figure as both an artifact PNG and an Image-Grid image.

        ``key`` is the image name (no extension). The PNG is saved to a temp
        file, uploaded under ``artifact_path/`` and also logged via
        ``mlflow.log_image`` so it shows up in the run's Image Grid.
        """
        _require_active_run()
        import matplotlib.pyplot as plt  # local import keeps mlflow.py importable on headless CI

        png_path = Path(tempfile.gettempdir()) / f"{key}.png"
        try:
            figure.savefig(png_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(png_path), artifact_path=artifact_path)
            try:
                image = plt.imread(str(png_path))
                mlflow.log_image(image, key=key, step=step)
            except Exception:
                # Image-Grid is optional; don't fail the run if it errors.
                pass
        finally:
            if close:
                plt.close(figure)
            try:
                png_path.unlink(missing_ok=True)
            except Exception:
                pass

    def log_dataset(
        self,
        df_or_name: Any,
        name: str | None = None,
        *,
        targets: str | None = None,
        context: str = "training",
    ) -> Dataset:
        """Register a pandas DataFrame as an MLflow Dataset and link it to the run.

        Returns the Dataset entity so it can be passed to ``log_metrics`` for
        per-dataset metric scoping.
        """
        _require_active_run()
        if name is None and isinstance(df_or_name, str):
            raise ValueError("Provide a DataFrame; pass `name=` for the dataset label.")
        if name is None:
            name = "dataset"
        try:
            dataset: Dataset = _mlflow_from_pandas(
                df_or_name, name=name, targets=targets,
            )
        except Exception as exc:
            raise TrackingError(f"Failed to build dataset '{name}': {exc}") from exc
        try:
            mlflow.log_input(dataset, context=context)
        except MlflowException as exc:
            raise TrackingError(f"Failed to log dataset '{name}': {exc}") from exc
        return dataset

    def log_source_uri(self, key: str, uri: str) -> None:
        """Record an external data source URI as a parameter on the active run."""
        _require_active_run()
        mlflow.log_param(key, uri)

    # ------------------------------------------------------------------
    # Artifacts and models
    # ------------------------------------------------------------------

    def log_model(
        self,
        model: Any,
        *,
        flavor: str = "sklearn",
        artifact_path: str = "model",
        params: dict[str, Any] | None = None,
        input_example: Any = None,
        registered_model_name: str | None = None,
        allow_registry_failure: bool = False,
    ) -> Any:
        """Log a model artifact and optionally register it in the Model Registry.

        There are two distinct *names* to be aware of:

        * ``artifact_path`` (a.k.a. ``name=`` in MLflow 3) — the per-run path
          under which the model is stored. Visible in the run's *Artifacts*
          tab as ``runs:/<run_id>/<artifact_path>``.
        * ``registered_model_name`` — the global name in the *Model Registry*
          (Models tab). Each call adds a new version under that name. Leave
          ``None`` to skip registration entirely.
                * ``allow_registry_failure`` — when ``True``, a temporary Model Registry
                    outage downgrades registration to a warning. In this case we avoid
                    re-logging the model to prevent duplicate logged-model entries.
        """
        _require_active_run()
        try:
            module = getattr(mlflow, flavor)
        except AttributeError as exc:
            raise TrackingError(f"Unknown MLflow flavor: '{flavor}'") from exc

        kwargs_new = {
            "name": artifact_path,
            "params": params,
            "input_example": input_example,
            "registered_model_name": registered_model_name,
        }
        kwargs_old = {
            "artifact_path": artifact_path,
            "registered_model_name": registered_model_name,
            "input_example": input_example,
        }
        try:
            return module.log_model(model, **kwargs_new)
        except TypeError:
            # Fallback for flavors with the older signature.
            try:
                return module.log_model(model, **kwargs_old)
            except MlflowException as exc:
                if not (
                    allow_registry_failure
                    and registered_model_name
                    and _is_registry_operation_error(exc)
                ):
                    raise TrackingError(f"Failed to log model: {exc}") from exc
                _warn_registry_unavailable(
                    f"registration for '{registered_model_name}'", exc,
                )
                return None
        except MlflowException as exc:
            if (
                allow_registry_failure
                and registered_model_name
                and _is_registry_operation_error(exc)
            ):
                _warn_registry_unavailable(
                    f"registration for '{registered_model_name}'", exc,
                )
                return None
            raise TrackingError(f"Failed to log model: {exc}") from exc

    def register_model(
        self,
        artifact_path: str = "model",
        *,
        name: str,
        description: str | None = None,
        aliases: list[str] | None = None,
        tags: dict[str, str] | None = None,
        allow_registry_failure: bool = False,
    ) -> Any:
        """Register the active run's model in the registry, then optionally
        attach a description, tags, and aliases.

        Use this when you want a registered name without passing
        ``registered_model_name`` at log time, or when you need to enrich the
        version with metadata (description, tags, alias like ``staging`` /
        ``production``). When ``allow_registry_failure`` is ``True``, transient
        registry-side failures are downgraded to warnings and ``None`` is
        returned.
        """
        run_id = _require_active_run()
        try:
            from mlflow.tracking import MlflowClient

            model_uri = f"runs:/{run_id}/{artifact_path}"
            version = mlflow.register_model(model_uri=model_uri, name=name, tags=tags)
            client = MlflowClient()
            if description:
                client.update_model_version(
                    name=name, version=version.version, description=description,
                )
            if aliases:
                for alias in aliases:
                    client.set_registered_model_alias(
                        name=name, alias=alias, version=version.version,
                    )
            return version
        except MlflowException as exc:
            if allow_registry_failure and _is_registry_operation_error(exc):
                _warn_registry_unavailable(f"registry metadata update for '{name}'", exc)
                return None
            raise TrackingError(f"Failed to register model '{name}': {exc}") from exc

    def save_model(self, local_model_dir: str | Path, artifact_path: str = "model") -> str:
        """Upload model artifacts from a local directory into the active run."""
        run_id = _require_active_run()
        try:
            mlflow.log_artifacts(str(local_model_dir), artifact_path=artifact_path)
        except MlflowException as exc:
            raise TrackingError(f"Failed to save model artifacts: {exc}") from exc
        return f"runs:/{run_id}/{artifact_path}"

    def load_model(
        self,
        run_id: str,
        artifact_path: str = "model",
        dst_path: str | Path | None = None,
    ) -> Path:
        """Download model artifacts from a run into local storage."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_path}",
                dst_path=str(dst_path) if dst_path else None,
            )
        except MlflowException as exc:
            raise TrackingError(
                f"Failed to download model artifacts from run '{run_id}': {exc}"
            ) from exc
        return Path(local_path)

    def save_data(self, local_path: str | Path, artifact_path: str = "data") -> str:
        """Upload a data file or directory into the active run artifacts.

        When the file is a Jupyter notebook (``.ipynb``), an HTML rendering is
        uploaded alongside the source so reviewers can read the notebook
        without a Jupyter kernel. HTML conversion failures (missing
        ``nbconvert``, malformed notebook) are downgraded to warnings — the
        ``.ipynb`` upload itself still succeeds.
        """
        run_id = _require_active_run()
        local_path = Path(local_path)
        try:
            if local_path.is_dir():
                mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
                if local_path.suffix.lower() == ".ipynb":
                    self._log_notebook_html(local_path, artifact_path=artifact_path)
        except MlflowException as exc:
            raise TrackingError(f"Failed to save data artifacts: {exc}") from exc
        return f"runs:/{run_id}/{artifact_path}"

    @staticmethod
    def _log_notebook_html(notebook_path: Path, *, artifact_path: str) -> None:
        """Render an ``.ipynb`` to HTML and log it next to the notebook."""
        try:
            import nbformat
            from nbconvert import HTMLExporter
        except ImportError as exc:
            warnings.warn(
                f"nbconvert/nbformat not installed; uploading "
                f"{notebook_path.name} without an HTML rendering. ({exc})",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        try:
            nb = nbformat.read(notebook_path, as_version=4)
            body, _ = HTMLExporter().from_notebook_node(nb)
        except Exception as exc:
            warnings.warn(
                f"Failed to render {notebook_path.name} to HTML; uploading "
                f"the notebook without an HTML rendering. ({exc})",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        html_path = Path(tempfile.gettempdir()) / f"{notebook_path.stem}.html"
        try:
            html_path.write_text(body, encoding="utf-8")
            mlflow.log_artifact(str(html_path), artifact_path=artifact_path)
        finally:
            try:
                html_path.unlink(missing_ok=True)
            except Exception:
                pass

    def load_data(
        self,
        run_id: str,
        artifact_path: str = "data",
        dst_path: str | Path | None = None,
    ) -> Path:
        """Download data artifacts from a run into local storage."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_path}",
                dst_path=str(dst_path) if dst_path else None,
            )
        except MlflowException as exc:
            raise TrackingError(
                f"Failed to download data artifacts from run '{run_id}': {exc}"
            ) from exc
        return Path(local_path)
