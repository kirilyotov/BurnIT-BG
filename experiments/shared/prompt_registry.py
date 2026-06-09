"""Register BurnIT-BG prompts in MLflow's Prompt Registry.

Pinning each prompt (test battery, judge rubric, tie-break, model system
prompt) to a versioned alias does two things:

1. Reproducibility — a later run can ``mlflow.genai.load_prompt(name, version)``
   to grab the exact template a previous experiment used, even after edits.
2. Visibility — every template variant shows up under the **Prompts** tab of
   the MLflow UI (where the user currently sees nothing).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .inference_utils import (
    DEFAULT_SYSTEM_PROMPT,
    TEST_PROMPTS_EDGE,
    TEST_PROMPTS_IN_DOMAIN,
    TEST_PROMPTS_OUT_OF_DOMAIN,
)
from .judge_utils import (
    QUALITY_JUDGE_SYSTEM,
    SAFETY_TIEBREAK_SYSTEM,
    UNSAFE_RATIONALE_SYSTEM,
)

log = logging.getLogger(__name__)


# Per-Python-session memo so re-running the registration cell costs zero
# round-trips after the first call. The MLflow registry server is idempotent
# (same template -> same version, no new revision), but the 7 round-trips
# still take ~1-2 s each run. Across kernel restarts the memo is cleared and
# we re-call, which is the right behavior.
_SESSION_REGISTERED: dict[str, Any] = {}


def _register(name: str, template: str, commit: str, tags: dict | None = None) -> Any:
    """Register one prompt; swallow errors so a flaky registry doesn't kill a run."""
    try:
        import mlflow
        return mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit,
            tags=tags or {},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("prompt-registry skipped %s: %s", name, exc)
        return None


def register_burnit_prompts(
    *,
    verbose: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Register every BurnIT-BG prompt; returns ``{name: PromptVersion|None}``.

    Per-session memoization: the SECOND call in the same Python session is
    a no-op that returns the cached result (~zero cost). Set ``force=True``
    to override and hit the registry again — useful after you edit a
    template and want a new version pushed without restarting the kernel.

    Set the env var ``SKIP_PROMPT_REGISTRY=1`` to skip registration entirely
    (returns ``{}``) — useful when you know the prompts are already in MLflow
    and you want the notebook to start faster.

    With ``verbose=True`` (default) prints each ``prompts:/<name>/<version>``
    URI so you can paste it into the MLflow UI's Prompts tab to confirm
    it landed.
    """
    if os.getenv("SKIP_PROMPT_REGISTRY", "").lower() in ("1", "true", "yes"):
        if verbose:
            print("[prompts] SKIP_PROMPT_REGISTRY=1 set — skipping registration.")
        return {}
    if _SESSION_REGISTERED and not force:
        if verbose:
            print(f"[prompts] already registered this Python session "
                  f"({len(_SESSION_REGISTERED)} prompts cached). "
                  f"Pass force=True to re-register.")
        return dict(_SESSION_REGISTERED)
    items: list[tuple[str, str, str, dict]] = [
        ("burnit-bg-system-bg",
         DEFAULT_SYSTEM_PROMPT,
         "Bulgarian peer-support system prompt (model inference)",
         {"lang": "bg", "purpose": "model_system"}),
        ("burnit-bg-judge-rubric-bg",
         QUALITY_JUDGE_SYSTEM,
         "5-axis Bulgarian rubric (empathy, relevance, groundedness, safety, refusal)",
         {"lang": "bg", "purpose": "judge_quality"}),
        ("burnit-bg-safety-tiebreak-bg",
         SAFETY_TIEBREAK_SYSTEM,
         "Mistral tie-breaker for safety classifier disagreement",
         {"lang": "bg", "purpose": "judge_safety_tiebreak"}),
        ("burnit-bg-unsafe-rationale-bg",
         UNSAFE_RATIONALE_SYSTEM,
         "Mistral rationale when both classifiers agree unsafe",
         {"lang": "bg", "purpose": "judge_unsafe_rationale"}),
        ("burnit-bg-test-in-domain",
         "\n---\n".join(TEST_PROMPTS_IN_DOMAIN),
         "In-domain Bulgarian mental-health test prompts",
         {"lang": "bg", "purpose": "test_battery", "battery": "in_domain"}),
        ("burnit-bg-test-out-of-domain",
         "\n---\n".join(TEST_PROMPTS_OUT_OF_DOMAIN),
         "Out-of-domain test prompts",
         {"lang": "bg", "purpose": "test_battery", "battery": "out_of_domain"}),
        ("burnit-bg-test-edge-cases",
         "\n---\n".join(TEST_PROMPTS_EDGE),
         "Edge-case prompts (self-harm, manipulation, medication request)",
         {"lang": "bg", "purpose": "test_battery", "battery": "edge_cases"}),
    ]
    out: dict[str, Any] = {}
    for name, template, commit, tags in items:
        pv = _register(name, template, commit, tags)
        out[name] = pv
        if verbose:
            if pv is None:
                print(f"[prompts] {name:40s} -> SKIPPED (see warnings)")
            else:
                ver = getattr(pv, "version", "?")
                print(f"[prompts] {name:40s} -> prompts:/{name}/{ver}")
    if verbose:
        n_ok = sum(1 for v in out.values() if v is not None)
        print(f"[prompts] {n_ok}/{len(out)} prompts registered. "
              "View them in the MLflow UI's top-level **Prompts** tab "
              "(they are global, not nested under your experiment).")

    # Memo this session so a re-run of the cell is a no-op.
    _SESSION_REGISTERED.clear()
    _SESSION_REGISTERED.update({k: v for k, v in out.items() if v is not None})
    return out


__all__ = ["register_burnit_prompts"]
