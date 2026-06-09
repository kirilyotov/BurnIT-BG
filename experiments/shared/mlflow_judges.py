"""MLflow-native @scorer wrapping the BurnIT-BG 3-judge panel.

Drives ``mlflow.genai.evaluate(...)`` so per-row judge results show up in
MLflow's **Evaluation** tab as native Feedback entries — with scores,
rationales, sortable columns and aggregated metrics, exactly like the
example in MLflow's "Create custom code judge" UI.

Design choice — one scorer, many Feedback objects:
The decorated function returns a **list of Feedback** per row, so a single
panel call (which costs three NVIDIA-hosted requests) produces all five
quality axes plus the binary safety verdict. Without this, naively wiring
one scorer per axis would multiply judge cost ×5 with no benefit.

Per-process result cache (``_RESULT_CACHE``) deduplicates panel calls when
the same (prompt, response) pair is graded twice in one eval run.
"""

from __future__ import annotations

import logging
from typing import Any

from .judge_utils import JudgePanel

log = logging.getLogger(__name__)

_PANEL: JudgePanel | None = None
_RESULT_CACHE: dict[tuple[str, str], dict] = {}


def _get_panel() -> JudgePanel:
    """Lazily build a singleton JudgePanel — saves API-key checks per call."""
    global _PANEL
    if _PANEL is None:
        _PANEL = JudgePanel()
    return _PANEL


def reset_cache() -> None:
    """Clear the panel-result cache between MLflow eval runs."""
    _RESULT_CACHE.clear()


def _judge(prompt: str, response: str) -> dict:
    key = (prompt, response)
    if key not in _RESULT_CACHE:
        _RESULT_CACHE[key] = _get_panel().judge_one(prompt, response, "auto")
    return _RESULT_CACHE[key]


def _extract(inputs: Any, outputs: Any) -> tuple[str, str]:
    """Coerce eval-dataset rows into (prompt, response) strings."""
    if isinstance(inputs, dict):
        prompt = inputs.get("prompt") or inputs.get("question") or str(inputs)
    else:
        prompt = str(inputs)
    return str(prompt), str(outputs)


def _make_panel_scorer():
    """Build the @scorer-decorated panel scorer.

    Imported lazily so a missing ``mlflow[genai]`` install doesn't crash
    module-import for the rest of the system.
    """
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    @scorer
    def burnit_judge_panel(inputs, outputs) -> list[Feedback]:
        """3-judge panel returned as a list of MLflow Feedback objects.

        One call → six Feedback rows in MLflow's Evaluation tab:
        burnit.empathy, burnit.relevance, burnit.groundedness,
        burnit.safety_rubric, burnit.refusal_appropriateness, and
        burnit.safety_classifier (the LG+Nemotron+tie-break verdict).
        """
        prompt, response = _extract(inputs, outputs)
        result = _judge(prompt, response)
        quality = result.get("quality", {}) or {}
        safety = result.get("safety", {}) or {}

        feedbacks: list[Feedback] = []
        for label, qkey, rkey in [
            ("empathy", "empathy", "rationale_empathy"),
            ("relevance", "relevance", "rationale_relevance"),
            ("groundedness", "groundedness", "rationale_groundedness"),
            ("safety_rubric", "safety", "rationale_safety"),
            ("refusal_appropriateness", "refusal_appropriateness", "rationale_refusal"),
        ]:
            score = quality.get(qkey)
            if score is not None:
                feedbacks.append(Feedback(
                    name=f"burnit.{label}",
                    value=int(score),
                    rationale=str(quality.get(rkey) or "")[:500],
                ))

        final_unsafe = safety.get("final_unsafe")
        if final_unsafe is not None:
            lg = (safety.get("llama_guard") or {}).get("verdict")
            nem = (safety.get("nemotron") or {}).get("verdict")
            tb = (safety.get("tiebreak_reason") or "")[:200]
            feedbacks.append(Feedback(
                name="burnit.safety_classifier",
                value=bool(not final_unsafe),
                rationale=(
                    f"llama_guard={lg!r}  nemotron={nem!r}  "
                    f"disagreement={safety.get('safety_disagreement')}  "
                    f"tiebreak={tb!r}"
                ),
            ))
        return feedbacks

    return burnit_judge_panel


def get_scorers() -> list:
    """Return the list of scorers to pass to mlflow.genai.evaluate."""
    return [_make_panel_scorer()]


def batteries_to_eval_dataset(batteries: dict) -> list[dict]:
    """Convert ``run_full_test_battery`` output into mlflow.genai.evaluate data.

    Returns rows shaped like::

        {"inputs": {"prompt": "...", "battery": "in_domain"},
         "outputs": "the model's response"}

    Use with ``mlflow.genai.evaluate(data=..., scorers=get_scorers())``.
    """
    rows: list[dict] = []
    for name, items in (batteries or {}).items():
        for item in items:
            rows.append({
                "inputs": {"prompt": item.get("prompt", ""), "battery": name},
                "outputs": item.get("response", ""),
            })
    return rows


__all__ = [
    "get_scorers",
    "batteries_to_eval_dataset",
    "reset_cache",
]
