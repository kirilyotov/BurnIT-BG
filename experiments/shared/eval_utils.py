"""Evaluation helpers: perplexity, calibration (ECE), inference benchmarks.

Everything here is pure ``torch`` / ``numpy`` and works for any
HuggingFace-compatible causal LM. No Unsloth dependency.
"""

import matplotlib.pyplot as plt

import math
import time
from typing import Any, Iterable

import numpy as np


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: Iterable[str],
    *,
    max_length: int = 2048,
    stride: int = 1024,
    device: str | None = None,
) -> float:
    """Sliding-window perplexity over a list of texts.

    Uses the standard HuggingFace cross-entropy approach: tokenize all
    texts as one sequence, slide a window with stride, sum the negative
    log-likelihood weighted by the new tokens per window, exponentiate.
    """
    import torch

    if device is None:
        device = "cuda" if hasattr(model, "device") and model.device.type == "cuda" else "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"

    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        chunk = input_ids[:, begin:end]
        target = chunk.clone()
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=target)
        # out.loss is mean NLL over (trg_len - 1) target tokens
        nlls.append(out.loss * trg_len)
        prev_end = end
        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum() / seq_len
    return float(torch.exp(total_nll).item())


def compute_ece(
    confidences: np.ndarray | list[float],
    correctness: np.ndarray | list[bool],
    *,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    ``confidences[i]`` is the model's confidence in its prediction,
    ``correctness[i]`` whether the prediction was correct. Lower is better.
    """
    conf = np.asarray(confidences, dtype=float).reshape(-1)
    corr = np.asarray(correctness, dtype=float).reshape(-1)
    if conf.size != corr.size:
        raise ValueError("confidences and correctness must have the same length.")
    if conf.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc_bin = corr[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


def benchmark_speed(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str = "Hello, how are you today?",
    new_tokens: int = 128,
    warmup: int = 1,
    runs: int = 3,
) -> dict[str, float]:
    """Average tokens/sec over a few generations of ``new_tokens`` tokens."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)

    samples: list[float] = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        produced = out.size(1) - inputs.input_ids.size(1)
        samples.append(produced / elapsed)

    return {
        "tokens_per_sec_mean": float(np.mean(samples)),
        "tokens_per_sec_std": float(np.std(samples)),
        "new_tokens": int(new_tokens),
        "runs": int(runs),
    }


def vram_snapshot() -> dict[str, float]:
    """Current and peak VRAM in MB. ``{}`` when CUDA isn't available."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def compute_bertscore(predictions, references, *, lang="bg", model_type=None) -> dict:
    """BERTScore P/R/F1. Lazy-imports evaluate; lang='bg' → mBERT; pass model_type='xlm-roberta-large' for stronger."""
    import evaluate
    kw = {"lang": lang} if model_type is None else {"model_type": model_type}
    bs = evaluate.load("bertscore")
    r = bs.compute(predictions=list(predictions), references=list(references), **kw)
    n = len(r["f1"]) or 1
    return {"bertscore_P": sum(r["precision"])/n,
            "bertscore_R": sum(r["recall"])/n,
            "bertscore_F1": sum(r["f1"])/n}


def compute_rouge(predictions, references) -> dict:
    """ROUGE-L F1. use_stemmer=False (Porter is English-only)."""
    import evaluate
    r = evaluate.load("rouge").compute(predictions=list(predictions), references=list(references), use_stemmer=False)
    return {"rougeL_F1": float(r["rougeL"]) if not hasattr(r["rougeL"], "mid") else float(r["rougeL"].mid.fmeasure)}


def plot_train_eval_loss(history, *, best_step=None, title="loss"):
    """train/eval loss over steps; optional vertical line at best checkpoint."""
    import matplotlib.pyplot as plt
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr: ax.plot(*zip(*tr), label="train_loss")
    if ev: ax.plot(*zip(*ev), label="eval_loss", marker="o")
    if best_step is not None: ax.axvline(best_step, ls="--", c="grey", label="best ckpt")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=.3)
    return fig


def plot_grad_norm(history, *, title="gradient norm"):
    """Plot grad_norm from trainer.state.log_history."""
    import matplotlib.pyplot as plt
    pts = [(h["step"], h.get("grad_norm")) for h in history if "grad_norm" in h]
    fig, ax = plt.subplots(figsize=(7, 3))
    if pts: ax.plot(*zip(*pts))
    ax.set_xlabel("step"); ax.set_ylabel("grad_norm"); ax.set_title(title); ax.grid(True, alpha=.3)
    return fig


def plot_metric_scorecard(metrics: dict, *, title="scorecard"):
    """Horizontal bar of normalized [0,1] metrics. Caller must normalize (e.g. 1/(1+ppl))."""
    keys = list(metrics.keys()); vals = [float(metrics[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 0.45*len(keys)+1))
    ax.barh(keys, vals); ax.set_xlim(0, 1.0); ax.set_title(title)
    for i, v in enumerate(vals): ax.text(min(v+0.01, 0.98), i, f"{v:.3f}", va="center")
    ax.grid(True, axis="x", alpha=.3)
    return fig


def overfit_summary(history) -> dict:
    """Return {'final_train_loss','final_eval_loss','overfit_gap','min_eval_loss','min_eval_step','underfit_warning':bool,'initial_train_loss':float}."""
    tr = [h["loss"] for h in history if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    init_tr = float(tr[0]) if tr else 0.0
    fin_tr = float(tr[-1]) if tr else 0.0
    fin_ev = float(ev[-1][1]) if ev else float("nan")
    min_step, min_ev = (min(ev, key=lambda x: x[1]) if ev else (None, float("nan")))
    gap = (fin_ev - fin_tr) if ev else float("nan")
    underfit = (init_tr > 0 and fin_tr > 0.6 * init_tr)
    return {
        "initial_train_loss": init_tr, "final_train_loss": fin_tr,
        "final_eval_loss": fin_ev, "overfit_gap": gap,
        "min_eval_loss": float(min_ev), "min_eval_step": min_step,
        "underfit_warning": bool(underfit),
    }
