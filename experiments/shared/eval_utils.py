"""Evaluation helpers: perplexity, calibration (ECE), inference benchmarks.

Everything here is pure ``torch`` / ``numpy`` and works for any
HuggingFace-compatible causal LM. No Unsloth dependency.
"""

from __future__ import annotations

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
