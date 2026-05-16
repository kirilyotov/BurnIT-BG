"""Model loading, saving, and export helpers for the LLM experiments.

These wrap **Unsloth** (preferred for fast 4-bit QLoRA fine-tuning) with
a graceful fallback to plain ``transformers + peft + bitsandbytes`` when
Unsloth isn't installed. Notebooks should use these helpers instead of
calling Unsloth directly so the code stays runnable on machines where
the fast path isn't available.

Default model: ``meta-llama/Llama-3.2-3B-Instruct``.

VRAM TIP — on a 4GB GPU (RTX 3050) inference works with 4-bit quant; for
fine-tuning prefer Google Colab T4 (15GB) or larger.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _try_import_unsloth() -> Any:
    """Return Unsloth's FastLanguageModel or ``None`` if unavailable."""
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
        return FastLanguageModel
    except ImportError:
        return None


def load_model_unsloth(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    dtype: Any = None,
    token: str | None = None,
) -> tuple[Any, Any]:
    """Load the base model in 4-bit, returning ``(model, tokenizer)``.

    Uses Unsloth when present; falls back to plain transformers+bitsandbytes.
    The returned objects are interchangeable for downstream code.
    """
    FastLanguageModel = _try_import_unsloth()
    if FastLanguageModel is not None:
        log.info("Loading %s via Unsloth (4bit=%s)", model_name, load_in_4bit)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
        )
        return model, tokenizer

    log.warning("Unsloth not installed; falling back to transformers + bitsandbytes.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=dtype or torch.bfloat16,
        device_map="auto",
        token=token,
    )
    model.config.max_position_embeddings = max(
        model.config.max_position_embeddings, max_seq_length
    )
    return model, tokenizer


def apply_qlora(
    model: Any,
    *,
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_gradient_checkpointing: str | bool = "unsloth",
    random_state: int = 42,
) -> Any:
    """Wrap a base model with QLoRA adapters (Unsloth fast path preferred)."""
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    FastLanguageModel = _try_import_unsloth()
    if FastLanguageModel is not None:
        log.info("Applying QLoRA via Unsloth: r=%d alpha=%d", r, lora_alpha)
        return FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
        )

    log.warning("Unsloth not installed; using peft + prepare_model_for_kbit_training.")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def count_trainable_params(model: Any) -> dict[str, int]:
    """Return ``{"trainable", "total", "trainable_pct"}`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable": int(trainable),
        "total": int(total),
        "trainable_pct": float(round(100.0 * trainable / max(total, 1), 4)),
    }


def save_model_local(model: Any, tokenizer: Any, output_dir: str | Path) -> Path:
    """Save model + tokenizer to ``output_dir`` (creates the directory)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def save_to_minio(
    local_dir: str | Path,
    *,
    remote_prefix: str,
    bucket: str | None = None,
) -> str:
    """Upload a saved-model directory to MinIO. Returns the ``s3://`` URI."""
    from data_platform.storage import MinioStorage

    storage = MinioStorage.from_env() if bucket is None else MinioStorage.from_env()
    return storage.save_directory(local_dir, remote_prefix, bucket=bucket)


def export_gguf(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    quantization: str = "q4_k_m",
) -> Path:
    """Export the model to GGUF format.

    Uses Unsloth's built-in GGUF export when available — it bundles
    llama.cpp under the hood. Without Unsloth, requires a separate
    ``llama.cpp`` build available on PATH (``llama-quantize`` /
    ``convert_hf_to_gguf.py``); raises a clear error otherwise so the
    notebook can skip the step.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained_gguf"):
        log.info("Exporting GGUF (%s) via Unsloth", quantization)
        model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quantization)
        return out

    # Fallback path — needs llama.cpp tooling on PATH
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if convert_script is None:
        raise RuntimeError(
            "GGUF export requires either Unsloth (preferred) or a llama.cpp "
            "checkout with `convert_hf_to_gguf.py` on PATH. Skipping export."
        )
    save_model_local(model, tokenizer, out / "hf")
    base_gguf = out / "model.f16.gguf"
    subprocess.check_call([
        convert_script,
        str(out / "hf"),
        "--outfile", str(base_gguf),
        "--outtype", "f16",
    ])
    quantize = shutil.which("llama-quantize") or shutil.which("quantize")
    if quantize is None:
        log.warning("llama-quantize not found; leaving f16 GGUF.")
        return base_gguf
    final = out / f"model.{quantization}.gguf"
    subprocess.check_call([quantize, str(base_gguf), str(final), quantization])
    return final


def cuda_device_info() -> dict[str, Any]:
    """Snapshot of CUDA availability + device properties (for MLflow params)."""
    try:
        import torch
    except ImportError:
        return {"cuda": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / 1024 ** 3, 2),
        "cc_major": props.major,
        "cc_minor": props.minor,
        "torch_version": torch.__version__,
    }
