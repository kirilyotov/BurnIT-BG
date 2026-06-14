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
LLAMA_MODEL = DEFAULT_MODEL_NAME
GEMMA3_MODEL = "INSAIT-Institute/BgGPT-Gemma-3-4B-IT"
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
        from unsloth import FastLanguageModel
        return FastLanguageModel
    except ImportError as exc:
        log.info("Unsloth ImportError (fast path disabled): %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.warning("Unsloth installed but failed to import (%s): %s",
                    type(exc).__name__, exc)
        return None


def _has_real_bf16_training() -> bool:
    """Return True only when bf16 *training* is actually viable.

    ``torch.cuda.is_bf16_supported()`` is unreliable — on recent PyTorch it
    returns True for Turing T4 (compute 7.5) too, even though SFTConfig will
    reject ``bf16=True`` with "Your setup doesn't support bf16/gpu. You need
    Ampere+". The only safe signal is the device's compute capability: bf16
    training needs Ampere (8.0+) or newer.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def _pick_compute_dtype() -> Any:
    """Pick bf16 on Ampere+ GPUs, fp16 elsewhere (e.g. Colab T4 = Turing 7.5).

    Mixing dtypes between the base model and the trainer's AMP path triggers
    ``"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for
    'BFloat16'`` — fp16 GradScaler can't unscale bf16 grads. Resolving the
    dtype once here and threading it through both the Unsloth and transformers
    paths keeps weights, compute, and AMP aligned.
    """
    import torch

    if _has_real_bf16_training():
        return torch.bfloat16
    return torch.float16


QUANTIZATIONS = ("4bit", "8bit", "none", "bf16", "fp16")


_GEMMA3_LEGACY_ATTRS = (
    "max_position_embeddings", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "vocab_size", "head_dim", "rms_norm_eps", "rope_theta", "rope_scaling",
    "attention_bias", "attention_dropout", "hidden_activation", "tie_word_embeddings",
)


def _patch_gemma3_config_for_legacy_attrs() -> None:
    """Make ``Gemma3Config.foo`` transparently fall back to ``text_config.foo``.

    Gemma 3 is multimodal — its language-model hyperparameters live on
    ``config.text_config``, not on the root config. Unsloth (and some older
    transformers paths) still read them from the root and raise
    ``AttributeError: 'Gemma3Config' object has no attribute 'max_position_embeddings'``.
    The simplest non-invasive fix is to override ``__getattribute__`` so a few
    well-known legacy attrs proxy through. Idempotent.
    """
    try:
        from transformers.models.gemma3 import Gemma3Config
    except ImportError:
        return
    if getattr(Gemma3Config, "_burnit_legacy_proxy", False):
        return
    _orig_getattribute = Gemma3Config.__getattribute__

    def __getattribute__(self, name):  # type: ignore[no-redef]
        try:
            return _orig_getattribute(self, name)
        except AttributeError:
            if name in _GEMMA3_LEGACY_ATTRS:
                try:
                    text_config = _orig_getattribute(self, "text_config")
                except AttributeError:
                    raise
                if text_config is not None and hasattr(text_config, name):
                    return getattr(text_config, name)
            raise

    Gemma3Config.__getattribute__ = __getattribute__  # type: ignore[assignment]
    Gemma3Config._burnit_legacy_proxy = True  # type: ignore[attr-defined]


def load_model_unsloth(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    quantization: str = "4bit",
    load_in_4bit: bool | None = None,
    dtype: Any = None,
    token: str | None = None,
) -> tuple[Any, Any]:
    """Load the base model with the requested quantization. Returns ``(model, tokenizer)``.

    ``quantization`` chooses how the weights are stored on the GPU:

    * ``"4bit"`` (default) — NF4 via bitsandbytes; ~1.7 GB for a 3B model. Slight
      quality loss; lowest VRAM. Pair with QLoRA/DoRA. Unsloth fast path supported.
    * ``"8bit"`` — 8-bit via bitsandbytes; ~3.4 GB for a 3B model. Near-bf16
      quality. **Forces the plain-transformers fallback path** — Unsloth doesn't
      expose an 8-bit option, so this loses the speed boost.
    * ``"none"`` / ``"bf16"`` / ``"fp16"`` — no quantization; full-precision weights
      (~6 GB for a 3B model). Best quality, fastest training, biggest VRAM.

    ``load_in_4bit`` is kept as a backwards-compatible alias for old callers:
    ``True`` resolves to ``"4bit"``, ``False`` to ``"none"``. New code should
    use ``quantization`` directly.
    """
    if load_in_4bit is not None:
        quantization = "4bit" if load_in_4bit else "none"
    quantization = quantization.lower()
    if quantization not in QUANTIZATIONS:
        raise ValueError(
            f"unknown quantization {quantization!r}; choose one of {QUANTIZATIONS}"
        )
    is_4bit = quantization == "4bit"
    is_8bit = quantization == "8bit"

    
    if dtype is None:
        dtype = _pick_compute_dtype()

    # Gemma 3 nests legacy hyperparameters under ``config.text_config``; unsloth
    # and some transformers paths still read them from the root. Patch once so
    # ``config.max_position_embeddings`` etc. transparently work.
    _patch_gemma3_config_for_legacy_attrs()

    FastLanguageModel = _try_import_unsloth()
    # Unsloth supports 4bit and full-precision; it does NOT expose 8-bit, so we
    # fall back to plain transformers + bitsandbytes for that case.
    if FastLanguageModel is not None and not is_8bit:
        log.info("Loading %s via Unsloth (quantization=%s, dtype=%s)",
                 model_name, quantization, dtype)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=is_4bit,
            token=token,
        )
        return model, tokenizer

    if is_8bit and FastLanguageModel is not None:
        log.info("Unsloth has no 8-bit path; using transformers + bitsandbytes")
    else:
        log.warning("Unsloth not installed; falling back to transformers + bitsandbytes.")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_cfg = None
    if is_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif is_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )
    # "none" / "bf16" / "fp16" -> quant_cfg stays None

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
    )
    model.config.max_position_embeddings = max(
        model.config.max_position_embeddings, max_seq_length
    )
    return model, tokenizer


def _align_trainable_dtype(model: Any, target_dtype: Any) -> None:
    """Force every trainable parameter onto ``target_dtype`` in-place."""
    import torch
    if not isinstance(target_dtype, torch.dtype):
        return
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != target_dtype:
            p.data = p.data.to(target_dtype)
            n += 1
    if n:
        log.info("Aligned %d trainable params to %s", n, target_dtype)


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
    dtype: Any = None,
) -> Any:
    """Wrap a base model with QLoRA adapters (Unsloth fast path preferred)."""
    target_modules = target_modules or DEFAULT_TARGET_MODULES

    FastLanguageModel = _try_import_unsloth()
    if FastLanguageModel is not None:
        log.info("Applying QLoRA via Unsloth: r=%d alpha=%d", r, lora_alpha)
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
        )
        target = dtype or _pick_compute_dtype()
        _align_trainable_dtype(model, target)
        return model

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
    model = get_peft_model(model, peft_config)
    target = dtype or _pick_compute_dtype()
    _align_trainable_dtype(model, target)
    return model


def apply_dora(
    model: Any,
    *,
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
) -> Any:
    """Wrap a (typically 4-bit) model with **DoRA** adapters via PEFT.

    DoRA (weight-decomposed low-rank adaptation) is a drop-in upgrade over
    LoRA that usually trains slightly better at the same rank — it's just
    ``use_dora=True`` in PEFT's ``LoraConfig``. Unsloth's fast path doesn't
    expose DoRA, so this always goes through PEFT.
    """
    target_modules = target_modules or DEFAULT_TARGET_MODULES
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:  # noqa: BLE001 — non-quantized models don't need this
        pass
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
        use_dora=True,
    )
    log.info("Applying DoRA via PEFT: r=%d alpha=%d", r, lora_alpha)
    return get_peft_model(model, peft_config)


def apply_lora(
    model: Any,
    *,
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
) -> Any:
    """Wrap a **non-quantized** (bf16/fp16) model with plain LoRA adapters.

    Use this with ``load_in_4bit=False`` as the control run that isolates
    the quality cost of QLoRA's 4-bit quantization.
    """
    target_modules = target_modules or DEFAULT_TARGET_MODULES
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    log.info("Applying plain LoRA (bf16) via PEFT: r=%d alpha=%d", r, lora_alpha)
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


MODEL_CARD_TEMPLATE = """---
license: {license}
library_name: {library_name}
base_model: {base_model}
pipeline_tag: text-generation
language:
- bg
- en
tags:
- mental-health
- bulgarian
- peer-support
- {experiment}{datasets_block}
---

# BurnIT-BG — {experiment}

Модел за **емоционална връстническа подкрепа на български език**, обучен с
техниката `{experiment}` върху `{base_model}`.

## ⚠️ Отказ от отговорност / Disclaimer

Този модел е **само за изследователски цели**. Той **не е медицински съвет**
и не заменя професионална психологическа или психиатрична помощ.
This model is for **research only** and is **not medical advice**.

### При криза се обадете / In a crisis, call:

- **112** (спешни случаи / emergency)
- **116 111** (Национална телефонна линия за деца — free 24/7)
- **02 492 02 04** (Български Червен кръст — psychosocial support)

## Употреба / Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
```

Tracked in MLflow as experiment `burnit-bg-experiments` (tag `experiment={experiment}`).
"""


def default_model_card(
    experiment: str,
    *,
    base_model: str = DEFAULT_MODEL_NAME,
    repo_id: str = "",
    license: str = "cc-by-nc-4.0",
    library_name: str = "peft",
    datasets_id: str = "",
) -> str:
    """Render a Bulgarian/English model card with the required disclaimer."""
    datasets_block = f"\ndatasets:\n- {datasets_id}" if datasets_id else ""
    return MODEL_CARD_TEMPLATE.format(
        experiment=experiment, base_model=base_model,
        repo_id=repo_id or "<your-repo>", license=license,
        library_name=library_name, datasets_block=datasets_block,
    )


def default_repo_for(experiment: str, *, user: str = "kiplayo", repo: str = "BurnIT-BG") -> tuple[str, str]:
    """Single-repo strategy: return (repo_id, revision) — one branch per experiment."""
    revision = experiment.replace("_", "-")
    return f"{user}/{repo}", revision


def push_to_hf(
    model: Any,
    tokenizer: Any,
    repo_id: str,
    *,
    revision: str | None = None,
    private: bool = False,
    experiment: str = "",
    merge_adapters: bool = False,
    model_card: str | None = None,
    commit_message: str = "Upload BurnIT-BG model",
) -> str:
    """Save ``model``+``tokenizer`` locally and upload to an HF model repo.

    By default this uploads whatever ``save_pretrained`` produces — LoRA/DoRA
    adapters for PEFT models (small, require the base model at load time).
    Set ``merge_adapters=True`` to push a standalone merged model when the
    model supports Unsloth's ``save_pretrained_merged``.

    When ``revision`` is provided (and not ``"main"``), uploads to that branch
    of the repo via ``HfApi`` — used by the single-repo / one-branch-per-
    experiment strategy. Otherwise falls back to ``HuggingFaceStorage``.

    Returns the ``hf://models/{repo_id}[@{revision}]`` URI.
    """
    import os
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="hf_push_"))
    if merge_adapters and hasattr(model, "save_pretrained_merged"):
        log.info("Saving merged model for HF push")
        model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
    else:
        save_model_local(model, tokenizer, out)

    card = model_card or default_model_card(
        experiment or "experiment", repo_id=repo_id,
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    if revision is None or revision == "main":
        from data_platform.storage.hugging_face import HuggingFaceStorage

        storage = HuggingFaceStorage.from_env()
        return storage.save_model(
            local_dir=out, repo_id=repo_id, private=private,
            create_repo_if_missing=True, commit_message=commit_message,
        )

    from huggingface_hub import HfApi

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    try:
        api.create_branch(repo_id=repo_id, repo_type="model", branch=revision, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("create_branch failed for %s@%s: %s", repo_id, revision, exc)
    api.upload_folder(
        folder_path=str(out), repo_id=repo_id, repo_type="model",
        revision=revision, commit_message=commit_message,
        ignore_patterns=["*.ipynb_checkpoints*", "checkpoint-*/", "*.bin"],
    )
    return f"hf://models/{repo_id}@{revision}"


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
