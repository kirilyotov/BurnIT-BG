"""Shared helpers used across the experiment notebooks.

These modules sit on top of :mod:`data_platform` (MLflow, MinIO, env
loading) and add experiment-specific conveniences: prompt templates,
test-prompt benchmarking, model load/save helpers, GGUF export, etc.

Import in notebooks via:

    import sys
    sys.path.insert(0, '../../../shared')
    from model_utils import load_model_unsloth, save_to_minio
    from eval_utils import compute_perplexity
    from inference_utils import run_test_prompts, TEST_PROMPTS_IN_DOMAIN
"""
