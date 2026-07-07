"""R-Tuning dataset pipeline.

Builds the out-of-domain refusal training datasets for BurnIT-BG from
TriviaQA and SQuAD v2:

1. ``download_raw``  — pull raw EN datasets from HF Hub to local jsonl.
2. ``upload_raw``    — mirror raw to MinIO (and optionally HF Hub).
3. ``translate``     — translate question + answer columns to Bulgarian.
4. ``build_rtuning`` — turn translated rows into Alpaca-style R-Tuning records.
5. ``combine``       — interleave triviaqa + squadv2 into a combined dataset.
6. ``publish_dataset`` — stage each curated dataset + push to MinIO + HF Hub.

End-to-end runner: ``run_scripts/data_transformation/rtuning_dataset.sh``.
"""
