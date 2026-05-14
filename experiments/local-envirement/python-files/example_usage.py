"""Minimal end-to-end example for the BurnIT-BG tracking helpers.

This script demonstrates the intended workflow:

1.  Pull credentials/tracking URI from a ``.env`` file (local) or from
    Google Colab secrets — the same call works in both runtimes.
2.  Configure MLflow once via ``MLflowTracking.from_env()``.
3.  Open a run that auto-logs hardware metadata and system metrics.
4.  Wrap logical sections in traces, register a Dataset entity, and
    attach metrics and the trained model to the run.

Run from the repo root with the project venv active:

    python experiments/example_usage.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_platform.common import set_env
from data_platform.tracking import MLflowTracking


def main() -> None:
    # 1. Populate os.environ from a local .env file or Google Colab userdata.
    #    No arguments needed — set_env() auto-detects the runtime.
    set_env()

    # 2. Configure MLflow once. Reads MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME /
    #    MLFLOW_TRACKING_INSECURE_TLS from the environment.
    tracking = MLflowTracking.from_env()
    tracking.check_connection()

    # 3. Prepare a tiny dataset so the example is self-contained.
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    # 4. Run the experiment. log_hardware=True records CPU/RAM/GPU/runtime
    #    automatically; system metrics also stream while the run is open.
    with tracking.run(run_name="wine-logreg-example", tags={"task": "classification"}):

        # Register the train DataFrame as an MLflow Dataset entity so metrics
        # can be scoped to it. `context="training"` is just a label.
        train_dataset = tracking.log_dataset(
            train_df, name="wine-train", targets="target", context="training",
        )

        with tracking.trace("preprocess", attributes={"rows": len(train_df)}):
            x_train = train_df.drop(columns=["target"])
            y_train = train_df["target"]
            x_test = test_df.drop(columns=["target"])
            y_test = test_df["target"]

        with tracking.trace("train", span_type="CHAIN"):
            model = LogisticRegression(max_iter=400, solver="lbfgs")
            model.fit(x_train, y_train)
            tracking.log_params({"max_iter": 400, "solver": "lbfgs", "algorithm": "LogReg"})

        with tracking.trace("evaluate", span_type="TOOL"):
            preds = model.predict(x_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1_macro": float(f1_score(y_test, preds, average="macro")),
            }
            tracking.log_metrics(metrics, dataset=train_dataset)
            print(f"Final metrics: {metrics}")

        # Final hardware snapshot at run end (lets you see RAM/GPU drift).
        tracking.log_hardware(step=1)

        # Persist the trained model as a run artifact.
        tracking.log_model(model, flavor="sklearn", artifact_path="model", input_example=x_train.head(2))


if __name__ == "__main__":
    main()
