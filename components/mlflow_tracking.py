"""MLflow Tracking Component.

Creates a parent MLflow run for the pipeline and logs all parameters, metrics, and artifacts.
"""

from kfp import dsl
from typing import Optional


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
    packages_to_install=["mlflow>=2.10.0", "boto3"],
)
def mlflow_tracking(
    pvc_mount_path: str,
    # MLflow connection
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str = "distributed-training",
    mlflow_run_name: str = "",
    mlflow_insecure_tls: bool = True,
    # ^ Skip SSL verification for self-signed certificates (default: True for OpenShift routes)
    mlflow_enable: bool = True,
    # ^ If False, no-op (avoids a Condition node in the UI)
    # Model artifact from training
    input_model: dsl.Input[dsl.Model] = None,
    # Metrics from training/eval
    input_training_metrics: dsl.Input[dsl.Metrics] = None,
    input_eval_metrics: dsl.Input[dsl.Metrics] = None,
    # Training parameters to log
    training_algorithm: str = "",
    training_base_model: str = "",
    training_learning_rate: float = 0.0,
    training_num_epochs: int = 0,
    training_effective_batch_size: int = 0,
    training_max_seq_len: int = 0,
    training_max_tokens_per_gpu: int = 0,
    training_unfreeze_rank_ratio: float = 0.0,
    # Pipeline metadata
    pipeline_run_id: str = "",
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Create an MLflow run and log all pipeline artifacts, parameters, and metrics.

    This component creates a single parent run representing the entire pipeline execution.
    All training parameters, evaluation metrics, and model artifacts are logged to this run.

    Args:
        pvc_mount_path: Path to shared PVC.
        mlflow_tracking_uri: MLflow tracking server URI (e.g., http://mlflow.mlflow.svc:5000).
        mlflow_experiment_name: Name of the MLflow experiment.
        mlflow_run_name: Optional name for the run (defaults to auto-generated).
        input_model: Model artifact from training component.
        input_training_metrics: Metrics from training component.
        input_eval_metrics: Metrics from evaluation component.
        training_algorithm: Training algorithm used (OSFT/SFT).
        training_base_model: Base model HuggingFace ID.
        training_learning_rate: Learning rate used.
        training_num_epochs: Number of epochs.
        training_effective_batch_size: Effective batch size.
        training_max_seq_len: Max sequence length.
        training_max_tokens_per_gpu: Max tokens per GPU.
        training_unfreeze_rank_ratio: OSFT unfreeze ratio (0 for SFT).
        pipeline_run_id: KFP pipeline run ID for lineage.
        shared_log_file: Shared log file name.

    Returns:
        MLflow run ID.
    """
    import os
    import stat
    import mlflow
    from urllib.parse import urlparse
    from datetime import datetime

    print("=" * 60)
    print("MLFLOW TRACKING COMPONENT")
    print("=" * 60)

    print(f"\n  Tracking URI: {mlflow_tracking_uri}")
    print(f"  Experiment: {mlflow_experiment_name}")
    print(f"  Run Name: {mlflow_run_name or '(auto)'}")
    print(f"  Insecure TLS: {mlflow_insecure_tls}")
    print(f"  Enabled: {mlflow_enable}")

    if not mlflow_enable:
        print("  MLflow tracking disabled; skipping.")
        # Write to shared log for visibility
        log_path = os.path.join(pvc_mount_path, shared_log_file)
        try:
            with open(log_path, "a") as f:
                f.write("MLflow Tracking: skipped (mlflow_enable=False)\n")
            print(f"  [Log written to {log_path}]")
        except Exception as e:
            print(f"  WARNING: Could not write skip log: {e}")
        print("\n" + "=" * 60)
        print("COMPLETE - MLflow skipped")
        print("=" * 60)
        return ""

    # Handle self-signed certificates (common in OpenShift)
    if mlflow_insecure_tls:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        # Also set for underlying requests library
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        print("  [SSL verification disabled for self-signed certs]")

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Default run name: use provided name, fall back to experiment name
    # The pipeline passes dsl.PIPELINE_JOB_NAME_PLACEHOLDER which resolves to KFP run name
    if not mlflow_run_name:
        mlflow_run_name = mlflow_experiment_name

    run_id = ""

    def get_pipeline_definition_id() -> str:
        """Get pipeline definition fingerprint for parent run grouping.
        
        All runs of the same experiment share one parent.
        Compare runs using MLflow tags (algorithm, base_model, etc.).
        """
        # Check for explicit env vars (optional, for advanced use cases)
        for key in [
            "PIPELINE_SPEC_MD5",
            "PIPELINE_VERSION",
            "PIPELINE_DEFINITION_HASH",
            "PIPELINE_DEFINITION_ID",
        ]:
            val = os.environ.get(key)
            if val:
                return f"{mlflow_experiment_name}-{val}"

        # Default: group by experiment name
        return mlflow_experiment_name

    def get_or_create_parent_run(exp_id: str, definition_id: str):
        """Find a parent run for this pipeline definition, create if missing."""
        try:
            parent_runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string=(
                    f"tags.pipeline_definition_id = '{definition_id}' "
                    "and tags.run_level = 'parent'"
                ),
                max_results=1,
            )
            if len(parent_runs) > 0:
                parent_run = parent_runs.iloc[0]
                print(f"  Reusing parent run: {parent_run.run_id}")
                return parent_run.run_id
        except Exception as e:
            print(f"  WARNING: search_runs failed, will create parent: {e}")

        # Create new parent
        parent = mlflow.start_run(
            run_name=f"pipeline-def-{definition_id}",
            nested=False,
        )
        mlflow.set_tags(
            {
                "run_level": "parent",
                "pipeline_definition_id": definition_id,
                "pipeline_experiment": mlflow_experiment_name,
            }
        )
        parent_id = parent.info.run_id
        mlflow.end_run()
        print(f"  Created parent run: {parent_id}")
        return parent_id

    # Resolve parent/child nesting (default on)
    exp = mlflow.get_experiment_by_name(mlflow_experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(mlflow_experiment_name)
    else:
        exp_id = exp.experiment_id

    pipeline_def_id = get_pipeline_definition_id()
    parent_run_id = get_or_create_parent_run(exp_id, pipeline_def_id)

    with mlflow.start_run(run_name=mlflow_run_name, nested=True, parent_run_id=parent_run_id) as run:
        run_id = run.info.run_id
        print(f"\n[MLflow Run Started] ID: {run_id} (parent: {parent_run_id})")
        mlflow.set_tags(
            {
                "run_level": "child",
                "pipeline_definition_id": pipeline_def_id,
                "parent_run_id": parent_run_id,
            }
        )

        # --------------------------------------------------------
        # Log Training Parameters
        # --------------------------------------------------------
        print("\n[Logging Parameters...]")
        params = {
            "algorithm": training_algorithm,
            "base_model": training_base_model,
            "learning_rate": training_learning_rate,
            "num_epochs": training_num_epochs,
            "effective_batch_size": training_effective_batch_size,
            "max_seq_len": training_max_seq_len,
            "max_tokens_per_gpu": training_max_tokens_per_gpu,
            "unfreeze_rank_ratio": training_unfreeze_rank_ratio,
            "pipeline_run_id": pipeline_run_id,
        }
        # Filter out empty/zero values for cleaner logging
        params = {k: v for k, v in params.items() if v}
        mlflow.log_params(params)
        print(f"  Logged {len(params)} parameters")
        for k, v in params.items():
            print(f"    - {k}: {v}")

        # --------------------------------------------------------
        # Log Training Metrics (final values from artifact metadata)
        # --------------------------------------------------------
        if input_training_metrics:
            print("\n[Logging Training Metrics (final)...]")
            try:
                training_metrics = dict(input_training_metrics.metadata) if hasattr(input_training_metrics, 'metadata') else {}
                for k, v in training_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"training_{k}", float(v))
                        print(f"    - training_{k}: {v}")
                print(f"  Logged {len(training_metrics)} training metrics")
            except Exception as e:
                print(f"  WARNING: Could not log training metrics: {e}")

        # --------------------------------------------------------
        # Log Training History (per-step metrics from JSONL file)
        # --------------------------------------------------------
        def log_training_history():
            """Read training history JSONL and log per-step metrics for curves."""
            import json
            import glob

            # Training metrics are in the checkpoints directory, not final_model
            ckpt_dir = os.path.join(pvc_mount_path, "checkpoints")

            print(f"\n[Logging Training History from {ckpt_dir}...]")

            # Determine metrics file based on algorithm
            metrics_file = None
            algorithm = training_algorithm.upper() if training_algorithm else ""

            if algorithm == "OSFT":
                # OSFT uses training_metrics_0.jsonl
                candidate = os.path.join(ckpt_dir, "training_metrics_0.jsonl")
                if os.path.exists(candidate):
                    metrics_file = candidate
            elif algorithm == "SFT":
                # SFT uses training_params_and_metrics_global0.jsonl
                candidate = os.path.join(ckpt_dir, "training_params_and_metrics_global0.jsonl")
                if os.path.exists(candidate):
                    metrics_file = candidate

            # Fallback: try both patterns
            if not metrics_file:
                for pattern in ["training_metrics_0.jsonl", "training_params_and_metrics_global0.jsonl"]:
                    candidate = os.path.join(ckpt_dir, pattern)
                    if os.path.exists(candidate):
                        metrics_file = candidate
                        break

            if not metrics_file:
                print(f"  No training history file found in {ckpt_dir}")
                return

            print(f"  Found: {metrics_file}")

            # Read and parse JSONL
            try:
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"  ERROR reading file: {e}")
                return

            if not lines:
                print("  File is empty")
                return

            # For SFT, first line is config; skip it
            start_idx = 0
            if "training_params_and_metrics" in metrics_file and len(lines) > 1:
                start_idx = 1

            logged_count = 0
            for line in lines[start_idx:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = entry.get("step", logged_count)

                # OSFT fields (check value is not None)
                if entry.get("loss") is not None:
                    mlflow.log_metric("train_loss", float(entry["loss"]), step=step)
                if entry.get("lr") is not None:
                    mlflow.log_metric("learning_rate", float(entry["lr"]), step=step)
                if entry.get("grad_norm") is not None:
                    mlflow.log_metric("grad_norm", float(entry["grad_norm"]), step=step)
                if entry.get("samples_per_second") is not None:
                    mlflow.log_metric("samples_per_second", float(entry["samples_per_second"]), step=step)
                if entry.get("val_loss") is not None:
                    mlflow.log_metric("val_loss", float(entry["val_loss"]), step=step)
                if entry.get("tokens_per_second") is not None:
                    mlflow.log_metric("tokens_per_second", float(entry["tokens_per_second"]), step=step)

                # SFT fields (different names)
                if entry.get("avg_loss") is not None:
                    mlflow.log_metric("train_loss", float(entry["avg_loss"]), step=step)
                if entry.get("gradnorm") is not None:
                    mlflow.log_metric("grad_norm", float(entry["gradnorm"]), step=step)
                if entry.get("overall_throughput") is not None:
                    mlflow.log_metric("throughput", float(entry["overall_throughput"]), step=step)

                logged_count += 1

            print(f"  Logged {logged_count} training steps with per-step metrics")

        print("\n[Attempting to log training history...]")
        try:
            log_training_history()
        except Exception as e:
            import traceback
            print(f"  WARNING: Could not log training history: {e}")
            traceback.print_exc()

        # --------------------------------------------------------
        # Log Evaluation Metrics
        # --------------------------------------------------------
        def sanitize_metric_name(name: str) -> str:
            """Sanitize metric name for MLflow (alphanumerics, _, -, ., :, /, space only)."""
            import re
            sanitized = re.sub(r"[,;!@#$%^&*()+=\[\]{}|\\<>?]", "_", name)
            sanitized = re.sub(r"_+", "_", sanitized)
            return sanitized.strip("_")

        if input_eval_metrics:
            print("\n[Logging Evaluation Metrics...]")
            try:
                eval_metrics = dict(input_eval_metrics.metadata) if hasattr(input_eval_metrics, 'metadata') else {}
                for k, v in eval_metrics.items():
                    if isinstance(v, (int, float)):
                        safe_name = sanitize_metric_name(f"eval_{k}")
                        mlflow.log_metric(safe_name, float(v))
                        print(f"    - {safe_name}: {v}")
                print(f"  Logged {len(eval_metrics)} evaluation metrics")
            except Exception as e:
                print(f"  WARNING: Could not log eval metrics: {e}")

        # --------------------------------------------------------
        # Log Model Artifact
        # --------------------------------------------------------
        if input_model:
            print("\n[Logging Model Artifact...]")
            try:
                artifact_uri = mlflow.get_artifact_uri()
                parsed = urlparse(artifact_uri)
                base_path = parsed.path or "/"
                can_write = parsed.scheme not in ("", "file") or os.access(base_path, os.W_OK | os.X_OK)

                model_path = input_model.path
                if can_write and os.path.exists(model_path):
                    mlflow.log_artifacts(model_path, artifact_path="model")
                    print(f"  Logged model from: {model_path}")
                else:
                    mlflow.log_param("model_uri", getattr(input_model, "uri", model_path))
                    reason = "artifact store not writable" if not can_write else "model path not found locally"
                    print(f"  Skipped model artifact upload ({reason}); logged URI instead")

                # Log model metadata
                if hasattr(input_model, 'metadata'):
                    for k, v in input_model.metadata.items():
                        if isinstance(v, (str, int, float, bool)):
                            mlflow.log_param(f"model_{k}", v)
            except Exception as e:
                print(f"  WARNING: Could not log model artifact: {e}")

        # --------------------------------------------------------
        # Log Pipeline Log File
        # --------------------------------------------------------
        log_path = os.path.join(pvc_mount_path, shared_log_file)
        if os.path.exists(log_path):
            print(f"\n[Logging Pipeline Log...]")
            try:
                mlflow.log_artifact(log_path)
                print(f"  Logged: {log_path}")
            except PermissionError:
                print("  WARNING: Artifact store not writable; skipping pipeline log upload")
            except Exception as e:
                print(f"  WARNING: Could not log pipeline log: {e}")

        # Add tags for easy filtering
        mlflow.set_tag("pipeline", "distributed-training")
        mlflow.set_tag("source", "kfp")
        if training_base_model:
            mlflow.set_tag("base_model", training_base_model)
        if training_algorithm:
            mlflow.set_tag("algorithm", training_algorithm)

    # Write to shared log
    log_entry = f"MLflow Tracking: run_id={run_id}, experiment={mlflow_experiment_name}\n"
    try:
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"\n[Log written to {log_path}]")
    except Exception as e:
        print(f"  WARNING: Could not write to log: {e}")

    print("\n" + "=" * 60)
    print(f"COMPLETE - MLflow Run ID: {run_id}")
    print("=" * 60)

    return run_id


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        mlflow_tracking,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: mlflow_tracking_component.yaml")

