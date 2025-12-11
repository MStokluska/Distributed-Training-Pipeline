"""Model Registry Component.

Registers a trained model to Kubeflow Model Registry using the official SDK.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
    packages_to_install=["model-registry==0.2.10"],
)
def model_registry(
    pvc_mount_path: str,
    input_model: dsl.Input[dsl.Model] = None,
    input_metrics: dsl.Input[dsl.Metrics] = None,
    eval_metrics: dsl.Input[dsl.Metrics] = None,
    eval_results: dsl.Input[dsl.Artifact] = None,
    registry_address: str = "",
    registry_port: int = 8080,
    model_name: str = "fine-tuned-model",
    model_version: str = "1.0.0",
    model_format_name: str = "pytorch",
    model_format_version: str = "1.0",
    model_description: str = "",
    author: str = "pipeline",
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Register model to Kubeflow Model Registry.

    Uses the upstream model artifact (input_model) produced by training,
    or falls back to PVC path if no artifact is provided.

    Args:
        input_model: Model artifact from training step.
        input_metrics: Training metrics.
        eval_metrics: Evaluation metrics from lm-eval.
        eval_results: Full evaluation results JSON artifact.
    """
    import os
    from model_registry import ModelRegistry
    from model_registry.exceptions import StoreError

    print("=" * 60)
    print("MODEL REGISTRY COMPONENT")
    print("=" * 60)

    # Derive model URI from upstream artifact; use user-provided name (prioritize user input)
    resolved_model_name = model_name  # User's parameter takes precedence
    model_uri = ""
    base_model_name = None
    if input_model:
        meta = getattr(input_model, "metadata", {}) or {}
        base_model_name = meta.get("model_name")  # e.g., "Qwen/Qwen2.5-1.5B-Instruct"
        # Only use metadata name if user didn't provide one (kept default)
        if model_name == "fine-tuned-model" and base_model_name:
            resolved_model_name = base_model_name
        model_uri = meta.get("artifact_path") or getattr(input_model, "path", "") or model_uri
        if not model_uri:
            model_uri = f"pvc://{pvc_mount_path}/final_model"
    else:
        model_uri = f"pvc://{pvc_mount_path}/final_model"

    print(f"\n  Model Name: {resolved_model_name}")
    if base_model_name:
        print(f"  Base Model: {base_model_name}")
    print(f"  Model Version: {model_version}")
    print(f"  Model URI: {model_uri}")
    print(f"  Registry: {registry_address}:{registry_port}")

    # Register to Model Registry
    model_id = "SKIPPED"
    if registry_address:
        print("\n[Registering to Model Registry...]")
        
        # Ensure address has scheme for client URL building
        server_addr = registry_address
        if not server_addr.startswith("http://") and not server_addr.startswith("https://"):
            server_addr = f"http://{server_addr}"
        # Create client (HTTP/insecure)
        client = ModelRegistry(
            server_address=server_addr,
            port=registry_port,
            author=author,
            is_secure=False,  # HTTP
        )

        # Collect metrics into metadata if provided
        version_metadata = {}
        eval_summary = {}
        try:
            # Add base model info
            if base_model_name:
                version_metadata["base_model"] = base_model_name

            # Add training metrics (hyperparameters)
            if input_metrics and getattr(input_metrics, "metadata", None):
                print("\n  Training Hyperparameters:")
                for k, v in input_metrics.metadata.items():
                    if k not in ("display_name", "store_session_info"):
                        version_metadata[f"training_{k}"] = str(v)
                        print(f"    - {k}: {v}")

            # Add evaluation metrics
            if eval_metrics and getattr(eval_metrics, "metadata", None):
                print("\n  Evaluation Metrics:")
                for k, v in eval_metrics.metadata.items():
                    if k not in ("display_name", "store_session_info"):
                        version_metadata[f"eval_{k}"] = str(v)
                        # Extract primary accuracy metrics for summary
                        if "_acc," in k or "_acc_norm," in k:
                            if "stderr" not in k:
                                eval_summary[k] = v
                                print(f"    - {k}: {v:.4f}" if isinstance(v, float) else f"    - {k}: {v}")

                # Print eval config
                eval_tasks = eval_metrics.metadata.get("eval_tasks", "")
                eval_duration = eval_metrics.metadata.get("eval_duration_seconds", "")
                if eval_tasks:
                    print(f"    - Tasks: {eval_tasks}")
                if eval_duration:
                    print(f"    - Duration: {eval_duration}s")

            # Add summary of best metrics
            if eval_summary:
                # Find the best accuracy metric
                best_metric = max(eval_summary.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                version_metadata["eval_best_metric"] = best_metric[0]
                version_metadata["eval_best_score"] = str(best_metric[1])
                print(f"\n  Best Eval Score: {best_metric[0]} = {best_metric[1]:.4f}" if isinstance(best_metric[1], float) else f"\n  Best: {best_metric}")

            print(f"\n  Total metadata keys: {len(version_metadata)}")
        except Exception as e:
            print(f"  Warning: Could not extract metrics: {e}")
            version_metadata = {}

        try:
            registered_model = client.register_model(
                name=resolved_model_name,
                uri=model_uri,
                version=model_version,
                model_format_name=model_format_name,
                model_format_version=model_format_version,
                author=author,
                owner=author,
                description=model_description or f"Registered via pipeline - {model_version}",
                metadata=version_metadata or None,
            )
            model_id = registered_model.id
            print(f"  Registered model: {registered_model.name} (ID: {model_id})")
        except StoreError as e:
            msg = str(e)
            if "already exists" in msg.lower():
                print(f"  Model version already exists; skipping create. Details: {msg}")
                model_id = f"{resolved_model_name}:{model_version}"
            else:
                raise

    # Write to shared log
    log_path = os.path.join(pvc_mount_path, shared_log_file)
    with open(log_path, "a") as f:
        f.write(f"Model Registry: {model_name} v{model_version} (ID: {model_id})\n")
    print(f"\n[Log written to {log_path}]")

    print("\n" + "=" * 60)
    print(f"COMPLETE - Model ID: {model_id}")
    print("=" * 60)

    return str(model_id)


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        model_registry,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: model_registry_component.yaml")
