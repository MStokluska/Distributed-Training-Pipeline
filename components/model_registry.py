"""Model Registry Component.

Registers a trained model to Kubeflow Model Registry using the official SDK.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
    packages_to_install=["boto3", "model-registry==0.2.10"],
)
def model_registry(
    pvc_mount_path: str,
    input_model: dsl.Input[dsl.Model] = None,
    input_metrics: dsl.Input[dsl.Metrics] = None,
    model_s3_bucket: str = "",
    model_s3_key: str = "",
    model_s3_endpoint: str = "",
    model_s3_access_key: str = "",
    model_s3_secret_key: str = "",
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

    Prefers the upstream model artifact (input_model) produced by training.
    Falls back to S3 parameters when no model artifact is provided.
    """
    import os
    import json
    from model_registry import ModelRegistry
    from model_registry.exceptions import StoreError

    print("=" * 60)
    print("MODEL REGISTRY COMPONENT")
    print("=" * 60)

    # Derive model URI and name from upstream artifact when available
    resolved_model_name = model_name
    model_uri = ""
    if input_model:
        # Prefer artifact_path metadata if present, else the artifact path
        meta = getattr(input_model, "metadata", {}) or {}
        resolved_model_name = meta.get("model_name", model_name)
        model_uri = meta.get("artifact_path") or getattr(input_model, "path", "") or model_uri
        if not model_uri:
            model_uri = f"pvc://{pvc_mount_path}/model"
    elif model_s3_bucket:
        model_uri = f"s3://{model_s3_bucket}/{model_s3_key}"
    else:
        model_uri = f"pvc://{pvc_mount_path}/model"

    print(f"\n  Model Name: {resolved_model_name}")
    print(f"  Model Version: {model_version}")
    print(f"  Model URI: {model_uri}")
    print(f"  Registry: {registry_address}:{registry_port}")

    # Verify S3 model exists
    if model_s3_bucket and model_s3_access_key and model_s3_secret_key:
        print("\n[Verifying S3 model...]")
        import boto3
        from botocore.client import Config
        s3 = boto3.client(
            "s3",
            endpoint_url=model_s3_endpoint,
            aws_access_key_id=model_s3_access_key,
            aws_secret_access_key=model_s3_secret_key,
            config=Config(signature_version="s3v4"),
        )
        try:
            response = s3.list_objects_v2(Bucket=model_s3_bucket, Prefix=model_s3_key)
            files = response.get("Contents", [])
            print(f"  Found {len(files)} files in S3")
            for obj in files[:5]:
                print(f"    - {obj['Key']}")
        except Exception as e:
            print(f"  WARNING: S3 verification failed: {e}")

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
        try:
            if input_metrics and getattr(input_metrics, "metadata", None):
                version_metadata = dict(input_metrics.metadata)
        except Exception:
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
