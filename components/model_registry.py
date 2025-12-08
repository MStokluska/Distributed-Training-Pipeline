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
    author: str = "pipeline",
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Register model to Kubeflow Model Registry.
    
    Args:
        pvc_mount_path: Path to shared PVC.
        model_s3_bucket: S3 bucket containing the model.
        model_s3_key: S3 key/prefix for the model.
        model_s3_endpoint: S3/MinIO endpoint URL.
        model_s3_access_key: S3 access key.
        model_s3_secret_key: S3 secret key.
        registry_address: Model Registry server address (e.g., pipeline-test.rhoai-model-registries.svc).
        registry_port: Model Registry port (default 8080 for HTTP).
        model_name: Name for the registered model.
        model_version: Version string for this model.
        model_format_name: Model format (e.g., pytorch, tensorflow, onnx).
        model_format_version: Model format version.
        author: Author name for the model.
        shared_log_file: Shared log file name.
    
    Returns:
        Registered model ID.
    """
    import os
    from model_registry import ModelRegistry

    print("=" * 60)
    print("MODEL REGISTRY COMPONENT")
    print("=" * 60)

    # Build model URI
    model_uri = f"s3://{model_s3_bucket}/{model_s3_key}" if model_s3_bucket else f"pvc://{pvc_mount_path}/model"
    print(f"\n  Model Name: {model_name}")
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
        response = s3.list_objects_v2(Bucket=model_s3_bucket, Prefix=model_s3_key)
        files = response.get("Contents", [])
        print(f"  Found {len(files)} files in S3")
        for obj in files[:5]:
            print(f"    - {obj['Key']}")

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
        
        # Register the model
        registered_model = client.register_model(
            name=model_name,
            uri=model_uri,
            version=model_version,
            model_format_name=model_format_name,
            model_format_version=model_format_version,
            author=author,
            description=f"Registered via pipeline - {model_version}",
        )
        
        model_id = registered_model.id
        print(f"  Registered model: {registered_model.name} (ID: {model_id})")

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
