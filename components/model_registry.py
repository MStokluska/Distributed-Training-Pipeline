"""Model Registry Component.

This is a skeleton component that will be replaced with actual model registry logic.
"""

from kfp import dsl


@dsl.component(
    # This is a Universal Image CPU based (I know the quay repo is confusing)
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
)
def model_registry(
    pvc_mount_path: str,
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Register model to the model registry.

    This skeleton component writes a hello world message to a shared file on the PVC,
    then reads and prints the entire log file to verify all 4 stages completed.

    Args:
        pvc_mount_path: Path where the shared PVC is mounted.
        shared_log_file: Name of the shared log file.

    Returns:
        Contents of the shared log file.
    """
    import os

    message = "Hello world from model registry"
    print(message)

    # Write to shared file on PVC
    log_path = os.path.join(pvc_mount_path, shared_log_file)
    with open(log_path, "a") as f:
        f.write(message + "\n")

    print(f"Message written to {log_path}")

    # Read and print the entire log file
    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION LOG:")
    print("=" * 50)

    with open(log_path, "r") as f:
        contents = f.read()

    print(contents)
    print("=" * 50)

    # Verify we have 4 hello worlds
    lines = [line for line in contents.strip().split("\n") if line.startswith("Hello world")]
    print(f"\nTotal hello world messages: {len(lines)}")

    if len(lines) == 4:
        print("[OK] All 4 pipeline stages completed successfully!")
    else:
        print(f"[FAIL] Expected 4 stages, but found {len(lines)}")

    return contents


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        model_registry,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: model_registry_component.yaml")

