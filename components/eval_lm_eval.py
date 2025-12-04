"""Evaluation Component using lm-eval.

This is a skeleton component that will be replaced with actual lm-eval evaluation logic.
"""

from kfp import dsl


@dsl.component(
    # This is a Universal Image CPU based (I know the quay repo is confusing)
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
)
def eval_lm_eval(
    pvc_mount_path: str,
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Evaluate model using lm-eval.

    This skeleton component writes a hello world message to a shared file on the PVC.

    Args:
        pvc_mount_path: Path where the shared PVC is mounted.
        shared_log_file: Name of the shared log file.

    Returns:
        Status message.
    """
    import os

    message = "Hello world from eval with lm-eval"
    print(message)

    # Write to shared file on PVC
    log_path = os.path.join(pvc_mount_path, shared_log_file)
    with open(log_path, "a") as f:
        f.write(message + "\n")

    print(f"Message written to {log_path}")
    return "eval_lm_eval completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        eval_lm_eval,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: eval_lm_eval_component.yaml")

