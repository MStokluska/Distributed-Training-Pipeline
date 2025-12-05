"""Training Component.

This is a skeleton component that will be replaced with actual training logic.
"""

from kfp import dsl


@dsl.component(
    # This is a Universal Image CPU based (I know the quay repo is confusing)
    base_image="quay.io/opendatahub/odh-training-th03-cuda128-torch28-py312-rhel9@sha256:84d05c5ef9dd3c6ff8173c93dca7e2e6a1cab290f416fb2c469574f89b8e6438",
    packages_to_install=["kubernetes"],
)
def training(
    pvc_mount_path: str,
    shared_log_file: str = "pipeline_log.txt",
) -> str:
    """Perform model training.

    This skeleton component writes a hello world message to a shared file on the PVC.

    Kubernetes credentials are read from environment variables (injected from secret):
        - KUBERNETES_SERVER_URL: The Kubernetes API server URL
        - KUBERNETES_AUTH_TOKEN: The bearer token for authentication
        - KUBERNETES_VERIFY_SSL: Whether to verify SSL (default: "true")

    Args:
        pvc_mount_path: Path where the shared PVC is mounted.
        shared_log_file: Name of the shared log file.

    Returns:
        Status message.
    """
    import os
    from kubernetes import client as k8s_client, config
    from kubernetes.client.rest import ApiException

    message = "Hello world from training"
    print(message)

    # Write to shared file on PVC
    log_path = os.path.join(pvc_mount_path, shared_log_file)
    with open(log_path, "a") as f:
        f.write(message + "\n")

    print(f"Message written to {log_path}")

    # =========================================================================
    # Kubernetes Client Setup
    # =========================================================================
    # Credentials are injected from a Kubernetes secret via environment variables
    # using kfp.kubernetes.use_secret_as_env() in the pipeline definition.
    # =========================================================================
    print("Loading Kubernetes configuration...")

    k8s_server_url = os.environ.get("KUBERNETES_SERVER_URL")
    k8s_auth_token = os.environ.get("KUBERNETES_AUTH_TOKEN")

    if k8s_server_url and k8s_auth_token:
        # Use explicit credentials from secret
        print(f"Using Kubernetes credentials from environment (server: {k8s_server_url})")
        configuration = k8s_client.Configuration()
        configuration.host = k8s_server_url
        configuration.api_key = {"authorization": f"Bearer {k8s_auth_token}"}
        # Control TLS verification via environment variable
        configuration.verify_ssl = os.environ.get("KUBERNETES_VERIFY_SSL", "true").lower() == "true"
        if not configuration.verify_ssl:
            print("Warning: TLS verification disabled for Kubernetes API")
        api_client = k8s_client.ApiClient(configuration)
        print("Loaded Kubernetes configuration from environment variables")
    else:
        # Fall back to in-cluster or kubeconfig
        print("No explicit credentials found, using in-cluster or kubeconfig...")
        try:
            config.load_incluster_config()
            print("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            config.load_kube_config()
            print("Loaded kubeconfig Kubernetes configuration")
        api_client = k8s_client.ApiClient()

    # Create the Custom Objects API client for TrainJob operations
    custom_objects_api = k8s_client.CustomObjectsApi(api_client)
    print("Successfully created Kubernetes API client")

    # TODO: Add your TrainJob creation logic here using custom_objects_api
    # Example:
    # train_job = {
    #     "apiVersion": "trainer.kubeflow.org/v1alpha1",
    #     "kind": "TrainJob",
    #     ...
    # }
    # custom_objects_api.create_namespaced_custom_object(...)

    return "training completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        training,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: training_component.yaml")

