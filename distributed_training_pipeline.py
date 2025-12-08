"""Distributed Training Pipeline.

A skeleton pipeline demonstrating shared PVC usage across 4 stages:
1. Dataset Download
2. Training  
3. Evaluation with lm-eval
4. Model Registry

Each component writes to a shared file on the PVC, and the final stage
prints the entire log to verify all stages completed successfully.
"""

import kfp
from kfp import dsl
import kfp.kubernetes

# Import components using relative path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.dataset_download import dataset_download
from components.training import training
from components.eval_lm_eval import eval_lm_eval
from components.model_registry import model_registry

# =============================================================================
# PVC Configuration (COMPILE-TIME settings - change these before compiling)
# =============================================================================
# NOTE: These settings are baked into the pipeline YAML at compile time.
# KFP's workspace feature configures the PVC as part of the pipeline spec,
# which means size/storage class cannot be changed at runtime.
# To modify these, edit the values below and recompile the pipeline.
# =============================================================================
PVC_SIZE = "10Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
# =============================================================================


@dsl.pipeline(
    name="dist-train",
    description="Skeleton pipeline with 4 stages sharing a PVC: dataset download, training, lm-eval, model registry",
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size=PVC_SIZE,
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "accessModes": PVC_ACCESS_MODES,
                    "storageClassName": PVC_STORAGE_CLASS,
                }
            )
        ),
    )
)
def distributed_training_pipeline(
    # =========================================================================
    # RUNTIME PARAMETERS
    # =========================================================================
    # These parameters can be configured when creating a pipeline run.
    # They are passed to components at runtime, unlike the PVC config above.
    # =========================================================================

    # -------------------------------------------------------------------------
    # Shared/Pipeline-wide Parameters
    # -------------------------------------------------------------------------
    shared_log_file: str = "pipeline_log.txt",  # Shared log file name

    # -------------------------------------------------------------------------
    # Dataset Download Parameters (EXAMPLE)
    # -------------------------------------------------------------------------
    # Source Configuration:
    # dataset_source_repo_id: str = "dataset/example",  # HuggingFace dataset repo ID
    # dataset_source_subset: str = "train",  # Dataset subset to download
    # dataset_source_revision: str = "main",  # Dataset revision/branch
    #
    # Authentication:
    # dataset_auth_hf_token: str = "",  # HuggingFace token for private datasets

    # -------------------------------------------------------------------------
    # Training Parameters (EXAMPLE)
    # -------------------------------------------------------------------------
    # Hyperparameters:
    # training_hyperparam_epochs: int = 3,  # Number of training epochs
    # training_hyperparam_batch_size: int = 32,  # Training batch size
    # training_hyperparam_learning_rate: float = 5e-5,  # Learning rate
    # training_hyperparam_use_liger: bool = False,  # Enable Liger kernel optimization
    # training_hyperparam_warmup_steps: int = 100,  # Warmup steps for scheduler
    #
    # Resource Configuration:
    # training_resource_cpu: str = "4",  # CPU cores to request
    # training_resource_memory: str = "16Gi",  # Memory to request
    # training_resource_gpu: int = 1,  # Number of GPUs to request
    # training_resource_gpu_type: str = "nvidia.com/gpu",  # GPU resource type

    # -------------------------------------------------------------------------
    # Evaluation Parameters (EXAMPLE)
    # -------------------------------------------------------------------------
    # Task Configuration:
    # eval_task_names: str = "hellaswag,arc_easy",  # Comma-separated lm-eval tasks
    # eval_task_limit: int = 100,  # Limit samples per task (use None for all)
    #
    # Inference Settings:
    # eval_inference_batch_size: int = 8,  # Evaluation batch size
    # eval_inference_max_tokens: int = 256,  # Max tokens to generate

    # -------------------------------------------------------------------------
    # S3 / Model location
    # -------------------------------------------------------------------------
    model_s3_bucket: str = "",
    model_s3_key: str = "",
    model_s3_endpoint: str = "",
    model_s3_access_key: str = "",
    model_s3_secret_key: str = "",

    # -------------------------------------------------------------------------
    # Model Registry (SDK client)
    # -------------------------------------------------------------------------
    registry_address: str = "",
    registry_port: int = 8080,
    model_name: str = "fine-tuned-model",
    model_version: str = "1.0.0",
    model_format_name: str = "pytorch",
    model_format_version: str = "1.0",
    author: str = "pipeline",
):
    """Distributed Training Pipeline with shared workspace PVC.

    This pipeline demonstrates a 4-stage workflow where all components share
    a workspace PVC for data exchange:

    1. Dataset Download - Prepares the training dataset
    2. Training - Trains the model
    3. Eval with lm-eval - Evaluates the trained model
    4. Model Registry - Registers the model and prints execution log

    Note:
        PVC size and storage class are COMPILE-TIME settings defined at the top
        of this file. They cannot be changed at runtime because KFP's workspace
        feature bakes these into the pipeline specification during compilation.
        To change PVC settings, modify PVC_SIZE/PVC_STORAGE_CLASS and recompile.

    Args:
        shared_log_file: Name of the shared log file for tracking completion.

    Example Parameters (uncomment in function signature to use):

        Dataset - Source Configuration:
            dataset_source_repo_id: HuggingFace dataset repository ID.
            dataset_source_subset: Dataset subset/split to download.
        Dataset - Authentication:
            dataset_auth_hf_token: HuggingFace API token for private datasets.

        Training - Hyperparameters:
            training_hyperparam_epochs: Number of training epochs.
            training_hyperparam_batch_size: Batch size for training.
            training_hyperparam_learning_rate: Learning rate for optimizer.
            training_hyperparam_use_liger: Enable Liger kernel optimization.
        Training - Resource Configuration:
            training_resource_cpu: CPU cores to request.
            training_resource_memory: Memory to request.
            training_resource_gpu: Number of GPUs to request.

        Evaluation - Task Configuration:
            eval_task_names: Comma-separated list of lm-eval tasks.
            eval_task_limit: Maximum samples per evaluation task.
        Evaluation - Inference Settings:
            eval_inference_batch_size: Batch size for evaluation.

        Model Registry - Model Metadata:
            registry_model_name: Name for the model in the registry.
            registry_model_version: Version string for the model.
        Model Registry - Connection:
            registry_connection_endpoint: Model registry API endpoint.
    """

    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    # TODO: Pass dataset parameters to your actual dataset download component
    # Example: dataset_download(hf_token=dataset_hf_token, repo_id=dataset_repo_id, ...)
    dataset_download_task = dataset_download(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        shared_log_file=shared_log_file,
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    # =========================================================================
    # Stage 2: Training
    # =========================================================================
    # TODO: Pass training parameters to your actual training component
    # Example: train_model(epochs=training_epochs, batch_size=training_batch_size, 
    #                      use_liger=training_use_liger, ...)
    training_task = training(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        shared_log_file=shared_log_file,
    )
    training_task.set_caching_options(False)
    training_task.after(dataset_download_task)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    # =========================================================================
    # Stage 3: Evaluation with lm-eval
    # =========================================================================
    # TODO: Pass evaluation parameters to your actual eval component
    # Example: evaluate_model(tasks=eval_tasks, batch_size=eval_batch_size, 
    #                         limit=eval_limit, ...)
    eval_task = eval_lm_eval(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        shared_log_file=shared_log_file,
    )
    eval_task.set_caching_options(False)
    eval_task.after(training_task)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    # =========================================================================
    # Stage 4: Model Registry
    # =========================================================================
    # TODO: Pass registry parameters to your actual model registry component
    # Example: register_model(model_name=registry_model_name, 
    #                         version=registry_model_version, endpoint=registry_endpoint, ...)
    model_registry_task = model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        model_s3_bucket=model_s3_bucket,
        model_s3_key=model_s3_key,
        model_s3_endpoint=model_s3_endpoint,
        model_s3_access_key=model_s3_access_key,
        model_s3_secret_key=model_s3_secret_key,
        registry_address=registry_address,
        registry_port=registry_port,
        model_name=model_name,
        model_version=model_version,
        model_format_name=model_format_name,
        model_format_version=model_format_version,
        author=author,
        shared_log_file=shared_log_file,
    )
    model_registry_task.set_caching_options(False)
    model_registry_task.after(eval_task)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=distributed_training_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
    print(f"Pipeline compiled successfully!")
    print(f"  PVC Size: {PVC_SIZE}")
    print(f"  Storage Class: {PVC_STORAGE_CLASS}")
    print(f"  Access Modes: {PVC_ACCESS_MODES}")
