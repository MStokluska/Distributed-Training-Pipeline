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
from typing import Literal

# Import components using relative path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.dataset_download import dataset_download
from components.training import train_model
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
    # Dataset Download Parameters (Required)
    # -------------------------------------------------------------------------
    # Dataset URI with scheme (hf://, s3://, pvc://, or absolute path)
    # Examples:
    #   - hf://HuggingFaceH4/ultrachat_200k
    #   - s3://my-bucket/datasets/chat_data.jsonl
    #   - pvc://datasets/local_data.jsonl
    dataset_uri: str,

    # -------------------------------------------------------------------------
    # Shared/Pipeline-wide Parameters
    # -------------------------------------------------------------------------
    shared_log_file: str = "pipeline_log.txt",  # Shared log file name

    # Training Component Parameters (prefixed with training_)
    training_algorithm: str = "OSFT",
    training_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    
    # Hyperparameters
    training_unfreeze_rank_ratio: float = 0.25,
    training_effective_batch_size: int = 128,
    training_num_epochs: int = 1,
    training_learning_rate: float = 5e-6,
    training_backend: str = "mini-trainer",
    training_lr_warmup_steps: int = 0,
    training_save_samples: int = None,
    training_accelerate_full_state_at_epoch: bool = None,
    training_seed: int = 42,
    training_max_tokens_per_gpu: int = 64000,
    training_max_seq_len: int = 8192,
    training_target_patterns: str = "",

    # Resources
    training_resources_num_nodes: int = 2,
    training_resource_cpu_per_worker: str = "8",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "32Gi",
    training_resource_num_procs_per_worker: int = 1,
    training_resource_num_workers: int = 1,

    training_use_liger: bool = True,
    training_use_processed_dataset: bool = None,
    training_unmask_messages: bool = None,
    training_lr_scheduler: str = "cosine",
    training_lr_scheduler_kwargs: str = "",
    
    training_checkpoint_at_epoch: bool = False,
    training_save_final_checkpoint: bool = True,
    # Runtime/resource/env parameters exposed for training
    training_envs: str = "",
    training_metadata_labels: str = "",
    training_metadata_annotations: str = "",

    # -------------------------------------------------------------------------
    # Dataset Download Parameters (Optional)
    # -------------------------------------------------------------------------
    dataset_train_split_ratio: float = 0.9,  # Train split ratio (0.9 = 90/10, 0.8 = 80/20)
    dataset_hf_token: str = "",  # HuggingFace token for gated/private datasets

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
    model_description: str = "",
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

        Algorithm common:
            training_algorithm: Training algorithm. "OSFT" = Orthogonal Subspace Fine-Tuning (continual learning), "SFT" = Standard Fine-Tuning.
            training_base_model: model_path: Path to the model to fine-tune
            training_learning_rate: learning_rate: Learning rate for training
            training_effective_batch_size: effective_batch_size: Effective batch size for training
            training_max_seq_len: max_seq_len: Maximum sequence length
            training_max_tokens_per_gpu: max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
            training_num_epochs: num_epochs: Number of training epochs
            training_lr_scheduler: lr_scheduler: Name of the PyTorch learning rate scheduler to use
            training_lr_warmup_steps: warmup_steps: Number of warmup steps
            training_lr_scheduler_kwargs: lr_scheduler_kwargs: Additional scheduler parameters (comma-delimited key=value pairs)
            training_accelerate_full_state_at_epoch: accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
            training_checkpoint_at_epoch: checkpoint_at_epoch: Whether to checkpoint at each epoch
            training_save_samples: save_samples: Number of samples to save after training (0 disables saving based on sample count)
            training_backend: backend: Backend implementation to use (default: "instructlab-training")
            training_target_patterns: target_patterns: Patterns to match when selecting modules for OSFT
            training_use_liger: use_liger: Whether to use Liger kernels for training
            training_use_processed_dataset: use_processed_dataset: Whether to use the processed dataset
            training_unmask_messages: unmask_messages: Whether to unmask messages during data processing

        Notes on paths (component-resolved):
            data_path: Path to the training data (resolved by component).
            ckpt_output_dir: Directory to save checkpoints (managed by component under the PVC).
            data_output_dir: Directory to save processed data (optional; component-managed when not provided).

    Example Parameters (uncomment in function signature to use):

        Dataset Download:
            dataset_uri: Dataset URI with scheme. Supported formats:
                - HuggingFace: hf://dataset-name or dataset-name
                - AWS S3: s3://bucket/path/file.jsonl (credentials from Kubernetes secret)
                - HTTP/HTTPS: https://... (e.g., MinIO shared links with embedded credentials)
                - Local/PVC: pvc://path/file.jsonl or /absolute/path/file.jsonl
            dataset_train_split_ratio: Train/eval split ratio (0.9 = 90/10, 0.8 = 80/20).
            dataset_hf_token: HuggingFace token for gated/private datasets.

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
    dataset_download_task = dataset_download(
        dataset_uri=dataset_uri,
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        train_split_ratio=dataset_train_split_ratio,
        hf_token=dataset_hf_token,
        shared_log_file=shared_log_file,
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    # Mount S3/MinIO credentials from Kubernetes secret
    kfp.kubernetes.use_secret_as_env(
        dataset_download_task,
        secret_name='minio-secret',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
        }
    )

    # =========================================================================
    # Stage 2: Training
    # =========================================================================
    # TODO: Pass training parameters to your actual training component
    # Example: train_model(epochs=training_epochs, batch_size=training_batch_size, 
    #                      use_liger=training_use_liger, ...)
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        training_base_model=training_base_model,
        training_algorithm=training_algorithm,
        training_unfreeze_rank_ratio=training_unfreeze_rank_ratio,
        training_effective_batch_size=training_effective_batch_size,
        training_max_tokens_per_gpu=training_max_tokens_per_gpu,
        training_max_seq_len=training_max_seq_len,
        training_learning_rate=training_learning_rate,
        training_backend=training_backend,
        training_target_patterns=training_target_patterns,
        training_seed=training_seed,
        training_use_liger=training_use_liger,
        training_use_processed_dataset=training_use_processed_dataset,
        training_unmask_messages=training_unmask_messages,
        training_lr_scheduler=training_lr_scheduler,
        training_lr_warmup_steps=training_lr_warmup_steps,
        training_save_samples=training_save_samples,
        training_accelerate_full_state_at_epoch=training_accelerate_full_state_at_epoch,
        training_lr_scheduler_kwargs=training_lr_scheduler_kwargs,
        training_checkpoint_at_epoch=training_checkpoint_at_epoch,
        training_save_final_checkpoint=training_save_final_checkpoint,
        training_num_epochs=training_num_epochs,
        training_envs=training_envs,
        training_resource_cpu_per_worker=training_resource_cpu_per_worker,
        training_resource_gpu_per_worker=training_resource_gpu_per_worker,
        training_resource_memory_per_worker=training_resource_memory_per_worker,
        training_resource_num_procs_per_worker=training_resource_num_procs_per_worker,
        training_resource_num_workers=training_resource_num_workers,
        training_metadata_labels=training_metadata_labels,
        training_metadata_annotations=training_metadata_annotations,
    )
    training_task.set_caching_options(False)
    training_task.after(dataset_download_task)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    # Inject Kubernetes credentials for TrainJob creation (from secret)
    # The training component reads these environment variables to connect to the K8s API.
    # Create the secret with:
    #   kubectl create secret generic kubernetes-credentials \
    #     --from-literal=server_url=https://your-k8s-api:6443 \
    #     --from-literal=auth_token=your-token
    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "server_url": "KUBERNETES_SERVER_URL",
            "auth_token": "KUBERNETES_AUTH_TOKEN",
        },
    )

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
        model_description=model_description,
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
