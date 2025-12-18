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
from components.eval_lm_eval import universal_llm_evaluator
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

# Pipeline identity (used in decorator AND provenance tracking)
PIPELINE_NAME = "dist-train"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
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
    # DATASET PARAMETERS
    # =========================================================================
    dataset_uri: str,
    # ^ Dataset URI with scheme. Supported formats:
    #   - hf://HuggingFaceH4/ultrachat_200k (HuggingFace dataset)
    #   - s3://bucket/path/file.jsonl (AWS S3, credentials from minio-secret)
    #   - https://url/file.jsonl (HTTP/HTTPS direct link)
    #   - pvc://path/file.jsonl or /absolute/path (local/PVC file)

    dataset_hf_token: str = "",
    # ^ HuggingFace token for gated/private datasets (e.g., Llama datasets).

    dataset_split_ratio: float = 0.9,
    # ^ Ratio for train/eval split. 0.9 = 90% train, 10% eval. 0.8 = 80/20.

    dataset_subset_count: int = 0,
    # ^ Number of examples to use from dataset. 0 = use all.
    #   Useful for quick testing (e.g., 100) or validation runs (e.g., 1000).

    # =========================================================================
    # TRAINING - Environment & Auth (training_env_*)
    # =========================================================================
    training_env_annotations: str = "",
    # ^ Kubernetes annotations for training pods as comma-delimited key=value pairs.

    training_env_hf_token: str = "",
    # ^ HuggingFace token for gated models (Llama, Mistral, etc.).
    #   Leave empty for public models.

    training_env_labels: str = "",
    # ^ Kubernetes labels for training pods as comma-delimited key=value pairs.

    training_env_vars: str = "",
    # ^ Additional environment variables as comma-delimited key=value pairs.
    #   Example: "NCCL_DEBUG=INFO,CUDA_VISIBLE_DEVICES=0,1"

    training_env_pull_secret: str = "",
    # ^ Pull secret for Red Hat Catalog, registry.redhat.io (Docker config.json content).

    # =========================================================================
    # TRAINING - Hyperparameters (training_hyper_*)
    # =========================================================================
    training_hyper_batch_size: int = 128,
    # ^ Effective batch size (samples per optimizer step). Automatically handles
    #   gradient accumulation. Guidance: 1 GPU: 16-32, 2 GPUs: 32-64, 4 GPUs: 64-128.

    training_hyper_epochs: int = 1,
    # ^ Number of training epochs. 1 = quick test, 3-5 = better convergence.

    training_hyper_learning_rate: float = 5e-6,
    # ^ Learning rate. Typical range: 1e-6 to 1e-4. 5e-6 is good for OSFT.

    training_hyper_max_seq_len: int = 8192,
    # ^ Maximum sequence length in tokens. Common values: 2048, 4096, 8192.

    training_hyper_max_tokens_per_gpu: int = 64000,
    # ^ Maximum tokens per GPU per batch (memory hard-cap). Used to auto-calculate
    #   micro-batch size and gradient accumulation while avoiding OOMs.

    training_hyper_seed: int = 42,
    # ^ Random seed for reproducibility.

    training_hyper_target_patterns: str = "",
    # ^ (OSFT only) Comma-separated patterns for selecting modules to train.
    #   Leave empty for default selection.

    # =========================================================================
    # TRAINING - Learning Rate Schedule (training_lr_*)
    # =========================================================================
    training_lr_scheduler: str = "cosine",
    # ^ (OSFT only) Learning rate scheduler: "cosine", "linear", "constant".

    training_lr_scheduler_kwargs: str = "",
    # ^ (OSFT only) Additional scheduler parameters as comma-delimited key=value pairs.
    #   Example: "num_cycles=1,num_warmup_steps=100"

    training_lr_warmup_steps: int = 0,
    # ^ Number of warmup steps before reaching full learning rate.

    # =========================================================================
    # TRAINING - Model Configuration (training_model_*)
    # =========================================================================
    training_model_algorithm: str = "OSFT",
    # ^ Training algorithm: "OSFT" (Orthogonal Subspace Fine-Tuning for continual
    #   learning) or "SFT" (Standard Fine-Tuning).

    training_model_backend: str = "mini-trainer",
    # ^ Training backend implementation: "mini-trainer" or "instructlab-training".

    training_model_base: str = "Qwen/Qwen2.5-1.5B-Instruct",
    # ^ HuggingFace model ID or path to fine-tune.

    training_pull_secret: str = "",
    # ^ Pull secret for Red Hat Catalog, registry.redhat.io (Docker config.json content).

    training_model_unfreeze_ratio: float = 0.25,
    # ^ (OSFT only) Ratio of parameters to unfreeze. Lower = more efficient,
    #   higher = more capacity. Typical range: 0.1-0.5.

    # =========================================================================
    # TRAINING - Optimizations (training_opt_*)
    # =========================================================================
    training_opt_processed_dataset: bool = False,
    # ^ (OSFT only) Set to True if dataset is already tokenized with input_ids.
    #   False (default) = process raw chat data.

    training_opt_unmask_messages: bool = False,
    # ^ (OSFT only) Whether to unmask messages during chat data processing.

    training_opt_use_liger: bool = True,
    # ^ (OSFT only) Enable Liger kernel optimizations for faster training.
    #   Requires Liger kernels in the training image.

    # =========================================================================
    # TRAINING - Resources (training_res_*)
    # =========================================================================
    training_res_cpu: str = "8",
    # ^ CPU cores per training worker pod.

    training_res_gpu: int = 1,
    # ^ GPUs per training worker pod. Usually equals num_procs.

    training_res_memory: str = "32Gi",
    # ^ Memory per training worker pod.

    training_res_num_procs: str = "auto",
    # ^ Number of processes (ranks) per worker. "auto" uses GPUs per worker.

    training_res_num_workers: int = 1,
    # ^ Total number of worker pods. 1 = single-node, 2+ = multi-node distributed.

    # =========================================================================
    # TRAINING - Saving & Checkpoints (training_save_*)
    # =========================================================================
    training_save_at_epoch: bool = False,
    # ^ Save a checkpoint at the end of each epoch. (Common)

    training_save_final: bool = True,
    # ^ (OSFT only) Save the final model checkpoint after training completes.

    training_save_full_state: bool = False,
    # ^ (SFT only) Save full Accelerate state at each epoch for resumption capability.

    training_save_samples: int = 0,
    # ^ (SFT only) Number of samples to save during training. 0 disables.

    # =========================================================================
    # EVALUATION - Configuration (eval_cfg_*)
    # =========================================================================
    eval_cfg_batch_size: str = "auto",
    # ^ Batch size for evaluation. "auto" lets vLLM determine optimal size.
    #   Can also be an integer like "8" or "16".

    eval_cfg_limit: int = -1,
    # ^ Maximum samples per task. -1 = evaluate all samples.
    #   Use smaller values (e.g., 100) for quick validation.

    eval_cfg_log_samples: bool = True,
    # ^ Whether to log individual sample predictions to output artifact.

    eval_cfg_verbosity: str = "INFO",
    # ^ Logging verbosity level: "DEBUG", "INFO", "WARNING", "ERROR".

    # =========================================================================
    # EVALUATION - Model & Generation (eval_model_*, eval_gen_*)
    # =========================================================================
    eval_gen_kwargs: dict = {},
    # ^ Generation kwargs for generative tasks.
    #   Example: {"max_tokens": 256, "temperature": 0.0}

    eval_model_args: dict = {},
    # ^ Additional model arguments as dictionary.
    #   Example: {"dtype": "float16", "gpu_memory_utilization": 0.9}

    # =========================================================================
    # EVALUATION - Tasks (eval_task_*)
    # =========================================================================
    eval_task_names: list = ["arc_easy"],
    # ^ List of lm-eval task names to run.
    #   Examples: ["mmlu"], ["gsm8k", "arc_easy"], ["hellaswag", "winogrande"]
    #   See: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks

    # =========================================================================
    # MODEL REGISTRY - Connection (registry_*)
    # =========================================================================
    registry_address: str = "",
    # ^ Model registry server address. Leave empty to skip registration.
    #   Example: "model-registry.kubeflow.svc.cluster.local"

    registry_port: int = 8080,
    # ^ Model registry server port.

    # =========================================================================
    # MODEL REGISTRY - Metadata (registry_model_*)
    # =========================================================================
    registry_model_author: str = "pipeline",
    # ^ Author/owner name for the registered model.

    registry_model_description: str = "",
    # ^ Human-readable description of the model.

    registry_model_format_name: str = "pytorch",
    # ^ Model format name (e.g., "pytorch", "onnx", "tensorflow").

    registry_model_format_version: str = "1.0",
    # ^ Model format version.

    registry_model_name: str = "fine-tuned-model",
    # ^ Name for the registered model in the registry.

    registry_model_version: str = "1.0.0",
    # ^ Version string for the model.

    registry_source_namespace: str = "",
    # ^ Namespace where the pipeline runs (e.g., "mstoklus", "pipeline-poc").
    #   Used for provenance - enables click-through link from Model Registry to pipeline run in UI.

    # =========================================================================
    # SHARED PARAMETERS
    # =========================================================================
    shared_log_file: str = "pipeline_log.txt",
    # ^ Name of the shared log file on PVC for tracking pipeline progress.
):
    """Distributed Training Pipeline with shared workspace PVC.

    A 4-stage ML pipeline for fine-tuning and evaluating language models:
    1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
    2) Training - Fine-tunes using OSFT or SFT with distributed training
    3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
    4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
        dataset_uri: Dataset URI (hf://, s3://, https://, pvc://, or absolute path)
        dataset_hf_token: HuggingFace token for gated/private datasets
        dataset_split_ratio: Train/eval split ratio (0.9 = 90% train, 10% eval)
        dataset_subset_count: Number of examples to use (0 = all)
        training_env_annotations: K8s annotations for training pods (key=val,...)
        training_env_hf_token: HuggingFace token for gated models (Llama, Mistral)
        training_env_labels: K8s labels for training pods (key=val,...)
        training_env_vars: Additional env vars (KEY=VAL,KEY=VAL)
        training_env_pull_secret: Pull secret for Red Hat Catalog, registry.redhat.io (Docker config.json content)
        training_hyper_batch_size: Effective batch size (samples per optimizer step)
        training_hyper_epochs: Number of training epochs
        training_hyper_learning_rate: Learning rate (typical: 1e-6 to 1e-4)
        training_hyper_max_seq_len: Maximum sequence length in tokens
        training_hyper_max_tokens_per_gpu: Max tokens per GPU (memory cap)
        training_hyper_seed: Random seed for reproducibility
        training_hyper_target_patterns: (OSFT only) OSFT module patterns (comma-separated)
        training_lr_scheduler: (OSFT only) LR scheduler type (cosine, linear, constant)
        training_lr_scheduler_kwargs: (OSFT only) Extra scheduler params (key=val,...)
        training_lr_warmup_steps: Warmup steps before full learning rate
        training_model_algorithm: Training algorithm - OSFT (continual learning) or SFT
        training_model_backend: Training backend - mini-trainer or instructlab-training
        training_model_base: HuggingFace model ID or path to fine-tune
        training_model_unfreeze_ratio: (OSFT only) OSFT ratio of parameters to unfreeze (0.1-0.5)
        training_opt_processed_dataset: (OSFT only) True if dataset already has input_ids (False=process raw)
        training_opt_unmask_messages: (OSFT only) Unmask messages during chat data processing
        training_opt_use_liger: (OSFT only) Enable Liger kernel optimizations
        training_res_cpu: CPU cores per training worker pod
        training_res_gpu: GPUs per training worker pod
        training_res_memory: Memory per training worker pod (e.g., 32Gi)
        training_res_num_procs: Processes per worker (usually equals GPUs)
        training_res_num_workers: Total worker pods (1=single-node, 2+=distributed)
        training_save_at_epoch: Save checkpoint at end of each epoch
        training_save_final: (OSFT only) Save final model checkpoint
        training_save_full_state: (SFT only) Save full Accelerate state for resumption
        training_save_samples: (SFT only) Number of samples to save (0 disables)
        eval_cfg_batch_size: Eval batch size (auto or integer)
        eval_cfg_limit: Max samples per task (-1 = all)
        eval_cfg_log_samples: Log individual sample predictions
        eval_cfg_verbosity: Logging level (DEBUG, INFO, WARNING, ERROR)
        eval_gen_kwargs: Generation kwargs dict (max_tokens, temperature)
        eval_model_args: Model init args dict (dtype, gpu_memory_utilization)
        eval_task_names: List of lm-eval tasks (mmlu, gsm8k, arc_easy, etc.)
        registry_address: Model registry server address (empty = skip)
        registry_port: Model registry server port
        registry_model_author: Author/owner name for registered model
        registry_model_description: Human-readable model description
        registry_model_format_name: Model format (pytorch, onnx, tensorflow)
        registry_model_format_version: Model format version
        registry_model_name: Name for the registered model
        registry_model_version: Version string for the model
        registry_source_namespace: Namespace where pipeline runs (enables UI link to pipeline run)
        shared_log_file: Shared log file on PVC for tracking progress
    """

    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=dataset_uri,
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        train_split_ratio=dataset_split_ratio,
        subset_count=dataset_subset_count,
        hf_token=dataset_hf_token,
        shared_log_file=shared_log_file,
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    # Mount S3/MinIO credentials from Kubernetes secret (OPTIONAL)
    # Only required if using s3:// URIs for dataset_uri
    # For hf://, https://, pvc://, or /path URIs - no secret needed!
    kfp.kubernetes.use_secret_as_env(
        dataset_download_task,
        secret_name='minio-secret',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
        },
        optional=True,  # Pod will start even if secret doesn't exist
    )

    # =========================================================================
    # Stage 2: Training
    # =========================================================================
    # Pass dataset artifact from download step and all training parameters
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        # Pass dataset artifact from previous step
        dataset=dataset_download_task.outputs["train_dataset"],
        # Model and algorithm (training_model_*)
        training_base_model=training_model_base,
        training_algorithm=training_model_algorithm,
        training_unfreeze_rank_ratio=training_model_unfreeze_ratio,
        training_backend=training_model_backend,
        # Hyperparameters (training_hyper_*)
        training_effective_batch_size=training_hyper_batch_size,
        training_max_tokens_per_gpu=training_hyper_max_tokens_per_gpu,
        training_max_seq_len=training_hyper_max_seq_len,
        training_learning_rate=training_hyper_learning_rate,
        training_target_patterns=training_hyper_target_patterns,
        training_seed=training_hyper_seed,
        training_num_epochs=training_hyper_epochs,
        # Optimizations (training_opt_*)
        training_use_liger=training_opt_use_liger,
        training_use_processed_dataset=training_opt_processed_dataset,
        training_unmask_messages=training_opt_unmask_messages,
        # Learning rate scheduler (training_lr_*)
        training_lr_scheduler=training_lr_scheduler,
        training_lr_warmup_steps=training_lr_warmup_steps,
        training_lr_scheduler_kwargs=training_lr_scheduler_kwargs,
        # Saving & Checkpointing (training_save_*)
        training_save_samples=training_save_samples,
        training_accelerate_full_state_at_epoch=training_save_full_state,
        training_checkpoint_at_epoch=training_save_at_epoch,
        training_save_final_checkpoint=training_save_final,
        # Environment and metadata (training_env_*)
        training_hf_token=training_env_hf_token,
        training_pull_secret=training_pull_secret,
        training_envs=training_env_vars,
        training_metadata_labels=training_env_labels,
        training_metadata_annotations=training_env_annotations,
        # Resources (training_res_*)
        training_resource_cpu_per_worker=training_res_cpu,
        training_resource_gpu_per_worker=training_res_gpu,
        training_resource_memory_per_worker=training_res_memory,
        training_resource_num_procs_per_worker=training_res_num_procs,
        training_resource_num_workers=training_res_num_workers,
    )
    training_task.set_caching_options(False)
    # Note: .after() not needed - KFP infers order from dataset artifact dependency
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

    # Note: HuggingFace token is passed via training_hf_token parameter (optional)
    # For gated models (Llama, Mistral, etc.), provide the token at runtime

    # =========================================================================
    # Stage 3: Evaluation with lm-eval
    # =========================================================================
    eval_task = universal_llm_evaluator(
        # Pass trained model from training step
        model_artifact=training_task.outputs["output_model"],
        # Pass eval dataset from dataset download (for tracking/lineage)
        eval_dataset=dataset_download_task.outputs["eval_dataset"],
        # Task configuration (eval_task_*)
        task_names=eval_task_names,
        # Configuration (eval_cfg_*)
        batch_size=eval_cfg_batch_size,
        limit=eval_cfg_limit,
        log_samples=eval_cfg_log_samples,
        verbosity=eval_cfg_verbosity,
        # Model and generation args (eval_model_*, eval_gen_*)
        model_args=eval_model_args,
        gen_kwargs=eval_gen_kwargs,
    )
    eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    # Eval requires GPU for vLLM inference
    kfp.kubernetes.add_node_selector(eval_task, "nvidia.com/gpu.present", "true")
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)

    # Inject HuggingFace token for model access during evaluation (OPTIONAL)
    # Only required for gated models (Llama, Mistral, etc.)
    kfp.kubernetes.use_secret_as_env(
        task=eval_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,  # Pod will start even if secret doesn't exist
    )

    # =========================================================================
    # Stage 4: Model Registry (waits for both training AND evaluation)
    # =========================================================================
    # Register the trained model to Kubeflow Model Registry with eval results
    model_registry_task = model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        # Pass model artifact from training
        input_model=training_task.outputs["output_model"],
        input_metrics=training_task.outputs["output_metrics"],
        # Pass evaluation results (this creates dependency on eval completing first)
        eval_metrics=eval_task.outputs["output_metrics"],
        eval_results=eval_task.outputs["output_results"],
        # Registry connection
        registry_address=registry_address,
        registry_port=registry_port,
        # Registry model metadata (registry_model_*)
        model_name=registry_model_name,
        model_version=registry_model_version,
        model_format_name=registry_model_format_name,
        model_format_version=registry_model_format_version,
        model_description=registry_model_description,
        author=registry_model_author,
        shared_log_file=shared_log_file,
        # Provenance / Lineage (auto-populated from KFP placeholders)
        source_pipeline_name=PIPELINE_NAME,  # Shared constant with @dsl.pipeline decorator
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace=registry_source_namespace,
    )
    model_registry_task.set_caching_options(False)
    # Dependency on eval_task is automatic via eval_metrics/eval_results artifacts
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
