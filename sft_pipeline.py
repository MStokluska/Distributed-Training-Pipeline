"""SFT (Supervised Fine-Tuning) Training Pipeline.

A 4-stage pipeline for standard supervised fine-tuning:
1. Dataset Download
2. SFT Training (instructlab-training backend)
3. Evaluation with lm-eval
4. Model Registry

SFT is the standard approach for adapting pre-trained language models
to new tasks or domains using labeled training data.
"""

import kfp
from kfp import dsl
import kfp.kubernetes

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from components.dataset_download import dataset_download
from components.training import train_model
from components.eval_lm_eval import universal_llm_evaluator
from components.model_registry import model_registry

# =============================================================================
# PVC Configuration (COMPILE-TIME settings)
# =============================================================================
PVC_SIZE = "10Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
PIPELINE_NAME = "sft-pipeline"
# =============================================================================


@dsl.pipeline(
    name=PIPELINE_NAME,
    description="SFT pipeline: standard supervised fine-tuning using instructlab-training",
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
def sft_pipeline(
    # =========================================================================
    # KEY PARAMETERS (Required/Important) - Sorted by step
    # =========================================================================
    key1_data_uri: str,
    key1_data_split: float = 0.9,
    key2_train_batch: int = 128,
    key2_train_epochs: int = 1,
    key2_train_gpu: int = 1,
    key2_train_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    key2_train_tokens: int = 10000,
    key2_train_workers: int = 4,
    key3_eval_tasks: list = ["arc_easy"],
    key4_reg_address: str = "",
    key4_reg_author: str = "pipeline",
    key4_reg_name: str = "sft-model",
    key4_reg_version: str = "1.0.0",
    # =========================================================================
    # OPTIONAL PARAMETERS - Sorted by step
    # =========================================================================
    opt1_data_hf_token: str = "",
    opt1_data_subset: int = 0,
    opt2_train_annotations: str = "",
    opt2_train_cpu: str = "4",
    opt2_train_env_vars: str = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,NCCL_DEBUG=INFO,INSTRUCTLAB_NCCL_TIMEOUT_MS=600000",
    opt2_train_hf_token: str = "",
    opt2_train_labels: str = "",
    opt2_train_learning_rate: float = 5e-6,
    opt2_train_lr_warmup: int = 0,
    opt2_train_lr_scheduler: str = "cosine",
    opt2_train_max_seq_len: int = 8192,
    opt2_train_memory: str = "64Gi",
    opt2_train_num_procs: str = "auto",
    opt2_train_pull_secret: str = "",
    opt2_train_save_epoch: bool = True,
    opt2_train_save_full_state: bool = False,
    opt2_train_fsdp_sharding: str = "FULL_SHARD",
    opt2_train_save_samples: int = 0,
    opt2_train_seed: int = 42,
    opt2_train_use_liger: bool = False,
    opt3_eval_batch: str = "auto",
    opt3_eval_gen_kwargs: dict = {},
    opt3_eval_limit: int = -1,
    opt3_eval_log_samples: bool = True,
    opt3_eval_model_args: dict = {},
    opt3_eval_verbosity: str = "INFO",
    opt4_reg_description: str = "",
    opt4_reg_format_name: str = "pytorch",
    opt4_reg_format_version: str = "1.0",
    opt4_reg_port: int = 8080,
):
    """SFT Training Pipeline - Standard supervised fine-tuning with instructlab-training.

    A 4-stage ML pipeline for fine-tuning language models:
    1) Dataset Download - Prepares training data from HuggingFace, S3, HTTP, or PVC
    2) SFT Training - Fine-tunes using instructlab-training backend
    3) Evaluation - Evaluates with lm-eval harness (MMLU, GSM8K, etc.)
    4) Model Registry - Registers trained model to Kubeflow Model Registry

    Args:
        key1_data_uri: [REQUIRED] Dataset location (hf://dataset, s3://bucket/path, https://url, pvc://path)
        key1_data_split: Train/eval split ratio (0.9 = 90% train, 10% eval)
        key2_train_batch: Effective batch size (samples per optimizer step). Start with 128
        key2_train_epochs: Number of training epochs. 1 is often sufficient
        key2_train_gpu: GPUs per worker. KEEP AT 1 to avoid /dev/shm issues
        key2_train_model: Base model (HuggingFace ID or path)
        key2_train_tokens: Max tokens per GPU (memory cap). 10000 for SFT
        key2_train_workers: Number of training pods. 4 pods × 1 GPU = 4 total GPUs
        key3_eval_tasks: lm-eval tasks (arc_easy, mmlu, gsm8k, hellaswag, etc.)
        key4_reg_address: Model Registry address (empty = skip registration)
        key4_reg_author: Author name for the registered model
        key4_reg_name: Model name in registry
        key4_reg_version: Semantic version (major.minor.patch)
        opt1_data_hf_token: HuggingFace token for gated/private datasets
        opt1_data_subset: Limit to first N examples (0 = all)
        opt2_train_annotations: K8s annotations (key=val,...)
        opt2_train_cpu: CPU cores per worker
        opt2_train_env_vars: Env vars (KEY=VAL,...) with NCCL timeout, memory optimization
        opt2_train_hf_token: HuggingFace token for gated models (Llama, Mistral)
        opt2_train_labels: K8s labels (key=val,...)
        opt2_train_learning_rate: Learning rate (1e-6 to 1e-4). 5e-6 recommended
        opt2_train_lr_warmup: Warmup steps before full LR
        opt2_train_lr_scheduler: LR schedule (cosine, linear, constant)
        opt2_train_max_seq_len: Max sequence length in tokens
        opt2_train_memory: RAM per worker. 64Gi matches tested config
        opt2_train_num_procs: Processes per worker ('auto' = one per GPU)
        opt2_train_pull_secret: K8s pull secret for private registries
        opt2_train_save_epoch: Save checkpoint at each epoch end
        opt2_train_save_full_state: Save full optimizer state for resumption
        opt2_train_fsdp_sharding: FSDP strategy (FULL_SHARD, HYBRID_SHARD, NO_SHARD)
        opt2_train_save_samples: Save every N samples (0 = disabled)
        opt2_train_seed: Random seed for reproducibility
        opt2_train_use_liger: Enable Liger kernel optimizations (experimental)
        opt3_eval_batch: Eval batch size ('auto' or integer)
        opt3_eval_gen_kwargs: Generation params dict (max_tokens, temperature)
        opt3_eval_limit: Max samples per task (-1 = all)
        opt3_eval_log_samples: Log individual predictions
        opt3_eval_model_args: Model init args dict (dtype, gpu_memory_utilization)
        opt3_eval_verbosity: Logging level (DEBUG, INFO, WARNING, ERROR)
        opt4_reg_description: Model description
        opt4_reg_format_name: Model format (pytorch, onnx, tensorflow)
        opt4_reg_format_version: Model format version
        opt4_reg_port: Model registry server port
    """

    # =========================================================================
    # Stage 1: Dataset Download
    # =========================================================================
    dataset_download_task = dataset_download(
        dataset_uri=key1_data_uri,
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        train_split_ratio=key1_data_split,
        subset_count=opt1_data_subset,
        hf_token=opt1_data_hf_token,
        shared_log_file="pipeline_log.txt",
    )
    dataset_download_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(dataset_download_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        dataset_download_task,
        secret_name='minio-secret',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
        },
        optional=True,
    )

    # =========================================================================
    # Stage 2: SFT Training
    # =========================================================================
    training_task = train_model(
        pvc_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        dataset=dataset_download_task.outputs["train_dataset"],
        # Model - SFT specific
        training_base_model=key2_train_model,
        training_algorithm="SFT",  # Hardcoded for SFT pipeline
        training_backend="instructlab-training",  # Hardcoded for SFT
        training_unfreeze_rank_ratio=0.0,  # Not used by SFT
        # Hyperparameters
        training_effective_batch_size=key2_train_batch,
        training_max_tokens_per_gpu=key2_train_tokens,
        training_max_seq_len=opt2_train_max_seq_len,
        training_learning_rate=opt2_train_learning_rate,
        training_target_patterns="",  # Not used by SFT
        training_seed=opt2_train_seed,
        training_num_epochs=key2_train_epochs,
        # SFT optimizations (use_liger supported, others not)
        training_use_liger=opt2_train_use_liger,
        training_use_processed_dataset=False,
        training_unmask_messages=False,
        # LR scheduler
        training_lr_scheduler=opt2_train_lr_scheduler,
        training_lr_warmup_steps=opt2_train_lr_warmup,
        training_lr_scheduler_kwargs="",
        # Saving (SFT-specific)
        training_checkpoint_at_epoch=opt2_train_save_epoch,
        training_save_final_checkpoint=False,  # Not used by SFT
        training_save_samples=opt2_train_save_samples,
        training_accelerate_full_state_at_epoch=opt2_train_save_full_state,
        training_fsdp_sharding_strategy=opt2_train_fsdp_sharding,
        # Environment
        training_hf_token=opt2_train_hf_token,
        training_pull_secret=opt2_train_pull_secret,
        training_envs=opt2_train_env_vars,
        training_metadata_labels=opt2_train_labels,
        training_metadata_annotations=opt2_train_annotations,
        # Resources
        training_resource_cpu_per_worker=opt2_train_cpu,
        training_resource_gpu_per_worker=key2_train_gpu,
        training_resource_memory_per_worker=opt2_train_memory,
        training_resource_num_procs_per_worker=opt2_train_num_procs,
        training_resource_num_workers=key2_train_workers,
    )
    training_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(training_task, "IfNotPresent")

    kfp.kubernetes.use_secret_as_env(
        task=training_task,
        secret_name="kubernetes-credentials",
        secret_key_to_env={
            "server_url": "KUBERNETES_SERVER_URL",
            "auth_token": "KUBERNETES_AUTH_TOKEN",
        },
    )

    # =========================================================================
    # Stage 3: Evaluation
    # =========================================================================
    eval_task = universal_llm_evaluator(
        model_artifact=training_task.outputs["output_model"],
        eval_dataset=dataset_download_task.outputs["eval_dataset"],
        task_names=key3_eval_tasks,
        batch_size=opt3_eval_batch,
        limit=opt3_eval_limit,
        log_samples=opt3_eval_log_samples,
        verbosity=opt3_eval_verbosity,
        model_args=opt3_eval_model_args,
        gen_kwargs=opt3_eval_gen_kwargs,
    )
    eval_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(eval_task, "IfNotPresent")

    kfp.kubernetes.add_node_selector(eval_task, "nvidia.com/gpu.present", "true")
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)

    kfp.kubernetes.use_secret_as_env(
        task=eval_task,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
        optional=True,
    )

    # =========================================================================
    # Stage 4: Model Registry
    # =========================================================================
    model_registry_task = model_registry(
        pvc_mount_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        input_model=training_task.outputs["output_model"],
        input_metrics=training_task.outputs["output_metrics"],
        eval_metrics=eval_task.outputs["output_metrics"],
        eval_results=eval_task.outputs["output_results"],
        registry_address=key4_reg_address,
        registry_port=opt4_reg_port,
        model_name=key4_reg_name,
        model_version=key4_reg_version,
        model_format_name=opt4_reg_format_name,
        model_format_version=opt4_reg_format_version,
        model_description=opt4_reg_description,
        author=key4_reg_author,
        shared_log_file="pipeline_log.txt",
        source_pipeline_name=PIPELINE_NAME,
        source_pipeline_run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        source_pipeline_run_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        source_namespace="",
    )
    model_registry_task.set_caching_options(False)
    kfp.kubernetes.set_image_pull_policy(model_registry_task, "IfNotPresent")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=sft_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
    print(f"SFT Pipeline compiled successfully!")
    print(f"  PVC Size: {PVC_SIZE}")
    print(f"  Storage Class: {PVC_STORAGE_CLASS}")
