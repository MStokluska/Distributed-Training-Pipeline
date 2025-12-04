# Distributed Training Pipeline

A skeleton Kubeflow Pipeline for distributed training with 4 stages sharing a common PVC.

## Pipeline Stages

1. **Dataset Download** - Prepares/downloads the training dataset
2. **Training** - Performs model training
3. **Eval with lm-eval** - Evaluates the trained model using lm-eval harness
4. **Model Registry** - Registers the model and verifies all stages completed

## Key Features

- **Shared Workspace PVC**: All components share a dynamically created PVC via KFP's workspace feature
- **Configurable Storage**: PVC size and storage class can be configured before compilation
- **Automatic Cleanup**: Workspace PVC is automatically managed by KFP
- **Execution Verification**: Final stage prints a log of all completed stages

## Project Structure

```
reusable-component-sample/
├── components/
│   ├── __init__.py
│   ├── dataset_download.py
│   ├── training.py
│   ├── eval_lm_eval.py
│   └── model_registry.py
├── distributed_training_pipeline.py
├── distributed_training_pipeline.yaml  # Generated
└── README.md
```

## Configuration

### PVC Settings (Compile-Time)

These are **compile-time settings** - they get baked into the pipeline YAML and cannot be changed at runtime. Modify these in `distributed_training_pipeline.py` and recompile:

```python
PVC_SIZE = "10Gi"
PVC_STORAGE_CLASS = "nfs-csi"
PVC_ACCESS_MODES = ["ReadWriteMany"]
```

> **Why compile-time?** KFP's workspace feature configures the PVC as part of the pipeline specification during compilation. This is different from pipeline parameters which are resolved at runtime.

### Runtime Parameters

These parameters can be configured when creating a pipeline run:

#### Dataset Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_hf_token` | str | `""` | HuggingFace token for private datasets |
| `dataset_repo_id` | str | `"dataset/example"` | HuggingFace dataset repo ID |
| `dataset_subset` | str | `"train"` | Dataset subset to download |

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_epochs` | int | `3` | Number of training epochs |
| `training_batch_size` | int | `32` | Training batch size |
| `training_learning_rate` | float | `5e-5` | Learning rate |
| `training_use_liger` | bool | `False` | Enable Liger kernel optimization |
| `training_cpu_request` | str | `"4"` | CPU cores to request |
| `training_memory_request` | str | `"16Gi"` | Memory to request |

#### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_tasks` | str | `"hellaswag,arc_easy"` | Comma-separated lm-eval tasks |
| `eval_batch_size` | int | `8` | Evaluation batch size |
| `eval_limit` | int | `100` | Limit samples per task |

#### Model Registry Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry_model_name` | str | `"my-model"` | Name for the registered model |
| `registry_model_version` | str | `"1.0.0"` | Model version |
| `registry_endpoint` | str | `""` | Model registry endpoint URL |

#### Shared Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shared_log_file` | str | `"pipeline_log.txt"` | Shared log file name |

## Prerequisites

- Python 3.9+
- KFP SDK 2.15+ (`pip install kfp kfp-kubernetes`)
- Access to a Kubernetes cluster with KFP 2.15+ installed
- A StorageClass that supports ReadWriteMany access mode (e.g., `nfs-csi`)

## Generate Pipeline YAML

```bash
# Modify PVC settings in distributed_training_pipeline.py if needed, then:
python distributed_training_pipeline.py
```

This generates `distributed_training_pipeline.yaml` which can be uploaded to Kubeflow Pipelines.

## Usage

### Option 1: Upload via UI

1. Modify PVC settings in `distributed_training_pipeline.py` if needed
2. Generate the YAML: `python distributed_training_pipeline.py`
3. Open Kubeflow Pipelines UI
4. Upload `distributed_training_pipeline.yaml`
5. Create a run and configure runtime parameters

### Option 2: Submit via SDK

```python
from kfp import Client

client = Client(host="<your-kfp-endpoint>")
client.create_run_from_pipeline_package(
    "distributed_training_pipeline.yaml",
    arguments={
        "dataset_hf_token": "hf_xxx",
        "training_epochs": 5,
        "training_use_liger": True,
        "eval_limit": 200,
    },
)
```

## Customization

This is a skeleton pipeline. To implement actual functionality:

1. **Dataset Download**: Replace with actual dataset download logic (S3, HuggingFace, etc.)
2. **Training**: Add your training code (distributed training with PyTorch, etc.)
3. **Eval**: Integrate lm-eval harness for model evaluation
4. **Model Registry**: Add logic to push model to your registry (MLflow, Model Registry, etc.)

## Expected Output

When the pipeline runs successfully, the final stage will print:

```
==================================================
PIPELINE EXECUTION LOG:
==================================================
Hello world from dataset download
Hello world from training
Hello world from eval with lm-eval
Hello world from model registry
==================================================

Total hello world messages: 4
[OK] All 4 pipeline stages completed successfully!
```
