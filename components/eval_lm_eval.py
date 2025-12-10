from kfp import dsl
import kfp

@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=[
        "lm-eval[vllm]",  # The core harness with vLLM backend
        "unitxt",         # For IBM/generic dataset recipes
        "sacrebleu",      # For translation metrics
        "datasets",
        "accelerate",
        "torch",
        "transformers"
    ],
)
def universal_llm_evaluator(
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    output_samples: dsl.Output[dsl.Artifact],
    # --- Generic Inputs ---
    task_names: list,
    model_path: str = None, # Optional: Use for HF Hub models (e.g. "ibm/granite-7b")
    model_artifact: dsl.Input[dsl.Model] = None, # Optional: Use for upstream pipeline models
    model_args: dict = {},
    gen_kwargs: dict = {},
    batch_size: str = "auto",
    limit: int = -1,
    log_samples: bool = True,
    verbosity: str = "INFO",
):
    """
    A Universal LLM Evaluator component using EleutherAI's lm-evaluation-harness.

    Args:
        model_path: String path or HF ID. Used if model_artifact is None.
        model_artifact: KFP Model artifact from a previous pipeline step.
        task_names: List of task names (e.g. ["mmlu", "gsm8k"]).
        model_args: Dictionary for model initialization (e.g. {"dtype": "float16"}).
        ...
    """
    import logging
    import json
    import os
    import time
    import torch

    # Delayed imports
    from lm_eval import evaluator

    # --- 1. Setup Logging ---
    logging.basicConfig(
        level=getattr(logging, verbosity.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("UniversalEval")

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! Evaluation will be extremely slow.")

    # --- 2. Resolve Model Path ---
    # Logic: Prefer the artifact from previous step; fallback to string path.
    if model_artifact:
        logger.info(f"Using model from pipeline artifact: {model_artifact.path}")
        final_model_path = model_artifact.path
    elif model_path:
        logger.info(f"Using model from string path/ID: {model_path}")
        final_model_path = model_path
    else:
        raise ValueError("No model provided! You must pass either 'model_path' (string) or 'model_artifact' (input).")

    # --- 3. Input Sanitization ---
    def parse_input(val, default):
        if val is None: return default
        if isinstance(val, str):
            try: return json.loads(val)
            except: return val
        return val

    tasks_list = parse_input(task_names, [])
    m_args = parse_input(model_args, {})
    g_kwargs = parse_input(gen_kwargs, {})
    limit_val = None if limit == -1 else limit

    # --- 4. Construct Model Arguments ---
    final_model_args = {
        "pretrained": final_model_path,  # The resolved path is used here
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.8,
        "dtype": "auto"
    }
    final_model_args.update(m_args)

    # --- 5. Execution ---
    logger.info("Starting evaluation...")
    start_time = time.time()

    try:
        results = evaluator.simple_evaluate(
            model="vllm",
            model_args=final_model_args,
            tasks=tasks_list,
            batch_size=batch_size,
            limit=limit_val,
            log_samples=log_samples,
            gen_kwargs=g_kwargs,
            verbosity=verbosity
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise RuntimeError(f"Fatal error in evaluation: {e}")

    # --- 6. Output Processing ---
    duration = time.time() - start_time
    logger.info(f"Evaluation completed in {duration:.2f}s")

    def clean_for_json(obj):
        if isinstance(obj, (int, float, str, bool, type(None))): return obj
        elif hasattr(obj, "item"): return obj.item()
        elif isinstance(obj, dict): return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [clean_for_json(item) for item in obj]
        return str(obj)

    clean_results = clean_for_json(results)

    # Save Metrics
    if "results" in clean_results:
        for task_name, metrics in clean_results["results"].items():
            display_name = metrics.get("alias", task_name)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key != "alias":
                    safe_key = f"{display_name}_{key}".replace(" ", "_").replace("/", "_")
                    output_metrics.log_metric(safe_key, value)

    # Save Artifacts
    output_results.name = "eval_results.json"
    with open(output_results.path, "w") as f:
        json.dump(clean_results, f, indent=2)

    if log_samples and "samples" in clean_results:
        output_samples.name = "eval_samples.json"
        with open(output_samples.path, "w") as f:
            json.dump(clean_results["samples"], f, indent=2)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        universal_llm_evaluator,
        package_path="universal_llm_evaluator.yaml"
    )
