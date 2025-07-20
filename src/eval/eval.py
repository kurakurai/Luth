import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from patch_lighteval.patch import patch_reasoning

patch_reasoning()


from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.utils import GenerationParameters
from pathlib import Path
import argparse
import nltk
import yaml
import logging

# Set the root logger to only show INFO and above
logging.basicConfig(level=logging.INFO)
logging.getLogger("vllm").setLevel(logging.INFO)


def get_tasks(task_keys):
    """
    Get the tasks based on the provided task keys.
    """
    return ",".join(task_keys)


def display_avg_metrics(results):
    """Calculate and display average metrics across multiple runs."""
    all_keys = results["all"].keys()
    base_keys = sorted([k for k in all_keys if not k.endswith("_stderr")])

    print(f"\nAVERAGE RESULTS ACROSS {len(results)} RUNS:")
    print("-" * 40)

    for key in base_keys:
        mean = results["all"][key]
        stderr = results["all"].get(f"{key}_stderr", 0.0)
        print(f"{key}: {mean:.4f} ± {stderr:.4f} = {mean + stderr:.4f}")


def main(args):
    """
    Main function to run the evaluation pipeline with vLLM backend.
    """
    # Ensure NLTK punkt tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab")

    # Loading configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_yaml = config.get("model", {})
    model_parameters_yaml = config.get("model_parameters", {})
    extras_yaml = config.get("extras", {})
    tasks_yaml = config.get("tasks", [])
    os.environ["answer_token"] = extras_yaml.get("answer_token", "")
    
    if not extras_yaml.get("use_chat_template"):
        extras_yaml["use_chat_template"] = False
    
    else : 
        os.environ["enable_thinking"] = str(extras_yaml.get("enable_thinking"))

    if extras_yaml.get("enable_thinking") and not extras_yaml.get("answer_token"):
        raise ValueError(
            "enable_thinking is set to True, but answer_token is not provided in the config."
        )

    # Prepare model configuration
    config_kwargs = dict(model_yaml)
    config_kwargs["generation_parameters"] = GenerationParameters(
        **model_parameters_yaml
    )
    model_config = VLLMModelConfig(**config_kwargs)

    # Set up pipeline parameters
    tasks_path = Path(f"src/eval/tasks.py")
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
        use_chat_template=extras_yaml.get("use_chat_template", True),
        system_prompt=extras_yaml.get("system_prompt", ""),
    )

    # Get the tasks to evaluate
    tasks = get_tasks(tasks_yaml)

    # Initialize the evaluation tracker
    evaluation_tracker = EvaluationTracker(
        output_dir=extras_yaml.get("output_dir", "results/"),
        save_details=extras_yaml.get("save_details", True),
        push_to_hub=extras_yaml.get("push_to_hub", False),
    )

    # Create the pipeline and run the evaluation for the specified number of runss
    for _ in range(extras_yaml.get("num_runs", 1)):
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            enable_thinking=extras_yaml.get("enable_thinking", False),
        )
        pipeline.evaluate()
        if len(tasks_yaml) > 1 and extras_yaml.get("num_runs", 1) > 1:
            print(f"num_runs > 1 isn't supported for multiple tasks.")
            break

    pipeline.save_and_push_results()
    pipeline.show_results()

    # Calculate average metrics across all runs for a single task
    if len(tasks_yaml) <= 1 and extras_yaml.get("num_runs", 1) > 1:
        display_avg_metrics(
            pipeline.evaluation_tracker.metrics_logger.metric_aggregated.copy()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline with vLLM backend."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to the evaluation configuration YAML file.",
    )
    args = parser.parse_args()
    main(args)
