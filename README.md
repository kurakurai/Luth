# Luth: Small French Language Model

Building a Small French Language Model.

## 1. Quick Setup

_Using [`uv`](https://github.com/astral-sh/uv) for fast and reliable dependency management._

```bash
# Basic environment setup
make env        # Set up environment for evaluation, mergekit, etc. (excluding training)
make env-train  # Set up environment with training dependencies
make clean      # Delete all .venv
```
That's it, you can now run any command you want!

## 2. Evaluation

Many custom French evals are supported, check `configs/eval/all_fr_tasks.txt`.

You can modify the evaluation configuration in the `eval_config.yaml` file. Also **don't forget** to set your `HF_TOKEN`.
```bash
# To run the CLI commands
make env
source .venv/bin/activate
python src/eval/eval.py --config 'configs/eval/eval_config.yaml'
```

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Evaluation French Benchmarks   | `make eval`       | `python src/eval/eval.py --config EVAL_CONFIG`                                                                                 | `EVAL_CONFIG=configs/eval/eval_config.yaml`                     |

⚠️ We use [LightEval](https://github.com/huggingface/lighteval) and [vLLM](https://github.com/vllm-project/vllm) for evaluation.

## 3. Training

You can modify the training configuration in the `sft_config.yaml` file. To enable multi-GPU training, prefix the `make` command with `CUDA_VISIBLE_DEVICES=0,1`.

```bash
# To run the CLI commands
make env-train
source .venv-train/bin/activate
accelerate launch src/train/sft.py --config 'configs/train/sft_config.yaml'
```

| Task        | Make Command       | Equivalent CLI Command                                                                                                                                               | Default Values                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Run SFT   | `make sft`       | `accelerate launch src/train/sft.py --config SFT_CONFIG`                                                                                 | `SFT_CONFIG=configs/train/sft_config.yaml`                     |

⚠️ We use [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for training. Includes support for Flash Attention.

## 4. Model Merging

We use [Mergekit](https://github.com/arcee-ai/mergekit/) for model merging.

```bash
# To run model merging
make env
source .venv/bin/activate
mergekit-yaml configs/mergekit/merge_linear.yaml ./merged-output --cuda
```

## 5. Results

### 5.1 Instruct mode

| Benchmarks                | Qwen3-0.6B | Qwen2.5-0.5B-Instruct |  LFM2-700M   |  LFM2-350M   |
|---------------------------|------------|-----------------------|--------------|--------------|
| IFEval_fr (prompt strict) |  43.62     |                       |              |              |
| GPQA_Diamond_fr           |  28.93     |                       |              |              |
| Math_500_fr               |  29.40     |                       |              |              |
| MMLU_fr                   |  27.16     |                       |              |              |
| AIME_2024_fr              |  -         |                       |              |              |
| Hellaswag_fr              |  25.11     |                       |              |              |
| ARC_Challenge_fr          |  31.31     |                       |              |              |
| IFEval_en (prompt strict) |  57.49     |                       |              |              |
| GPQA_Diamond_en           |  29.80     |                       |              |              |
| Math_500_en               |  45.00     |                       |              |              |
| MMLU_en                   |  36.85     |                       |              |              |
| AIME_2024_en              |  3.33      |                       |              |              |
| Hellaswag_en              |  42.91     |                       |              |              |
| ARC_Challenge_en          |  33.62     |                       |              |              |


We used:
- French bench: `temperature=0.0` and `system_prompt="Vous êtes un assistant utile."`.
- English bench: `temperature=0.0` and `system_prompt="You are a helpful assistant."`.