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

Custom French evals supported: `IFEval-fr`, `GPQA-Diamond-fr`, `BBH-fr`, `Math-HARD-fr`, `BoolQ-fr`, `MMLU-fr`, `MuSR-fr`, `Hellaswag-fr`

You can modify the evaluation configuration in the `eval_config.yaml` file. Also don't forget to set your `HF_TOKEN`.
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

### 5.1 Non-Thinking mode

| Benchmark                 | Qwen3-0.6B | Qwen2.5-0.5B-Instruct | SmolLM2-360M-Instruct |  LFM2-700M   |  LFM2-350M   |
|---------------------------|------------|-----------------------|-----------------------|--------------|--------------|
| IFEval-fr (prompt strict) |  **43.44** | 20.70                 | 20.89                 |              |              |
| GPQA-Diamond-fr           |  **29.95** | 22.84                 | 28.43                 |              |              |
| BBH-fr                    |    24.60   | 24.35                 |  **40.41**            |              |              |
| Math-HARD-fr              |  **10.86** | 3.04                  |   1.17                |              |              |
| BoolQ-fr                  |  **74.72** | 59.55                 |   26.97               |              |              |
| MMLU-fr                   |  27.16     | **35.04**             |   26.14               |              |              |
| MuSR-fr                   |  33.34     |   34.17               |   **36.41**           |              |              |
| Hellaswag-fr              |  24.90     | **51.21**             |   32.78               |              |              |

We used `temperature=0.0` and `system_prompt="Vous êtes un assistant utile."`.