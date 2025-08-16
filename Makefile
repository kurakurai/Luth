PYTHON_VERSION := 3.11

# Define virtualenv paths
TRAIN_VENV := .venv-train
EVAL_VENV  := .venv

# Scripts
EVAL_SCRIPT := src/eval/eval.py
SFT_SCRIPT  := src/train/sft.py

# Configs (can be overridden from CLI)
EVAL_CONFIG ?= configs/eval/eval_config.yaml
SFT_CONFIG  ?= configs/train/sft_config.yaml

.PHONY: env-train env eval sft clean

# Create and set up training environment
env-train:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up training environment..."
	@uv venv $(TRAIN_VENV) --python $(PYTHON_VERSION) --no-project
	@echo "Installing build dependencies..."
	@uv pip install setuptools wheel build --python $(TRAIN_VENV)/bin/python
	@uv pip install torch==2.6.0 --python $(TRAIN_VENV)/bin/python
	@echo "Installing base requirements..."
	@uv pip install -r requirements-train.txt --python $(TRAIN_VENV)/bin/python
	@uv pip install --no-build-isolation axolotl[deepspeed]>=0.12.0 --python $(TRAIN_VENV)/bin/python
	@echo "Training environment ready."

# Create and set up evaluation environment
env:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Setting up eval environment..."
	@uv venv $(EVAL_VENV) --python $(PYTHON_VERSION) --no-project
	@uv pip install -r requirements.txt --python $(EVAL_VENV)/bin/python
	@VLLM_USE_PRECOMPILED=1 uv pip install git+https://github.com/paulpak58/vllm.git@lfm2_v2 --python $(EVAL_VENV)/bin/python
	@uv pip install git+https://github.com/flashinfer-ai/flashinfer.git@v0.2.12 --no-deps --python $(EVAL_VENV)/bin/python
	@echo "Evaluation environment ready."
	
# Run supervised fine-tuning
sft:
	@$(TRAIN_VENV)/bin/accelerate launch $(SFT_SCRIPT) --config $(SFT_CONFIG)

# Run evaluation
eval:
	@$(EVAL_VENV)/bin/python $(EVAL_SCRIPT) --config $(EVAL_CONFIG)

# Clean virtual environments
clean:
	rm -rf $(TRAIN_VENV) $(EVAL_VENV)
