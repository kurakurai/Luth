import os
from axolotl.utils.dict import DictDefault
from axolotl.cli.config import load_cfg
from axolotl.utils import patch_optimized_env
from axolotl.common.datasets import load_datasets
from axolotl.train import setup_model_and_tokenizer
from axolotl.utils.trainer import setup_trainer
import argparse
import yaml


def main(args):
    # Load the configuration from the specified YAML file
    with open(args.config, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Convert the YAML configuration to a dictionary with defaults
    config = DictDefault(**yaml_config)
    cfg = load_cfg(config)

    # Speedup downloads from HF 🤗 and set "PYTORCH_CUDA_ALLOC_CONF" env to save memory
    patch_optimized_env()

    # Drop long samples from the dataset that overflow the max sequence length
    dataset_meta = load_datasets(cfg=cfg)

    # cfg.max_steps = 100

    model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)
    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps
    trainer = setup_trainer(
        cfg=cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        total_num_steps=total_num_steps,
        peft_config=peft_config,
    )
    trainer.train()

    # Save the model and tokenizer after training
    base_model = yaml_config.get("base_model")
    base_model_name = base_model.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, base_model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT with Axolotl")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/sft_config.yaml",
        help="Path to the SFT configuration YAML file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/",
        help="Directory to save the model and tokenizer.",
    )
    args = parser.parse_args()
    main(args)
