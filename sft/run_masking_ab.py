import argparse
import json
import os

import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from core.metrics import steps_to_threshold
from core.tokenizer_utils import load_tokenizer
from sft.collator import ChatDataCollator
from sft.data import build_chat_sample, build_chat_sample_full, load_multi_turn_dataset
from sft.train import SFTConfig


class LossCollector(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def _build_tokenized(config, tokenizer, mode):
    dataset = load_multi_turn_dataset(config.dataset_name, field=config.field,
                                      max_samples=config.max_samples)
    fn = build_chat_sample if mode == "masked" else build_chat_sample_full
    tokenized = dataset.map(lambda ex: fn(ex["messages"], tokenizer, config.max_length),
                            remove_columns=dataset.column_names)
    return tokenized


def run_ab(config: SFTConfig, steps: int, threshold: float) -> dict:
    tokenizer = load_tokenizer(config.model_id)
    results = {}
    for mode in ("masked", "full"):
        tokenized = _build_tokenized(config, tokenizer, mode)
        args = TrainingArguments(
            output_dir=f"outputs/ab-{mode}",
            max_steps=steps,
            per_device_train_batch_size=config.per_device_batch,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.lr,
            bf16=config.bf16,
            logging_steps=1,
            report_to=[],
            remove_unused_columns=False,
            save_strategy="no",
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto")
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target)))
        collector = LossCollector()
        trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                          data_collator=ChatDataCollator(tokenizer, max_length=config.max_length),
                          callbacks=[collector])
        trainer.train()
        results[mode] = {"losses": collector.losses,
                         "steps_to_threshold": steps_to_threshold(collector.losses, threshold)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft/config/sft_lora.yaml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=1.5)
    args = p.parse_args()
    cfg = SFTConfig.from_yaml(args.config)
    results = run_ab(cfg, args.steps, args.threshold)

    os.makedirs("docs/experiments", exist_ok=True)
    with open("docs/experiments/masking_ab.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for mode, r in results.items():
        plt.plot(r["losses"], label=mode)
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("docs/experiments/masking_ab.png")
    print(json.dumps({k: v["steps_to_threshold"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
