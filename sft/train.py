import argparse
from dataclasses import dataclass, field as dc_field

import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from core.tokenizer_utils import load_tokenizer
from sft.collator import ChatDataCollator
from sft.data import build_chat_sample, load_multi_turn_dataset


@dataclass
class SFTConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "BelleGroup/train_3.5M_CN"
    field: str = "conversations"
    max_samples: int | None = None
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: tuple = dc_field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    per_device_batch: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    output_dir: str = "outputs/sft-lora"
    bf16: bool = True
    grad_checkpoint: bool = True
    use_flash_attn: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def tokenize_fn(example, tokenizer, max_length):
    return build_chat_sample(example["messages"], tokenizer, max_length=max_length)


def main(config: SFTConfig):
    tokenizer = load_tokenizer(config.model_id)
    dataset = load_multi_turn_dataset(config.dataset_name, field=config.field,
                                      max_samples=config.max_samples)
    tokenized = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, config.max_length),
                            remove_columns=dataset.column_names)

    attn = "flash_attention_2" if config.use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto",
                                                 attn_implementation=attn)
    if config.grad_checkpoint:
        model.enable_input_require_grads()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=list(config.lora_target))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.grad_checkpoint,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      data_collator=ChatDataCollator(tokenizer, max_length=config.max_length))
    trainer.train()
    model.save_pretrained(config.output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft/config/sft_lora.yaml")
    cfg = SFTConfig.from_yaml(p.parse_args().config)
    main(cfg)
