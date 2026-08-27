"""GRPO 训练循环：生成→奖励→优势→loss→更新（GPU-defer）。"""
import argparse
from dataclasses import dataclass, field as dc_field

import yaml


@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "openai/gsm8k"
    max_samples: int | None = None
    output_dir: str = "outputs/grpo-lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: tuple = dc_field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    group_size: int = 4
    prompts_per_step: int = 4
    grad_accum: int = 2
    clip_epsilon: float = 0.2
    beta: float = 0.01
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    lr: float = 1e-5
    num_epochs: int = 1
    bf16: bool = True
    grad_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_prompt(question: str) -> str:
    """把 GSM8K 题目包成固定指令模板（格式奖励的依据）。"""
    return (f"Solve the following math problem step by step, then give the final "
            f'answer after "####".\n\nQuestion: {question}\n')


def main(config: GRPOConfig):
    """GRPO 训练循环。上卡后运行；本函数不在 CPU 上执行。"""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model

    from core.tokenizer_utils import load_tokenizer
    from grpo.reward import combined_reward, extract_answer
    from grpo.advantage import group_advantage
    from grpo.loss import grpo_loss
    from grpo.cache import CompletionCache
    from grpo.generate import compute_seq_log_probs, sample_completions

    tokenizer = load_tokenizer(config.model_id)
    device = "cuda"
    ds = load_dataset(config.dataset_name, "main", split="train")
    if config.max_samples is not None:
        ds = ds.select(range(config.max_samples))

    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
    if config.grad_checkpoint:
        model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=list(config.lora_target))
    model = get_peft_model(model, lora).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16).eval().to(device)

    cache = CompletionCache(f"{config.output_dir}/completion_cache.jsonl")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    prompts = [build_prompt(ex["question"]) for ex in ds]
    golds = [extract_answer(ex["answer"]) for ex in ds]

    model.train()
    for epoch in range(config.num_epochs):
        for start in range(0, len(prompts), config.prompts_per_step):
            batch_prompts = prompts[start:start + config.prompts_per_step]
            batch_golds = golds[start:start + config.prompts_per_step]

            # 1) 组采样（G 条 / prompt）
            completions, old_log_probs = sample_completions(
                model, tokenizer, batch_prompts, group_size=config.group_size,
                temperature=config.temperature, top_p=config.top_p,
                max_new_tokens=config.max_new_tokens, device=device)

            # 2) 奖励（带缓存）：completion 展平布局 = prompt0 的 G 条在前
            gold_rep = [g for g in batch_golds for _ in range(config.group_size)]
            prompt_rep = [p for p in batch_prompts for _ in range(config.group_size)]
            rewards = []
            for i, (prompt, completion, gold) in enumerate(zip(prompt_rep, completions, gold_rep)):
                hit = cache.get(prompt, completion)
                if hit is not None:
                    rewards.append(hit["reward"])
                else:
                    r = combined_reward(completion, gold)
                    cache.put(prompt, completion, r, old_log_probs[i].item())
                    rewards.append(r)
            rewards = torch.tensor(rewards, device=device)

            # 3) 优势 + 三路 log-prob
            advantages = group_advantage(rewards, group_size=config.group_size)
            log_probs = compute_seq_log_probs(model, tokenizer, prompt_rep, completions, device=device)
            with torch.no_grad():
                ref_log_probs = compute_seq_log_probs(ref_model, tokenizer, prompt_rep, completions, device=device)

            # 4) loss（grad_accum）
            loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages,
                             clip_epsilon=config.clip_epsilon, beta=config.beta)
            (loss / config.grad_accum).backward()

            if (start // config.prompts_per_step + 1) % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(f"epoch={epoch} step={start // config.prompts_per_step} "
                  f"loss={loss.item():.4f} mean_reward={rewards.mean().item():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="grpo/config/grpo_lora.yaml")
    main(GRPOConfig.from_yaml(p.parse_args().config))
