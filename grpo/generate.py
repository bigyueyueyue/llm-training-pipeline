"""GRPO 采样生成：对每个 prompt 采样 G 条 completion（GPU-defer）。"""
from __future__ import annotations

import torch

from grpo.sampling import sum_log_probs


@torch.no_grad()
def sample_completions(model, tokenizer, prompts, *, group_size: int,
                       temperature: float = 1.0, top_p: float = 0.95,
                       max_new_tokens: int = 256, device: str = "cuda"):
    """对 prompts 各采样 group_size 条 completion，返回 (completions, old_log_probs)。

    completions: (len(prompts)*group_size,) 展平，prompt 0 的 G 条在前。
    old_log_probs: (len(prompts)*group_size,) 每条 completion 的序列 sum log-prob（生成时刻、已 detach）。
    """
    # generate 需要 KV cache + eval 态：训练态下 gradient checkpointing 会关 use_cache，
    # 导致每步重算全部历史 attention（O(n²)），生成极慢。这里临时开 cache、结束后恢复。
    model.eval()
    model.config.use_cache = True
    completions: list[str] = []
    all_log_probs: list[float] = []
    for prompt in prompts:
        inputs = tokenizer([prompt] * group_size, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[:, prompt_len:]
        for i in range(group_size):
            completions.append(tokenizer.decode(gen_ids[i], skip_special_tokens=True))
        all_log_probs.extend(_seq_log_probs(out, prompt_len, tokenizer.eos_token_id).detach().tolist())
    model.train()                 # 恢复训练态，供后续 compute_seq_log_probs（带梯度）
    model.config.use_cache = False
    return completions, torch.tensor(all_log_probs, device=device)


def compute_seq_log_probs(model, tokenizer, prompts, completions, device: str = "cuda"):
    """对 (prompt, completion) 拼接序列前向，返回每条 completion 部分的 sum log-prob。

    用于计算「当前策略」与「参考模型」的序列 log-prob（loss 里的 log_probs / ref_log_probs）。
    """
    import torch.nn.functional as F

    full_texts = [p + c for p, c in zip(prompts, completions)]
    enc = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    input_ids = enc["input_ids"]
    mask = enc["attention_mask"]
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1)  # (B,)

    logits = model(input_ids=input_ids, attention_mask=mask).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)

    totals = []
    for i in range(input_ids.shape[0]):
        start = int(prompt_lens[i].item()) - 1      # 第一个 completion token 的预测位置
        end = int(mask[i].sum().item()) - 1         # 最后一个有效 token 的预测位置
        if end <= start:
            totals.append(torch.zeros((), device=device))
            continue
        target = input_ids[i, start + 1:end + 1]
        lp = log_probs[i, start:end].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        totals.append(lp.sum())
    return torch.stack(totals)


def _seq_log_probs(gen_output, prompt_len: int, eos_token_id: int) -> torch.Tensor:
    """从 generate 的 scores 反算每条序列的 sum log-prob（只累加 EOS 之前的真实 token）。

    generate 返回的 sequences 在 EOS 之后用 pad_token 补齐（本项目 pad==eos），
    且 output_scores 覆盖到整批最长序列。若把这些 EOS/padding 位也算进 sum，
    会与 compute_seq_log_probs（按 decode 后文本重编码、不含 EOS）口径不一致，
    exp(log_probs - old_log_probs) 溢出为 inf，再乘以 advantage==0 的组 → NaN loss。
    """
    import torch.nn.functional as F

    scores = torch.stack(gen_output.scores, dim=1)  # (B, new_tokens, vocab)
    log_softmax = F.log_softmax(scores.float(), dim=-1)
    gen_ids = gen_output.sequences[:, prompt_len:]
    per_token = log_softmax.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    mask = (gen_ids != eos_token_id).to(per_token.dtype)
    return sum_log_probs(per_token * mask)
