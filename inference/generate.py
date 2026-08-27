"""贪心 decode 循环（GPU-deferred 参考实现）。

上卡后：把真实 Qwen2.5-7B 每层注意力接入 attention_forward，K/V 由 KVCache
承接。本脚本演示 prefill / decode 两阶段与 KV cache 交互，依赖真实模型权重
与 GPU，CPU 不跑。
"""
import torch


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, kv_cache, *,
             temperature=1.0, top_k=None, device="cuda"):
    """贪心/采样解码。

    model 需提供增量接口 `model(input_ids, positions, cache) -> logits`：
    对给定 token 与 KV cache 位置做一次前向，返回 (batch, seq, vocab) logits。
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = []

    # 1) prefill：整个 prompt 一次 forward，KV 全量写入 cache
    positions = torch.arange(input_ids.shape[1], device=device)
    logits = model(input_ids, positions, cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])

    # 2) decode：逐 token，每步只算最后 1 个 token，KV append 到 cache 尾
    next_id = _sample(logits[0, -1], temperature, top_k)
    for _ in range(max_new_tokens):
        generated.append(int(next_id))
        if next_id == tokenizer.eos_token_id:
            break
        positions = torch.tensor([kv_cache.seq_len], device=device)
        logits = model(next_id.reshape(1, 1), positions, cache=kv_cache)
        kv_cache.advance(1)
        next_id = _sample(logits[0, -1], temperature, top_k)

    return generated


def _sample(logits, temperature=1.0, top_k=None):
    """贪心（top_k=None）或 top-k 采样，返回标量 LongTensor。"""
    if top_k is None:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, top_k)
    probs = torch.softmax(topk_vals, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1)
    return topk_idx[chosen].reshape(1)
