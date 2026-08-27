import math

import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 (batch, num_kv_heads, seq_len, head_dim) 扩到 (batch, num_kv_heads*n_rep, seq_len, head_dim)。"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def grouped_query_attention(query, key, value, num_key_value_heads,
                            attn_mask=None, is_causal=False, dropout=0.0):
    """手写分组查询注意力：Q·K^T/√d → mask → softmax → ·V。"""
    bsz, num_heads, q_len, head_dim = query.shape
    n_rep = num_heads // num_key_value_heads
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if is_causal:
        kv_len = key.shape[-2]
        causal = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

    if attn_mask is not None:
        scores = scores + attn_mask

    # 低精度输入升到 fp32 再 softmax；fp32/fp64 输入保持原精度（避免 fp64→fp32 降精度破坏 gradcheck）
    if query.dtype in (torch.float16, torch.bfloat16):
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    else:
        attn_weights = torch.softmax(scores, dim=-1)

    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    return torch.matmul(attn_weights, value)
