from enum import Enum

import torch
import torch.nn.functional as F

from operators.gqa import grouped_query_attention


class AttentionBackend(str, Enum):
    FLASH_ATTN_2 = "flash_attn_2"
    SDPA = "sdpa"
    MANUAL = "manual"


def flash_attn_available() -> bool:
    """flash-attn 是否可用：可 import 且导出 flash_attn_func。

    注意：真实 flash-attn 包没有 is_available()，只能用「可 import + 关键
    符号存在」来探测，否则健康安装下会抛 AttributeError（且不被 except
    ImportError 捕获），破坏 auto 路径的优雅降级。
    """
    try:
        import flash_attn
    except ImportError:
        return False
    return hasattr(flash_attn, "flash_attn_func")


def select_backend(device, dtype, head_dim, *, flash_available, backend="auto") -> AttentionBackend:
    """纯函数：给定 device/dtype/head_dim/flash 可用性，返回选中的后端。"""
    if backend != "auto":
        return AttentionBackend(backend)
    if (flash_available
            and device.type == "cuda"
            and dtype in (torch.float16, torch.bfloat16)
            and head_dim <= 256):
        return AttentionBackend.FLASH_ATTN_2
    return AttentionBackend.SDPA


def attention_forward(query, key, value, num_key_value_heads, *,
                      is_causal=True, backend="auto", softmax_scale=None) -> tuple[torch.Tensor, str]:
    """统一注意力入口，返回 (output, used_backend)。

    query: (batch, num_heads, seq_len, head_dim)
    key/value: (batch, num_kv_heads, seq_len, head_dim)
    """
    head_dim = query.shape[-1]
    num_heads = query.shape[1]
    resolved = select_backend(query.device, query.dtype, head_dim,
                              flash_available=flash_attn_available(),
                              backend=backend)

    if resolved == AttentionBackend.FLASH_ATTN_2:
        from flash_attn import flash_attn_func
        # FA2 布局 (batch, seq_len, num_heads, head_dim)，进出各转置一次
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=softmax_scale)
        return out.transpose(1, 2), resolved.value

    if resolved == AttentionBackend.SDPA:
        # torch 2.13 默认不广播 KV 头，GQA 必须显式 enable_gqa
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, scale=softmax_scale,
            enable_gqa=(num_heads != num_key_value_heads))
        return out, resolved.value

    out = grouped_query_attention(query, key, value, num_key_value_heads,
                                  is_causal=is_causal)
    return out, resolved.value
