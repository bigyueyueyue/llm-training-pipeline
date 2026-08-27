import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from operators.gqa import grouped_query_attention, repeat_kv
from operators.rope import RotaryEmbedding, apply_rotary_pos_emb


def _head_dim(config):
    return getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)


def _rope_theta(config):
    return config.rope_parameters["rope_theta"]


def test_config_geometry(qwen_config):
    assert qwen_config.num_attention_heads == 28
    assert qwen_config.num_key_value_heads == 4
    assert _head_dim(qwen_config) == 128


def test_rope_matches_hf_cos_sin(qwen_config):
    head_dim = _head_dim(qwen_config)
    mine = RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=qwen_config.max_position_embeddings,
        base=_rope_theta(qwen_config),
    )
    hf = Qwen2RotaryEmbedding(qwen_config)
    x = torch.randn(1, 8, head_dim)
    pos = torch.arange(8).unsqueeze(0)
    my_cos, my_sin = mine(x, pos)
    hf_cos, hf_sin = hf(x, pos)
    assert torch.allclose(my_cos, hf_cos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my_sin, hf_sin, atol=1e-5, rtol=1e-5)


def test_rope_apply_matches_hf(qwen_config):
    head_dim = _head_dim(qwen_config)
    hf = Qwen2RotaryEmbedding(qwen_config)
    q = torch.randn(1, 28, 8, head_dim)
    k = torch.randn(1, 28, 8, head_dim)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = hf(q, pos)
    my_q, my_k = apply_rotary_pos_emb(q, k, cos, sin)
    hf_q, hf_k = hf_apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(my_q, hf_q, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my_k, hf_k, atol=1e-5, rtol=1e-5)


def test_gqa_matches_sdpa_real_geometry(qwen_config):
    num_heads = qwen_config.num_attention_heads
    num_kv_heads = qwen_config.num_key_value_heads
    head_dim = _head_dim(qwen_config)
    ratio = num_heads // num_kv_heads
    assert ratio == 7
    torch.manual_seed(0)
    q = torch.randn(2, num_heads, 32, head_dim)
    k = torch.randn(2, num_kv_heads, 32, head_dim)
    v = torch.randn(2, num_kv_heads, 32, head_dim)
    mine = grouped_query_attention(q, k, v, num_key_value_heads=num_kv_heads, is_causal=True)
    ref = F.scaled_dot_product_attention(q, repeat_kv(k, ratio), repeat_kv(v, ratio), is_causal=True)
    assert torch.allclose(mine, ref, rtol=1e-4, atol=1e-5)
