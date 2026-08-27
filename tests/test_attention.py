import torch
import torch.nn.functional as F

from inference.attention import (AttentionBackend, attention_forward,
                                 flash_attn_available, select_backend)
from operators.gqa import repeat_kv


def test_select_backend_auto_cpu():
    backend = select_backend(torch.device("cpu"), torch.float32, 128,
                             flash_available=False)
    assert backend == AttentionBackend.SDPA


def test_select_backend_auto_flash():
    backend = select_backend(torch.device("cuda"), torch.float16, 128,
                             flash_available=True)
    assert backend == AttentionBackend.FLASH_ATTN_2


def test_select_backend_flash_requires_fp16_bf16():
    backend = select_backend(torch.device("cuda"), torch.float32, 128,
                             flash_available=True)
    assert backend == AttentionBackend.SDPA


def test_select_backend_flash_head_dim_limit():
    backend = select_backend(torch.device("cuda"), torch.float16, 512,
                             flash_available=True)
    assert backend == AttentionBackend.SDPA


def test_select_backend_explicit_manual():
    backend = select_backend(torch.device("cpu"), torch.float32, 128,
                             flash_available=False, backend="manual")
    assert backend == AttentionBackend.MANUAL


def test_flash_attn_available_returns_bool():
    assert isinstance(flash_attn_available(), bool)


def _gqa_inputs(seed=0, b=2, nh=28, nkv=4, s=16, d=128):
    torch.manual_seed(seed)
    q = torch.randn(b, nh, s, d)
    k = torch.randn(b, nkv, s, d)
    v = torch.randn(b, nkv, s, d)
    return q, k, v, nkv


def test_attention_forward_sdpa_matches_manual():
    q, k, v, nkv = _gqa_inputs()
    out_sdpa, backend_sdpa = attention_forward(q, k, v, nkv, backend="sdpa")
    out_manual, backend_manual = attention_forward(q, k, v, nkv, backend="manual")
    assert backend_sdpa == "sdpa"
    assert backend_manual == "manual"
    assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4)


def test_attention_forward_auto_matches_sdpa_reference():
    q, k, v, nkv = _gqa_inputs()
    out, backend = attention_forward(q, k, v, nkv)  # auto -> CPU 落 SDPA
    kr = repeat_kv(k, q.shape[1] // nkv)
    vr = repeat_kv(v, q.shape[1] // nkv)
    ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    assert backend == "sdpa"
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_forward_causal():
    q, k, v, nkv = _gqa_inputs(s=8)
    out, _ = attention_forward(q, k, v, nkv, backend="manual", is_causal=True)
    out_nocausal, _ = attention_forward(q, k, v, nkv, backend="manual", is_causal=False)
    assert not torch.allclose(out, out_nocausal)
    kr = repeat_kv(k, q.shape[1] // nkv)
    vr = repeat_kv(v, q.shape[1] // nkv)
    ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_forward_output_shape():
    q, k, v, nkv = _gqa_inputs()
    out, _ = attention_forward(q, k, v, nkv)
    assert out.shape == q.shape


def test_attention_forward_mha_no_gqa():
    b, s, d = 2, 8, 32
    q = torch.randn(b, 4, s, d)
    k = torch.randn(b, 4, s, d)
    v = torch.randn(b, 4, s, d)
    out, backend = attention_forward(q, k, v, 4, backend="sdpa")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert backend == "sdpa"
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
