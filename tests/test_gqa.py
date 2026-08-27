import torch
import torch.nn.functional as F

from operators.gqa import grouped_query_attention, repeat_kv


def test_repeat_kv_shape():
    x = torch.randn(2, 4, 8, 16)
    out = repeat_kv(x, n_rep=7)
    assert out.shape == (2, 28, 8, 16)


def test_repeat_kv_grouping():
    x = torch.randn(1, 4, 3, 8)
    out = repeat_kv(x, n_rep=7)
    for j in range(4):
        for r in range(7):
            assert torch.equal(out[:, j * 7 + r], x[:, j])


def test_gqa_output_shape():
    q = torch.randn(2, 28, 8, 128)
    k = torch.randn(2, 4, 8, 128)
    v = torch.randn(2, 4, 8, 128)
    out = grouped_query_attention(q, k, v, num_key_value_heads=4, is_causal=True)
    assert out.shape == (2, 28, 8, 128)


def test_gqa_matches_sdpa_causal():
    torch.manual_seed(0)
    q = torch.randn(1, 28, 16, 128)
    k = torch.randn(1, 4, 16, 128)
    v = torch.randn(1, 4, 16, 128)
    mine = grouped_query_attention(q, k, v, num_key_value_heads=4, is_causal=True)
    ref = F.scaled_dot_product_attention(q, repeat_kv(k, 7), repeat_kv(v, 7), is_causal=True)
    assert torch.allclose(mine, ref, rtol=1e-4, atol=1e-5)


def test_gqa_matches_sdpa_explicit_mask():
    torch.manual_seed(0)
    q = torch.randn(1, 6, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    mask = torch.zeros(8, 8)
    mask[:, 4:] = float("-inf")
    mine = grouped_query_attention(q, k, v, num_key_value_heads=2, attn_mask=mask)
    ref = F.scaled_dot_product_attention(q, repeat_kv(k, 3), repeat_kv(v, 3), attn_mask=mask)
    assert torch.allclose(mine, ref, rtol=1e-4, atol=1e-5)


def test_gqa_gradcheck():
    q = torch.randn(1, 4, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda qq, kk, vv: grouped_query_attention(qq, kk, vv, num_key_value_heads=2),
        (q, k, v), eps=1e-6, atol=1e-4,
    )
