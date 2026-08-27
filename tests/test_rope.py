import torch

from operators.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


def test_rotate_half():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.allclose(rotate_half(x), expected)


def test_apply_preserves_shape():
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    cos = torch.randn(2, 8, 32)
    sin = torch.randn(2, 8, 32)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_relative_position_invariance():
    """RoPE 核心不变量：<R_m q, R_n k> 只取决于相对位置 n-m（平移不变）。"""
    head_dim = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=64, base=10000.0)
    torch.manual_seed(0)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    def rotate_at(x, p):
        pos = torch.tensor([[p]], dtype=torch.long)
        x4 = x.view(1, 1, 1, -1)
        cos, sin = rope(x4, pos)
        xr, _ = apply_rotary_pos_emb(x4, x4, cos, sin, unsqueeze_dim=1)
        return xr.view(-1)

    def score(m, n):
        return torch.dot(rotate_at(q, m), rotate_at(k, n))

    for delta in [1, 2, 4]:
        assert torch.allclose(score(0, delta), score(3, 3 + delta), atol=1e-5)


def test_apply_rotary_gradcheck():
    q = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float64, requires_grad=True)
    cos = torch.randn(1, 4, 8, dtype=torch.float64)
    sin = torch.randn(1, 4, 8, dtype=torch.float64)

    def fn(qq, kk):
        qe, ke = apply_rotary_pos_emb(qq, kk, cos, sin, unsqueeze_dim=1)
        return torch.cat([qe, ke], dim=0)

    assert torch.autograd.gradcheck(fn, (q, k), eps=1e-6, atol=1e-4)
