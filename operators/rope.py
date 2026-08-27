import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后半维取负拼到前半，用于 RoPE 的成对旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """对 (batch, num_heads, seq_len, head_dim) 的 q/k 施加旋转，返回同形状。"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE 频率表：由 position_ids 生成 cos/sin，形状 (batch, seq_len, head_dim)。"""

    def __init__(self, head_dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)  # (batch, seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (batch, seq_len, head_dim)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)
