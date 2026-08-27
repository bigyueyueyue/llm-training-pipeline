"""GRPO 组采样：纯函数核心（log-prob 聚合 + 分组索引）；生成编排在 generate.py。"""
import torch


def sum_log_probs(token_log_probs: torch.Tensor) -> torch.Tensor:
    """把逐 token log-prob 求和为序列级 log-prob（求和，非平均）。

    token_log_probs: (N, seq_len)。返回 (N,)。
    """
    return token_log_probs.sum(dim=1)


def group_indices(num_prompts: int, group_size: int) -> torch.Tensor:
    """返回每个 completion 所属 prompt 组的下标。

    展平布局：prompt 0 的 G 条在前、prompt 1 的 G 条在后，依次类推。
    返回 (num_prompts*group_size,) 的组下标（0..num_prompts-1）。
    """
    return torch.arange(num_prompts).repeat_interleave(group_size)
