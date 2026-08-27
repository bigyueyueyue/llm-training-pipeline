"""GRPO 组内相对优势归一化（免 Critic 的关键）。"""
import torch


def group_advantage(rewards: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """按每组 group_size 做 (R − mean) / std 归一化。

    rewards: (N*G,) 展平，前 group_size 个属第 0 组，依次类推。
    返回同形状的 advantage。std=0（组内奖励全同）的组退化为 0，不除零。
    """
    rewards = rewards.float()
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards 长度 {rewards.numel()} 不能被 group_size {group_size} 整除"
        )
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    adv = (grouped - mean) / safe_std
    return adv.view(-1)
