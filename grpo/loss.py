"""GRPO 策略梯度目标：clip 目标 + KL 惩罚（对冻结参考模型）。"""
import torch


def grpo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
              ref_log_probs: torch.Tensor, advantages: torch.Tensor, *,
              clip_epsilon: float = 0.2, beta: float = 0.01) -> torch.Tensor:
    """逐 completion 的 GRPO 目标，返回标量 loss。

    log_probs / old_log_probs / ref_log_probs: (N*G,) 当前策略 / 旧策略 / 参考模型的序列 sum log-prob。
    advantages: (N*G,) 组内优势。
    old_log_probs 与 ref_log_probs 应已 detach（由调用方保证）。
    """
    log_ratio = torch.clamp(log_probs - old_log_probs, -20.0, 20.0)  # 数值护栏：防 exp 溢出 inf
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl = log_probs - ref_log_probs  # 序列级 log-prob 差之和
    return policy_loss + beta * kl.mean()
