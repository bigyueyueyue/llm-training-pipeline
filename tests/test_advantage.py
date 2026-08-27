import pytest
import torch

from grpo.advantage import group_advantage


def test_group_normalizes_to_zero_mean_unit_std():
    rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])
    adv = group_advantage(rewards, group_size=2)
    assert adv.shape == rewards.shape
    for g in range(2):
        group = adv[g * 2:(g + 1) * 2]
        assert torch.allclose(group.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(group.std(), torch.tensor(1.0), atol=1e-6)


def test_groups_are_isolated():
    rewards = torch.tensor([0.0, 1.0, 100.0, 101.0])
    adv = group_advantage(rewards, group_size=2)
    # 两组内部归一化互不影响：归一化结果相同
    assert torch.allclose(adv[:2], adv[2:])


def test_degenerate_zero_std():
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = group_advantage(rewards, group_size=2)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_raises_on_non_divisible():
    with pytest.raises(ValueError):
        group_advantage(torch.tensor([1.0, 2.0, 3.0]), group_size=2)
