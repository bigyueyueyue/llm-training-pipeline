import torch

from grpo.loss import grpo_loss


def test_loss_reduces_to_negative_advantage_when_ratio_one():
    adv = torch.tensor([1.0, 2.0])
    lp = torch.tensor([-0.5, 0.3])
    loss = grpo_loss(lp, lp, lp, adv, beta=0.0)
    assert loss.ndim == 0
    assert torch.allclose(loss, torch.tensor(-1.5))


def test_loss_clips_ratio_at_upper_bound():
    adv = torch.tensor([1.0, 1.0])
    old_lp = torch.tensor([0.0, 0.0])
    log_probs = torch.tensor([0.5, 0.5])  # ratio = exp(0.5) ≈ 1.648 > 1.2
    loss = grpo_loss(log_probs, old_lp, torch.tensor([0.0, 0.0]), adv, beta=0.0)
    assert torch.allclose(loss, torch.tensor(-1.2))


def test_loss_adds_kl_penalty():
    adv = torch.tensor([0.0, 0.0])
    log_probs = torch.tensor([0.1, 0.2])
    ref_lp = torch.tensor([0.0, 0.0])
    loss = grpo_loss(log_probs, torch.tensor([0.0, 0.0]), ref_lp, adv, beta=0.5)
    assert torch.allclose(loss, torch.tensor(0.075))


def test_loss_backprop_through_log_probs():
    log_probs = torch.tensor([0.1, 0.2], requires_grad=True)
    loss = grpo_loss(log_probs, torch.tensor([0.0, 0.0]),
                     torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0]))
    loss.backward()
    assert log_probs.grad is not None
