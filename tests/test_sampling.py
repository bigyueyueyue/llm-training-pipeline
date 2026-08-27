import torch

from grpo.sampling import group_indices, sum_log_probs


def test_sum_log_probs_sums_over_tokens():
    token_lp = torch.tensor([[0.1, 0.2, 0.3],
                             [-0.5, 0.0, 0.25]])
    out = sum_log_probs(token_lp)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([0.6, -0.25]))


def test_sum_not_mean():
    token_lp = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    assert torch.allclose(sum_log_probs(token_lp), torch.tensor([0.4]))


def test_group_indices_contiguous_layout():
    idx = group_indices(num_prompts=3, group_size=4)
    assert idx.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
