import torch

from grpo.generate import compute_seq_log_probs


class _StubLogits:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    def __call__(self, input_ids=None, attention_mask=None):
        b, t = input_ids.shape
        return _StubLogits(torch.randn(b, t, 8, requires_grad=True))


class _Batch:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, key):
        return self._d[key]

    def to(self, device):
        return self


class _StubTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = torch.tensor([[i + 1] * len(t) for i, t in enumerate(texts)])
        mask = torch.ones_like(ids)
        return _Batch({"input_ids": ids, "attention_mask": mask})


def test_compute_seq_log_probs_carries_grad():
    model = _StubModel()
    tok = _StubTokenizer()
    lp = compute_seq_log_probs(model, tok, ["p"], ["c"], device="cpu")
    assert lp.requires_grad
    lp.sum().backward()  # 不应抛错：梯度能反传（曾因 in-place fill 切断 autograd）
