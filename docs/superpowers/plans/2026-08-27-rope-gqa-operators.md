# 手写 RoPE / GQA 算子 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 手写并单测 RoPE / GQA 算子，在真实 Qwen2.5-7B 超参下与 HF / SDPA 对拍一致。

**Architecture:** 纯 PyTorch 手写 `operators/rope.py`（`RotaryEmbedding` + `apply_rotary_pos_emb`）与 `operators/gqa.py`（`repeat_kv` + `grouped_query_attention`）；单测先验证解析性质（旋转相对位置不变性、KV 头分组映射）与数值对齐（vs HF `Qwen2RotaryEmbedding`、vs `F.scaled_dot_product_attention`），再 `gradcheck` 验证反向可导。

**Tech Stack:** PyTorch 2.13（CPU）、transformers 5.15、pytest、`torch.autograd.gradcheck`。

## Global Constraints

- Python 3.12，`.venv` 已装 `torch==2.13.0`、`transformers==5.15.1`。
- 运行单测：`.venv/bin/python -m pytest -v`（`pyproject.toml` 已配 `testpaths=["tests"]`）。
- 全部单测 CPU 可跑，**不依赖 GPU / flash-attn**。
- 对拍容差：RoPE 用 `atol=1e-5, rtol=1e-5`；GQA 用 `rtol=1e-4, atol=1e-5`；`gradcheck` 用 fp64、`eps=1e-6, atol=1e-4`。
- `rope_theta` / `head_dim` 一律从真实 config 读，**不硬编码**（Qwen2.5-7B 的 `rope_theta=1e6`，不能假设成 1e4）。
- 函数签名与 HF 对齐：`apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)`、`repeat_kv(hidden_states, n_rep)`。
- 实现应在 feature branch 上进行（与子项目 1 一致），不直接在 main 写代码。

---

### Task 1: RoPE 算子（`operators/rope.py`）+ 单测

**Files:**
- Create: `operators/__init__.py`
- Create: `operators/rope.py`
- Test: `tests/test_rope.py`

**Interfaces:**
- Produces: `rotate_half(x) -> Tensor`；`apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1) -> (Tensor, Tensor)`；`RotaryEmbedding(head_dim, max_position_embeddings=2048, base=10000.0)`，`forward(x, position_ids) -> (cos, sin)`，cos/sin 形状 `(batch, seq_len, head_dim)`。

- [ ] **Step 1: 写失败单测**

创建 `tests/test_rope.py`：

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_rope.py -v`
Expected: FAIL，报 `ModuleNotFoundError: No module named 'operators'`（`operators/` 与 `operators/rope.py` 尚不存在）。

- [ ] **Step 3: 写最小实现**

创建 `operators/__init__.py`：

```python
"""手写注意力算子：RoPE 与 GQA。"""
```

创建 `operators/rope.py`：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_rope.py -v`
Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
git add operators/__init__.py operators/rope.py tests/test_rope.py
git commit -m "feat(operators): 手写 RoPE（RotaryEmbedding + apply_rotary_pos_emb）"
```

---

### Task 2: GQA 算子（`operators/gqa.py`）+ 单测

**Files:**
- Create: `operators/gqa.py`
- Test: `tests/test_gqa.py`

**Interfaces:**
- Consumes: 无（仅依赖 torch；`repeat_kv` 独立，`grouped_query_attention` 内部调用 `repeat_kv`）。
- Produces: `repeat_kv(hidden_states, n_rep) -> Tensor`；`grouped_query_attention(query, key, value, num_key_value_heads, attn_mask=None, is_causal=False, dropout=0.0) -> Tensor`。

- [ ] **Step 1: 写失败单测**

创建 `tests/test_gqa.py`：

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_gqa.py -v`
Expected: FAIL，报 `ModuleNotFoundError: No module named 'operators.gqa'`。

- [ ] **Step 3: 写最小实现**

创建 `operators/gqa.py`：

```python
import math

import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 (batch, num_kv_heads, seq_len, head_dim) 扩到 (batch, num_kv_heads*n_rep, seq_len, head_dim)。"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def grouped_query_attention(query, key, value, num_key_value_heads,
                            attn_mask=None, is_causal=False, dropout=0.0):
    """手写分组查询注意力：Q·K^T/√d → mask → softmax → ·V。"""
    bsz, num_heads, q_len, head_dim = query.shape
    n_rep = num_heads // num_key_value_heads
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if is_causal:
        kv_len = key.shape[-2]
        causal = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))

    if attn_mask is not None:
        scores = scores + attn_mask

    # 低精度输入升到 fp32 再 softmax；fp32/fp64 输入保持原精度（避免 fp64→fp32 降精度破坏 gradcheck）
    if query.dtype in (torch.float16, torch.bfloat16):
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    else:
        attn_weights = torch.softmax(scores, dim=-1)

    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=True)

    return torch.matmul(attn_weights, value)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_gqa.py -v`
Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
git add operators/gqa.py tests/test_gqa.py
git commit -m "feat(operators): 手写 GQA（repeat_kv + grouped_query_attention）"
```

---

### Task 3: 真实 Qwen2.5-7B config 对拍

**Files:**
- Modify: `tests/conftest.py`（新增 `qwen_config` fixture）
- Test: `tests/test_ops_against_model.py`

**Interfaces:**
- Consumes: `RotaryEmbedding`、`apply_rotary_pos_emb`（Task 1）；`grouped_query_attention`、`repeat_kv`（Task 2）。
- Produces: `_head_dim(config)`、`_rope_theta(config)` 两个私有辅助；session 级 `qwen_config` fixture。

- [ ] **Step 1: 新增 `qwen_config` fixture**

在 `tests/conftest.py` 末尾追加：

```python
@pytest.fixture(scope="session")
def qwen_config():
    from transformers import Qwen2Config

    return Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

（`Qwen2Config.from_pretrained` 只下 config.json，KB 级，会缓存；离线后走缓存。）

- [ ] **Step 2: 写失败单测**

创建 `tests/test_ops_against_model.py`：

```python
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

from operators.gqa import grouped_query_attention, repeat_kv
from operators.rope import RotaryEmbedding, apply_rotary_pos_emb


def _head_dim(config):
    return getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)


def _rope_theta(config):
    return config.rope_parameters["rope_theta"]


def test_config_geometry(qwen_config):
    assert qwen_config.num_attention_heads == 28
    assert qwen_config.num_key_value_heads == 4
    assert _head_dim(qwen_config) == 128


def test_rope_matches_hf_cos_sin(qwen_config):
    head_dim = _head_dim(qwen_config)
    mine = RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=qwen_config.max_position_embeddings,
        base=_rope_theta(qwen_config),
    )
    hf = Qwen2RotaryEmbedding(qwen_config)
    x = torch.randn(1, 8, head_dim)
    pos = torch.arange(8).unsqueeze(0)
    my_cos, my_sin = mine(x, pos)
    hf_cos, hf_sin = hf(x, pos)
    assert torch.allclose(my_cos, hf_cos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my_sin, hf_sin, atol=1e-5, rtol=1e-5)


def test_rope_apply_matches_hf(qwen_config):
    head_dim = _head_dim(qwen_config)
    hf = Qwen2RotaryEmbedding(qwen_config)
    q = torch.randn(1, 28, 8, head_dim)
    k = torch.randn(1, 28, 8, head_dim)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = hf(q, pos)
    my_q, my_k = apply_rotary_pos_emb(q, k, cos, sin)
    hf_q, hf_k = hf_apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(my_q, hf_q, atol=1e-5, rtol=1e-5)
    assert torch.allclose(my_k, hf_k, atol=1e-5, rtol=1e-5)


def test_gqa_matches_sdpa_real_geometry(qwen_config):
    num_heads = qwen_config.num_attention_heads
    num_kv_heads = qwen_config.num_key_value_heads
    head_dim = _head_dim(qwen_config)
    ratio = num_heads // num_kv_heads
    assert ratio == 7
    torch.manual_seed(0)
    q = torch.randn(2, num_heads, 32, head_dim)
    k = torch.randn(2, num_kv_heads, 32, head_dim)
    v = torch.randn(2, num_kv_heads, 32, head_dim)
    mine = grouped_query_attention(q, k, v, num_key_value_heads=num_kv_heads, is_causal=True)
    ref = F.scaled_dot_product_attention(q, repeat_kv(k, ratio), repeat_kv(v, ratio), is_causal=True)
    assert torch.allclose(mine, ref, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 3: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_ops_against_model.py -v`
Expected: 4 passed（首次运行会下载 config.json，约几秒）。

- [ ] **Step 4: 全量回归**

Run: `.venv/bin/python -m pytest -v`
Expected: 全部通过（子项目 1 的 14 个 + 本计划 14 个 = 28 个）。

- [ ] **Step 5: 提交**

```bash
git add tests/conftest.py tests/test_ops_against_model.py
git commit -m "test(operators): 真实 Qwen2.5-7B config 对拍 RoPE/GQA"
```

---

## Self-Review

- **Spec 覆盖**：§4 RoPE（Task 1 实现 + 旋转性质/HF 对拍单测）、§5 GQA（Task 2 实现 + 分组映射/SDPA 对拍/gradcheck）、§6 真实模型对拍（Task 3）、§7 测试清单（三个 test 文件逐条对应）、§8 gradcheck（Task 1/2）。全部覆盖。
- **占位符扫描**：无 TBD/TODO，所有步骤含完整代码。
- **类型一致性**：`RotaryEmbedding(head_dim, max_position_embeddings, base)`、`apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1)`、`repeat_kv(hidden_states,n_rep)`、`grouped_query_attention(query,key,value,num_key_value_heads,attn_mask=None,is_causal=False,dropout=0.0)` 三处（实现 + 两个测试文件）签名一致。
