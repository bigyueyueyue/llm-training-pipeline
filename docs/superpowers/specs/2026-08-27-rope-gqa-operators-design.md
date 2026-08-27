# 子项目 2（第一部分）：手写 RoPE / GQA 算子 — 设计规格

- 日期：2026-08-27
- 状态：待用户审阅
- 范围：本规格只覆盖「子项目 2 的手写 RoPE/GQA + 单测」部分。FlashAttention-2 集成、KV Cache 管理、推理吞吐压测属子项目 2 后续 GPU 部分，各自独立成规格。

---

## 1. 背景与目标

在子项目 1（SFT）之后，转向**推理侧的核心算子**。目标是把 LLM 注意力里最关键的三个「不变量」从「调库」升级为「手写 + 对拍」：

1. **RoPE（旋转位置编码）**：手写频率表构造与旋转变换，用旋转性质 + HF 实现双重对拍。
2. **GQA（分组查询注意力）**：手写 `repeat_kv` 的 KV 头共享 + 显式 `Q·K^T/√d → mask → softmax → ·V` 的注意力计算，用 SDPA 对拍。
3. **真实模型对拍**：用 `Qwen/Qwen2.5-7B-Instruct` 的真实超参（28 头 / 4 KV 头 / head_dim 128，GQA ratio 7）验证手写算子在真实尺度下正确。

**性质**：求职作品集 / 可复现实验。所有算子纯 PyTorch 实现、全部 CPU 可单测；不依赖 GPU、不依赖 flash-attn。

**价值主张**：证明「我不仅会用 `attention_implementation="flash_attention_2"`，还能讲清 RoPE 的旋转不变性、GQA 的 KV 头共享映射、以及手写实现如何与优化内核数值对齐」。

---

## 2. 技术栈

| 组件 | 选择 |
|---|---|
| 算子实现 | 纯 PyTorch（`torch.nn.Module` + 张量运算），不依赖 flash-attn |
| 参考实现 | HF `transformers` 的 `Qwen2RotaryEmbedding`；`torch.nn.functional.scaled_dot_product_attention`（SDPA） |
| 真实超参 | `Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct")`（仅下 config.json，不下载权重） |
| 测试 | `pytest` + `torch.autograd.gradcheck` |
| 精度 | CPU 单测用 fp32；gradcheck 用 fp64 |

**明确不选**：不加载 7B 权重（CPU 上无意义且过重）；不在本部分集成 flash-attn（GPU-only，属后续部分）。

---

## 3. 目录结构

```
Project2/
├── operators/                 # 新增子包
│   ├── __init__.py
│   ├── rope.py                # RotaryEmbedding + apply_rotary_pos_emb
│   └── gqa.py                 # repeat_kv + grouped_query_attention
├── tests/
│   ├── test_rope.py           # RoPE 单测
│   ├── test_gqa.py            # GQA 单测
│   └── test_ops_against_model.py  # 真实 Qwen2.5 config 对拍
└── docs/superpowers/specs/    # 本规格
```

---

## 4. RoPE 算子（`operators/rope.py`）

### 4.1 算法

对 head_dim `d`（偶数，Qwen2.5 为 128）、base `θ`：

- `inv_freq[i] = 1 / θ^(2i/d)`，`i ∈ [0, d/2)`。
- 位置 `m` 的角度：`m · inv_freq`（外积，shape `(seq_len, d/2)`）。
- 旋转按相邻两维成对进行，高效实现用 `rotate_half`：
  - `rotate_half(x)`：`x1 = x[..., :d//2]`、`x2 = x[..., d//2:]`，返回 `cat([-x2, x1], dim=-1)`。
  - `apply(x) = x·cos + rotate_half(x)·sin`。

### 4.2 接口

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int = 2048, base: float = 10000.0): ...
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (cos, sin)，shape (batch, seq_len, head_dim)。"""

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """对 (batch, num_heads, seq_len, head_dim) 的 q/k 施加旋转，返回同形状。"""
```

约定与 HF 对齐：`cos/sin` 形状为 `(batch, seq_len, head_dim)`；`apply_rotary_pos_emb` 默认 `unsqueeze_dim=1`，广播到多头维度。`base`/`head_dim`/`max_position_embeddings` 由调用方从真实 config 读入，**不写死**。

### 4.3 验证（三重）

1. **形状**：apply 后 q/k 形状与输入一致。
2. **旋转性质（解析）**：RoPE 的核心不变量——`⟨R_m q, R_n k⟩` 只取决于相对位置 `n−m`。断言：对任意平移 δ，`s(m, m+δ) == s(m+Δ, m+Δ+δ)`。
3. **HF 对拍**：与 `Qwen2RotaryEmbedding`（同 config）的 `cos/sin` 及 apply 输出在容差内一致。

---

## 5. GQA 算子（`operators/gqa.py`）

### 5.1 算法

GQA 把 `num_heads` 个 query 头分到 `num_key_value_heads` 组，`ratio = num_heads / num_key_value_heads`（Qwen2.5-7B 为 `28/4 = 7`）。query 头 `i` 共享 KV 头 `i // ratio`。

- `repeat_kv`：把 KV 从 `(batch, num_kv_heads, seq_len, head_dim)` 扩到 `(batch, num_heads, seq_len, head_dim)`（每个 KV 头重复 `ratio` 次，保证组内共享）。
- 注意力（手写，显式张量运算）：
  1. `scores = Q · K^T / √head_dim`
  2. 施加 mask（`is_causal` 上三角置 `-inf`；额外 `attn_mask` 相加）
  3. `softmax`（在 fp32 下计算，再 cast 回原 dtype）
  4. `output = scores · V`

### 5.2 接口

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, num_kv_heads, seq_len, head_dim) -> (batch, num_kv_heads*n_rep, seq_len, head_dim)"""

def grouped_query_attention(query, key, value, num_key_value_heads,
                            attn_mask=None, is_causal=False, dropout=0.0):
    """手写分组注意力。query: (batch, num_heads, seq, head_dim)；key/value: (batch, num_kv_heads, seq, head_dim)。"""
```

### 5.3 验证（三重）

1. **形状 + 分组映射**：输出形状 `(batch, num_heads, seq, head_dim)`；`repeat_kv` 后 KV 头 `j` 与 `[j·ratio, (j+1)·ratio)` 的 query 头对应同一源 KV 头（组内完全相同）。
2. **SDPA 对拍**：手写输出 ≈ `F.scaled_dot_product_attention(Q, repeat_kv(K), repeat_kv(V), is_causal=...)`，容差内一致（因果 mask 生效）。
3. **gradcheck**：`torch.autograd.gradcheck` 验证反向梯度正确（softmax 光滑可导）。

---

## 6. 真实模型对拍（`tests/test_ops_against_model.py`）

用 `Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct")` 读真实超参（**只下 config.json，不下权重**），然后：

| 对拍 | 做法 |
|---|---|
| RoPE | 手写 `RotaryEmbedding(head_dim, max_position_embeddings, base=rope_theta)` vs HF `Qwen2RotaryEmbedding(config)`，比对 cos/sin 与 apply 输出 |
| GQA | 真实形状 `(batch, 28, seq, 128)` / `(batch, 4, seq, 128)` 下，手写 vs SDPA（`repeat_kv` 扩到 28） |

关键真实超参（Qwen2.5-7B-Instruct）：

| 项 | 值 |
|---|---|
| `num_attention_heads` | 28 |
| `num_key_value_heads` | 4 |
| `head_dim` | 128 |
| `rope_theta` | 1000000.0 |
| GQA ratio | 7 |

> 注意：这些值从 config 读，不硬编码；`rope_theta` 尤其不能假设成 10000（Qwen2.5 是 1e6）。

---

## 7. 测试策略（全部 CPU 可跑）

| 测试 | 断言 |
|---|---|
| RoPE 形状 | apply 后 q/k 维度不变 |
| RoPE 旋转性质 | `⟨R_m q, R_n k⟩` 只随 `n−m` 变化（平移不变） |
| RoPE vs HF | cos/sin、apply 输出容差一致（atol=1e-5, rtol=1e-5） |
| RoPE gradcheck | `apply_rotary_pos_emb` 反向梯度正确 |
| GQA 形状 | 输出形状 `(batch, num_heads, seq, head_dim)` |
| GQA 分组映射 | `repeat_kv` 后同组 KV 头张量相同 |
| GQA vs SDPA | 手写 ≈ SDPA（rtol=1e-4, atol=1e-5），因果 mask 生效 |
| GQA gradcheck | 注意力反向梯度正确 |
| 真实模型对拍 | 28/4/128 + rope_theta 下两两一致 |

单测依赖：`torch`(CPU)、`transformers`、`pytest`。不依赖 GPU、不依赖 flash-attn。真实模型对拍测试需要一次性下载 `config.json`（KB 级，会缓存），离线后走缓存。

---

## 8. 成功标准

1. `operators/rope.py` 与 `operators/gqa.py` 全部单测通过（CPU）。
2. 旋转性质（相对位置不变性）解析测试通过——这是「讲得清原理」的关键证据。
3. 手写 RoPE / GQA 在真实 Qwen2.5-7B 超参下与 HF / SDPA 对拍一致。
4. 两个算子的 `gradcheck` 通过，证明可无缝接入端到端训练/推理图。

---

## 9. 非目标（YAGNI）

- 不集成 FlashAttention-2（GPU-only，属子项目 2 后续部分）。
- 不做 KV Cache 管理与推理吞吐压测（属子项目 2 后续部分）。
- 不写 C++/CUDA 扩展（纯 PyTorch 手写，定位是「讲原理 + 数值对拍」，不是「性能优化」）。
- 不做推理时的 full model 端到端（仅算子级验证，不加载 7B 权重）。
- 不涉及 GRPO / 奖励函数（属子项目 3）。
