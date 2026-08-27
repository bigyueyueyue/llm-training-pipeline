# FlashAttention-2 集成 + KV Cache + 吞吐压测 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `inference/` 子包实现三层注意力 dispatch（FA2/SDPA/手写）+ 手写 KV Cache + GPU-deferred 的 decode 循环与吞吐压测脚本。

**Architecture:** `attention_forward` 按 device/dtype/head_dim/flash 可用性运行时选后端并返回命中后端；`KVCache` 预分配 buffer + 位置游标支撑 prefill 全量写 / decode 增量 append；`generate.py`/`benchmark.py` 串起二者，上卡运行。CPU 只测 dispatch 选择与缓存逻辑，FA2 路径与吞吐数字 GPU-deferred。

**Tech Stack:** torch 2.13（CPU）；`flash_attn`（可选、运行时守卫）；复用 `operators.gqa`；pytest。

## Global Constraints

- 本机 torch 2.13.0，**无 CUDA、无 flash-attn**（`flash_attn_available()` 恒为 False）。
- CPU 单测用 fp32，不依赖 GPU / flash-attn；所有单测必须 CPU 可跑。
- SDPA 路径**必须**传 `enable_gqa=(num_heads != num_key_value_heads)`（torch 2.13 默认不广播 KV 头，GQA 下不传会 RuntimeError）。
- FA2 选择条件：flash 可用 + `device.type=="cuda"` + `dtype ∈ {fp16, bf16}` + `head_dim ≤ 256`。
- 手动兜底复用 `operators.gqa.grouped_query_attention`（第一部分产物，**不得修改**）。
- 接口签名与规格 §4.2 / §5.2 严格一致：`attention_forward(query, key, value, num_key_value_heads, *, is_causal=True, backend="auto", softmax_scale=None) -> (Tensor, str)`；`KVCache.update(layer_idx, key, value, positions)` 的 `positions` 为 `torch.LongTensor`。
- 只提交本部分新文件（`inference/`、`tests/test_attention.py`、`tests/test_kv_cache.py`）；**不 git-add** 未决的 `docs/interview-prep-sft.md`、`docs/interview-prep-rope-gqa.md`。
- 实现须在功能分支上进行，不得直接在 main 上写实现代码。

---

### Task 1: 手写 KV Cache（`inference/kv_cache.py`）

**Files:**
- Create: `inference/__init__.py`
- Create: `inference/kv_cache.py`
- Test: `tests/test_kv_cache.py`

**Interfaces:**
- Consumes: 无（纯 PyTorch 张量，无外部依赖）
- Produces: `KVCache(num_layers, batch, num_kv_heads, head_dim, max_seq_len, dtype=torch.float32, device="cpu")`，方法 `update(layer_idx, key, value, positions)`、`get(layer_idx)`、`advance(n)`，属性 `seq_len`。后续 Task 3 的 `benchmark.py` 依赖此接口。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_kv_cache.py`：

```python
import torch
import pytest

from inference.kv_cache import KVCache


@pytest.fixture
def cache():
    # num_layers=2, batch=1, num_kv_heads=4, head_dim=8, max_seq_len=16
    return KVCache(num_layers=2, batch=1, num_kv_heads=4, head_dim=8,
                   max_seq_len=16, dtype=torch.float32, device="cpu")


def test_init_shapes(cache):
    for layer in range(cache.num_layers):
        k, v = cache.get(layer)
        assert k.shape == (1, 4, 16, 8)
        assert v.shape == (1, 4, 16, 8)
        assert k.dtype == torch.float32
    assert cache.seq_len == 0


def test_prefill_write_read_roundtrip(cache):
    L = 5
    k = torch.randn(1, 4, L, 8)
    v = torch.randn(1, 4, L, 8)
    cache.update(0, k, v, torch.arange(L))
    cache.advance(L)
    got_k, got_v = cache.get(0)
    assert got_k.shape == (1, 4, L, 8)
    assert torch.allclose(got_k, k)
    assert torch.allclose(got_v, v)


def test_decode_incremental_append(cache):
    L = 5
    k_pre = torch.randn(1, 4, L, 8)
    v_pre = torch.randn(1, 4, L, 8)
    cache.update(0, k_pre, v_pre, torch.arange(L))
    cache.advance(L)
    k_dec = torch.randn(1, 4, 1, 8)
    v_dec = torch.randn(1, 4, 1, 8)
    cache.update(0, k_dec, v_dec, torch.tensor([L]))
    cache.advance(1)
    got_k, got_v = cache.get(0)
    assert got_k.shape == (1, 4, L + 1, 8)
    assert torch.allclose(got_k[:, :, :L], k_pre)
    assert torch.allclose(got_k[:, :, L:], k_dec)
    assert torch.allclose(got_v[:, :, L:], v_dec)


def test_get_respects_seq_len(cache):
    L = 3
    k = torch.randn(1, 4, L, 8)
    cache.update(0, k, torch.zeros_like(k), torch.arange(L))
    cache.advance(L)
    # 越过当前 seq_len=3 写第 6 位，get 不应返回它
    extra = torch.randn(1, 4, 1, 8)
    cache.update(0, extra, extra, torch.tensor([6]))
    got_k, _ = cache.get(0)
    assert got_k.shape == (1, 4, 3, 8)
    assert torch.allclose(got_k, k)


def test_update_at_noncontiguous_positions(cache):
    positions = torch.tensor([0, 2, 4])
    k = torch.randn(1, 4, 3, 8)
    cache.update(1, k, k, positions)
    cache.advance(5)
    got_k, _ = cache.get(1)
    assert torch.allclose(got_k[:, :, 0], k[:, :, 0])
    assert torch.allclose(got_k[:, :, 2], k[:, :, 1])
    assert torch.allclose(got_k[:, :, 4], k[:, :, 2])
    assert torch.all(got_k[:, :, 1] == 0)
    assert torch.all(got_k[:, :, 3] == 0)


def test_layer_isolation(cache):
    k0 = torch.randn(1, 4, 1, 8)
    cache.update(0, k0, k0, torch.tensor([0]))
    cache.advance(1)
    got_k1, _ = cache.get(1)
    assert torch.all(got_k1 == 0)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_kv_cache.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'inference'`）

- [ ] **Step 3: 实现**

创建 `inference/__init__.py`：

```python
"""推理侧：注意力 dispatch 与手写 KV Cache。"""
```

创建 `inference/kv_cache.py`：

```python
import torch


class KVCache:
    """手写 KV Cache：预分配连续 buffer + 位置游标。

    prefill 一次写入 prompt 全部位置的 K/V；decode 每步只 append 最后一个
    token 的 K/V，其余位置复用，避免重算历史 KV。
    """

    def __init__(self, num_layers, batch, num_kv_heads, head_dim, max_seq_len,
                 dtype=torch.float32, device="cpu"):
        self.num_layers = num_layers
        self.batch = batch
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        shape = (batch, num_kv_heads, max_seq_len, head_dim)
        self._keys = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._values = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self._seq_len = 0

    @property
    def seq_len(self):
        return self._seq_len

    def update(self, layer_idx, key, value, positions):
        """写入 key/value（shape (batch, num_kv_heads, len(positions), head_dim)）到 positions 指定位置。"""
        self._keys[layer_idx][:, :, positions] = key
        self._values[layer_idx][:, :, positions] = value

    def get(self, layer_idx):
        """返回当前有效 KV 切片，各 (batch, num_kv_heads, seq_len, head_dim)。"""
        return (self._keys[layer_idx][:, :, : self._seq_len],
                self._values[layer_idx][:, :, : self._seq_len])

    def advance(self, n):
        self._seq_len += n
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_kv_cache.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add inference/__init__.py inference/kv_cache.py tests/test_kv_cache.py
git commit -m "feat(inference): 手写 KV Cache（预分配 + 位置游标）"
```

---

### Task 2: 注意力 dispatch 层（`inference/attention.py`）

**Files:**
- Create: `inference/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: `operators.gqa.grouped_query_attention`（第一部分产物，已存在）
- Produces: `AttentionBackend(str, Enum)`；`flash_attn_available() -> bool`；`select_backend(device, dtype, head_dim, *, flash_available, backend="auto") -> AttentionBackend`；`attention_forward(query, key, value, num_key_value_heads, *, is_causal=True, backend="auto", softmax_scale=None) -> tuple[Tensor, str]`。Task 3 的 `generate.py`/`benchmark.py` 依赖 `attention_forward`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_attention.py`：

```python
import torch
import torch.nn.functional as F

from inference.attention import (AttentionBackend, attention_forward,
                                 flash_attn_available, select_backend)
from operators.gqa import repeat_kv


def test_select_backend_auto_cpu():
    backend = select_backend(torch.device("cpu"), torch.float32, 128,
                             flash_available=False)
    assert backend == AttentionBackend.SDPA


def test_select_backend_auto_flash():
    backend = select_backend(torch.device("cuda"), torch.float16, 128,
                             flash_available=True)
    assert backend == AttentionBackend.FLASH_ATTN_2


def test_select_backend_flash_requires_fp16_bf16():
    backend = select_backend(torch.device("cuda"), torch.float32, 128,
                             flash_available=True)
    assert backend == AttentionBackend.SDPA


def test_select_backend_flash_head_dim_limit():
    backend = select_backend(torch.device("cuda"), torch.float16, 512,
                             flash_available=True)
    assert backend == AttentionBackend.SDPA


def test_select_backend_explicit_manual():
    backend = select_backend(torch.device("cpu"), torch.float32, 128,
                             flash_available=False, backend="manual")
    assert backend == AttentionBackend.MANUAL


def test_flash_attn_available_returns_bool():
    assert isinstance(flash_attn_available(), bool)


def _gqa_inputs(seed=0, b=2, nh=28, nkv=4, s=16, d=128):
    torch.manual_seed(seed)
    q = torch.randn(b, nh, s, d)
    k = torch.randn(b, nkv, s, d)
    v = torch.randn(b, nkv, s, d)
    return q, k, v, nkv


def test_attention_forward_sdpa_matches_manual():
    q, k, v, nkv = _gqa_inputs()
    out_sdpa, backend_sdpa = attention_forward(q, k, v, nkv, backend="sdpa")
    out_manual, backend_manual = attention_forward(q, k, v, nkv, backend="manual")
    assert backend_sdpa == "sdpa"
    assert backend_manual == "manual"
    assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4)


def test_attention_forward_auto_matches_sdpa_reference():
    q, k, v, nkv = _gqa_inputs()
    out, backend = attention_forward(q, k, v, nkv)  # auto -> CPU 落 SDPA
    kr = repeat_kv(k, q.shape[1] // nkv)
    vr = repeat_kv(v, q.shape[1] // nkv)
    ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    assert backend == "sdpa"
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_forward_causal():
    q, k, v, nkv = _gqa_inputs(s=8)
    out, _ = attention_forward(q, k, v, nkv, backend="manual", is_causal=True)
    out_nocausal, _ = attention_forward(q, k, v, nkv, backend="manual", is_causal=False)
    assert not torch.allclose(out, out_nocausal)
    kr = repeat_kv(k, q.shape[1] // nkv)
    vr = repeat_kv(v, q.shape[1] // nkv)
    ref = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_attention_forward_output_shape():
    q, k, v, nkv = _gqa_inputs()
    out, _ = attention_forward(q, k, v, nkv)
    assert out.shape == q.shape


def test_attention_forward_mha_no_gqa():
    b, s, d = 2, 8, 32
    q = torch.randn(b, 4, s, d)
    k = torch.randn(b, 4, s, d)
    v = torch.randn(b, 4, s, d)
    out, backend = attention_forward(q, k, v, 4, backend="sdpa")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert backend == "sdpa"
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'inference.attention'`）

- [ ] **Step 3: 实现**

创建 `inference/attention.py`：

```python
from enum import Enum

import torch
import torch.nn.functional as F

from operators.gqa import grouped_query_attention


class AttentionBackend(str, Enum):
    FLASH_ATTN_2 = "flash_attn_2"
    SDPA = "sdpa"
    MANUAL = "manual"


def flash_attn_available():
    """flash-attn 是否可用（import 成功且 is_available() 为 True）。"""
    try:
        import flash_attn
        return flash_attn.is_available()
    except ImportError:
        return False


def select_backend(device, dtype, head_dim, *, flash_available, backend="auto"):
    """纯函数：给定 device/dtype/head_dim/flash 可用性，返回选中的后端。"""
    if backend != "auto":
        return AttentionBackend(backend)
    if (flash_available
            and device.type == "cuda"
            and dtype in (torch.float16, torch.bfloat16)
            and head_dim <= 256):
        return AttentionBackend.FLASH_ATTN_2
    return AttentionBackend.SDPA


def attention_forward(query, key, value, num_key_value_heads, *,
                      is_causal=True, backend="auto", softmax_scale=None):
    """统一注意力入口，返回 (output, used_backend)。

    query: (batch, num_heads, seq_len, head_dim)
    key/value: (batch, num_kv_heads, seq_len, head_dim)
    """
    head_dim = query.shape[-1]
    num_heads = query.shape[1]
    resolved = select_backend(query.device, query.dtype, head_dim,
                              flash_available=flash_attn_available(),
                              backend=backend)

    if resolved == AttentionBackend.FLASH_ATTN_2:
        from flash_attn import flash_attn_func
        # FA2 布局 (batch, seq_len, num_heads, head_dim)，进出各转置一次
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=softmax_scale)
        return out.transpose(1, 2), resolved.value

    if resolved == AttentionBackend.SDPA:
        # torch 2.13 默认不广播 KV 头，GQA 必须显式 enable_gqa
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, scale=softmax_scale,
            enable_gqa=(num_heads != num_key_value_heads))
        return out, resolved.value

    out = grouped_query_attention(query, key, value, num_key_value_heads,
                                  is_causal=is_causal)
    return out, resolved.value
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_attention.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add inference/attention.py tests/test_attention.py
git commit -m "feat(inference): 注意力三层 dispatch（FA2/SDPA/手写）"
```

---

### Task 3: decode 循环 + 吞吐压测（`generate.py` / `benchmark.py`，GPU-deferred）

**Files:**
- Create: `inference/generate.py`
- Create: `inference/benchmark.py`

**Interfaces:**
- Consumes: Task 1 的 `KVCache`；Task 2 的 `attention_forward`
- Produces: `generate(model, tokenizer, prompt, max_new_tokens, kv_cache, *, temperature=1.0, top_k=None, device="cuda") -> list[int]`；`benchmark.main(argv=None)`（`python -m inference.benchmark` 入口）。二者为 GPU-deferred 参考脚本，**不写单测**，CPU 只做导入 + 语法检查。

> 说明：本任务是 GPU-deferred 参考实现，TDD 的「失败测试」不适用；用「导入 + 语法编译」作为 CPU 可跑的验证门槛。真实 FA2 路径与吞吐数字上卡后再跑。

- [ ] **Step 1: 写 decode 循环**

创建 `inference/generate.py`：

```python
"""贪心 decode 循环（GPU-deferred 参考实现）。

上卡后：把真实 Qwen2.5-7B 每层注意力接入 attention_forward，K/V 由 KVCache
承接。本脚本演示 prefill / decode 两阶段与 KV cache 交互，依赖真实模型权重
与 GPU，CPU 不跑。
"""
import torch


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, kv_cache, *,
             temperature=1.0, top_k=None, device="cuda"):
    """贪心/采样解码。

    model 需提供增量接口 `model(input_ids, positions, cache) -> logits`：
    对给定 token 与 KV cache 位置做一次前向，返回 (batch, seq, vocab) logits。
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = []

    # 1) prefill：整个 prompt 一次 forward，KV 全量写入 cache
    positions = torch.arange(input_ids.shape[1], device=device)
    logits = model(input_ids, positions, cache=kv_cache)
    kv_cache.advance(input_ids.shape[1])

    # 2) decode：逐 token，每步只算最后 1 个 token，KV append 到 cache 尾
    next_id = _sample(logits[0, -1], temperature, top_k)
    for _ in range(max_new_tokens):
        generated.append(int(next_id))
        if next_id == tokenizer.eos_token_id:
            break
        positions = torch.tensor([kv_cache.seq_len], device=device)
        logits = model(next_id.reshape(1, 1), positions, cache=kv_cache)
        kv_cache.advance(1)
        next_id = _sample(logits[0, -1], temperature, top_k)

    return generated


def _sample(logits, temperature=1.0, top_k=None):
    """贪心（top_k=None）或 top-k 采样，返回标量 LongTensor。"""
    if top_k is None:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, top_k)
    probs = torch.softmax(topk_vals, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1)
    return topk_idx[chosen].reshape(1)
```

- [ ] **Step 2: 写吞吐压测**

创建 `inference/benchmark.py`：

```python
"""吞吐压测（GPU-deferred）。

上卡后运行，测量 prefill（批量前向）与 decode（单 token 增量）两阶段吞吐，
并打印实际命中的注意力后端，确保 FA2 真跑了而非静默降级。

用法：python -m inference.benchmark --batch 1 2 4 8 --seq-len 128 256 512 1024 2048
"""
import argparse
import time

import torch

from inference.attention import attention_forward
from inference.kv_cache import KVCache


def _gqa_inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device):
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def bench_prefill(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device, repeats=10):
    q, k, v = _gqa_inputs(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device)
    for _ in range(3):
        attention_forward(q, k, v, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        attention_forward(q, k, v, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * seq_len / (dt / repeats)  # tokens/s


def bench_decode(batch, seq_len, num_heads, num_kv_heads, head_dim, dtype, device, repeats=10):
    cache = KVCache(num_layers=1, batch=batch, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, max_seq_len=seq_len + 1, dtype=dtype, device=device)
    k = torch.randn(batch, num_kv_heads, seq_len - 1, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_kv_heads, seq_len - 1, head_dim, dtype=dtype, device=device)
    cache.update(0, k, v, torch.arange(seq_len - 1, device=device))
    cache.advance(seq_len - 1)
    q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype, device=device)
    kk = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    vv = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=dtype, device=device)
    for _ in range(3):
        attention_forward(q, kk, vv, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        attention_forward(q, kk, vv, num_kv_heads, is_causal=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 1.0 / (dt / repeats)  # tokens/s（单 token 步）


def main(argv=None):
    parser = argparse.ArgumentParser(description="FA2/KV Cache 吞吐压测（GPU）")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    q, k, v = _gqa_inputs(1, 16, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device)
    _, backend = attention_forward(q, k, v, args.num_kv_heads, is_causal=True)
    print(f"backend: {backend}")

    print("\nprefill tokens/s (batch x seq_len):")
    for b in args.batch:
        cells = [f"{s}:{bench_prefill(b, s, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device):.0f}"
                 for s in args.seq_len]
        print(f"  batch={b}: " + "  ".join(cells))

    print("\ndecode tokens/s (batch, cache grows to seq_len):")
    for b in args.batch:
        for s in args.seq_len:
            t = bench_decode(b, s, args.num_heads, args.num_kv_heads, args.head_dim, dtype, device)
            print(f"  batch={b} seq_len={s}: {t:.1f} tokens/s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 语法 + 导入验证（CPU）**

Run:
```bash
python -m py_compile inference/generate.py inference/benchmark.py
python -c "import inference.generate, inference.benchmark; print('import ok')"
```
Expected: 无输出（py_compile）+ `import ok`（导入成功）

- [ ] **Step 4: 提交**

```bash
git add inference/generate.py inference/benchmark.py
git commit -m "feat(inference): decode 循环 + 吞吐压测脚本（GPU-deferred）"
```

---

## 完成后

- 运行全量单测：`pytest -q`（预期 14 旧 + 17 新 = 31 passed）。
- 用 superpowers:finishing-a-development-branch 收尾（合并回 main 或按用户选择）。
- 更新 README 子项目 2 状态（从「规划中」→「已完成手写算子 + FA2/KV Cache 集成（吞吐上卡待跑）」）——可选，另行确认。
