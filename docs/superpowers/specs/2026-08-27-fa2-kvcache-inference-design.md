# 子项目 2（第二部分）：FlashAttention-2 集成 + KV Cache + 吞吐压测 — 设计规格

- 日期：2026-08-27
- 状态：待用户审阅
- 范围：本规格覆盖「子项目 2 的推理侧加速」——FlashAttention-2 运行时 dispatch、手写 KV Cache、以及吞吐压测。手写 RoPE/GQA 属第一部分（已完成），见 `2026-08-27-rope-gqa-operators-design.md`。

---

## 1. 背景与目标

第一部分把注意力里的 RoPE / GQA 从「调库」升级为「手写 + 对拍」，定位是「讲清原理」。本部分转向**推理性能**：把三种注意力后端（FlashAttention-2 优化内核 / SDPA / 手写 GQA）接入一个统一的 dispatch 层，并手写 KV Cache 支撑「prefill 全量写 + decode 增量 append」的解码循环，最后给出吞吐压测脚本。

三个交付物：

1. **运行时 dispatch**（`inference/attention.py`）：一个 `attention_forward`，运行时按「可用性 + device + dtype + head_dim」自动选择 FlashAttention-2 / SDPA / 手写 GQA，并**返回实际命中的后端**（证明 FA2 真跑了，不是静默 fallback）。
2. **手写 KV Cache**（`inference/kv_cache.py`）：预分配 buffer + 位置游标，prefill 全量写、decode 增量 append、按游标切片复用，避免重算历史 KV。
3. **吞吐压测**（`inference/benchmark.py`）：prefill / decode 的 tokens/s、KV cache 显存、batch/seq_len 扩展性。

**性质**：求职作品集 / 可复现实验。**诚实口径**——当前无 GPU，本部分代码「先写 + CPU 可测部分单测」；FA2 实际路径与吞吐数字在用户配置 GPU 后于 RTX 6000 上跑，**不虚构任何吞吐数据**。

**价值主张**：证明「我不仅能写对注意力算子，还懂推理引擎的核心——KV cache 为何省、prefill/decode 两阶段差异、以及如何把 FA2 这类 GPU-only 内核做成可降级、可验证的集成层」。

---

## 2. 技术栈

| 组件 | 选择 |
|---|---|
| dispatch 层 | 纯 PyTorch；`flash_attn`（可选依赖，运行时 `import` 守卫） |
| FA2 后端 | `flash_attn.flash_attn_func`（GPU-only，fp16/bf16） |
| SDPA 后端 | `torch.nn.functional.scaled_dot_product_attention`（torch 2.11，CPU/GPU 都可用，原生支持 GQA） |
| 手写后端 | 复用第一部分 `operators.gqa.grouped_query_attention` |
| KV Cache | 纯 PyTorch 张量预分配 + 位置游标，手写 |
| 测试 | `pytest`；CPU 单测用 fp32 |
| 环境事实 | torch 2.11.0；本机无 CUDA、无 flash-attn（`flash_attn_available()` 为 False） |

**明确不选**：不引入 `torch.compile`、不写 CUDA/C++ 扩展、不依赖 `transformers` 的 `past_key_values` 机制（本部分 KV Cache 手写）。

---

## 3. 目录结构

```
Project2/
├── inference/                 # 新增子包
│   ├── __init__.py
│   ├── attention.py           # attention_forward + 三层 dispatch
│   ├── kv_cache.py            # 手写 KVCache
│   ├── generate.py            # 贪心 decode 循环（GPU-deferred）
│   └── benchmark.py           # 吞吐压测脚本（GPU-deferred）
├── tests/
│   ├── test_attention.py      # dispatch 选择 + 后端输出对拍（CPU）
│   └── test_kv_cache.py       # 写入/读取/游标/切片（CPU）
└── docs/superpowers/specs/    # 本规格
```

> 依赖方向：`inference.attention` → `operators.gqa`（复用第一部分手写算子，作手动兜底）。`inference.kv_cache` 无外部依赖。

---

## 4. dispatch 层（`inference/attention.py`）

### 4.1 算法

统一入口 `attention_forward`，输入 q/k/v（`(batch, num_heads, seq_len, head_dim)` 与 `(batch, num_kv_heads, seq_len, head_dim)`），按 `backend` 参数分发：

| backend | 条件 | 实现 |
|---|---|---|
| `flash_attn_2` | `flash_attn` 可用 + `device.type=="cuda"` + `dtype ∈ {fp16, bf16}` + `head_dim ≤ 256` | `flash_attn_func`（注意其布局为 `(b, s, h, d)`，需 `transpose(1,2)` 进出） |
| `sdpa` | 默认 | `F.scaled_dot_product_attention`，需显式传 `enable_gqa=True`（torch 2.11 默认**不**自动广播 KV 头，GQA 下不传会报错） |
| `manual` | 显式指定 | `operators.gqa.grouped_query_attention`（`repeat_kv` 扩到 num_heads） |

`backend="auto"` 时按上述条件**自上而下**选第一个满足者；`backend` 显式传值时强制走指定后端（供测试/对拍/压测）。

三个后端在 `softmax_scale=None` 时都默认 `1/√head_dim`，保证对拍一致性。

### 4.2 接口

```python
class AttentionBackend(str, Enum):
    FLASH_ATTN_2 = "flash_attn_2"
    SDPA = "sdpa"
    MANUAL = "manual"

def flash_attn_available() -> bool:
    """flash-attn 是否可用：import 成功且 is_available() 为 True。"""

def select_backend(device, dtype, head_dim, *, flash_available, backend="auto") -> AttentionBackend:
    """纯函数：给定 device/dtype/head_dim/flash_available，返回选中的后端。"""

def attention_forward(query, key, value, num_key_value_heads, *,
                      is_causal=True, backend="auto", softmax_scale=None) -> tuple[torch.Tensor, str]:
    """返回 (output, used_backend)。output 形状 (batch, num_heads, seq_len, head_dim)；used_backend 为实际命中后端的字符串。"""
```

### 4.3 验证（三重）

1. **dispatch 选择（纯函数，CPU 可测）**：`select_backend` 在「CPU/无 flash → SDPA」「CUDA+fp16+head_dim≤256+flash → FLASH_ATTN_2」「CUDA+fp32 → SDPA」「head_dim>256 → SDPA」「显式 manual → MANUAL」各分支正确。
2. **后端输出对拍（CPU 可测）**：`attention_forward`（auto，CPU 落 SDPA）与 `backend="manual"`、与直接调 `F.scaled_dot_product_attention` 三者输出一致（GQA 形状 28/4，因果 mask 生效）。
3. **返回值**：`attention_forward` 返回的 `used_backend` 与实际后端一致（本机为 `"sdpa"`）。

---

## 5. 手写 KV Cache（`inference/kv_cache.py`）

### 5.1 算法

KV cache 是每层一组 (K, V)，按位置索引存储。prefill 阶段一次写入 prompt 全部位置的 KV；decode 阶段每步只算最后一个 token 的 KV、append 到缓存尾部，其余位置直接复用——这是推理从「每步重算全序列」降到「每步只算 1 个 token」的关键。

- **预分配**：初始化即分配 `(batch, num_kv_heads, max_seq_len, head_dim)` 的 K、V 张量（连续显存，避免动态 concat 的反复搬迁），配一个 `seq_len` 游标记录当前有效长度。
- **写（update）**：`positions` 指定写入位置，prefill 传 `arange(0, L)`，decode 传 `[seq_len]`。
- **读（get）**：返回 `[:seq_len]` 切片——只暴露有效长度，越界位置不外泄。
- **游标（advance）**：显式推进 `seq_len`（prefill 后 `advance(L)`，decode 后 `advance(1)`）。

### 5.2 接口

```python
class KVCache:
    def __init__(self, num_layers: int, batch: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int, dtype=torch.float32, device="cpu"): ...
    @property
    def seq_len(self) -> int: ...
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor,
               positions: torch.LongTensor) -> None:
        """把 key/value（shape (batch, num_kv_heads, len(positions), head_dim)）写入 positions 指定位置。"""
    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回当前有效 KV 切片，各 (batch, num_kv_heads, seq_len, head_dim)。"""
    def advance(self, n: int) -> None: ...
```

> 约定：`positions` 为 `torch.LongTensor`（沿 seq 维的索引）。每层独立一组 (K, V)，layer_idx 隔离。显存占用 = `2 × num_layers × batch × num_kv_heads × max_seq_len × head_dim × dtype_size`（本部分用 `max_seq_len` 预分配；真实部署可改 chunked 增长，属后续优化，非目标）。

### 5.3 验证（CPU 可测）

1. **预分配形状**：init 后各层 K/V 形状正确、初值为 0。
2. **prefill 往返**：`update(layer, k, v, arange(L))` + `advance(L)` 后，`get` 返回与写入一致。
3. **decode 增量 append**：prefill L 后 `update(layer, k1, v1, [L])` + `advance(1)`，`get` 长度变 L+1 且末位是新增值、前 L 位不变。
4. **get 尊重游标**：写入超过 `seq_len` 的位置后，`get` 只返回 `[:seq_len]`，不越界。
5. **多位置 / 非连续写入**：`positions=[0,2,4]` 时正确落位，中间位置保持 0。
6. **层间隔离**：写 layer 0 不影响 layer 1。

---

## 6. 贪心 decode 循环（`inference/generate.py`，GPU-deferred）

把 `attention_forward` + `KVCache` 串起来的端到端解码参考实现，结构如下（**上卡跑**，CPU 不加载 7B 权重）：

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, kv_cache, *,
             temperature=1.0, top_k=None, device="cuda") -> list[int]:
    # 1) prefill：整个 prompt 一次 forward，KV 全量写入 cache（positions = arange(prompt_len)），advance(prompt_len)
    # 2) decode：循环 max_new_tokens 步，每步只 forward 最后 1 个 token，
    #    新 KV append 到 cache 尾（positions = [seq_len]），advance(1)，其余位置复用
    # 3) 采样：temperature / top_k（贪心 = temperature 1.0 + top_k None）
    # 4) 直到命中 eos_token_id 或达到 max_new_tokens
```

- **依赖真实模型权重与 GPU**，`model` 为接入 dispatch 后的 Qwen2.5-7B（预填充/增量 KV 由 `KVCache` 承接，attention 由 `attention_forward` 计算）。
- **非目标**：不实现 beam search / speculative decoding；不把 7B 模型集成代码写进本部分（留到上卡时接 `transformers` 模型，作 GPU 步骤说明）。

---

## 7. 吞吐压测（`inference/benchmark.py`，GPU-deferred）

脚本测量注意力/解码两阶段吞吐，输出表格 + 可选图。**记录 `used_backend`**，确保压测确实命中 FA2 而非静默降级。

| 场景 | 指标 | 扫描 |
|---|---|---|
| prefill | 单次前向 wall-time → `tokens/s = batch × seq_len / t` | `batch ∈ {1,2,4,8}`，`seq_len ∈ {128,256,512,1024,2048}` |
| decode | 每步单 token 延迟 → `tokens/s = 1 / t`、`ms/token` | 同 batch，KV cache 从 0 增到 seq_len |
| 显存 | KV cache 峰值 = 上文公式 × dtype_size | 随 seq_len/batch |

输出：console 表格 + 可选 `matplotlib` 折线图（prefill vs decode 吞吐随 seq_len 变化）。GPU-deferred：**本机无卡，脚本写好、上卡运行，数字不虚构**。

---

## 8. 测试策略

| 层 | 测试 | 断言 |
|---|---|---|
| CPU | `select_backend` 各分支 | 每个条件组合返回预期后端 |
| CPU | `attention_forward` vs `manual` / 直接 SDPA | GQA 形状 28/4 下三者输出一致（因果 mask 生效） |
| CPU | `attention_forward` 返回值 | `used_backend == "sdpa"`（本机） |
| CPU | `KVCache` 预分配/往返/append/切片/层隔离 | 见 §5.3 |
| GPU（deferred） | FA2 真实路径 | `flash_attn_func` 输出 vs SDPA 对拍（fp16/bf16） |
| GPU（deferred） | generate / benchmark | 端到端 decode + 吞吐数字（上卡） |

单测依赖：`torch`(CPU)、`pytest`。不依赖 GPU、不依赖 flash-attn（CPU 单测只走 SDPA/manual 分支）。FA2 集成测试与吞吐数字标注 GPU-deferred，不阻塞合并。

---

## 9. 成功标准

1. `inference/attention.py` 与 `inference/kv_cache.py` 全部 CPU 单测通过。
2. `select_backend` 纯函数逻辑正确（含 FA2 的选择条件：CUDA + fp16/bf16 + head_dim≤256）。
3. `attention_forward` 三后端（本机 SDPA/manual）输出对拍一致，且返回真实命中的后端。
4. `KVCache` 的 prefill 全量写 / decode 增量 append / 游标切片逻辑正确。
5. `generate.py` / `benchmark.py` 结构完整、标注 GPU-deferred，上卡即可运行，不虚构数字。

---

## 10. 非目标（YAGNI）

- 不写 CUDA/C++ 扩展、不用 `torch.compile`（FA2 本身就是优化内核，本部分做「集成 + 降级 + 验证」）。
- 不实现 beam search / speculative decoding / 连续批处理（continuous batching）。
- 不把真实 Qwen2.5-7B 模型集成代码写进本部分（generate/benchmark 留接口，上卡时接 `transformers`）。
- 不做 KV cache 的 chunked 动态增长 / paged attention（属后续优化）。
- 不涉及 GRPO / 奖励函数（属子项目 3）。
