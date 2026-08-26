# 子项目 1：SFT 微调链路 — 设计规格

- 日期：2026-08-26
- 状态：待用户审阅
- 范围：本规格只覆盖「子项目 1（SFT 微调链路）」。子项目 2（手写算子 + 推理加速）与子项目 3（GRPO RL 对齐）各自独立成规格。

---

## 1. 背景与目标

基于开源基座 **Qwen2.5-7B-Instruct**，在 **NVIDIA RTX 6000（48GB）** 单卡上搭建监督微调（SFT）链路。核心目标是：

1. 实现并验证一个**自定义多轮对话掩码 DataCollator**——loss 只落在 assistant 回答上，规避在历史 prompt 上产生无效梯度，提升收敛效率。
2. 通过可复现的 A/B 实验量化这一收益（目标：收敛效率提升约 20%）。
3. 产出可加载、可继续用于后续 GRPO 对齐的 **LoRA adapter**。

**性质**：求职作品集 / 可复现实验。代码结构与文档清晰度优先，所有核心逻辑必须在 CPU 上可单测；全量训练留待上卡后执行。

---

## 2. 技术栈

| 组件 | 选择 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 训练框架 | `transformers.Trainer`（或 `trl.SFTTrainer`），由自定义 collator 注入 |
| 微调方式 | **LoRA**（`peft`），`r=16, alpha=32, dropout=0.05`，target=`[q,k,v,o,gate,up,down]_proj` |
| 精度 | `bf16`（上卡后启用）；本地 CPU 单测用 `fp32`/`float64` |
| 加速 | `accelerate` + `gradient_checkpointing` + `flash_attn_2`（仅 GPU 启用，本地跳过） |
| 数据 | `datasets`，默认 `BelleGroup/train_3.5M_CN`（可配置替换） |

**明确不选**：TRL 内置 `DataCollatorForCompletionOnlyLM`（避免「定制」退化为「配置现成类」）；全量微调（7B 全量优化器状态在 48GB 放不下）。

---

## 3. 目录结构（Monorepo 中 sft 子包 + 共享 core）

```
Project2/
├── core/
│   ├── __init__.py
│   ├── tokenizer_utils.py   # tokenizer 加载、chat 模板辅助
│   └── metrics.py           # loss 记录、收敛阈值计算
├── sft/
│   ├── __init__.py
│   ├── data.py              # 数据加载 + build_chat_sample（labels 生成）
│   ├── collator.py          # ChatDataCollator（核心交付物）
│   ├── train.py             # 训练入口：LoRA + Trainer + 自定义 collator
│   ├── run_masking_ab.py    # 掩码 vs 全序列 loss 的 A/B 收敛实验
│   └── config/
│       └── sft_lora.yaml    # 超参配置
├── tests/
│   ├── test_labels.py       # 掩码规则单测
│   └── test_collator.py     # collator 单测
├── scripts/
│   └── setup_env.sh         # 环境清单（Mac/GPU 两套，flash-attn 可选）
├── docs/
│   ├── superpowers/specs/   # 本规格
│   └── experiments/         # 实验记录（loss 曲线、收敛表）
└── README.md
```

---

## 4. 数据流

1. **加载**：`load_dataset("BelleGroup/train_3.5M_CN")`，取子集（可配置 `--max_samples`）。
2. **样本形态**：每条样本为多轮 `messages` 列表：
   ```json
   [{"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}]
   ```
3. **tokenize + 生成 labels**：见 §5。

---

## 5. 自定义 DataCollator（核心交付物）

### 5.1 接口

```python
# sft/data.py
def build_chat_sample(messages: list[dict], tokenizer, max_length: int = 2048) -> dict:
    """返回 {"input_ids": list[int], "attention_mask": list[int], "labels": list[int]}"""
    ...

# sft/collator.py
class ChatDataCollator:
    def __init__(self, tokenizer, max_length: int = 2048,
                 pad_token_id=None, label_pad_token_id: int = -100):
        ...
    def __call__(self, features: list[dict]) -> dict:
        """padding 到批次内最长（截断到 max_length），返回 batched tensors"""
        ...
```

### 5.2 掩码规则

对 Qwen2.5 的 chat 格式 `<|im_start|>role\ncontent<|im_end|>\n`，按角色设置 labels：

| 片段 | labels |
|---|---|
| `<|im_start|>system\n…<|im_end|>` | 全部 `-100` |
| `<|im_start|>user\n…<|im_end|>` | 全部 `-100` |
| `<|im_start|>assistant\n`（assistant 段头部） | `-100` |
| assistant 的 `content` | 保留原 token id |
| assistant 段结尾 `<|im_end|>` | 保留原 token id |
| padding | `-100` |

要点：

- **`<|im_end|>` 必须归入 assistant 段**（模型要学着自己输出结束符，这是常见漏点）。
- `input_ids` 与 `labels` 等长，`labels` 中 `-100` 处不参与 loss。

### 5.3 实现方式

用 `tokenizer.apply_chat_template(messages, tokenize=True)` 得到正确的 `input_ids`（保证格式与官方一致），再基于 Qwen 特殊标记定位 assistant 段（`<|im_start|>assistant` 到 `<|im_end|>`）生成 `labels`。定位逻辑封装成纯函数，便于单测。

### 5.4 诚实澄清（写入 README，防面试穿帮）

掩码优化的是 **loss / 梯度信号**：梯度只来自回答，不在 prompt 上产生无效更新，因此收敛更快。它**不直接省 FLOPs**。真正节省「历史 prompt 计算」的技术是 **sequence packing**（合并短样本 + attention mask 隔离），列为**可选增强**，不在本子项目第一版范围。

---

## 6. LoRA + 训练配置（48GB 单卡，bf16）

| 项 | 值 |
|---|---|
| LoRA | `r=16, alpha=32, dropout=0.05`，target=`[q,k,v,o,gate,up,down]_proj` |
| 精度 | `bf16`；`gradient_checkpointing=True` |
| 注意力 | `flash_attn_2`（上卡后启用） |
| 有效 batch | ≈ 32（per-device 4 × grad_accum 8） |
| 学习率 | `2e-4`，cosine 调度，warmup ratio 0.03 |
| 序列长度 | 2048 |
| epochs | 1–2 |
| 优化器 | AdamW（`weight_decay=0.0`，与 LoRA 惯例一致） |

---

## 7. 收敛效率实验（"20%" 的可复现来源）

- **A/B 对照**：同一数据、同一超参，分别用：
  - **A**：掩码 collator（`labels` 仅在 assistant 段）
  - **B**：全序列 loss（`labels` = 全部 `input_ids`）
- 各训固定 step 数（或到达同一 eval-loss 阈值）。
- **记录指标**：训练 loss 曲线、验证集 assistant-token loss、到达阈值所需 step 数、每 step 吞吐。
- **"20%" 定义** = 达到同等收敛（同一 eval-loss 阈值）所需 step 的减少比例。
- 产物：`docs/experiments/` 下的曲线图 + 表格，README 引用。

---

## 8. 测试策略（全部 CPU 可跑）

| 测试 | 断言 |
|---|---|
| `test_roles_masked` | system/user token → `-100`；assistant content → 原 id |
| `test_eos_kept` | assistant 段 `<|im_end|>` 保留 |
| `test_padding_masked` | padding → `-100` |
| `test_length_match` | `len(input_ids) == len(labels)` |
| `test_collator_batch` | 变长样本 padding 后形状正确、mask 不被 padding 破坏 |

单测依赖：`torch`(CPU)、`transformers`、`pytest`。不依赖 GPU、不依赖 flash-attn。

---

## 9. 成功标准

1. `ChatDataCollator` 通过全部单测。
2. 一条完整 SFT 样本能正确生成 `input_ids/labels`，人工抽查掩码正确。
3. A/B 实验脚本可运行、可复现，输出 loss 曲线与收敛对比表。
4. 上卡后能完整跑通 LoRA SFT，保存 adapter 供子项目 3 加载。

---

## 10. 非目标（YAGNI）

- 不做 sequence packing（可选增强，后续再议）。
- 不做全量微调、不做 4/8-bit 量化加载（LoRA + bf16 已足够）。
- 不实现自己的训练循环（交给 Trainer）。
- 不在本规格内涉及 GRPO / 手写算子（属子项目 2、3）。
