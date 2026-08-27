# 子项目 3：GRPO RL 对齐 — 设计规格

- 日期：2026-08-27
- 状态：待用户审阅
- 范围：本规格只覆盖「子项目 3（GRPO RL 对齐）」。子项目 1（SFT）与子项目 2（手写算子 + 推理加速）各自独立成规格。

---

## 1. 背景与目标

基于开源基座 **Qwen2.5-7B-Instruct**，在 **NVIDIA RTX 6000（48GB）** 单卡上，用 **GRPO（Group Relative Policy Optimization）** 做数学推理的 RLVR（Reinforcement Learning with Verifiable Rewards）对齐。

核心目标：

1. **手写 GRPO 核心**：组采样、组内优势归一化、clip 目标 + KL 惩罚，不依赖 TRL 的 `GRPOTrainer`（延续子项目 1「不配置现成类」的定位，面试能讲清每个公式）。
2. **规则奖励函数**：GSM8K 数学题，奖励 = 答案正确性 + 格式合规，纯规则可判，**无需奖励模型**。
3. **讲清「免 Critic」**：GRPO 用「组内相对优势」替代 PPO 的价值网络，省一个价值模型 = 省显存——这是本子项目的核心卖点。

**性质**：求职作品集 / 可复现实验。**诚实口径**——与子项目 2 一致：核心逻辑（奖励、优势、loss、采样）CPU 可测；7B 真实训练 GPU-defer，在用户配置 GPU 后上卡跑，**不虚构任何 reward / accuracy 数字**。

---

## 2. 技术栈

| 组件 | 选择 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-7B-Instruct`（策略与参考模型都从此初始化） |
| 训练框架 | `transformers` + `accelerate`；**手写 GRPO 核心**（loss/advantage/采样），不用 `trl.GRPOTrainer` |
| 微调方式 | **LoRA**（`peft`），`r=16, alpha=32, dropout=0.05`（沿用子项目 1，策略与参考共享冻结基座、只差 adapter） |
| 精度 | `bf16`（上卡后）；本地 CPU 单测用 `fp32` |
| 数据 | `openai/gsm8k`（数学应用题，答案格式 `#### <num>`） |
| 生成 | `transformers` 自回归生成（temperature=1.0, top_p=0.95）；vLLM 列为非目标 |

**明确不选**：TRL `GRPOTrainer`（核心退化为「配置现成类」）；奖励模型 / ORM（规则奖励已够、且 48GB 放不下 RM + 策略 + 参考）；全量微调（7B 全量优化器状态 + 参考模型在 48GB 放不下）。

---

## 3. 目录结构（Monorepo 中 grpo 子包）

```
Project2/
├── grpo/
│   ├── __init__.py
│   ├── reward.py        # 规则奖励：答案抽取 + 正确性 + 格式（组合）
│   ├── advantage.py     # 组内优势归一化（GRPO 免 Critic 的来源）
│   ├── loss.py          # 策略梯度 loss（clip 目标 + KL 惩罚）
│   ├── sampling.py      # 组采样：G 条 completion 的生成编排（GPU-defer）
│   ├── cache.py         # completion/reward 缓存（resume + 离线 replay）
│   ├── generate.py      # 采样生成（GPU-defer，复用 transformers）
│   ├── train.py         # 训练循环：生成→奖励→优势→loss→更新（GPU-defer）
│   └── config/
│       └── grpo_lora.yaml
├── tests/
│   ├── test_reward.py      # 答案抽取 / 正确性 / 格式规则
│   ├── test_advantage.py   # 组内归一化：均值0、方差1、组隔离、退化 std 保护
│   ├── test_loss.py        # clip / KL / 数值（CPU 小张量）
│   └── test_sampling.py    # 组采样形状 / log-prob 聚合
└── docs/superpowers/specs/ # 本规格
```

> 依赖方向：`grpo.loss → grpo.advantage`（loss 消费优势）；`grpo.reward`、`grpo.cache`、`grpo.advantage` 为纯函数模块、无外部依赖、CPU 可测。`grpo.sampling` / `grpo.generate` / `grpo.train` 依赖真实模型权重，GPU-defer。

---

## 4. 数据流

```
GSM8K 题目 prompt
    → 策略模型采样 G 条 completion（同 temperature，组内需有区分度）
    → 每条算规则奖励（正确性 + 格式）
    → 组内算 advantage（Â_i = (R_i − mean(R)) / std(R)）
    → 算 GRPO loss（clip 目标 + β·KL vs 冻结参考模型）
    → 反传更新策略（LoRA adapter）
    → 循环下一批 prompt
```

一条样本的形态：GSM8K 每条 = `{"question": "…", "answer": "…#### <num>"}`。prompt 由 question 套一个固定指令模板构成，例如：

```
Solve the following math problem step by step, then give the final answer
after "####".\n\nQuestion: {question}\n
```

> 指令模板是格式奖励的依据（§6.3）：模板引导模型用 `#### <num>` 或 `<answer>...</answer>` 输出，奖励端才能可靠抽取。

---

## 5. GRPO 核心算法（本子项目中心）

### 5.1 组采样

对每个 prompt `q`，用当前策略采样 G 条 completion `o_1..o_G`（同 temperature）。G 越大优势方差估计越稳，但生成成本线性上升。

### 5.2 组内相对优势（免 Critic 的关键）

对同一 prompt 的 G 条 completion，按组内奖励算相对优势：

```
Â_i = (R_i − mean(R_1..R_G)) / std(R_1..R_G)
```

- **组内做**：每条 advantage 只相对同 prompt 的其他 completion，不需要绝对价值。
- **这就是 GRPO 对比 PPO 的关键**：PPO 需要一个 Critic/价值网络来估计优势 `A = R − V(s)`；GRPO 用「同组奖励的相对位置」替代，省掉整个价值模型 = 省显存、省一个模型的训练。

### 5.3 策略梯度目标（clip + KL）

```
J_GRPO(θ) = E_q,o ~ π_θ_old [ (1/G) Σ_i ( min(ρ_i Â_i, clip(ρ_i, 1−ε, 1+ε) Â_i) − β·KL(π_θ ‖ π_ref) ) ]
```

- `ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)`：重要性采样比。`π_θ_old` 是**生成时刻**的旧策略（log-prob 冻结、detach），`π_θ` 是当前策略（参与梯度）。
- `clip(ρ, 1−ε, 1+ε)`，`ε=0.2`：限制更新幅度，防单步跑偏。
- `KL(π_θ ‖ π_ref)`：对**冻结参考模型** `π_ref` 的 KL 惩罚，防策略离初始太远（reward hacking / 模式崩塌）。
- **只更新策略**（LoRA adapter）；`π_ref` 冻结、只前向。

### 5.4 KL 估计（v1 实现）

用逐 token 的 log-prob 差估计：`KL ≈ Σ_t (log π_θ(o_t) − log π_ref(o_t))`（序列内求和）。更稳的 k3 估计器 `π_ref/π_θ − log(π_ref/π_θ) − 1` 列为后续可选，不在 v1 范围。

### 5.5 接口

```python
# grpo/advantage.py
def group_advantage(rewards: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """rewards: (N*G,) 展平，按每组 group_size 归一化，返回 (N*G,)。"""

# grpo/loss.py
def grpo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
              ref_log_probs: torch.Tensor, advantages: torch.Tensor, *,
              clip_epsilon: float = 0.2, beta: float = 0.01) -> torch.Tensor:
    """逐 completion 的 clip 目标 − β·KL，返回标量 loss。"""
```

---

## 6. 奖励函数（规则、组合）

### 6.1 答案抽取（优先 GSM8K 原生，fallback 结构化标签）

GSM8K 原生答案是 `#### <num>`。抽取顺序：

1. 优先正则匹配 `####\s*(-?\d+(?:\.\d+)?)` 取数值；
2. 找不到再 fallback 匹配 `<answer>...</answer>` 内的文本。

两者都解析，取第一个命中；都未命中则正确性 0。

### 6.2 正确性奖励（主导）

抽取答案与 ground-truth 数值/字符串匹配（数值做小数归一化：`42` == `42.0`）→ 1 或 0。

### 6.3 格式奖励（引导可解析性）

回答落在约定结构（`<answer>...</answer>` 或直接 `#### <num>`）→ 固定小加分 **+0.05**。加分远小于正确性 1.0，避免模型牺牲正确性去凑格式。

### 6.4 组合奖励

```
reward = correctness (0/1) + format (0/0.05)，范围 [0, 1.05]
```

纯规则可判，**无需奖励模型**。

---

## 7. 训练配置（48GB 单卡，bf16）

| 项 | 值 |
|---|---|
| LoRA | `r=16, alpha=32, dropout=0.05`（策略与参考共享冻结基座、只差 adapter） |
| 组大小 G | **默认 4**，可配；**最低 4**（G=2 方差估计不稳）；显存余量足可调至 8（生成成本翻倍） |
| 每步 prompt 数 | 4（每步 = 4 × G = 16 条 completion） |
| 梯度累积 | `gradient_accumulation_steps=2`（等效 batch 32 条 completion） |
| KL 系数 β | **0.01 起**，可调；reward 升但 KL 爆炸则调高 |
| clip ε | 0.2 |
| 采样温度 / top_p | temperature=1.0, top_p=0.95（组内需有区分度；温度太低 → 组内奖励方差小 → 优势归一化失效） |
| 学习率 | `1e-5`（RL 阶段低于 SFT 的 2e-4，防跑偏），cosine 调度 |
| 精度 / 显存 | `bf16` + `gradient_checkpointing`；策略 + 参考共享冻结基座，只差 LoRA adapter |
| 数据 | `openai/gsm8k`（train split 约 7.5k，可配置 `--max_samples` 取子集） |

> **参考模型**：本子项目不接子项目 1 的 SFT adapter（那是中文通用对话，对数学不是好 init，已确认）。`π_ref` = 冻结的 base `Qwen2.5-7B-Instruct`。诚实注明：Qwen 真实数学 RLVR 在 GRPO 前还有一步「数学 SFT」，本子项目跳过该步、直接从 Instruct 做 GRPO，作为「免 Critic RLVR 机制」的可复现演示。

---

## 8. completion/reward 缓存（resume + 离线 replay）

`grpo/cache.py` 提供轻量缓存，把 `(prompt, completion) → {reward, log_prob}` 持久化到 JSONL：

- **resume**：重启训练时，同一 prompt+completion 组合已评过分则直接取缓存，不重复生成/重复打分。
- **离线 replay**：加载缓存轨迹，反复重算 advantage / loss 做调试，**无需重新生成**（生成是 GPU 上最贵的一步）。
- 实现极轻：内存 dict + JSONL 追加写，不引入数据库/序列化框架。

> 诚实澄清：奖励函数本身是规则解析（近乎免费），真正贵的是**生成**。缓存存的是「completion + 其奖励 + 其 log-prob」，而非只缓存奖励——这样 resume 才能跳过生成、debug 才能离线 replay。

---

## 9. 测试策略（全 CPU 可跑）

| 测试 | 断言 |
|---|---|
| `test_reward.py` | 抽取：`#### 42` / `<answer>42</answer>` / 无答案各分支正确；数值归一化 `42 == 42.0`；格式加分 +0.05；组合 reward 范围 [0,1.05] |
| `test_advantage.py` | 组内归一化后每组均值≈0、std≈1；不同组互不干扰；`std=0`（G 条同奖励）退化为 0 不除零 |
| `test_loss.py` | 与论文公式**逐项数值对拍**（手算小张量 case）：clip 生效时取 clip 值、未生效时取原值；KL 项为 log-prob 差之和 × β；`ρ=1` 时退化为纯优势 |
| `test_sampling.py` | 组采样形状 `(N*G, seq_len)`、`(N*G,)` 的 log-prob 为**序列 token log-prob 之和**（求和，非平均）、分组索引正确 |

单测依赖：`torch`(CPU)、`pytest`。不依赖 GPU、不依赖 `transformers` 权重（loss/advantage/reward 用纯张量）。真实生成/训练 GPU-defer。

---

## 10. 成功标准

1. `grpo/reward.py`、`advantage.py`、`loss.py`、`sampling.py` 全部 CPU 单测通过。
2. 手写 loss/advantage 与论文公式逐项数值对拍一致（手算小张量 case）。
3. 上卡后能跑通至少 1 个 GSM8K 子集的 GRPO 迭代，reward 有提升趋势（数字真实，不虚构）。

---

## 11. 非目标（YAGNI）

- 不用 TRL `GRPOTrainer`、不用 vLLM / 分布式 / 多卡。
- 不做奖励模型 / ORM / 过程监督（PRM）。
- 不做全量微调、不做 4/8-bit 量化加载（LoRA + bf16 已足够）。
- 不做 length penalty / repetition penalty / 复杂组合奖励（v1 只正确性 + 格式）。
- 不接子项目 1 的 SFT adapter（域不匹配，已确认）。
- 不做 beam search / speculative decoding（生成走 HF 采样）。
