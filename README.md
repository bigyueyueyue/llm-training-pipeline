# Qwen2.5-7B-Instruct 微调 + 推理加速 + GRPO 对齐

基于开源基座 **Qwen2.5-7B-Instruct**，在 **NVIDIA RTX 6000（48GB）** 单卡上搭建的 LLM 微调与对齐工程。求职作品集项目，兼顾「可复现实验」与「讲得清原理」。

Monorepo 分三个子项目（当前已实现子项目 1）：

| 子项目 | 内容 | 状态 |
|---|---|---|
| 1. SFT 微调链路 | 自定义多轮掩码 DataCollator + LoRA + 收敛对比实验 | ✅ 已实现 |
| 2. 手写算子 + 推理加速 | RoPE / GQA 手写验证、FlashAttention-2、KV Cache 压测 | 规划中 |
| 3. GRPO RL 对齐 | 组合奖励函数、免 Critic 低显存对齐 | 规划中 |

---

## 子项目 1：SFT 微调链路

### 核心亮点

- **自定义多轮对话掩码 `ChatDataCollator`**：loss 仅落在 assistant 回答上，system/user 与格式头部全部 `-100`，`<|im_end|>` 保留可学习。手写 per-turn tokenization，并用「与官方 chat 模板输出一致」的守卫测试保证正确性。
- **可复现收敛实验**：掩码 vs 全序列 loss 的 A/B 对比（`run_masking_ab.py`），用「达到同一 eval-loss 阈值所需 step 数」量化收敛收益。

### 诚实声明（避免面试穿帮）

掩码优化的是 **loss / 梯度信号**——梯度只来自回答、不在 prompt 上产生无效更新，因此收敛更快。它**不直接省 FLOPs**。真正节省「历史 prompt 计算」的技术是 **sequence packing**（合并短样本 + attention mask 隔离），列为后续可选增强。

### 目录结构

```
core/    tokenizer 加载、收敛指标（共享）
sft/     数据、collator、训练、A/B 实验（子项目 1）
tests/   全部 CPU 可跑的单元测试
scripts/ 环境脚本
docs/    设计规格、实施计划、实验记录
```

### 环境

```bash
bash scripts/setup_env.sh dev    # 本地 CPU 单测依赖
bash scripts/setup_env.sh train  # 训练依赖（不含 flash-attn，GPU 上单独装）
```

### 运行

```bash
# 单测（CPU）
.venv/bin/python -m pytest -v

# SFT 训练（GPU，上卡后）
.venv/bin/python sft/train.py --config sft/config/sft_lora.yaml

# 收敛 A/B 实验（GPU，输出 docs/experiments/）
.venv/bin/python sft/run_masking_ab.py --steps 200
```
