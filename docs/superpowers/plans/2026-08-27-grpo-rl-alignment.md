# GRPO RL 对齐 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 手写 GRPO 核心（组内优势归一化 + clip 目标 + KL 惩罚）、规则奖励（正确性 + 格式）、completion 缓存，以及 GPU-defer 的生成/训练循环，全部 CPU 单测覆盖核心逻辑。

**Architecture:** `grpo/` 子包按职责拆分——`reward`/`advantage`/`loss`/`cache`/`sampling` 为纯函数模块（CPU 可测），`generate`/`train` 为 GPU-defer 脚手架（上卡运行，不虚构数字）。任务 1–5 互相独立、各自带测试；任务 6 消费前五个模块串成训练循环。

**Tech Stack:** PyTorch（CPU 单测）、pytest、PyYAML（config）、transformers/peft/datasets（仅 GPU-defer 的 generate/train，惰性导入）。

## Global Constraints

- 基座 `Qwen/Qwen2.5-7B-Instruct`；LoRA `r=16, alpha=32, dropout=0.05`，target=`[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`。
- GRPO 超参：`group_size` 默认 4（最低 4）、`clip_epsilon=0.2`、`beta=0.01`、`temperature=1.0`、`top_p=0.95`。
- 组合奖励 = 正确性 (0/1) + 格式 (`+0.05`)，范围 `[0, 1.05]`；答案抽取**优先 `#### <num>`，fallback `<answer>...</answer>`**。
- log-prob 聚合用**求和**（非平均）；优势归一化**组内**做、`std=0` 退化不除零。
- **不引入** `trl.GRPOTrainer`、vLLM、奖励模型/ORM、全量微调（见 spec §11）。
- 核心逻辑 CPU 可测、不依赖 GPU、不依赖模型权重；生成/训练 GPU-defer、不虚构数字。
- 单测命令：`.venv/bin/python -m pytest`；提交只 `git add` 计划内文件，**绝不 `git add docs/interview-prep-*.md`**（保持 untracked）。

---

### Task 1: 规则奖励函数（reward.py）

**Files:**
- Create: `grpo/__init__.py`
- Create: `grpo/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: 无。
- Produces: `extract_answer(completion) -> str | None`、`correctness_reward(completion, gold) -> float`、`format_reward(completion) -> float`、`combined_reward(completion, gold) -> float`、常量 `FORMAT_BONUS = 0.05`。

- [ ] **Step 1: 写失败测试**

创建 `grpo/__init__.py`：

```python
"""GRPO RL 对齐（子项目 3）：手写 GRPO 核心 + 规则奖励 + 训练循环。"""
```

创建 `tests/test_reward.py`：

```python
from grpo.reward import (FORMAT_BONUS, combined_reward, correctness_reward,
                         extract_answer, format_reward)


def test_extract_hash_answer():
    assert extract_answer("Some reasoning... #### 42") == "42"


def test_extract_answer_tag_fallback():
    assert extract_answer("<answer> 42 </answer>") == "42"


def test_extract_negative_decimal():
    assert extract_answer("#### -3.5") == "-3.5"


def test_extract_missing_returns_none():
    assert extract_answer("no answer here") is None


def test_correctness_normalizes_decimals():
    assert correctness_reward("#### 42.0", "42") == 1.0
    assert correctness_reward("#### 42", "42.0") == 1.0


def test_correctness_wrong_answer():
    assert correctness_reward("#### 43", "42") == 0.0


def test_correctness_missing_answer():
    assert correctness_reward("no answer", "42") == 0.0


def test_format_reward_bonus():
    assert format_reward("#### 42") == FORMAT_BONUS
    assert format_reward("<answer>42</answer>") == FORMAT_BONUS
    assert format_reward("no answer") == 0.0


def test_combined_reward_range():
    assert combined_reward("#### 42", "42") == 1.0 + FORMAT_BONUS   # 1.05
    assert combined_reward("#### 43", "42") == 0.0 + FORMAT_BONUS   # 0.05 错但格式对
    assert combined_reward("no answer", "42") == 0.0
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_reward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.reward'`

- [ ] **Step 3: 写实现**

创建 `grpo/reward.py`：

```python
"""GRPO 规则奖励：答案抽取 + 正确性 + 格式（组合，无需奖励模型）。"""
import re

# GSM8K 原生答案：#### <num>（优先级最高）
_NUM_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
# 结构化 fallback：<answer>...</answer>
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

FORMAT_BONUS = 0.05


def _normalize_number(s: str) -> str:
    """数值归一化：去千分位逗号，42 == 42.0。"""
    s = s.strip().replace(",", "")
    try:
        return f"{float(s):.10g}"
    except ValueError:
        return s.strip()


def extract_answer(completion: str) -> str | None:
    """抽取最终答案：优先 #### <num>，fallback <answer>...</answer>。未命中返回 None。"""
    m = _NUM_RE.search(completion)
    if m:
        return m.group(1)
    m = _ANSWER_TAG_RE.search(completion)
    return m.group(1).strip() if m else None


def correctness_reward(completion: str, gold: str) -> float:
    """正确性奖励：抽取答案与 gold 数值归一化比对 → 1 或 0。"""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_number(pred) == _normalize_number(gold) else 0.0


def format_reward(completion: str) -> float:
    """格式奖励：命中 #### 或 <answer> 结构 → +FORMAT_BONUS，否则 0。"""
    return FORMAT_BONUS if extract_answer(completion) is not None else 0.0


def combined_reward(completion: str, gold: str) -> float:
    """组合奖励 = 正确性 (0/1) + 格式 (0/0.05)，范围 [0, 1.05]。"""
    return correctness_reward(completion, gold) + format_reward(completion)
```

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_reward.py -q`
Expected: `9 passed`

- [ ] **Step 5: 提交**

```bash
git add grpo/__init__.py grpo/reward.py tests/test_reward.py
git commit -m "feat(grpo): 规则奖励（答案抽取 + 正确性 + 格式组合）"
```

---

### Task 2: 组内优势归一化（advantage.py）

**Files:**
- Create: `grpo/advantage.py`
- Test: `tests/test_advantage.py`

**Interfaces:**
- Consumes: 无（纯张量）。
- Produces: `group_advantage(rewards: torch.Tensor, *, group_size: int) -> torch.Tensor`（`(N*G,) -> (N*G,)`，按组归一化）。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_advantage.py`：

```python
import pytest
import torch

from grpo.advantage import group_advantage


def test_group_normalizes_to_zero_mean_unit_std():
    rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])
    adv = group_advantage(rewards, group_size=2)
    assert adv.shape == rewards.shape
    for g in range(2):
        group = adv[g * 2:(g + 1) * 2]
        assert torch.allclose(group.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(group.std(), torch.tensor(1.0), atol=1e-6)


def test_groups_are_isolated():
    rewards = torch.tensor([0.0, 1.0, 100.0, 101.0])
    adv = group_advantage(rewards, group_size=2)
    # 两组内部归一化互不影响：归一化结果相同
    assert torch.allclose(adv[:2], adv[2:])


def test_degenerate_zero_std():
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = group_advantage(rewards, group_size=2)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_raises_on_non_divisible():
    with pytest.raises(ValueError):
        group_advantage(torch.tensor([1.0, 2.0, 3.0]), group_size=2)
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_advantage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.advantage'`

- [ ] **Step 3: 写实现**

创建 `grpo/advantage.py`：

```python
"""GRPO 组内相对优势归一化（免 Critic 的关键）。"""
import torch


def group_advantage(rewards: torch.Tensor, *, group_size: int) -> torch.Tensor:
    """按每组 group_size 做 (R − mean) / std 归一化。

    rewards: (N*G,) 展平，前 group_size 个属第 0 组，依次类推。
    返回同形状的 advantage。std=0（组内奖励全同）的组退化为 0，不除零。
    """
    rewards = rewards.float()
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards 长度 {rewards.numel()} 不能被 group_size {group_size} 整除"
        )
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    adv = (grouped - mean) / safe_std
    return adv.view(-1)
```

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_advantage.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add grpo/advantage.py tests/test_advantage.py
git commit -m "feat(grpo): 组内相对优势归一化（免 Critic）"
```

---

### Task 3: 策略梯度 loss（loss.py）

**Files:**
- Create: `grpo/loss.py`
- Test: `tests/test_loss.py`

**Interfaces:**
- Consumes: 优势张量（由 `advantage.group_advantage` 产出，本任务以入参接收）。
- Produces: `grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, *, clip_epsilon=0.2, beta=0.01) -> torch.Tensor`（标量）。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_loss.py`：

```python
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
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_loss.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.loss'`

- [ ] **Step 3: 写实现**

创建 `grpo/loss.py`：

```python
"""GRPO 策略梯度目标：clip 目标 + KL 惩罚（对冻结参考模型）。"""
import torch


def grpo_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor,
              ref_log_probs: torch.Tensor, advantages: torch.Tensor, *,
              clip_epsilon: float = 0.2, beta: float = 0.01) -> torch.Tensor:
    """逐 completion 的 GRPO 目标，返回标量 loss。

    log_probs / old_log_probs / ref_log_probs: (N*G,) 当前策略 / 旧策略 / 参考模型的序列 sum log-prob。
    advantages: (N*G,) 组内优势。
    old_log_probs 与 ref_log_probs 应已 detach（由调用方保证）。
    """
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl = log_probs - ref_log_probs  # 序列级 log-prob 差之和
    return policy_loss + beta * kl.mean()
```

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_loss.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add grpo/loss.py tests/test_loss.py
git commit -m "feat(grpo): 策略梯度 loss（clip 目标 + KL 惩罚）"
```

---

### Task 4: completion/reward 缓存（cache.py）

**Files:**
- Create: `grpo/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: 无。
- Produces: `CompletionCache(path)`，方法 `get(prompt, completion) -> dict | None`、`put(prompt, completion, reward, log_prob) -> None`、`__len__`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_cache.py`：

```python
from grpo.cache import CompletionCache


def test_put_get_roundtrip(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    c.put("q", "a", 1.05, -3.2)
    assert c.get("q", "a") == {"reward": 1.05, "log_prob": -3.2}


def test_persistence_across_instances(tmp_path):
    p = str(tmp_path / "cache.jsonl")
    CompletionCache(p).put("q", "a", 1.0, -1.0)
    c2 = CompletionCache(p)  # 从磁盘重新加载
    assert c2.get("q", "a") == {"reward": 1.0, "log_prob": -1.0}


def test_miss_returns_none(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    assert c.get("q", "missing") is None


def test_len(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    c.put("q1", "a", 1.0, 0.0)
    c.put("q2", "b", 0.5, -1.0)
    assert len(c) == 2
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.cache'`

- [ ] **Step 3: 写实现**

创建 `grpo/cache.py`：

```python
"""completion/reward 缓存：resume + 离线 replay。"""
import json
import os


class CompletionCache:
    """把 (prompt, completion) → {"reward", "log_prob"} 持久化到 JSONL。

    - resume：重启训练时同一 prompt+completion 已评过分则直接取缓存，不重复生成/打分。
    - 离线 replay：加载缓存轨迹重算 advantage/loss 做调试，无需重新生成。
    """

    def __init__(self, path: str):
        self.path = path
        self._data: dict[tuple[str, str], dict] = {}
        self._load()

    @staticmethod
    def _key(prompt: str, completion: str) -> tuple[str, str]:
        return (prompt, completion)

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._data[(rec["prompt"], rec["completion"])] = {
                    "reward": rec["reward"], "log_prob": rec["log_prob"]}

    def get(self, prompt: str, completion: str):
        return self._data.get(self._key(prompt, completion))

    def put(self, prompt: str, completion: str, reward: float, log_prob: float):
        self._data[self._key(prompt, completion)] = {"reward": reward, "log_prob": log_prob}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "reward": reward, "log_prob": log_prob}) + "\n")

    def __len__(self):
        return len(self._data)
```

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add grpo/cache.py tests/test_cache.py
git commit -m "feat(grpo): completion/reward 缓存（resume + 离线 replay）"
```

---

### Task 5: 组采样纯函数核心（sampling.py）

**Files:**
- Create: `grpo/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: 无（纯张量）。
- Produces: `sum_log_probs(token_log_probs) -> torch.Tensor`（`(N, T) -> (N,)`，求和）、`group_indices(num_prompts, group_size) -> torch.Tensor`（`(N*G,)`）。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_sampling.py`：

```python
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
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_sampling.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.sampling'`

- [ ] **Step 3: 写实现**

创建 `grpo/sampling.py`：

```python
"""GRPO 组采样：纯函数核心（log-prob 聚合 + 分组索引）；生成编排在 generate.py。"""
import torch


def sum_log_probs(token_log_probs: torch.Tensor) -> torch.Tensor:
    """把逐 token log-prob 求和为序列级 log-prob（求和，非平均）。

    token_log_probs: (N, seq_len)。返回 (N,)。
    """
    return token_log_probs.sum(dim=1)


def group_indices(num_prompts: int, group_size: int) -> torch.Tensor:
    """返回每个 completion 所属 prompt 组的下标。

    展平布局：prompt 0 的 G 条在前、prompt 1 的 G 条在后，依次类推。
    返回 (num_prompts*group_size,) 的组下标（0..num_prompts-1）。
    """
    return torch.arange(num_prompts).repeat_interleave(group_size)
```

- [ ] **Step 4: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_sampling.py -q`
Expected: `3 passed`

- [ ] **Step 5: 提交**

```bash
git add grpo/sampling.py tests/test_sampling.py
git commit -m "feat(grpo): 组采样纯函数核心（log-prob 求和 + 分组索引）"
```

---

### Task 6: 配置 + 生成 + 训练循环（GPU-defer）

**Files:**
- Create: `grpo/config/grpo_lora.yaml`
- Create: `grpo/generate.py`
- Create: `grpo/train.py`
- Test: `tests/test_grpo_config.py`

**Interfaces:**
- Consumes: `grpo.reward.combined_reward` / `extract_answer`、`grpo.advantage.group_advantage`、`grpo.loss.grpo_loss`、`grpo.cache.CompletionCache`、`grpo.generate.sample_completions`。
- Produces: `GRPOConfig.from_yaml(path)`、`build_prompt(question) -> str`、`main(config)`、`sample_completions(...)`。

> 说明：`generate.py` / `train.py` 依赖真实模型权重，本任务 CPU 上只验证「config 解析正确」与「模块可导入（不加载权重）」，其余 GPU-defer。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_grpo_config.py`：

```python
from grpo.train import GRPOConfig


def test_from_yaml():
    cfg = GRPOConfig.from_yaml("grpo/config/grpo_lora.yaml")
    assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.group_size == 4
    assert cfg.beta == 0.01
    assert cfg.clip_epsilon == 0.2
    assert "q_proj" in cfg.lora_target
```

- [ ] **Step 2: 跑测试确认失败（RED）**

Run: `.venv/bin/python -m pytest tests/test_grpo_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grpo.train'`

- [ ] **Step 3: 写配置**

创建 `grpo/config/grpo_lora.yaml`：

```yaml
model_id: "Qwen/Qwen2.5-7B-Instruct"
dataset_name: "openai/gsm8k"
max_samples: 1000
output_dir: "outputs/grpo-lora"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

group_size: 4
prompts_per_step: 4
grad_accum: 2
clip_epsilon: 0.2
beta: 0.01

temperature: 1.0
top_p: 0.95
max_new_tokens: 256

lr: 1.0e-5
num_epochs: 1
bf16: true
grad_checkpoint: true
```

- [ ] **Step 4: 写生成模块（GPU-defer）**

创建 `grpo/generate.py`：

```python
"""GRPO 采样生成：对每个 prompt 采样 G 条 completion（GPU-defer）。"""
from __future__ import annotations

import torch


@torch.no_grad()
def sample_completions(model, tokenizer, prompts, *, group_size: int,
                       temperature: float = 1.0, top_p: float = 0.95,
                       max_new_tokens: int = 256, device: str = "cuda"):
    """对 prompts 各采样 group_size 条 completion，返回 (completions, old_log_probs)。

    completions: (len(prompts)*group_size,) 展平，prompt 0 的 G 条在前。
    old_log_probs: (len(prompts)*group_size,) 每条 completion 的序列 sum log-prob（生成时刻、已 detach）。
    """
    completions: list[str] = []
    all_log_probs: list[float] = []
    for prompt in prompts:
        inputs = tokenizer([prompt] * group_size, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[:, prompt_len:]
        for i in range(group_size):
            completions.append(tokenizer.decode(gen_ids[i], skip_special_tokens=True))
        all_log_probs.extend(_seq_log_probs(out, prompt_len).detach().tolist())
    return completions, torch.tensor(all_log_probs)


@torch.no_grad()
def compute_seq_log_probs(model, tokenizer, prompts, completions, device: str = "cuda"):
    """对 (prompt, completion) 拼接序列前向，返回每条 completion 部分的 sum log-prob。

    用于计算「当前策略」与「参考模型」的序列 log-prob（loss 里的 log_probs / ref_log_probs）。
    """
    import torch.nn.functional as F

    full_texts = [p + c for p, c in zip(prompts, completions)]
    enc = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    input_ids = enc["input_ids"]
    mask = enc["attention_mask"]
    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1)  # (B,)

    logits = model(input_ids=input_ids, attention_mask=mask).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)

    totals = torch.zeros(input_ids.shape[0], device=device)
    for i in range(input_ids.shape[0]):
        start = int(prompt_lens[i].item()) - 1      # 第一个 completion token 的预测位置
        end = int(mask[i].sum().item()) - 1         # 最后一个有效 token 的预测位置
        if end <= start:
            continue
        target = input_ids[i, start + 1:end + 1]
        lp = log_probs[i, start:end].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        totals[i] = lp.sum()
    return totals


def _seq_log_probs(gen_output, prompt_len: int) -> torch.Tensor:
    """从 generate 的 scores 反算每条序列的 sum log-prob。"""
    import torch.nn.functional as F

    scores = torch.stack(gen_output.scores, dim=1)  # (B, new_tokens, vocab)
    log_softmax = F.log_softmax(scores.float(), dim=-1)
    gen_ids = gen_output.sequences[:, prompt_len:]
    per_token = log_softmax.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    return per_token.sum(dim=1)
```

- [ ] **Step 5: 写训练循环（GPU-defer）**

创建 `grpo/train.py`：

```python
"""GRPO 训练循环：生成→奖励→优势→loss→更新（GPU-defer）。"""
import argparse
from dataclasses import dataclass, field as dc_field

import yaml


@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "openai/gsm8k"
    max_samples: int | None = None
    output_dir: str = "outputs/grpo-lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: tuple = dc_field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    group_size: int = 4
    prompts_per_step: int = 4
    grad_accum: int = 2
    clip_epsilon: float = 0.2
    beta: float = 0.01
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    lr: float = 1e-5
    num_epochs: int = 1
    bf16: bool = True
    grad_checkpoint: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_prompt(question: str) -> str:
    """把 GSM8K 题目包成固定指令模板（格式奖励的依据）。"""
    return (f"Solve the following math problem step by step, then give the final "
            f'answer after "####".\n\nQuestion: {question}\n')


def main(config: GRPOConfig):
    """GRPO 训练循环。上卡后运行；本函数不在 CPU 上执行。"""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model

    from core.tokenizer_utils import load_tokenizer
    from grpo.reward import combined_reward, extract_answer
    from grpo.advantage import group_advantage
    from grpo.loss import grpo_loss
    from grpo.cache import CompletionCache
    from grpo.generate import compute_seq_log_probs, sample_completions

    tokenizer = load_tokenizer(config.model_id)
    ds = load_dataset(config.dataset_name, "main", split="train")
    if config.max_samples is not None:
        ds = ds.select(range(config.max_samples))

    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.bfloat16)
    if config.grad_checkpoint:
        model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=list(config.lora_target))
    model = get_peft_model(model, lora)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16).eval()

    cache = CompletionCache(f"{config.output_dir}/completion_cache.jsonl")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    prompts = [build_prompt(ex["question"]) for ex in ds]
    golds = [extract_answer(ex["answer"]) for ex in ds]

    model.train()
    for epoch in range(config.num_epochs):
        for start in range(0, len(prompts), config.prompts_per_step):
            batch_prompts = prompts[start:start + config.prompts_per_step]
            batch_golds = golds[start:start + config.prompts_per_step]

            # 1) 组采样（G 条 / prompt）
            completions, old_log_probs = sample_completions(
                model, tokenizer, batch_prompts, group_size=config.group_size,
                temperature=config.temperature, top_p=config.top_p,
                max_new_tokens=config.max_new_tokens)

            # 2) 奖励（带缓存）：completion 展平布局 = prompt0 的 G 条在前
            gold_rep = [g for g in batch_golds for _ in range(config.group_size)]
            prompt_rep = [p for p in batch_prompts for _ in range(config.group_size)]
            rewards = []
            for prompt, completion, gold in zip(prompt_rep, completions, gold_rep):
                hit = cache.get(prompt, completion)
                if hit is not None:
                    rewards.append(hit["reward"])
                else:
                    r = combined_reward(completion, gold)
                    cache.put(prompt, completion, r, 0.0)  # log_prob 稍后回填
                    rewards.append(r)
            rewards = torch.tensor(rewards)

            # 3) 优势 + 三路 log-prob
            advantages = group_advantage(rewards, group_size=config.group_size)
            log_probs = compute_seq_log_probs(model, tokenizer, prompt_rep, completions)
            ref_log_probs = compute_seq_log_probs(ref_model, tokenizer, prompt_rep, completions)

            # 4) loss（grad_accum）
            loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages,
                             clip_epsilon=config.clip_epsilon, beta=config.beta)
            (loss / config.grad_accum).backward()

            if (start // config.prompts_per_step + 1) % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(f"epoch={epoch} step={start // config.prompts_per_step} "
                  f"loss={loss.item():.4f} mean_reward={rewards.mean().item():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="grpo/config/grpo_lora.yaml")
    main(GRPOConfig.from_yaml(p.parse_args().config))
```

- [ ] **Step 6: 跑测试确认通过（GREEN）**

Run: `.venv/bin/python -m pytest tests/test_grpo_config.py -q`
Expected: `1 passed`

- [ ] **Step 7: 模块可导入 smoke 检查（CPU，不加载权重）**

Run: `.venv/bin/python -c "import grpo.generate, grpo.train; from grpo.train import GRPOConfig, build_prompt; print(build_prompt('What is 2+2?'))"`
Expected: 打印出包了模板的 prompt，无 ImportError。

- [ ] **Step 8: 全量回归**

Run: `.venv/bin/python -m pytest -q`
Expected: `70 passed`（原 45 + 新 25 = 9+4+4+4+3+1；以实际计数为准，全部通过即可）

- [ ] **Step 9: 提交**

```bash
git add grpo/config/grpo_lora.yaml grpo/generate.py grpo/train.py tests/test_grpo_config.py
git commit -m "feat(grpo): 训练配置 + 采样生成 + 训练循环（GPU-defer）"
```

---

## 自审（plan vs spec）

- **spec 覆盖**：§5（GRPO 核心）→ Task 2/3/5；§6（奖励）→ Task 1；§8（缓存）→ Task 4；§3（generate/train/config）→ Task 6；§9（测试）→ 各 Task 的 test 文件。
- **占位扫描**：无 TBD/TODO，所有超参、公式、代码均完整给出。
- **类型一致性**：`group_advantage(rewards, *, group_size)`、`grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, *, clip_epsilon, beta)`、`sample_completions`、`compute_seq_log_probs` 在任务间签名一致；`GRPOConfig` 字段名与 yaml key 一一对应。
- **已知取舍（与 spec 的轻微差异，不阻塞）**：spec §3 把「组采样编排」归 sampling.py，本计划把「生成编排」落到 generate.py（`sample_completions`），sampling.py 只保留纯函数核心——职责更清晰；GPU-defer 的 `train.py` 中 `cache.put(..., log_prob=0.0)` 是占位回填，上卡时用 `old_log_probs` 回填。
