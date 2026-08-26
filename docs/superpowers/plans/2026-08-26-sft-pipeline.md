# SFT 微调链路 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建基于 Qwen2.5-7B-Instruct 的 LoRA SFT 链路，核心交付一个自定义多轮对话掩码 `ChatDataCollator`，并配可复现的「掩码 vs 全序列 loss」收敛对比实验。

**Architecture:** Monorepo 内拆 `core/`（共享：tokenizer 加载、收敛指标）与 `sft/`（数据、collator、训练、A/B 实验）。手写掩码逻辑用 per-turn tokenization 生成 `input_ids`+`labels`，`ChatDataCollator` 只负责 padding；训练循环交给 `transformers.Trainer`。全部核心逻辑 CPU 可单测；全量训练 GPU 后置。

**Tech Stack:** PyTorch(CPU) + transformers + datasets + peft + accelerate；pytest；uv（Python 3.12）。GPU 侧另加 flash-attn-2。

## Global Constraints

- Python 版本固定 **3.12**（3.13/3.14 无稳定 torch wheel）；用 `uv venv --python 3.12` 建环境。
- 基座模型 ID：`Qwen/Qwen2.5-7B-Instruct`（tokenizer 无需鉴权，可直接下载）。
- 掩码忽略值 `IGNORE_INDEX = -100`；assistant 段 `<|im_end|>` 必须保留（可学习）。
- 目录结构严格按 [设计规格](2026-08-26-sft-pipeline-design.md) §3。
- 所有单测在 CPU 上跑通；训练/压测脚本上卡后跑，本地只做 smoke test（不加载 7B 权重）。
- 依赖声明：`requirements-dev.txt`（CPU 单测）与 `requirements-train.txt`（训练）；flash-attn 仅 GPU 装，不在本机装。

---

## File Structure

```
Project2/
├── pyproject.toml              # pytest 配置（pythonpath=root）
├── requirements-dev.txt        # torch/transformers/pytest
├── requirements-train.txt      # + datasets/peft/accelerate/matplotlib/pyyaml
├── core/
│   ├── __init__.py
│   ├── tokenizer_utils.py      # load_tokenizer
│   └── metrics.py              # ema_smooth / steps_to_threshold
├── sft/
│   ├── __init__.py
│   ├── data.py                 # build_chat_sample / build_chat_sample_full / load_multi_turn_dataset
│   ├── collator.py             # ChatDataCollator
│   ├── train.py                # SFTConfig + LoRA 训练入口
│   ├── run_masking_ab.py       # 掩码 vs 全序列 loss A/B 实验
│   └── config/sft_lora.yaml
├── tests/
│   ├── conftest.py             # tokenizer fixture
│   ├── test_labels.py
│   ├── test_collator.py
│   └── test_metrics.py
└── scripts/setup_env.sh
```

---

### Task 1: 环境与项目脚手架

**Files:**
- Create: `pyproject.toml`, `requirements-dev.txt`, `requirements-train.txt`, `scripts/setup_env.sh`, `core/__init__.py`, `sft/__init__.py`

**Interfaces:**
- Produces: 可导入的 `core`/`sft` 包（pytest 从根目录 `pythonpath=["."]` 解析）；`.venv` 虚拟环境（Python 3.12 + dev 依赖）。

- [ ] **Step 1: 写配置文件**

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`requirements-dev.txt`（能跑全部 CPU 单测的最小集）:
```
torch>=2.4
transformers>=4.45
tokenizers>=0.20
datasets>=2.20
peft>=0.12
pyyaml>=6.0
pytest>=8.0
```

`requirements-train.txt`（上卡训练所需；GPU-only 的 flash-attn 单独装）:
```
-r requirements-dev.txt
accelerate>=0.34
matplotlib>=3.8
# GPU only（上卡后手动装）：
# flash-attn>=2.6
```

`scripts/setup_env.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-dev}"

uv venv --python 3.12 .venv
source .venv/bin/activate

case "$MODE" in
  dev)   uv pip install -r requirements-dev.txt ;;
  train) uv pip install -r requirements-train.txt ;;
  *) echo "usage: $0 [dev|train]" >&2; exit 1 ;;
esac
```

`core/__init__.py` 与 `sft/__init__.py` 为空文件。

- [ ] **Step 2: 建环境并装 dev 依赖**

Run: `bash scripts/setup_env.sh dev`
Expected: 创建 `.venv`（Python 3.12），安装 torch/transformers/pytest 成功。

- [ ] **Step 3: 验证导入**

Run: `.venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`
Expected: 打印版本号，无报错。

- [ ] **Step 4: 提交**

```bash
git add pyproject.toml requirements-dev.txt requirements-train.txt scripts/setup_env.sh core/__init__.py sft/__init__.py
git commit -m "chore: 项目脚手架 + 依赖清单 + uv 环境脚本"
```

---

### Task 2: `build_chat_sample` —— 多轮掩码核心

**Files:**
- Create: `core/tokenizer_utils.py`, `sft/data.py`, `tests/conftest.py`, `tests/test_labels.py`

**Interfaces:**
- Produces:
  - `core.tokenizer_utils.load_tokenizer(model_id="Qwen/Qwen2.5-7B-Instruct") -> tokenizer`
  - `sft.data.build_chat_sample(messages: list[dict], tokenizer, max_length=2048) -> {"input_ids": list[int], "attention_mask": list[int], "labels": list[int]}`
  - `sft.data.IGNORE_INDEX = -100`

- [ ] **Step 1: 写失败测试**

`tests/conftest.py`:
```python
import pytest
from core.tokenizer_utils import load_tokenizer

@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer("Qwen/Qwen2.5-7B-Instruct")
```

`tests/test_labels.py`:
```python
from sft.data import build_chat_sample, IGNORE_INDEX


def test_roles_masked(tokenizer):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
    ]
    s = build_chat_sample(msgs, tokenizer)
    assert len(s["input_ids"]) == len(s["labels"]) == len(s["attention_mask"])
    kept = [i for i, l in enumerate(s["labels"]) if l != IGNORE_INDEX]
    assert kept  # 至少 assistant 内容被保留
    for i in kept:
        assert s["labels"][i] == s["input_ids"][i]
    assert "你好！有什么可以帮你？" in tokenizer.decode([s["input_ids"][i] for i in kept])


def test_system_user_masked(tokenizer):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "hi"},
    ]
    s = build_chat_sample(msgs, tokenizer)
    masked = [s["input_ids"][i] for i, l in enumerate(s["labels"]) if l == IGNORE_INDEX]
    assert "hi" not in tokenizer.decode(masked)  # assistant 内容不应出现在被掩码区域


def test_eos_kept(tokenizer):
    s = build_chat_sample([{"role": "assistant", "content": "hi"}], tokenizer)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pos = s["input_ids"].index(im_end)
    assert s["labels"][pos] == im_end  # <|im_end|> 必须可学习


def test_matches_chat_template(tokenizer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "thank you"},
        {"role": "assistant", "content": "you're welcome"},
    ]
    s = build_chat_sample(msgs, tokenizer)
    ref = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    assert s["input_ids"] == ref
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_labels.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'sft'` 或 `build_chat_sample` 未定义）

- [ ] **Step 3: 写实现**

`core/tokenizer_utils.py`:
```python
from transformers import AutoTokenizer


def load_tokenizer(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """加载 Qwen2.5 tokenizer；缺失 pad_token 时回退到 eos_token。"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
```

`sft/data.py`:
```python
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE_INDEX = -100


def _tokenize(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_chat_sample(messages: list[dict], tokenizer, max_length: int = 2048) -> dict:
    """把多轮对话转成 input_ids/labels，loss 只落在 assistant 回答上。

    每轮格式为 `<|im_start|>role\\ncontent<|im_end|>\\n`，按 turn 分别 tokenize 并
    生成 labels：system/user 与所有格式头部 → -100；assistant 的 content 与
    `<|im_end|>` → 保留原 token id。
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        is_assistant = role == "assistant"

        header_ids = _tokenize(f"{IM_START}{role}\n", tokenizer)
        content_ids = _tokenize(content, tokenizer)
        footer_ids = _tokenize(f"{IM_END}\n", tokenizer)

        input_ids.extend(header_ids)
        input_ids.extend(content_ids)
        input_ids.extend(footer_ids)

        labels.extend([IGNORE_INDEX] * len(header_ids))
        labels.extend(content_ids if is_assistant else [IGNORE_INDEX] * len(content_ids))
        labels.extend(footer_ids if is_assistant else [IGNORE_INDEX] * len(footer_ids))

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_labels.py -v`
Expected: PASS（4 个测试全过；首次会下载 Qwen2.5 tokenizer）

- [ ] **Step 5: 提交**

```bash
git add core/tokenizer_utils.py sft/data.py tests/conftest.py tests/test_labels.py
git commit -m "feat: build_chat_sample 多轮对话掩码（labels 仅落 assistant）"
```

---

### Task 3: `ChatDataCollator` —— 批次 padding

**Files:**
- Create: `sft/collator.py`, `tests/test_collator.py`

**Interfaces:**
- Consumes: `sft.data.build_chat_sample`（Task 2）
- Produces: `sft.collator.ChatDataCollator(tokenizer, max_length=2048, label_pad_token_id=-100)`，`__call__(features) -> {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}`

- [ ] **Step 1: 写失败测试**

`tests/test_collator.py`:
```python
import torch
from sft.data import build_chat_sample, IGNORE_INDEX
from sft.collator import ChatDataCollator


def _samples(tokenizer):
    a = build_chat_sample([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}], tokenizer)
    b = build_chat_sample([{"role": "user", "content": "hi"}], tokenizer)  # 更短
    return [a, b]


def test_collator_batch_shape(tokenizer):
    collator = ChatDataCollator(tokenizer)
    batch = collator(_samples(tokenizer))
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].dim() == 2


def test_padding_masked(tokenizer):
    collator = ChatDataCollator(tokenizer)
    batch = collator(_samples(tokenizer))
    pad = tokenizer.pad_token_id
    for row in range(batch["input_ids"].shape[0]):
        pad_positions = (batch["input_ids"][row] == pad)
        # 所有 padding 位置：attention_mask=0 且 label=-100
        assert (batch["attention_mask"][row][pad_positions] == 0).all()
        assert (batch["labels"][row][pad_positions] == IGNORE_INDEX).all()
        # 非 padding 位置 attention_mask=1
        assert (batch["attention_mask"][row][~pad_positions] == 1).all()


def test_truncation(tokenizer):
    collator = ChatDataCollator(tokenizer, max_length=8)
    batch = collator(_samples(tokenizer))
    assert batch["input_ids"].shape[1] <= 8
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_collator.py -v`
Expected: FAIL（`No module named 'sft.collator'`）

- [ ] **Step 3: 写实现**

`sft/collator.py`:
```python
import torch


class ChatDataCollator:
    """把 build_chat_sample 的输出 padding 到批次内最长（截断到 max_length）。"""

    def __init__(self, tokenizer, max_length: int = 2048, label_pad_token_id: int = -100):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        features = [{k: v[: self.max_length] for k, v in f.items()} for f in features]
        max_len = max((len(f["input_ids"]) for f in features), default=0)

        input_ids = [f["input_ids"] + [self.pad_token_id] * (max_len - len(f["input_ids"])) for f in features]
        attention_mask = [f["attention_mask"] + [0] * (max_len - len(f["attention_mask"])) for f in features]
        labels = [f["labels"] + [self.label_pad_token_id] * (max_len - len(f["labels"])) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_collator.py -v`
Expected: PASS（3 个测试全过）

- [ ] **Step 5: 提交**

```bash
git add sft/collator.py tests/test_collator.py
git commit -m "feat: ChatDataCollator 批次 padding（padding 掩码 -100）"
```

---

### Task 4: 多轮数据集加载

**Files:**
- Modify: `sft/data.py`（追加 `build_chat_sample_full`、`load_multi_turn_dataset`）
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: Task 2 的 `build_chat_sample`
- Produces:
  - `sft.data.load_multi_turn_dataset(dataset_name="BelleGroup/train_3.5M_CN", split="train", max_samples=None, field="conversations") -> datasets.Dataset`（每行含 `messages: list[dict]`）
  - `sft.data.build_chat_sample_full(messages, tokenizer, max_length=2048) -> dict`（labels 全置为 input_ids，A/B 对照组）

- [ ] **Step 1: 写失败测试**

`tests/test_data.py`:
```python
from sft.data import build_chat_sample, build_chat_sample_full, load_multi_turn_dataset


def test_full_labels(tokenizer):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    s = build_chat_sample_full(msgs, tokenizer)
    assert s["labels"] == s["input_ids"]


def test_normalize_roles(tokenizer):
    # 用 BelleGroup 的 from/value 结构，验证归一化到 role/content
    from sft.data import _normalize_conversation
    conv = [
        {"from": "human", "value": "你好"},
        {"from": "gpt", "value": "你好！"},
    ]
    msgs = _normalize_conversation(conv)
    assert msgs == [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
    s = build_chat_sample(msgs, tokenizer)
    assert len(s["input_ids"]) == len(s["labels"])


def test_load_small_subset():
    ds = load_multi_turn_dataset("BelleGroup/train_3.5M_CN", max_samples=4)
    assert len(ds) == 4
    assert all(isinstance(row["messages"], list) for row in ds)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: FAIL（`build_chat_sample_full` / `load_multi_turn_dataset` 未定义）

- [ ] **Step 3: 写实现**

在 `sft/data.py` 顶部加 `from datasets import Dataset, load_dataset`，并在末尾追加：
```python
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def _normalize_conversation(conv: list[dict]) -> list[dict]:
    """把 BelleGroup 的 {"from", "value"} 或通用 {"role", "content"} 统一成 role/content。"""
    out = []
    for turn in conv:
        if "role" in turn and "content" in turn:
            out.append({"role": turn["role"], "content": turn["content"]})
        elif "from" in turn and "value" in turn:
            out.append({"role": ROLE_MAP.get(turn["from"], turn["from"]), "content": turn["value"]})
        else:
            raise ValueError(f"无法识别的 turn 结构: {turn}")
    return out


def load_multi_turn_dataset(dataset_name: str = "BelleGroup/train_3.5M_CN", split: str = "train",
                            max_samples: int | None = None, field: str = "conversations"):
    if max_samples is not None:
        # 用 streaming 只取前 max_samples 条，避免全量下载（BelleGroup 有 3.5M 行）
        it = load_dataset(dataset_name, split=split, streaming=True)
        ds = Dataset.from_list(list(it.take(max_samples)))
    else:
        ds = load_dataset(dataset_name, split=split)

    def _map(example):
        return {"messages": _normalize_conversation(example[field])}

    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "messages"])
    return ds


def build_chat_sample_full(messages: list[dict], tokenizer, max_length: int = 2048) -> dict:
    """A/B 对照组：loss 落在全部 token 上（labels == input_ids）。"""
    sample = build_chat_sample(messages, tokenizer, max_length=max_length)
    sample["labels"] = sample["input_ids"]
    return sample
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_data.py -v`
Expected: PASS（`test_load_small_subset` 首次会下载 BelleGroup 子集，其余 2 个即时通过）

- [ ] **Step 5: 提交**

```bash
git add sft/data.py tests/test_data.py
git commit -m "feat: 多轮数据集加载 + 全序列 loss 对照组"
```

---

### Task 5: LoRA 训练入口 + 配置

**Files:**
- Create: `sft/config/sft_lora.yaml`, `sft/train.py`, `tests/test_train_config.py`

**Interfaces:**
- Consumes: `core.tokenizer_utils.load_tokenizer`、`sft.data.load_multi_turn_dataset`/`build_chat_sample`、`sft.collator.ChatDataCollator`
- Produces: `sft.train.SFTConfig`（dataclass，含 `from_yaml`）与 `sft.train.main(config)`；训练产物为 `outputs/sft-lora/` 下的 LoRA adapter。

- [ ] **Step 1: 写配置与失败测试**

`sft/config/sft_lora.yaml`:
```yaml
model_id: "Qwen/Qwen2.5-7B-Instruct"
dataset_name: "BelleGroup/train_3.5M_CN"
field: "conversations"
max_samples: 20000
max_length: 2048

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

per_device_batch: 4
grad_accum: 8
lr: 2.0e-4
num_epochs: 1
warmup_ratio: 0.03
output_dir: "outputs/sft-lora"
bf16: true
grad_checkpoint: true
use_flash_attn: true
```

`tests/test_train_config.py`:
```python
from sft.train import SFTConfig


def test_from_yaml():
    cfg = SFTConfig.from_yaml("sft/config/sft_lora.yaml")
    assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.bf16 is True
    assert "q_proj" in cfg.lora_target
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_train_config.py -v`
Expected: FAIL（`No module named 'sft.train'`）

- [ ] **Step 3: 写实现**

`sft/train.py`:
```python
import argparse
from dataclasses import dataclass, field

import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from core.tokenizer_utils import load_tokenizer
from sft.collator import ChatDataCollator
from sft.data import build_chat_sample, load_multi_turn_dataset


@dataclass
class SFTConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "BelleGroup/train_3.5M_CN"
    field: str = "conversations"
    max_samples: int | None = None
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target: tuple = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    per_device_batch: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    output_dir: str = "outputs/sft-lora"
    bf16: bool = True
    grad_checkpoint: bool = True
    use_flash_attn: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def tokenize_fn(example, tokenizer, max_length):
    return build_chat_sample(example["messages"], tokenizer, max_length=max_length)


def main(config: SFTConfig):
    tokenizer = load_tokenizer(config.model_id)
    dataset = load_multi_turn_dataset(config.dataset_name, field=config.field,
                                      max_samples=config.max_samples)
    tokenized = dataset.map(lambda ex: tokenize_fn(ex, tokenizer, config.max_length),
                            remove_columns=dataset.column_names)

    attn = "flash_attention_2" if config.use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto",
                                                 attn_implementation=attn)
    if config.grad_checkpoint:
        model.enable_input_require_grads()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=list(config.lora_target))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.grad_checkpoint,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      data_collator=ChatDataCollator(tokenizer, max_length=config.max_length))
    trainer.train()
    model.save_pretrained(config.output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft/config/sft_lora.yaml")
    cfg = SFTConfig.from_yaml(p.parse_args().config)
    main(cfg)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_train_config.py -v`
Expected: PASS（`peft`/`datasets`/`pyyaml` 已随 dev 依赖装好；不加载 7B 权重）

- [ ] **Step 5: 提交**

```bash
git add sft/config/sft_lora.yaml sft/train.py tests/test_train_config.py
git commit -m "feat: LoRA SFT 训练入口 + yaml 配置"
```

---

### Task 6: 收敛指标 + A/B 实验

**Files:**
- Create: `core/metrics.py`, `sft/run_masking_ab.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: Task 2/4 的 `build_chat_sample`/`build_chat_sample_full`、Task 3 的 `ChatDataCollator`、Task 5 的 `SFTConfig`
- Produces:
  - `core.metrics.ema_smooth(values, window=10) -> list[float]`
  - `core.metrics.steps_to_threshold(losses, threshold, window=10) -> int | None`
  - `sft.run_masking_ab.run_ab(config, steps, threshold) -> dict`（含两种模式的 loss 曲线与 step 对比）

- [ ] **Step 1: 写失败测试**

`tests/test_metrics.py`:
```python
from core.metrics import ema_smooth, steps_to_threshold


def test_steps_to_threshold():
    losses = [3.0, 2.5, 2.0, 1.8, 1.5]
    assert steps_to_threshold(losses, threshold=1.7, window=1) == 4


def test_steps_to_threshold_none():
    losses = [3.0, 2.5, 2.0]
    assert steps_to_threshold(losses, threshold=1.0, window=1) is None


def test_ema_smooth_empty():
    assert ema_smooth([]) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL（`No module named 'core.metrics'`）

- [ ] **Step 3: 写实现**

`core/metrics.py`:
```python
def ema_smooth(values: list[float], window: int = 10) -> list[float]:
    """指数滑动平均，平滑 loss 曲线。"""
    if not values:
        return []
    alpha = 2 / (window + 1)
    out, ema = [], None
    for v in values:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def steps_to_threshold(losses: list[float], threshold: float, window: int = 10) -> int | None:
    """平滑后首个 <= threshold 的步下标；未达阈值返回 None。"""
    for i, v in enumerate(ema_smooth(losses, window=window)):
        if v <= threshold:
            return i
    return None
```

`sft/run_masking_ab.py`:
```python
import argparse
import json

import matplotlib.pyplot as plt
from transformers import Trainer, TrainerCallback, TrainingArguments

from core.metrics import steps_to_threshold
from sft.train import SFTConfig, tokenize_fn
from core.tokenizer_utils import load_tokenizer
from sft.collator import ChatDataCollator
from sft.data import build_chat_sample, build_chat_sample_full, load_multi_turn_dataset


class LossCollector(TrainerCallback):
    def __init__(self):
        self.losses = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def _build_tokenized(config, tokenizer, mode):
    dataset = load_multi_turn_dataset(config.dataset_name, field=config.field,
                                      max_samples=config.max_samples)
    fn = build_chat_sample if mode == "masked" else build_chat_sample_full
    tokenized = dataset.map(lambda ex: fn(ex["messages"], tokenizer, config.max_length),
                            remove_columns=dataset.column_names)
    return tokenized


def run_ab(config: SFTConfig, steps: int, threshold: float) -> dict:
    tokenizer = load_tokenizer(config.model_id)
    results = {}
    for mode in ("masked", "full"):
        tokenized = _build_tokenized(config, tokenizer, mode)
        args = TrainingArguments(output_dir=f"outputs/ab-{mode}", max_steps=steps,
                                 per_device_train_batch_size=config.per_device_batch,
                                 gradient_accumulation_steps=config.grad_accum,
                                 learning_rate=config.lr, bf16=config.bf16,
                                 logging_steps=1, report_to=[],
                                 remove_unused_columns=False, save_strategy="no")
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, TaskType, get_peft_model
        model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto")
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_r,
                                                 lora_alpha=config.lora_alpha,
                                                 target_modules=list(config.lora_target)))
        collector = LossCollector()
        trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                          data_collator=ChatDataCollator(tokenizer, max_length=config.max_length),
                          callbacks=[collector])
        trainer.train()
        results[mode] = {"losses": collector.losses,
                         "steps_to_threshold": steps_to_threshold(collector.losses, threshold)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sft/config/sft_lora.yaml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--threshold", type=float, default=1.5)
    args = p.parse_args()
    cfg = SFTConfig.from_yaml(args.config)
    results = run_ab(cfg, args.steps, args.threshold)

    with open("docs/experiments/masking_ab.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for mode, r in results.items():
        plt.plot(r["losses"], label=mode)
    plt.legend(); plt.xlabel("step"); plt.ylabel("loss")
    plt.savefig("docs/experiments/masking_ab.png")
    print(json.dumps({k: v["steps_to_threshold"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS。另做语法检查：`.venv/bin/python -m py_compile sft/run_masking_ab.py sft/train.py`（不加载模型）。

- [ ] **Step 5: 提交**

```bash
git add core/metrics.py sft/run_masking_ab.py tests/test_metrics.py
git commit -m "feat: 收敛指标 + 掩码 vs 全序列 loss 的 A/B 实验脚本"
```

---

### Task 7: README + 实验记录骨架

**Files:**
- Create: `README.md`, `docs/experiments/.gitkeep`

**Interfaces:**
- Consumes: 全部已实现模块。

- [ ] **Step 1: 写 README**

`README.md` 内容要点（完整撰写，非占位）：
```markdown
# Qwen2.5-7B-Instruct 微调 + 推理加速 + GRPO 对齐

求职作品集项目。Monorepo 三子项目：SFT 微调链路（本目录当前实现）、手写算子+推理加速、GRPO RL 对齐。

## 子项目 1：SFT 微调链路（已实现）

### 核心亮点
- **自定义多轮对话掩码 `ChatDataCollator`**：loss 仅落在 assistant 回答上，system/user 与格式头部全部 `-100`，`<|im_end|>` 保留可学习。
- **可复现收敛实验**：掩码 vs 全序列 loss 的 A/B 对比（`run_masking_ab.py`）。

### 诚实声明（避免面试穿帮）
掩码优化的是 loss/梯度信号，**不直接省 FLOPs**；真正省算力的是 sequence packing（列为后续可选增强）。

### 环境
\`\`\`bash
bash scripts/setup_env.sh dev    # 本地 CPU 单测
bash scripts/setup_env.sh train  # 训练依赖
\`\`\`

### 运行
\`\`\`bash
# 单测（CPU）
.venv/bin/python -m pytest -v
# SFT 训练（GPU，上卡后）
.venv/bin/python sft/train.py --config sft/config/sft_lora.yaml
# 收敛 A/B 实验（GPU）
.venv/bin/python sft/run_masking_ab.py --steps 200
\`\`\`
```

`docs/experiments/.gitkeep` 为空文件。

- [ ] **Step 2: 全量单测回归**

Run: `.venv/bin/python -m pytest -v`
Expected: 全部 PASS（labels/collator/data/train_config/metrics）。

- [ ] **Step 3: 提交**

```bash
git add README.md docs/experiments/.gitkeep
git commit -m "docs: README + 实验记录骨架"
```

---

## Self-Review

- **Spec coverage**：§5 掩码规则→Task 2/3；§6 训练配置→Task 5；§7 收敛实验→Task 6；§8 测试→Task 2/3/4/6 的 test 文件；§3 目录→Task 1 脚手架。无缺口。
- **Placeholder scan**：无 TBD/TODO；每个代码步骤都有完整实现。
- **Type consistency**：`build_chat_sample`/`build_chat_sample_full`/`ChatDataCollator`/`SFTConfig`/`ema_smooth`/`steps_to_threshold` 的签名在定义处与消费处一致。
