from datasets import Dataset, load_dataset

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
