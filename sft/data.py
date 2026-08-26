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
