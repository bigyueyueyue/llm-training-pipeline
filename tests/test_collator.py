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
