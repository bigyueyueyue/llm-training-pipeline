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
