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
    # transformers 5.x 返回 BatchEncoding（Mapping），4.x 返回 list，两种都兼容
    ref_ids = ref if isinstance(ref, list) else list(ref["input_ids"])
    assert s["input_ids"] == ref_ids
