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
