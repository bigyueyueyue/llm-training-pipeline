from transformers import AutoTokenizer


def load_tokenizer(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """加载 Qwen2.5 tokenizer；缺失 pad_token 时回退到 eos_token。"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
