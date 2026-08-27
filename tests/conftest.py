import pytest
from core.tokenizer_utils import load_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer("Qwen/Qwen2.5-7B-Instruct")


@pytest.fixture(scope="session")
def qwen_config():
    from transformers import Qwen2Config

    return Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
