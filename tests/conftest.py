import pytest
from core.tokenizer_utils import load_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer("Qwen/Qwen2.5-7B-Instruct")
