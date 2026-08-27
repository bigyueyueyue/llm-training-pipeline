from grpo.reward import (FORMAT_BONUS, combined_reward, correctness_reward,
                         extract_answer, format_reward)


def test_extract_hash_answer():
    assert extract_answer("Some reasoning... #### 42") == "42"


def test_extract_answer_tag_fallback():
    assert extract_answer("<answer> 42 </answer>") == "42"


def test_extract_negative_decimal():
    assert extract_answer("#### -3.5") == "-3.5"


def test_extract_missing_returns_none():
    assert extract_answer("no answer here") is None


def test_correctness_normalizes_decimals():
    assert correctness_reward("#### 42.0", "42") == 1.0
    assert correctness_reward("#### 42", "42.0") == 1.0


def test_correctness_wrong_answer():
    assert correctness_reward("#### 43", "42") == 0.0


def test_correctness_missing_answer():
    assert correctness_reward("no answer", "42") == 0.0


def test_format_reward_bonus():
    assert format_reward("#### 42") == FORMAT_BONUS
    assert format_reward("<answer>42</answer>") == FORMAT_BONUS
    assert format_reward("no answer") == 0.0


def test_combined_reward_range():
    assert combined_reward("#### 42", "42") == 1.0 + FORMAT_BONUS   # 1.05
    assert combined_reward("#### 43", "42") == 0.0 + FORMAT_BONUS   # 0.05 错但格式对
    assert combined_reward("no answer", "42") == 0.0
