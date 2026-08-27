"""GRPO 规则奖励：答案抽取 + 正确性 + 格式（组合，无需奖励模型）。"""
import re

# GSM8K 原生答案：#### <num>（优先级最高）
_NUM_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
# 结构化 fallback：<answer>...</answer>
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

FORMAT_BONUS = 0.05


def _normalize_number(s: str) -> str:
    """数值归一化：去千分位逗号，42 == 42.0。"""
    s = s.strip().replace(",", "")
    try:
        return f"{float(s):.10g}"
    except ValueError:
        return s.strip()


def extract_answer(completion: str) -> str | None:
    """抽取最终答案：优先 #### <num>，fallback <answer>...</answer>。未命中返回 None。"""
    m = _NUM_RE.search(completion)
    if m:
        return m.group(1)
    m = _ANSWER_TAG_RE.search(completion)
    return m.group(1).strip() if m else None


def correctness_reward(completion: str, gold: str) -> float:
    """正确性奖励：抽取答案与 gold 数值归一化比对 → 1 或 0。"""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_number(pred) == _normalize_number(gold) else 0.0


def format_reward(completion: str) -> float:
    """格式奖励：命中 #### 或 <answer> 结构 → +FORMAT_BONUS，否则 0。"""
    return FORMAT_BONUS if extract_answer(completion) is not None else 0.0


def combined_reward(completion: str, gold: str) -> float:
    """组合奖励 = 正确性 (0/1) + 格式 (0/0.05)，范围 [0, 1.05]。"""
    return correctness_reward(completion, gold) + format_reward(completion)
