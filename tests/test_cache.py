from grpo.cache import CompletionCache


def test_put_get_roundtrip(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    c.put("q", "a", 1.05, -3.2)
    assert c.get("q", "a") == {"reward": 1.05, "log_prob": -3.2}


def test_persistence_across_instances(tmp_path):
    p = str(tmp_path / "cache.jsonl")
    CompletionCache(p).put("q", "a", 1.0, -1.0)
    c2 = CompletionCache(p)  # 从磁盘重新加载
    assert c2.get("q", "a") == {"reward": 1.0, "log_prob": -1.0}


def test_miss_returns_none(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    assert c.get("q", "missing") is None


def test_len(tmp_path):
    c = CompletionCache(str(tmp_path / "cache.jsonl"))
    c.put("q1", "a", 1.0, 0.0)
    c.put("q2", "b", 0.5, -1.0)
    assert len(c) == 2
