"""completion/reward 缓存：resume + 离线 replay。"""
import json
import os


class CompletionCache:
    """把 (prompt, completion) → {"reward", "log_prob"} 持久化到 JSONL。

    - resume：重启训练时同一 prompt+completion 已评过分则直接取缓存，不重复生成/打分。
    - 离线 replay：加载缓存轨迹重算 advantage/loss 做调试，无需重新生成。
    """

    def __init__(self, path: str):
        self.path = path
        self._data: dict[tuple[str, str], dict] = {}
        self._load()

    @staticmethod
    def _key(prompt: str, completion: str) -> tuple[str, str]:
        return (prompt, completion)

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._data[(rec["prompt"], rec["completion"])] = {
                    "reward": rec["reward"], "log_prob": rec["log_prob"]}

    def get(self, prompt: str, completion: str):
        return self._data.get(self._key(prompt, completion))

    def put(self, prompt: str, completion: str, reward: float, log_prob: float):
        self._data[self._key(prompt, completion)] = {"reward": reward, "log_prob": log_prob}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                "reward": reward, "log_prob": log_prob}) + "\n")

    def __len__(self):
        return len(self._data)
