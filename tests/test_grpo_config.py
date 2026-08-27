from grpo.train import GRPOConfig


def test_from_yaml():
    cfg = GRPOConfig.from_yaml("grpo/config/grpo_lora.yaml")
    assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.group_size == 4
    assert cfg.beta == 0.01
    assert cfg.clip_epsilon == 0.2
    assert "q_proj" in cfg.lora_target
