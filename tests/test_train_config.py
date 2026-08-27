from sft.train import SFTConfig


def test_from_yaml():
    cfg = SFTConfig.from_yaml("sft/config/sft_lora.yaml")
    assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.bf16 is True
    assert "q_proj" in cfg.lora_target
