from transformers import Qwen3Config

from sglang_omni.utils.hf import use_text_rope


def test_use_text_rope_drops_only_mrope_parameters() -> None:
    config = Qwen3Config(
        rope_theta=1_000_000,
        rope_scaling={
            "rope_type": "default",
            "interleaved": True,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        },
    )

    use_text_rope(config)

    assert config.rope_scaling == {"rope_type": "default", "rope_theta": 1_000_000}
