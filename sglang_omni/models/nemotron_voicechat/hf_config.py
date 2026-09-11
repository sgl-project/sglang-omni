import json
import pathlib

from huggingface_hub import hf_hub_download
from sglang.srt.configs.nemotron_h import NemotronHConfig
from transformers import AutoConfig

VOICECHAT_MODEL_ARCH_OVERRIDE = "NemotronVoiceChatForCausalLM"
_voicechat_hf_config_registered = False


def _load_backbone_config(repo_id):
    path = hf_hub_download(repo_id, "config.json")
    backbone = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for key in ("architectures", "model_type", "torch_dtype", "dtype"):
        backbone.pop(key, None)
    return backbone


class NemotronVoiceChatConfig(NemotronHConfig):
    model_type = "nemotron_voicechat"

    def __init__(
        self,
        perception=None,
        speech_generation=None,
        duplex=None,
        text_backbone_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = dict(perception or {})
        self.speech_generation = dict(speech_generation or {})
        self.duplex = dict(duplex or {})
        self.text_backbone_path = text_backbone_path

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Already a voice chat config
        if "perception" in config_dict:
            return super().from_dict(config_dict, **kwargs)

        stt_config = config_dict["model"]["stt"]["model"]
        backbone_path = stt_config["pretrained_llm"]
        return super().from_dict(
            {
                **_load_backbone_config(backbone_path),
                "architectures": [VOICECHAT_MODEL_ARCH_OVERRIDE],
                "perception": stt_config["perception"],
                "speech_generation": config_dict["model"]["speech_generation"]["model"],
                "duplex": {
                    key: value
                    for key, value in stt_config.items()
                    if key.startswith("duplex_")
                    or key in ("use_function_head", "predict_user_text")
                },
                "text_backbone_path": backbone_path,
            },
            **kwargs,
        )


def register_voicechat_hf_config():
    global _voicechat_hf_config_registered
    if _voicechat_hf_config_registered:
        return
    AutoConfig.register("nemotron_voicechat", NemotronVoiceChatConfig, exist_ok=True)
    _voicechat_hf_config_registered = True


__all__ = [
    "VOICECHAT_MODEL_ARCH_OVERRIDE",
    "register_voicechat_hf_config",
    "NemotronVoiceChatConfig",
]
