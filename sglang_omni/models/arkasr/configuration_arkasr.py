# SPDX-License-Identifier: Apache-2.0
"""Config + processor + audio-token-length helper for ARK-ASR-3B.

ARK-ASR (AutoArk-AI/ARK-ASR-3B) is a Whisper-style audio tower (with RoPE
self-attention) + MLP frame-merge adapter feeding a dense Qwen2 LM. The
checkpoint's ``ArkasrConfig`` subclasses ``Qwen2Config`` and carries a nested
``whisper_config``. Serving loads the checkpoint tokenizer and config with
``trust_remote_code=True``, while model execution is implemented locally
because the checkpoint model code targets the transformers-4 encoder-layer API,
which is incompatible with transformers 5.
"""

from __future__ import annotations

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    ProcessorMixin,
    Qwen2Config,
    WhisperConfig,
)

from .audio_lengths import arkasr_audio_token_lengths


class ArkasrConfig(Qwen2Config):
    """Native ARK-ASR config: Qwen2 LM params at top level + nested whisper_config."""

    model_type = "arkasr"
    is_composition = True

    def __init__(
        self,
        whisper_config=None,
        adapter_type: str = "mlp",
        merge_factor: int = 4,
        spec_aug: bool = False,
        use_rope: bool = True,
        max_whisper_length: int = 1500,
        mlp_adapter_act: str = "gelu",
        audio_token_id: int = 151663,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(whisper_config, dict):
            self.whisper_config = WhisperConfig(**whisper_config)
        elif isinstance(whisper_config, WhisperConfig):
            self.whisper_config = whisper_config
        else:
            self.whisper_config = WhisperConfig()
        self.adapter_type = adapter_type
        self.merge_factor = int(merge_factor)
        self.spec_aug = bool(spec_aug)
        self.use_rope = bool(use_rope)
        self.max_whisper_length = int(max_whisper_length)
        self.mlp_adapter_act = mlp_adapter_act
        self.audio_token_id = int(audio_token_id)

    def to_dict(self):
        output = super().to_dict()
        output["whisper_config"] = self.whisper_config.to_dict()
        return output


class ArkasrProcessor(ProcessorMixin):
    """Composite processor: WhisperFeatureExtractor + fast tokenizer.

    The stock ARK processor lives in remote code; SGLang's multimodal pipeline
    only needs mel extraction + tokenization, so we provide a minimal one.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


# Register the processor without changing Transformers' process-global config mapping.
register_customized_processor(ArkasrProcessor)(ArkasrConfig)


__all__ = ["ArkasrConfig", "ArkasrProcessor", "arkasr_audio_token_lengths"]
