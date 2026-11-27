import torch
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.utils import logger
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    PretrainedConfig,
    ProcessorMixin,
)

from .audio_lengths import qwen3_asr_audio_token_lengths


class Qwen3ASRProcessor(ProcessorMixin):
    """Minimal composite processor: WhisperFeatureExtractor + Qwen2Tokenizer.

    AutoProcessor.from_pretrained() for Qwen3-ASR returns just a tokenizer,
    but SGLang's multimodal pipeline needs a processor that handles audio.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if "revision" in kwargs:
            load_kwargs["revision"] = kwargs["revision"]
        else:
            pass
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def get_feat_extract_output_lengths(self, input_lengths):
        return qwen3_asr_audio_token_lengths(input_lengths)

    def __call__(self, text=None, audio=None, audio_kwargs=None, **kwargs):
        inputs = {}
        if audio is not None:
            audio_kwargs = audio_kwargs or {}
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors=kwargs.get("return_tensors"),
                **audio_kwargs,
            )
            inputs["input_features"] = audio_inputs["input_features"]
            if "attention_mask" in audio_inputs:
                inputs["feature_attention_mask"] = audio_inputs["attention_mask"]
            else:
                pass
        else:
            pass

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
            )
            input_ids = text_inputs["input_ids"]

            # Expand the single <|audio_pad|> placeholder in the prompt to N
            # copies, where N is the audio encoder's output length for this clip.
            # Without this, the model only sees 1 audio token for hundreds of
            # feature frames and can't align audio embeddings with token positions.
            if audio is not None and "feature_attention_mask" in inputs:
                audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
                feat_lengths = inputs["feature_attention_mask"].sum(dim=-1)
                audio_token_counts = self.get_feat_extract_output_lengths(feat_lengths)
                expanded = []
                for seq_idx in range(input_ids.shape[0]):
                    ids = input_ids[seq_idx].tolist()
                    audio_idx = 0
                    new_ids = []
                    for tid in ids:
                        if tid == audio_pad_id and audio_idx < len(audio_token_counts):
                            n = int(audio_token_counts[audio_idx].item())
                            new_ids.extend([audio_pad_id] * n)
                            audio_idx += 1
                        else:
                            new_ids.append(tid)
                    expanded.append(new_ids)
                max_len = max(len(s) for s in expanded)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = [s + [pad_id] * (max_len - len(s)) for s in expanded]
                input_ids = torch.tensor(padded, dtype=torch.long)
            else:
                pass

            inputs["input_ids"] = input_ids
        else:
            pass
        return inputs


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"
    sub_configs = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151676,
        audio_start_token_id=151669,
        audio_end_token_id=151670,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        else:
            pass
        self.audio_config = audio_config

        from transformers.models.qwen3.configuration_qwen3 import (
            Qwen3Config as HFQwen3Config,
        )

        if isinstance(text_config, dict):
            text_config = HFQwen3Config(**text_config)
        elif text_config is None:
            text_config = HFQwen3Config()
        else:
            pass

        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id


@register_customized_processor(Qwen3ASRProcessor)
class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config: Qwen3ASRThinkerConfig | dict[str, object] | None = None,
        audio_config: PretrainedConfig | dict[str, object] | None = None,
        text_config: PretrainedConfig | dict[str, object] | None = None,
        audio_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        **kwargs: object,
    ) -> None:
        if thinker_config is None:
            if audio_config is not None or text_config is not None:
                # note (Nathan Roll): native Hugging Face Qwen3-ASR checkpoints place audio_config and text_config at the top level without a thinker_config wrapper.
                thinker_kwargs: dict[str, object] = {}
                if audio_config is not None:
                    thinker_kwargs["audio_config"] = audio_config
                else:
                    pass
                if text_config is not None:
                    if isinstance(text_config, dict):
                        copied_text_config = dict(text_config)
                        for attribute in (
                            "tie_word_embeddings",
                            "eos_token_id",
                            "pad_token_id",
                        ):
                            if (
                                copied_text_config.get(attribute) is None
                                and kwargs.get(attribute) is not None
                            ):
                                copied_text_config[attribute] = kwargs[attribute]
                            else:
                                pass
                        thinker_kwargs["text_config"] = copied_text_config
                    else:
                        thinker_kwargs["text_config"] = text_config
                else:
                    pass
                if audio_token_id is not None:
                    thinker_kwargs["audio_token_id"] = audio_token_id
                else:
                    pass
                if audio_start_token_id is not None:
                    thinker_kwargs["audio_start_token_id"] = audio_start_token_id
                else:
                    pass
                if audio_end_token_id is not None:
                    thinker_kwargs["audio_end_token_id"] = audio_end_token_id
                else:
                    pass
                for attribute in ("dtype", "torch_dtype", "initializer_range"):
                    if attribute in kwargs:
                        thinker_kwargs[attribute] = kwargs[attribute]
                    else:
                        pass
                thinker_config = thinker_kwargs
            else:
                thinker_config = {}
                logger.info(
                    "thinker_config is None. "
                    "Initializing Qwen3-ASR thinker with default values"
                )
        else:
            pass
        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config
        super().__init__(**kwargs)

    @property
    def audio_config(self) -> PretrainedConfig:
        return self.thinker_config.audio_config

    @audio_config.setter
    def audio_config(self, value: PretrainedConfig) -> None:
        self.thinker_config.audio_config = value

    @property
    def text_config(self) -> PretrainedConfig:
        return self.thinker_config.text_config

    @text_config.setter
    def text_config(self, value: PretrainedConfig) -> None:
        self.thinker_config.text_config = value

    @property
    def audio_token_id(self) -> int:
        return self.thinker_config.audio_token_id

    @audio_token_id.setter
    def audio_token_id(self, value: int) -> None:
        self.thinker_config.audio_token_id = value

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.thinker_config.text_config


def register_qwen3_asr_hf_config() -> None:
    """Register Qwen3-ASR config classes in AutoConfig."""
    # note (ratish): the installed transformers may map these types to its own
    # classes; the model reads this package's layout, so the mapping is replaced.
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig, exist_ok=True)


register_qwen3_asr_hf_config()
