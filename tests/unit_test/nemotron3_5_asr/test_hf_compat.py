# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch


def test_import_does_not_mutate_transformers_auto_mappings() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.feature_extraction_auto import (
        FEATURE_EXTRACTOR_MAPPING,
    )
    from transformers.models.auto.modeling_auto import MODEL_FOR_RNNT_MAPPING

    mappings = (CONFIG_MAPPING, FEATURE_EXTRACTOR_MAPPING, MODEL_FOR_RNNT_MAPPING)
    before = [dict(mapping._extra_content) for mapping in mappings]

    module = importlib.import_module("sglang_omni.models.nemotron3_5_asr.hf_compat")
    importlib.reload(module)

    assert [dict(mapping._extra_content) for mapping in mappings] == before


def test_processor_loads_nested_feature_extractor_without_auto_registration(
    tmp_path, monkeypatch
) -> None:
    from sglang_omni.models.nemotron3_5_asr.hf_compat import (
        Nemotron3_5AsrProcessor,
        NemotronAsrStreamingFeatureExtractor,
    )
    from sglang_omni.models.nemotron3_5_asr.hf_compat import (
        processing_nemotron3_5_asr as processing,
    )

    processor_config = {
        "blank_token": "<blank>",
        "default_num_lookahead_tokens": 3,
        "feature_extractor": {
            "feature_extractor_type": "NemotronAsrStreamingFeatureExtractor",
            "feature_size": 4,
            "hop_length": 4,
            "n_fft": 16,
            "sampling_rate": 16000,
            "win_length": 8,
        },
        "num_prompts": 128,
        "processor_class": "Nemotron3_5AsrProcessor",
        "prompt_dictionary": {"en-US": 0, "auto": 101},
        "supported_num_lookahead_tokens": [3, 0, 6, 13],
    }
    (tmp_path / "processor_config.json").write_text(
        json.dumps(processor_config), encoding="utf-8"
    )

    tokenizer = SimpleNamespace(
        init_kwargs={},
        convert_tokens_to_ids=lambda token: 13087 if token == "<blank>" else 0,
    )
    tokenizer_loads: list[tuple[object, dict[str, object]]] = []

    def load_tokenizer(path, **kwargs):
        tokenizer_loads.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(
        processing.ParakeetTokenizer,
        "from_pretrained",
        load_tokenizer,
    )
    monkeypatch.setattr(
        Nemotron3_5AsrProcessor,
        "check_argument_for_proper_class",
        lambda self, name, value: object,
    )

    processor = Nemotron3_5AsrProcessor.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(processor.feature_extractor, NemotronAsrStreamingFeatureExtractor)
    assert processor.feature_extractor.feature_size == 4
    assert processor.tokenizer is tokenizer
    assert processor.blank_token_id == 13087
    assert processor.default_num_lookahead_tokens == 3
    assert tokenizer_loads[0][0] == tmp_path
    assert tokenizer_loads[0][1]["local_files_only"] is True


def test_processor_rejects_checkpoint_without_nested_feature_extractor() -> None:
    from sglang_omni.models.nemotron3_5_asr.hf_compat import Nemotron3_5AsrProcessor

    with pytest.raises(ValueError, match="nested.*feature_extractor"):
        Nemotron3_5AsrProcessor._get_arguments_from_pretrained(
            "unused", processor_dict={}
        )


def test_config_and_model_support_local_from_pretrained(tmp_path) -> None:
    from sglang_omni.models.nemotron3_5_asr.hf_compat import (
        Nemotron3_5AsrConfig,
        Nemotron3_5AsrForRNNT,
    )
    from sglang_omni.models.nemotron3_5_asr.hf_compat.configuration_nemotron_asr_streaming import (
        NemotronAsrStreamingEncoderConfig,
    )

    config = Nemotron3_5AsrConfig(
        vocab_size=16,
        decoder_hidden_size=8,
        num_decoder_layers=1,
        blank_token_id=15,
        num_prompts=4,
        prompt_intermediate_size=8,
        default_prompt_id=1,
        encoder_config={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "subsampling_factor": 2,
            "subsampling_conv_channels": 2,
            "num_mel_bins": 4,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "conv_kernel_size": 3,
            "sliding_window": 5,
            "default_num_lookahead_tokens": 0,
        },
    )
    model = Nemotron3_5AsrForRNNT(config)
    model.save_pretrained(tmp_path)

    loaded_config = Nemotron3_5AsrConfig.from_pretrained(
        tmp_path, local_files_only=True
    )
    loaded_model = Nemotron3_5AsrForRNNT.from_pretrained(
        tmp_path,
        config=loaded_config,
        local_files_only=True,
    )

    assert isinstance(loaded_config.encoder_config, NemotronAsrStreamingEncoderConfig)
    assert isinstance(loaded_model, Nemotron3_5AsrForRNNT)
    assert isinstance(loaded_model.config, Nemotron3_5AsrConfig)
    assert loaded_model.config.vocab_size == 16


def test_parakeet_compat_forwards_cache_aware_encoder_kwargs(monkeypatch) -> None:
    from transformers.generation import GenerationMixin

    from sglang_omni.models.nemotron3_5_asr.hf_compat.generation_parakeet import (
        ParakeetRNNTGenerationMixin,
    )

    input_features = torch.zeros(2, 5, 4)
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    model_kwargs = {
        "attention_mask": attention_mask,
        "past_key_values": object(),
        "padding_cache": "encoder-padding-cache",
        "num_lookahead_tokens": 3,
        "use_cache": True,
    }
    monkeypatch.setattr(
        GenerationMixin,
        "_prepare_model_inputs",
        lambda self, *args, **kwargs: (
            input_features,
            "input_features",
            dict(model_kwargs),
        ),
    )

    calls = []

    class FakeModel(ParakeetRNNTGenerationMixin):
        def get_audio_features(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                attention_mask=torch.ones(2, 3, dtype=torch.long),
                last_hidden_state=torch.zeros(2, 3, 4),
            )

    _, input_name, prepared = FakeModel()._prepare_model_inputs()

    assert input_name == "input_features"
    assert calls == [
        {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "output_attention_mask": True,
            "padding_cache": "encoder-padding-cache",
            "num_lookahead_tokens": 3,
        }
    ]
    assert prepared["encoder_valid_lengths"].tolist() == [3, 3]
    assert prepared["encoder_frame_idxs"].tolist() == [0, 0]
