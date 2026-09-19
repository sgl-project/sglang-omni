# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingFeatureExtractor,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    processing_nemotron3_5_asr as processing,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat.configuration_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderConfig,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat.generation_parakeet import (
    ParakeetRNNTGenerationMixin,
)
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
)


def test_processor_text_decode_preserves_repeated_tokens() -> None:
    tokenizer = processing.ParakeetTokenizer(
        tokenizer_object=Tokenizer(WordLevel({"<blank>": 0, "hello": 1})),
        pad_token="<blank>",
    )
    processor = Nemotron3_5AsrProcessor(
        feature_extractor=NemotronAsrStreamingFeatureExtractor(), tokenizer=tokenizer
    )
    token_ids = [1, 1, 0, 1]
    assert processor.decode(token_ids) == "hello hello hello"
    assert processor.batch_decode([token_ids]) == ["hello hello hello"]
    assert processor.decode(token_ids, group_tokens=True) == "hello hello"
    assert processor.batch_decode([token_ids], group_tokens=True) == ["hello hello"]
    with pytest.raises(ValueError, match="timestamps are not supported"):
        processor.decode(token_ids, durations=torch.ones(len(token_ids)))


def test_local_model_preserves_streaming_results_and_caches_when_batched(
    tmp_path,
) -> None:

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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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

    runner = object.__new__(Nemotron3_5ASRModelRunner)
    runner.model = loaded_model.eval()
    runner.device = torch.device("cpu")
    runner.model_lock = threading.Lock()
    runner.processor = SimpleNamespace(
        default_num_lookahead_tokens=0,
        batch_decode=lambda rows, **kwargs: [str(row.tolist()) for row in rows],
    )
    serial = [runner.new_streaming_decode_state() for _ in range(2)]
    batched = [runner.new_streaming_decode_state() for _ in range(2)]
    for chunk_index in range(2):
        chunks = [
            Nemotron3_5ASRPreparedChunk(
                input_features=torch.full((1, 8, 4), float(index + chunk_index)),
                prompt_ids=torch.tensor([index]),
            )
            for index in range(2)
        ]
        for state, chunk in zip(serial, chunks):
            runner.run_streaming_batch([state], [chunk], requested_languages=["auto"])
        runner.run_streaming_batch(
            batched, chunks, requested_languages=["auto", "auto"]
        )
        for expected, actual in zip(serial, batched):
            assert actual.tokens == expected.tokens
            assert actual.durations == expected.durations
            torch.testing.assert_close(
                actual.decoder_cache.cache, expected.decoder_cache.cache
            )
            for left, right in zip(
                actual.attention_cache.layers, expected.attention_cache.layers
            ):
                torch.testing.assert_close(left.keys, right.keys)
            for key in actual.padding_cache.layers:
                torch.testing.assert_close(
                    actual.padding_cache.layers[key].cache,
                    expected.padding_cache.layers[key].cache,
                )
        assert (
            batched[0].decoder_cache.cache.data_ptr()
            != batched[1].decoder_cache.cache.data_ptr()
        )
        for left, right in zip(
            batched[0].attention_cache.layers, batched[1].attention_cache.layers
        ):
            assert left.keys.data_ptr() != right.keys.data_ptr()
        for key in batched[0].padding_cache.layers:
            assert (
                batched[0].padding_cache.layers[key].cache.data_ptr()
                != batched[1].padding_cache.layers[key].cache.data_ptr()
            )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_attention_cache_split_merge_preserves_sliding_window_state(
    batch_size: int,
) -> None:
    config = NemotronAsrStreamingEncoderConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        sliding_window=5,
    )
    runner = object.__new__(Nemotron3_5ASRModelRunner)
    request_caches = []
    for request_index in range(batch_size):
        cache = DynamicCache(config=config)
        for update_index in range(3):
            key_states = torch.full(
                (1, 2, 2, 4),
                float(request_index + update_index),
            )
            cache.layers[0].update(key_states, key_states)
        request_caches.append(cache)

    merged = runner.merge_attention_caches(
        request_caches,
        cache_config=config,
    )
    assert merged is not None
    merged_layer = merged.layers[0]
    assert merged_layer.is_sliding
    assert merged_layer.keys.shape[0] == batch_size
    assert merged_layer.keys.shape[-2] == 4
    assert merged_layer.get_seq_length() == 6

    split = runner.split_attention_cache(
        merged,
        batch_size,
        cache_config=config,
    )
    assert len(split) == batch_size
    for request_cache in split:
        request_layer = request_cache.layers[0]
        assert request_layer.is_sliding
        assert request_layer.keys.shape[-2] == 4
        assert request_layer.get_seq_length() == 6

        new_key_states = torch.zeros((1, 2, 2, 4))
        request_layer.update(new_key_states, new_key_states)
        assert request_layer.keys.shape[-2] == 4
        assert request_layer.get_seq_length() == 8


def test_parakeet_compat_forwards_cache_aware_encoder_kwargs(monkeypatch) -> None:

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
