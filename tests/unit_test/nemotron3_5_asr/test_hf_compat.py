# SPDX-License-Identifier: Apache-2.0
"""Compatibility and batch equivalence for the vendored Nemotron model."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

from sglang_omni.models.nemotron3_5_asr.encoder import (
    NemotronStreamingEncoderGraphRunner,
    encode_pooled_streaming_batch,
)
from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderPoolLayout,
    NemotronEncoderStatePool,
)
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
)
from sglang_omni.vendor.nemotron3_5_asr import processing_nemotron3_5_asr as processing
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)
from sglang_omni.vendor.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from sglang_omni.vendor.nemotron3_5_asr.generation_parakeet import (
    ParakeetRNNTGenerationMixin,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
    NemotronAsrStreamingEncoderModelOutput,
)
from sglang_omni.vendor.nemotron3_5_asr.processing_nemotron3_5_asr import (
    Nemotron3_5AsrProcessor,
)


@dataclass(kw_only=True)
class ReferenceDecodeState:
    tokens: list[int]
    durations: list[int]
    attention_cache: DynamicCache
    decoder_cache: Nemotron3_5AsrRNNTDecoderCache
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache = field(
        default_factory=NemotronAsrStreamingEncoderCausalConvPaddingCache
    )
    symbols_at_frame: int = 0
    encoder_frames: int = 0
    decoder_steps: int = 0

    @classmethod
    def create(cls, model: Nemotron3_5AsrForRNNT) -> ReferenceDecodeState:
        return cls(
            tokens=[model.config.blank_token_id],
            durations=[0],
            attention_cache=DynamicCache(config=model.config.encoder_config),
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(model.config),
        )


@torch.inference_mode()
def run_reference_chunk(
    runner: Nemotron3_5ASRModelRunner,
    state: ReferenceDecodeState,
    chunk: Nemotron3_5ASRPreparedChunk,
) -> None:
    encoder_output = runner.model.get_audio_features(
        input_features=chunk.input_features,
        prompt_ids=chunk.prompt_ids,
        past_key_values=state.attention_cache,
        padding_cache=state.padding_cache,
        num_lookahead_tokens=runner.processor.default_num_lookahead_tokens,
        use_cache=True,
    )
    state.attention_cache = encoder_output.past_key_values
    state.padding_cache = encoder_output.padding_cache
    state.encoder_frames += encoder_output.pooler_output.shape[1]
    frame_index = 0
    while frame_index < encoder_output.pooler_output.shape[1]:
        output = runner.model(
            encoder_outputs=NemotronAsrStreamingEncoderModelOutput(
                pooler_output=encoder_output.pooler_output[
                    :, frame_index : frame_index + 1
                ]
            ),
            decoder_input_ids=torch.tensor([[state.tokens[-1]]], device=runner.device),
            decoder_cache=state.decoder_cache,
            use_decoder_cache=True,
        )
        state.decoder_cache = output.decoder_cache
        token = output.logits[0, -1].argmax().item()
        state.tokens.append(token)
        state.decoder_steps += 1
        is_blank = token == runner.model.config.blank_token_id
        symbols = 0 if is_blank else state.symbols_at_frame + 1
        advance = is_blank or symbols >= runner.model.max_symbols_per_step
        state.symbols_at_frame = 0 if advance else symbols
        state.durations.append(int(advance))
        frame_index += int(advance)


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


@pytest.fixture
def tiny_model_config() -> Nemotron3_5AsrConfig:
    return Nemotron3_5AsrConfig(
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


@pytest.mark.parametrize("blank_only", [False, True])
@pytest.mark.parametrize(
    "prior_chunks,frames,lookahead,first_frames",
    [
        ((0, 0), 8, 0, 8),
        ((1, 3), 8, 0, 8),
        ((3, 5), 8, 0, 8),
        ((1, 3), 6, 3, 6),
        ((3, 5), 6, 3, 6),
        ((0, 3), 8, 3, 7),
        ((3, 0), 8, 3, 7),
        ((0, 6), 2, 0, 1),
    ],
)
def test_local_model_preserves_streaming_results_and_caches_when_batched(
    tmp_path: Path,
    tiny_model_config: Nemotron3_5AsrConfig,
    prior_chunks: tuple[int, int],
    frames: int,
    lookahead: int,
    first_frames: int,
    blank_only: bool,
) -> None:
    config = tiny_model_config
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
    if blank_only:
        with torch.no_grad():
            runner.model.joint.head.weight.zero_()
            runner.model.joint.head.bias.fill_(-10)
            runner.model.joint.head.bias[config.blank_token_id] = 10
    else:
        pass
    runner.device = torch.device("cpu")
    runner.model_lock = threading.Lock()
    runner.encoder_pool_layout = EncoderPoolLayout.from_model(runner.model)
    runner.encoder_state_pool = None
    runner.enable_encoder_cuda_graph = False
    runner.encoder_graph_runner = None
    runner.configure_encoder_state_pool(2)

    def decode_rows(
        rows: list[torch.Tensor], *, skip_special_tokens: bool
    ) -> list[str]:
        return [str(row.tolist()) for row in rows]

    runner.processor = SimpleNamespace(
        default_num_lookahead_tokens=lookahead,
        batch_decode=decode_rows,
    )
    serial = [ReferenceDecodeState.create(runner.model) for _ in range(2)]
    batched = [runner.new_streaming_decode_state() for _ in range(2)]
    for index, count in enumerate(prior_chunks):
        for chunk_index in range(count):
            chunk = Nemotron3_5ASRPreparedChunk(
                input_features=torch.full(
                    (1, first_frames if chunk_index == 0 else frames, 4),
                    float(index + chunk_index),
                ),
                prompt_ids=torch.tensor([index]),
            )
            run_reference_chunk(runner, serial[index], chunk)
            runner.run_streaming_batch(
                [batched[index]], [chunk], requested_languages=["auto"]
            )
    for chunk_index in range(2):
        chunks = [
            Nemotron3_5ASRPreparedChunk(
                input_features=torch.full(
                    (
                        1,
                        (
                            first_frames
                            if prior_chunks[index] == 0 and chunk_index == 0
                            else frames
                        ),
                        4,
                    ),
                    float(index + chunk_index),
                ),
                prompt_ids=torch.tensor([index]),
            )
            for index in range(2)
        ]
        for state, chunk in zip(serial, chunks):
            run_reference_chunk(runner, state, chunk)
        order = [0, 1] if chunk_index == 0 else [1, 0]
        runner.run_streaming_batch(
            [batched[index] for index in order],
            [chunks[index] for index in order],
            requested_languages=["auto", "auto"],
        )
        for expected, actual in zip(serial, batched):
            assert actual.tokens == expected.tokens
            assert actual.durations == expected.durations
            assert actual.decoder_steps == expected.decoder_steps
            assert actual.encoder_frames == expected.encoder_frames
            assert actual.symbols_at_frame == expected.symbols_at_frame
            torch.testing.assert_close(
                actual.decoder_cache.cache, expected.decoder_cache.cache
            )
            torch.testing.assert_close(
                actual.decoder_cache.hidden_state, expected.decoder_cache.hidden_state
            )
            torch.testing.assert_close(
                actual.decoder_cache.cell_state, expected.decoder_cache.cell_state
            )
            assert actual.encoder_slot.seen_frames == expected.encoder_frames
        assert (
            batched[0].decoder_cache.cache.data_ptr()
            != batched[1].decoder_cache.cache.data_ptr()
        )
    runner.close()


@torch.inference_mode()
def test_encoder_outputs_after_slot_reuse(
    tiny_model_config: Nemotron3_5AsrConfig,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Nemotron3_5AsrForRNNT(tiny_model_config).eval()
    generator = torch.Generator().manual_seed(11)
    pool = NemotronEncoderStatePool(EncoderPoolLayout.from_model(model), 2)
    slots = [pool.acquire(), pool.acquire()]
    references = [ReferenceDecodeState.create(model) for _ in slots]
    for step, rows in enumerate(((0,), (1, 0), (0, 1), (1, 0), (0, 1))):
        if step == 3:
            released_slot_id = slots[0].slot_id
            slots[0].release()
            slots[0] = pool.acquire()
            assert slots[0].slot_id == released_slot_id
            references[0] = ReferenceDecodeState.create(model)
        else:
            pass
        features, expected = [], []
        prompt_ids = torch.tensor(rows)
        for index, row in enumerate(rows):
            reference = references[row]
            mel_frames = 7 if reference.attention_cache.get_seq_length() == 0 else 8
            features.append(torch.randn(1, mel_frames, 4, generator=generator))
            expected.append(
                model.get_audio_features(
                    input_features=features[index],
                    prompt_ids=prompt_ids[index : index + 1],
                    past_key_values=reference.attention_cache,
                    padding_cache=reference.padding_cache,
                    num_lookahead_tokens=3,
                    use_cache=True,
                ).pooler_output
            )
        actual = encode_pooled_streaming_batch(
            model,
            features,
            prompt_ids,
            encoder_slots=[slots[row] for row in rows],
            num_lookahead_tokens=3,
        )
        # note (wirybeaver): Batch size and padded history change floating-point reduction order.
        torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-4, atol=1e-5)
    pool.close()


@torch.inference_mode()
def test_conformer_pointwise_linear_matches_conv1d(
    tiny_model_config: Nemotron3_5AsrConfig,
) -> None:
    """Both pointwise paths keep the Conv1d result of the convolution module."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Nemotron3_5AsrForRNNT(tiny_model_config).eval()
    convolution_module = model.encoder.layers[0].conv
    hidden_size = tiny_model_config.encoder_config.hidden_size
    # note (Emre): The model initializes the biases to zero, so a dropped bias argument would still pass.
    convolution_module.pointwise_conv1.bias.copy_(
        torch.linspace(-1.0, 1.5, 2 * hidden_size)
    )
    convolution_module.pointwise_conv2.bias.copy_(
        torch.linspace(0.75, -0.5, hidden_size)
    )
    generator = torch.Generator().manual_seed(11)
    hidden_states = torch.randn(2, 6, hidden_size, generator=generator)
    all_masked_rows = torch.zeros(2, 1, 6, dtype=torch.bool)
    all_masked_rows[0, 0, -1] = True

    expected = convolution_module.pointwise_conv1(hidden_states.transpose(1, 2))
    expected = torch.nn.functional.glu(expected, dim=1)
    expected = expected.masked_fill(all_masked_rows, 0.0)
    expected = convolution_module.depthwise_conv(expected)
    expected = convolution_module.norm(expected.transpose(1, 2)).transpose(1, 2)
    expected = convolution_module.activation(expected)
    expected = convolution_module.pointwise_conv2(expected).transpose(1, 2)

    for use_linear_pointwise in (False, True):
        actual = convolution_module(
            hidden_states,
            all_masked_rows=all_masked_rows,
            use_linear_pointwise=use_linear_pointwise,
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_streaming_encoder_graph_replay_matches_eager(
    tiny_model_config: Nemotron3_5AsrConfig,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Nemotron3_5AsrForRNNT(tiny_model_config).eval().cuda()
    pool = NemotronEncoderStatePool(EncoderPoolLayout.from_model(model), 6)
    graph_runner = NemotronStreamingEncoderGraphRunner(
        model,
        pool,
        subsequent_mel_frames=8,
        num_lookahead_tokens=3,
        max_batch_size=2,
    )
    assert set(graph_runner.captured_batches) == {1, 2}
    graph_slots = [pool.acquire(), pool.acquire()]
    eager_slots = [pool.acquire(), pool.acquire()]
    prompt_ids = torch.tensor([0, 1], device="cuda")
    generator = torch.Generator().manual_seed(11)
    for step in range(5):
        mel_frames = 7 if step == 0 else 8
        features = []
        for _ in graph_slots:
            features.append(torch.randn(1, mel_frames, 4, generator=generator).cuda())
        graph_output = encode_pooled_streaming_batch(
            model,
            features,
            prompt_ids,
            encoder_slots=graph_slots,
            num_lookahead_tokens=3,
            graph_runner=graph_runner,
        )
        eager_output = encode_pooled_streaming_batch(
            model,
            features,
            prompt_ids,
            encoder_slots=eager_slots,
            num_lookahead_tokens=3,
        )
        assert torch.equal(graph_output, eager_output)
    for graph_slot, eager_slot in zip(graph_slots, eager_slots, strict=True):
        assert torch.equal(
            pool.key_states[:, graph_slot.slot_id],
            pool.key_states[:, eager_slot.slot_id],
        )
        assert torch.equal(
            pool.value_states[:, graph_slot.slot_id],
            pool.value_states[:, eager_slot.slot_id],
        )
        for convolution_states in pool.convolution_states.values():
            assert torch.equal(
                convolution_states[graph_slot.slot_id],
                convolution_states[eager_slot.slot_id],
            )
        assert (
            pool.seen_frames[graph_slot.slot_id] == pool.seen_frames[eager_slot.slot_id]
        )
    graph_runner.close()
    pool.close()


def test_parakeet_compat_forwards_cache_aware_encoder_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

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
        Mock(
            return_value=(
                input_features,
                "input_features",
                dict(model_kwargs),
            )
        ),
    )

    calls = []

    class FakeModel(ParakeetRNNTGenerationMixin):
        def get_audio_features(
            self,
            *,
            input_features: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attention_mask: bool,
            padding_cache: str,
            num_lookahead_tokens: int,
        ) -> NemotronAsrStreamingEncoderModelOutput:
            calls.append(
                dict(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_attention_mask=output_attention_mask,
                    padding_cache=padding_cache,
                    num_lookahead_tokens=num_lookahead_tokens,
                )
            )
            return NemotronAsrStreamingEncoderModelOutput(
                attention_mask=torch.ones(2, 3, dtype=torch.long),
                last_hidden_state=torch.zeros(2, 3, 4),
            )

    _, input_name, prepared = (
        FakeModel()._prepare_model_inputs()
    )  # noqa: leading-underscore  # Required Transformers hook

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
