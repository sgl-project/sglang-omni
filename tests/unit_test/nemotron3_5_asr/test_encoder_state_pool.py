# SPDX-License-Identifier: Apache-2.0
"""Persistent encoder slots preserve eager outputs, decoder results and ownership."""

import math
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.nemotron3_5_asr.decoder import Nemotron3_5ASRDecodeState
from sglang_omni.models.nemotron3_5_asr.encoder import encode_pooled_streaming_batch
from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderPoolLayout,
    NemotronEncoderStatePool,
)
from sglang_omni.models.nemotron3_5_asr.model_runner import (
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRPreparedChunk,
)
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from tests.unit_test.nemotron3_5_asr.test_hf_compat import (
    ReferenceDecodeState,
    run_reference_chunk,
)


@pytest.fixture
def model() -> Iterator[Nemotron3_5AsrForRNNT]:
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
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 2,
            "num_mel_bins": 4,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "conv_kernel_size": 9,
            "sliding_window": 9,
        },
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield Nemotron3_5AsrForRNNT(config).eval()


def new_state(
    model: Nemotron3_5AsrForRNNT, pool: NemotronEncoderStatePool
) -> Nemotron3_5ASRDecodeState:
    return Nemotron3_5ASRDecodeState(
        tokens=[model.config.blank_token_id],
        durations=[0],
        decoder_cache=Nemotron3_5AsrRNNTDecoderCache(model.config),
        encoder_slot=pool.acquire(),
    )


@pytest.mark.parametrize("lookahead_tokens", [0, 3, 6, 13])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="requires CUDA encoder state parity",
                ),
            ],
        ),
    ],
)
@torch.inference_mode()
def test_pooled_encoder_preserves_mixed_age_batches_and_slot_reuse(
    model: Nemotron3_5AsrForRNNT, lookahead_tokens: int, device: str
) -> None:
    model = model.to(device)
    pool = NemotronEncoderStatePool(
        EncoderPoolLayout.from_model(model), capacity_slots=2
    )
    reference = [ReferenceDecodeState.create(model) for _ in range(2)]
    pooled = [new_state(model, pool) for _ in range(2)]
    generator = torch.Generator(device=device).manual_seed(43)
    retained_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def check_batch(rows: list[int]) -> None:
        features = [
            torch.randn(
                1,
                (
                    1 + 8 * lookahead_tokens
                    if pooled[row].encoder_slot.seen_frames == 0
                    else 8 * (lookahead_tokens + 1)
                ),
                4,
                device=device,
                generator=generator,
            )
            for row in rows
        ]
        prompt_ids = torch.tensor(rows, device=device)
        expected = torch.cat(
            [
                model.get_audio_features(
                    input_features=features[index],
                    prompt_ids=prompt_ids[index : index + 1],
                    past_key_values=reference[row].attention_cache,
                    padding_cache=reference[row].padding_cache,
                    num_lookahead_tokens=lookahead_tokens,
                    use_cache=True,
                ).pooler_output
                for index, row in enumerate(rows)
            ]
        )
        actual = encode_pooled_streaming_batch(
            model,
            features,
            prompt_ids,
            encoder_slots=[pooled[row].encoder_slot for row in rows],
            num_lookahead_tokens=lookahead_tokens,
        )
        torch.testing.assert_close(actual, expected)
        retained_outputs.append((actual, expected))

    try:
        rollover_steps = (
            math.ceil(
                model.config.encoder_config.sliding_window / (lookahead_tokens + 1)
            )
            + 1
        )
        for _ in range(rollover_steps):
            check_batch([0])
        check_batch([1, 0])
        check_batch([0, 1])
        pooled[0].release_encoder_state()
        pooled[0] = new_state(model, pool)
        reference[0] = ReferenceDecodeState.create(model)
        check_batch([1, 0])
        check_batch([0, 1])
        for actual, expected in retained_outputs:
            torch.testing.assert_close(actual, expected)
    finally:
        pool.close()
    assert not pool.active_slots and pool.nbytes == 0


@torch.inference_mode()
def test_pool_accounting_and_idempotent_release(model: Nemotron3_5AsrForRNNT) -> None:
    pool = NemotronEncoderStatePool(
        EncoderPoolLayout.from_model(model), capacity_slots=2
    )
    first, second = pool.acquire(), pool.acquire()
    expected_bytes = pool.nbytes
    assert expected_bytes == first.nbytes + second.nbytes
    first.release()
    assert first.nbytes == 0 and pool.nbytes == expected_bytes
    replacement = pool.acquire()
    assert replacement.seen_frames == 0 and not replacement.is_released
    first.release()
    assert not replacement.is_released and replacement.nbytes == second.nbytes
    pool.close()
    assert replacement.is_released and second.is_released
    assert pool.nbytes == 0


@torch.inference_mode()
def test_pooled_runner_matches_reference_and_releases_model_state(
    model: Nemotron3_5AsrForRNNT,
) -> None:
    runner = object.__new__(Nemotron3_5ASRModelRunner)
    runner.model = model
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    runner.model_lock = threading.Lock()
    runner.encoder_pool_layout = EncoderPoolLayout.from_model(model)
    runner.encoder_state_pool = None

    def decode_tokens(
        token_ids: list[torch.Tensor], *, skip_special_tokens: bool
    ) -> list[str]:
        return [str(tokens.tolist()) for tokens in token_ids]

    runner.processor = SimpleNamespace(
        default_num_lookahead_tokens=3, batch_decode=decode_tokens
    )
    runner.configure_encoder_state_pool(2)
    pool = runner.encoder_state_pool
    assert pool is not None
    pooled = [runner.new_streaming_decode_state() for _ in range(2)]
    reference = [ReferenceDecodeState.create(model) for _ in range(2)]
    try:
        for rows in ([0], [1, 0], [0, 1], [1, 0]):
            chunks = [
                Nemotron3_5ASRPreparedChunk(
                    input_features=torch.full(
                        (1, 25 if pooled[row].encoder_slot.seen_frames == 0 else 32, 4),
                        float(row),
                    ),
                    prompt_ids=torch.tensor([row]),
                )
                for row in rows
            ]
            for row, chunk in zip(rows, chunks, strict=True):
                run_reference_chunk(runner, reference[row], chunk)
            result = runner.run_streaming_batch(
                [pooled[row] for row in rows],
                chunks,
                requested_languages=["auto"] * len(rows),
            )
            assert result.raw_texts == [str(reference[row].tokens) for row in rows]
            for row in rows:
                assert pooled[row].tokens == reference[row].tokens
                assert pooled[row].durations == reference[row].durations
    finally:
        runner.close()
    assert pool.nbytes == 0 and not pool.active_slots
    assert all(state.encoder_slot.is_released for state in pooled)
