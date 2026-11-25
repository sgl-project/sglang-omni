# SPDX-License-Identifier: Apache-2.0
"""Parity tests for ``GreedyRNNTDecoder`` against ``decode_streaming_batch``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.nemotron3_5_asr.decoder import (
    Nemotron3_5ASRDecodeState,
    decode_streaming_batch,
)
from sglang_omni.models.nemotron3_5_asr.rnnt_greedy import GreedyRNNTDecoder
from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoder,
    Nemotron3_5AsrRNNTJointNetwork,
)


class TinyRNNTModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 32,
        hidden_size: int = 16,
        num_layers: int = 2,
        max_symbols_per_step: int = 4,
    ) -> None:
        super().__init__()
        self.config = Nemotron3_5AsrConfig(
            vocab_size=vocab_size,
            blank_token_id=vocab_size - 1,
            decoder_hidden_size=hidden_size,
            num_decoder_layers=num_layers,
            hidden_act="relu",
        )
        self.max_symbols_per_step = max_symbols_per_step
        self.decoder = Nemotron3_5AsrRNNTDecoder(self.config)
        self.joint = Nemotron3_5AsrRNNTJointNetwork(self.config)


def make_states(
    config: Nemotron3_5AsrConfig, count: int
) -> list[Nemotron3_5ASRDecodeState]:
    blank = int(config.blank_token_id)
    return [
        Nemotron3_5ASRDecodeState(
            tokens=[blank],
            durations=[0],
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(config),
            encoder_slot=SimpleNamespace(release=lambda: None),  # type: ignore[arg-type]
        )
        for _ in range(count)
    ]


def test_greedy_rnnt_decoder_matches_reference_across_streaming_chunks() -> None:
    torch.manual_seed(7)
    model = TinyRNNTModel(vocab_size=24, hidden_size=16, max_symbols_per_step=3).eval()
    # Bias non-blank emissions on select tokens so we exercise multi-symbol
    # emissions, forced frame advances, and blank skips.
    with torch.no_grad():
        model.joint.head.bias.fill_(-0.2)
        model.joint.head.bias[model.config.blank_token_id] = 0.05

    ref_states = make_states(model.config, 4)
    fast_states = make_states(model.config, 4)
    fast_decoder = GreedyRNNTDecoder(model, use_cuda_graphs=False)  # type: ignore[arg-type]

    for chunk_idx, lengths in enumerate(([7, 7, 5, 0], [7, 4, 7, 3])):
        max_t = max(lengths)
        frames = torch.randn(4, max_t, model.config.decoder_hidden_size) * 2.5
        with torch.inference_mode():
            for row, length in enumerate(lengths):
                if length > 0:
                    decode_streaming_batch(
                        model,  # type: ignore[arg-type]
                        [ref_states[row]],
                        frames[row : row + 1, :length],
                        [None],
                    )
                else:
                    pass
            fast_decoder.decode(fast_states, frames, lengths)

        for row, (ref, fast) in enumerate(zip(ref_states, fast_states, strict=True)):
            assert (
                fast.tokens == ref.tokens
            ), f"chunk={chunk_idx} row={row} tokens mismatch"
            assert (
                fast.durations == ref.durations
            ), f"chunk={chunk_idx} row={row} durations mismatch"
            assert fast.encoder_frames == ref.encoder_frames
            assert fast.decoder_steps == ref.decoder_steps
            assert fast.decoder_cache.is_initialized == ref.decoder_cache.is_initialized
            if ref.decoder_cache.is_initialized:
                torch.testing.assert_close(
                    fast.decoder_cache.hidden_state,
                    ref.decoder_cache.hidden_state,
                    atol=1e-5,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    fast.decoder_cache.cell_state,
                    ref.decoder_cache.cell_state,
                    atol=1e-5,
                    rtol=1e-5,
                )
            else:
                pass


def test_greedy_rnnt_can_decode_guards_token_limits_and_mid_frame() -> None:
    model = TinyRNNTModel(max_symbols_per_step=4).eval()
    decoder = GreedyRNNTDecoder(model, use_cuda_graphs=False)  # type: ignore[arg-type]
    state = make_states(model.config, 1)[0]

    assert decoder.can_decode(state, frames=7, limit=None)
    assert decoder.can_decode(state, frames=7, limit=29)
    assert not decoder.can_decode(state, frames=7, limit=28)

    state.symbols_at_frame = 1
    assert not decoder.can_decode(state, frames=7, limit=None)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_rnnt_decoder_cuda_graph_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    model = (
        TinyRNNTModel(vocab_size=64, hidden_size=64, max_symbols_per_step=4)
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    # Give select tokens clear logit margins so bf16 rounding never causes
    # random tie-breaks while still exercising non-blank emissions and blanks.
    with torch.no_grad():
        model.joint.head.weight.mul_(0.5)
        model.joint.head.bias.fill_(-1.0)
        model.joint.head.bias[model.config.blank_token_id] = 0.2
        model.joint.head.bias[1] = 0.0
        model.joint.head.bias[2] = -0.1
    ref_states = make_states(model.config, 3)
    fast_states = make_states(model.config, 3)
    fast_decoder = GreedyRNNTDecoder(model, use_cuda_graphs=True)  # type: ignore[arg-type]
    try:
        fast_decoder.capture_common_graphs(
            torch.device("cuda"), dtype, batch_buckets=(4,), frame_buckets=(8,)
        )
        for lengths in ([7, 7, 5], [7, 3, 7]):
            frames = (
                torch.randn(
                    3, 7, model.config.decoder_hidden_size, device="cuda", dtype=dtype
                )
                * 1.5
            )
            with torch.inference_mode():
                for row, length in enumerate(lengths):
                    decode_streaming_batch(
                        model,  # type: ignore[arg-type]
                        [ref_states[row]],
                        frames[row : row + 1, :length],
                        [None],
                    )
                fast_decoder.decode(fast_states, frames, lengths)
            for ref, fast in zip(ref_states, fast_states, strict=True):
                assert fast.tokens == ref.tokens
                assert fast.durations == ref.durations
                assert fast.decoder_steps == ref.decoder_steps
    finally:
        fast_decoder.close()
