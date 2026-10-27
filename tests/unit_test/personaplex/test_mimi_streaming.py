# SPDX-License-Identifier: Apache-2.0
"""Chunked Mimi must land on the samples a whole-sequence pass produces."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang_omni.models.personaplex.architecture import MIMI
from sglang_omni.models.personaplex.components.causal_conv import (
    CausalConv1d,
    CausalConvTranspose1d,
)
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.models.personaplex.components.mimi_loader import (
    load_mimi_codec,
    rename_mimi_key,
)
from sglang_omni.models.personaplex.components.mimi_transformer import (
    AttentionState,
    MimiAttention,
    MimiTransformer,
    TransformerState,
)
from sglang_omni.models.personaplex.mimi_encode_graph import MimiEncodeSlots


def test_causal_conv_chunks_match_whole():
    torch.manual_seed(1)
    for kernel, stride, dilation, mode in (
        (7, 1, 1, "constant"),
        (8, 4, 1, "constant"),
        (3, 1, 2, "constant"),
        (4, 2, 1, "replicate"),
    ):
        conv = CausalConv1d(
            3, 5, kernel, stride=stride, dilation=dilation, pad_mode=mode
        )
        x = torch.randn(2, 3, 48)
        whole = conv(x)
        state = conv.init_state(batch_size=2)
        chunks = [conv.step(x[..., i : i + 8], state) for i in range(0, 48, 8)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_causal_conv_transpose_chunks_match_whole():
    torch.manual_seed(2)
    for kernel, stride, groups in ((16, 8, 1), (4, 2, 6)):
        convtr = CausalConvTranspose1d(6, 6, kernel, stride=stride, groups=groups)
        x = torch.randn(1, 6, 10)
        whole = convtr(x)
        state = convtr.init_state(batch_size=1)
        chunks = [convtr.step(x[..., i : i + 1], state) for i in range(10)]
        torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_codec_encode_and_decode_step_match_whole(random_codec):
    codec = random_codec
    frames = 5
    x = torch.randn(1, 1, codec.samples_per_frame * frames)
    codes = codec.encode(x)
    assert codes.shape == (1, 8, frames)
    state = codec.init_encode_state(batch_size=1)
    step = codec.samples_per_frame
    chunked = torch.cat(
        [
            codec.encode_step(x[..., i : i + step], state)
            for i in range(0, x.shape[-1], step)
        ],
        -1,
    )
    assert torch.equal(chunked, codes)

    whole = codec.decode(codes)
    assert whole.shape == (1, 1, codec.samples_per_frame * frames)
    state = codec.init_decode_state(batch_size=1, max_chunk_frames=1)
    chunked = torch.cat(
        [codec.decode_step(codes[..., f : f + 1], state) for f in range(frames)], -1
    )
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


def test_a_reset_decode_state_decodes_like_a_fresh_one(random_codec):
    codec = random_codec
    generator = torch.Generator().manual_seed(4)
    earlier, codes = torch.randint(0, 2048, (2, 1, 8, 3), generator=generator)

    def stream(codes_BKF: torch.Tensor, state: MimiDecodeState) -> torch.Tensor:
        return torch.cat(
            [codec.decode_step(codes_BKF[..., f : f + 1], state) for f in range(3)],
            -1,
        )

    fresh = stream(codes, codec.init_decode_state(batch_size=1, max_chunk_frames=1))
    state = codec.init_decode_state(batch_size=1, max_chunk_frames=1)
    stream(earlier, state)
    state.reset()
    assert torch.equal(stream(codes, state), fresh)


def test_encode_rows_match_separate_streams_after_reset_and_reordering() -> None:
    """A new or reused session must not inherit another row's padding or history."""
    torch.manual_seed(5)
    spec = replace(
        MIMI,
        n_filters=2,
        ratios=(2, 2),
        dim=16,
        num_heads=2,
        num_layers=2,
        ffn_dim=8,
        context=6,
        codebook_dim=8,
        codebook_size=32,
        num_codebooks=2,
    )
    codec = MimiCodec(spec, mode="encode").eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.3)
        for quantizer in (codec.quantizer.rvq_first, codec.quantizer.rvq_rest):
            for layer in quantizer.vq.layers:
                layer.codebook.embedding_sum.normal_(std=0.1)
    state = codec.init_encode_state(batch_size=3)
    references = [codec.init_encode_state(batch_size=1) for _ in range(3)]
    for iteration, row_order in enumerate(
        ([2], [0, 2], [1, 2, 0], [2, 0], [0, 2], [1, 2, 0])
    ):
        if iteration == 4:
            state.reset_row(0)
            references[0] = codec.init_encode_state(batch_size=1)
        elif iteration == 5:
            state.reset()
            references = [codec.init_encode_state(batch_size=1) for _ in range(3)]
        else:
            pass
        rows = torch.tensor(row_order)
        waveform = torch.randn(len(row_order), 1, codec.samples_per_frame)
        selected = state.select_rows(rows)
        batched_codes = codec.encode_step(waveform, selected)
        state.store_rows(rows, selected)
        expected_codes = torch.cat(
            [
                codec.encode_step(waveform[index : index + 1], references[row])
                for index, row in enumerate(row_order)
            ]
        )
        torch.testing.assert_close(batched_codes, expected_codes, atol=0, rtol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_encode_graph_matches_eager_across_mixed_rows_and_slot_reuse(
    random_codec: MimiCodec,
) -> None:
    codec = random_codec.cuda()
    graphed = MimiEncodeSlots(codec, num_slots=3, max_batch_size=2, enable_graphs=True)
    eager = MimiEncodeSlots(codec, num_slots=3, max_batch_size=2, enable_graphs=False)
    assert set(graphed.graphs) == {1, 2}
    plan = [((0,), 1), ((1, 0), 1), ((2, 0), 2), ((0, 2), 1), ((1,), 1)]
    generator = torch.Generator().manual_seed(6)
    waveforms = [
        torch.randn(
            len(streams),
            1,
            codec.samples_per_frame * frame_count,
            generator=generator,
        ).cuda()
        for streams, frame_count in plan
    ]
    first_round_codes: list[torch.Tensor] = []
    for round_index in range(2):
        pool_slots: list[list[int]] = []
        for pool in (graphed, eager):
            acquired_slots: list[int] = []
            for _ in range(3):
                slot_index = pool.acquire_slot()
                assert slot_index is not None
                acquired_slots.append(slot_index)
            pool_slots.append(acquired_slots)
        for step_index, ((streams, _), waveform) in enumerate(
            zip(plan, waveforms, strict=True)
        ):
            graphed_codes = graphed.encode_step(
                waveform, [pool_slots[0][stream] for stream in streams]
            ).clone()
            eager_codes = eager.encode_step(
                waveform, [pool_slots[1][stream] for stream in streams]
            )
            torch.testing.assert_close(graphed_codes, eager_codes, atol=0, rtol=0)
            if round_index == 0:
                first_round_codes.append(graphed_codes)
            else:
                torch.testing.assert_close(
                    graphed_codes, first_round_codes[step_index], atol=0, rtol=0
                )
        for pool, acquired_slots in zip((graphed, eager), pool_slots, strict=True):
            for slot_index in acquired_slots:
                pool.release_slot(slot_index)


@pytest.mark.parametrize("mode", ["encode", "decode", "both"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
def test_selectively_loaded_codec_matches_full_codec(
    random_codec: MimiCodec,
    mimi_checkpoint: Path,
    mode: Literal["encode", "decode", "both"],
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    else:
        pass
    reference_codec = random_codec.to(device=device)
    codec = load_mimi_codec(mimi_checkpoint, device=device, mode=mode)
    assert codec.device == reference_codec.device
    assert not codec.training
    assert all(parameter.dtype == torch.float32 for parameter in codec.parameters())
    if mode != "both":
        assert sum(tensor.numel() for tensor in codec.state_dict().values()) < sum(
            tensor.numel() for tensor in reference_codec.state_dict().values()
        )
    else:
        pass

    frame_count = 5
    waveform = torch.randn(1, 1, frame_count * codec.samples_per_frame, device=device)
    codes = reference_codec.encode(waveform)
    if mode != "decode":
        torch.testing.assert_close(codec.encode(waveform), codes, atol=0, rtol=0)
        encode_state = codec.init_encode_state(batch_size=1)
        reference_encode_state = reference_codec.init_encode_state(batch_size=1)
        for waveform_chunk in waveform.split(codec.samples_per_frame, dim=-1):
            torch.testing.assert_close(
                codec.encode_step(waveform_chunk, encode_state),
                reference_codec.encode_step(waveform_chunk, reference_encode_state),
                atol=0,
                rtol=0,
            )
    else:
        pass
    if mode != "encode":
        expected_waveform = reference_codec.decode(codes)
        torch.testing.assert_close(
            codec.decode(codes), expected_waveform, atol=0, rtol=0
        )
        for chunk_frames in (1, 4):
            decode_state = codec.init_decode_state(
                batch_size=1, max_chunk_frames=chunk_frames
            )
            reference_decode_state = reference_codec.init_decode_state(
                batch_size=1, max_chunk_frames=chunk_frames
            )
            for code_chunk in codes.split(chunk_frames, dim=-1):
                torch.testing.assert_close(
                    codec.decode_step(code_chunk, decode_state),
                    reference_codec.decode_step(code_chunk, reference_decode_state),
                    atol=0,
                    rtol=0,
                )
    else:
        pass


@pytest.mark.parametrize(
    "mode,missing_weight",
    [
        ("encode", "quantizer.rvq_first.input_proj.weight"),
        ("decode", "quantizer.rvq_first.output_proj.weight"),
        ("encode", "quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum"),
        ("decode", "quantizer.rvq_rest.vq.layers.0._codebook.cluster_usage"),
    ],
)
def test_selective_loading_requires_quantizer_weights(
    mimi_checkpoint: Path,
    tmp_path: Path,
    mode: Literal["encode", "decode"],
    missing_weight: str,
) -> None:
    checkpoint_weights = load_file(str(mimi_checkpoint))
    checkpoint_weights.pop(missing_weight)
    with TemporaryDirectory(dir=tmp_path) as checkpoint_directory:
        incomplete_checkpoint = Path(checkpoint_directory) / "incomplete.safetensors"
        save_file(checkpoint_weights, str(incomplete_checkpoint))
        with pytest.raises(RuntimeError, match="Missing key"):
            load_mimi_codec(incomplete_checkpoint, device="cpu", mode=mode)


@pytest.mark.parametrize("mode", ["encode", "decode"])
def test_selective_loading_rejects_unexpected_active_weights(
    mimi_checkpoint: Path,
    tmp_path: Path,
    mode: Literal["encode", "decode"],
) -> None:
    checkpoint_weights = load_file(str(mimi_checkpoint))
    checkpoint_weights[f"{mode}r.unexpected.weight"] = torch.zeros(1)
    with TemporaryDirectory(dir=tmp_path) as checkpoint_directory:
        invalid_checkpoint = Path(checkpoint_directory) / "invalid.safetensors"
        save_file(checkpoint_weights, str(invalid_checkpoint))
        with pytest.raises(RuntimeError, match="Unexpected key"):
            load_mimi_codec(invalid_checkpoint, device="cpu", mode=mode)


def test_checkpoint_names_map_onto_the_module_tree():
    codec = MimiCodec()
    expected = set(codec.state_dict())
    checkpoint_names = [
        "encoder.model.0.conv.conv.weight",
        "encoder.model.1.block.1.conv.conv.bias",
        "decoder.model.2.convtr.convtr.weight",
        "downsample.conv.conv.conv.weight",
        "upsample.convtr.convtr.convtr.weight",
        "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight",
        "decoder_transformer.transformer.layers.7.layer_scale_2.scale",
        "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        "quantizer.rvq_rest.vq.layers.6._codebook.cluster_usage",
    ]
    for name in checkpoint_names:
        assert rename_mimi_key(name) in expected, name
    assert (
        rename_mimi_key("quantizer.rvq_rest.vq.layers.7._codebook.embedding_sum")
        is None
    )


SMALL = replace(MIMI, context=6, num_layers=2, dim=16, num_heads=2, ffn_dim=8)


def small_transformer() -> MimiTransformer:
    torch.manual_seed(4)
    transformer = MimiTransformer(SMALL).eval()
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.2)
    return transformer


def influenced_steps(chunk: int) -> list[int]:
    """Which steps still depend on step 0, feeding chunk steps at a time."""
    torch.manual_seed(0)
    attention = MimiAttention(
        dim=8, num_heads=2, context=SMALL.context, max_period=1e4, write_chunk=chunk
    )
    with torch.no_grad():
        attention.in_proj_weight.normal_()
        attention.out_proj.weight.normal_()
    x = torch.randn(1, 4 * SMALL.context, 8)
    changed = x.clone()
    changed[:, 0] += 1.0

    def run(inp):
        state = attention.init_state(batch_size=1, max_chunk_steps=chunk)
        with torch.no_grad():
            parts = [
                attention(inp[:, t : t + chunk], offset=t, state=state)
                for t in range(0, inp.shape[1], chunk)
            ]
        return torch.cat(parts, 1)

    differs = (run(x) - run(changed)).abs().amax(dim=(0, 2)) > 1e-6
    return differs.nonzero().flatten().tolist()


def test_the_ring_drops_its_oldest_step_as_the_reference_does():
    """History at or before each frame's end minus the context is excluded."""
    assert influenced_steps(1) == list(range(SMALL.context - 1))
    assert influenced_steps(SMALL.frame_ratio) == list(
        range(SMALL.context - SMALL.frame_ratio)
    )


@pytest.mark.parametrize("max_chunk_frames", [1, 3, 8])
@pytest.mark.parametrize("length_in_steps", [5, 6, 67])
def test_aggregated_transformer_chunks_match_whole_across_ring_wraps(
    max_chunk_frames: int, length_in_steps: int
) -> None:
    transformer = small_transformer()
    latent = torch.randn(1, SMALL.dim, length_in_steps)
    max_chunk_steps = max_chunk_frames * SMALL.frame_ratio
    state = transformer.init_state(batch_size=1, max_chunk_steps=max_chunk_steps)
    with torch.inference_mode():
        whole = transformer(latent)
        chunks = [
            transformer.step(latent[..., start : start + max_chunk_steps], state)
            for start in range(0, length_in_steps, max_chunk_steps)
        ]
    torch.testing.assert_close(torch.cat(chunks, -1), whole, atol=1e-6, rtol=1e-5)


def test_rows_at_different_offsets_step_together_like_their_own_streams():
    transformer = small_transformer()
    chunk = SMALL.frame_ratio
    # Steps each row has taken: none, a few, and enough to wrap its ring.
    taken = [0, 2 * chunk, 3 * SMALL.context]
    x = torch.randn(len(taken), SMALL.dim, max(taken) + chunk)
    with torch.no_grad():
        states = []
        for row, steps in enumerate(taken):
            state = transformer.init_state(
                batch_size=1, max_chunk_steps=SMALL.frame_ratio
            )
            for t in range(0, steps, chunk):
                transformer.step(x[row : row + 1, :, t : t + chunk], state)
            states.append(state)
        stacked = TransformerState(
            offset=torch.cat([state.offset for state in states]),
            layers=[
                AttentionState(
                    keys=torch.cat([state.layers[layer].keys for state in states]),
                    values=torch.cat([state.layers[layer].values for state in states]),
                )
                for layer in range(SMALL.num_layers)
            ],
        )
        next_chunks = [
            x[row : row + 1, :, steps : steps + chunk]
            for row, steps in enumerate(taken)
        ]
        batched = transformer.step(torch.cat(next_chunks), stacked)
        separate = [
            transformer.step(next_chunk, state)
            for next_chunk, state in zip(next_chunks, states, strict=True)
        ]
    torch.testing.assert_close(batched, torch.cat(separate), atol=1e-6, rtol=1e-5)
    assert stacked.offset.tolist() == [steps + chunk for steps in taken]
