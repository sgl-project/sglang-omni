# SPDX-License-Identifier: Apache-2.0
"""Drama 3 audio decoder against hand-written formulas; cache slots; weight loading."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Literal

import pytest
import torch
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.model_loader.utils import set_default_torch_dtype

from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.weights import (
    checkpoint_part,
    expected_parameter_shards,
    interleave_query_and_gate,
    load_audio_decoder_weights,
    parameter_shard,
)
from tests.unit_test.fishaudio_drama3.dense_attention import CacheEntry, DenseAttention
from tests.unit_test.fishaudio_drama3.test_hf_config import fish_config_dict

# Geometry of fish_config_dict(), written out so the reference never reads the module.
DIM = 32
HEAD_COUNT = 4
KV_HEAD_COUNT = 2
HEAD_DIM = 12
ROTARY_DIM = 6
ROPE_BASE = 1000.0
NORM_EPS = 1e-6
INTERMEDIATE_SIZE = 24
CODEBOOK_SIZE = 8
NUM_CODEBOOKS = 4
SHARED_LAYER_COUNT = 2
RESIDUAL_LAYER_COUNT = NUM_CODEBOOKS - 1
BACKBONE_CODEBOOK_COUNT = 1
TEXT_VOCAB_SIZE = 96
SEMANTIC_BEGIN_ID = 80
SEMANTIC_END_ID = 87
EOS_TOKEN_ID = 3
WINDOW_FRAMES = 16
# No token id is negative, so a window of these never triggers resampling.
EMPTY_WINDOW_TOKEN = -1
BFLOAT16_CACHE_DRIFT = 0.05

DeviceType = Literal["cpu", "cuda"]
CheckpointProblem = Literal[
    "missing-up-half",
    "duplicate",
    "unknown-key",
    "unknown-namespace",
    "fused-gate-up-after-shards",
    "fused-gate-up-before-shards",
    "wrong-shape",
]

# Frames are [main token, code 0, code 1, code 2, code 3]. Request "a" walks
# through text -> semantic -> semantic -> eos -> text -> semantic; its text rows
# carry out-of-range placeholder codes. "b" has one row (no prefill row at all)
# and "c" two.
REQUEST_FRAMES = {
    "a": [
        [10, 99, -4, 99, 99],
        [83, 3, 5, 1, 6],
        [86, 6, 0, 7, 2],
        [EOS_TOKEN_ID, 0, 0, 0, 0],
        [20, -1, -1, -1, -1],
        [81, 1, 4, 4, 3],
    ],
    "b": [[12, 0, 0, 0, 0]],
    "c": [[84, 4, 2, 2, 2], [85, 5, 3, 3, 3]],
}
FIRST_POSITIONS = {"a": 3, "b": 7, "c": 0}


def small_config() -> FishOmniConfig:
    return FishOmniConfig(**fish_config_dict())


def random_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Decoder tensors with the checkpoint names and layouts (flat wqkv, w1, w3)."""
    generator = torch.Generator().manual_seed(seed)

    def random_tensor(*shape: int, scale: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * scale

    checkpoint = {
        "audio_decoder.codebook_embeddings.weight": random_tensor(
            CODEBOOK_SIZE * NUM_CODEBOOKS, DIM, scale=1.0
        )
    }
    layers = [
        (f"audio_decoder_mtp_shared.layers.{index}.", "semantic_embedding")
        for index in range(SHARED_LAYER_COUNT)
    ] + [
        (f"audio_decoder_mtp.layers.{index}.", "codebook_embedding")
        for index in range(RESIDUAL_LAYER_COUNT)
    ]
    for prefix, embedding_name in layers:
        checkpoint[f"{prefix}{embedding_name}.weight"] = random_tensor(
            CODEBOOK_SIZE, DIM, scale=1.0
        )
        for norm_name in (
            "enorm",
            "hnorm",
            "final_layernorm",
            "transformer_layer.attention_norm",
            "transformer_layer.ffn_norm",
        ):
            checkpoint[f"{prefix}{norm_name}.weight"] = random_tensor(DIM, scale=0.1)
        for norm_name in ("q_norm", "k_norm"):
            checkpoint[f"{prefix}transformer_layer.attention.{norm_name}.weight"] = (
                random_tensor(HEAD_DIM, scale=0.1)
            )
        checkpoint[f"{prefix}eh_proj.weight"] = random_tensor(DIM, 2 * DIM, scale=0.2)
        checkpoint[f"{prefix}transformer_layer.attention.wqkv.weight"] = random_tensor(
            (2 * HEAD_COUNT + 2 * KV_HEAD_COUNT) * HEAD_DIM, DIM, scale=0.2
        )
        checkpoint[f"{prefix}transformer_layer.attention.wo.weight"] = random_tensor(
            DIM, HEAD_COUNT * HEAD_DIM, scale=0.2
        )
        for projection_name in ("w1", "w3"):
            checkpoint[
                f"{prefix}transformer_layer.feed_forward.{projection_name}.weight"
            ] = random_tensor(INTERMEDIATE_SIZE, DIM, scale=0.2)
        checkpoint[f"{prefix}transformer_layer.feed_forward.w2.weight"] = random_tensor(
            DIM, INTERMEDIATE_SIZE, scale=0.2
        )
    for index in range(RESIDUAL_LAYER_COUNT):
        checkpoint[f"audio_decoder_mtp.heads.{index}.weight"] = random_tensor(
            CODEBOOK_SIZE, DIM, scale=1.0
        )
    return checkpoint


def loaded_decoder(
    checkpoint: dict[str, torch.Tensor], dtype: torch.dtype, device: DeviceType
) -> AudioDecoder:
    with set_default_torch_dtype(dtype):
        decoder = AudioDecoder(small_config())
    load_audio_decoder_weights(decoder, checkpoint.items())
    return decoder.to(device=device)


@contextmanager
def torch_fused_op_backend() -> Iterator[None]:
    previous_backend = get_fused_op_backend()
    set_fused_op_backend(KernelBackend.TORCH)
    try:
        yield
    finally:
        set_fused_op_backend(previous_backend)


def reference_norm(values: torch.Tensor, gain_minus_one: torch.Tensor) -> torch.Tensor:
    return (
        values
        / torch.sqrt(values.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS)
        * (1.0 + gain_minus_one)
    )


def reference_rope(head: torch.Tensor, position: int) -> torch.Tensor:
    rotated = head.clone()
    for pair in range(ROTARY_DIM // 2):
        angle = position / ROPE_BASE ** (2 * pair / ROTARY_DIM)
        even = head[2 * pair]
        odd = head[2 * pair + 1]
        rotated[2 * pair] = even * math.cos(angle) - odd * math.sin(angle)
        rotated[2 * pair + 1] = odd * math.cos(angle) + even * math.sin(angle)
    return rotated


def is_semantic(token_id: int) -> bool:
    return SEMANTIC_BEGIN_ID <= token_id <= SEMANTIC_END_ID


def reference_codebook_sum(
    checkpoint: dict[str, torch.Tensor], frame: list[int], codebooks: range
) -> torch.Tensor:
    table = checkpoint["audio_decoder.codebook_embeddings.weight"]
    summed = torch.zeros(DIM)
    for codebook in codebooks:
        code = min(max(frame[1 + codebook], 0), CODEBOOK_SIZE - 1)
        summed = summed + table[codebook * CODEBOOK_SIZE + code]
    return summed


def reference_shared_input(
    checkpoint: dict[str, torch.Tensor], frame: list[int]
) -> torch.Tensor:
    if is_semantic(frame[0]):
        return reference_codebook_sum(
            checkpoint, frame, range(1 + BACKBONE_CODEBOOK_COUNT, NUM_CODEBOOKS)
        )
    else:
        return torch.zeros(DIM)


def reference_layer(
    checkpoint: dict[str, torch.Tensor],
    prefix: str,
    hidden: torch.Tensor,
    code: int,
    label_is_semantic: bool,
    shared_input: torch.Tensor | None,
    position: int,
    cache: list[CacheEntry],
) -> torch.Tensor:
    """One tower layer for one row, read straight off the checkpoint tensors."""
    weight = {
        name[len(prefix) :]: tensor
        for name, tensor in checkpoint.items()
        if name.startswith(prefix)
    }
    embedding_name = (
        "semantic_embedding" if "mtp_shared" in prefix else "codebook_embedding"
    )
    embedding = weight[f"{embedding_name}.weight"][min(max(code, 0), CODEBOOK_SIZE - 1)]
    embedding = embedding * (1.0 if label_is_semantic else 0.0)
    if shared_input is not None:
        embedding = embedding + shared_input
    else:
        pass
    mixed = weight["eh_proj.weight"] @ torch.cat(
        (
            reference_norm(embedding, weight["enorm.weight"]),
            reference_norm(hidden, weight["hnorm.weight"]),
        )
    )
    projected = weight["transformer_layer.attention.wqkv.weight"] @ reference_norm(
        mixed, weight["transformer_layer.attention_norm.weight"]
    )
    queries, gates, keys, values = projected.split(
        [
            HEAD_COUNT * HEAD_DIM,
            HEAD_COUNT * HEAD_DIM,
            KV_HEAD_COUNT * HEAD_DIM,
            KV_HEAD_COUNT * HEAD_DIM,
        ]
    )
    queries = torch.stack(
        [
            reference_rope(
                reference_norm(
                    query, weight["transformer_layer.attention.q_norm.weight"]
                ),
                position,
            )
            for query in queries.view(HEAD_COUNT, HEAD_DIM)
        ]
    )
    keys = torch.stack(
        [
            reference_rope(
                reference_norm(
                    key, weight["transformer_layer.attention.k_norm.weight"]
                ),
                position,
            )
            for key in keys.view(KV_HEAD_COUNT, HEAD_DIM)
        ]
    )
    cache.append((position, keys, values.view(KV_HEAD_COUNT, HEAD_DIM)))
    head_outputs = []
    for head in range(HEAD_COUNT):
        kv_head = head // (HEAD_COUNT // KV_HEAD_COUNT)
        visible = [entry for entry in cache if entry[0] <= position]
        scores = torch.stack(
            [
                torch.dot(queries[head], entry[1][kv_head]) / math.sqrt(HEAD_DIM)
                for entry in visible
            ]
        )
        attended = sum(
            probability * entry[2][kv_head]
            for probability, entry in zip(torch.softmax(scores, dim=0), visible)
        )
        head_outputs.append(
            attended * torch.sigmoid(gates.view(HEAD_COUNT, HEAD_DIM)[head])
        )
    residual = mixed + weight["transformer_layer.attention.wo.weight"] @ torch.cat(
        head_outputs
    )
    mlp_input = reference_norm(residual, weight["transformer_layer.ffn_norm.weight"])
    mlp_output = weight["transformer_layer.feed_forward.w2.weight"] @ (
        torch.nn.functional.silu(
            weight["transformer_layer.feed_forward.w1.weight"] @ mlp_input
        )
        * (weight["transformer_layer.feed_forward.w3.weight"] @ mlp_input)
    )
    return reference_norm(residual + mlp_output, weight["final_layernorm.weight"])


class ReferenceTower:
    """Row-at-a-time tower over checkpoint tensors; a cache per (layer, request)."""

    def __init__(self, checkpoint: dict[str, torch.Tensor]) -> None:
        self.checkpoint = checkpoint
        self.caches: dict[tuple[int, str], list[CacheEntry]] = {}

    def layer_prefixes(self) -> list[str]:
        return [
            f"audio_decoder_mtp_shared.layers.{index}."
            for index in range(SHARED_LAYER_COUNT)
        ] + [
            f"audio_decoder_mtp.layers.{index}."
            for index in range(RESIDUAL_LAYER_COUNT)
        ]

    def run(
        self,
        request_id: str,
        hidden: torch.Tensor,
        position: int,
        input_frame: list[int],
        label_frame: list[int],
    ) -> torch.Tensor:
        """All tower layers for one row, given every code of its label frame."""
        shared_input = reference_shared_input(self.checkpoint, input_frame)
        for layer_index, prefix in enumerate(self.layer_prefixes()):
            code_column = (
                1
                if layer_index < SHARED_LAYER_COUNT
                else 1 + layer_index - SHARED_LAYER_COUNT
            )
            hidden = reference_layer(
                self.checkpoint,
                prefix,
                hidden,
                label_frame[code_column],
                is_semantic(label_frame[0]),
                shared_input if layer_index == 0 else None,
                position,
                self.caches.setdefault((layer_index, request_id), []),
            )
        return hidden

    def decode(
        self,
        request_id: str,
        hidden: torch.Tensor,
        position: int,
        input_frame: list[int],
        semantic_logits: torch.Tensor,
    ) -> list[int]:
        """Greedy decode step (top_k 1, empty window)."""
        bias = torch.full((TEXT_VOCAB_SIZE,), float("-inf"))
        bias[SEMANTIC_BEGIN_ID : SEMANTIC_END_ID + 1] = 0.0
        bias[EOS_TOKEN_ID] = 0.0
        main_token = int(torch.argmax(semantic_logits + bias))
        frame = [
            main_token,
            min(max(main_token - SEMANTIC_BEGIN_ID, 0), CODEBOOK_SIZE - 1),
        ]
        shared_input = reference_shared_input(self.checkpoint, input_frame)
        for layer_index, prefix in enumerate(self.layer_prefixes()):
            hidden = reference_layer(
                self.checkpoint,
                prefix,
                hidden,
                frame[-1] if layer_index >= SHARED_LAYER_COUNT else frame[1],
                is_semantic(main_token),
                shared_input if layer_index == 0 else None,
                position,
                self.caches.setdefault((layer_index, request_id), []),
            )
            if layer_index >= SHARED_LAYER_COUNT:
                head_index = layer_index - SHARED_LAYER_COUNT
                logits = (
                    self.checkpoint[f"audio_decoder_mtp.heads.{head_index}.weight"]
                    @ hidden
                )
                frame.append(int(torch.argmax(logits)))
            else:
                pass
        return frame


def prefill_rows() -> tuple[list[str], list[int], list[list[int]], list[list[int]]]:
    """Every request's rows but the last, each paired with the next frame."""
    request_ids, positions, input_frames, next_frames = [], [], [], []
    for request_id, frames in REQUEST_FRAMES.items():
        for row in range(len(frames) - 1):
            request_ids.append(request_id)
            positions.append(FIRST_POSITIONS[request_id] + row)
            input_frames.append(frames[row])
            next_frames.append(frames[row + 1])
    return request_ids, positions, input_frames, next_frames


def decode_rows() -> tuple[list[str], list[int], list[list[int]]]:
    request_ids = list(REQUEST_FRAMES)
    positions = [
        FIRST_POSITIONS[request_id] + len(REQUEST_FRAMES[request_id]) - 1
        for request_id in request_ids
    ]
    return (
        request_ids,
        positions,
        [REQUEST_FRAMES[request_id][-1] for request_id in request_ids],
    )


def run_decoder(
    decoder: AudioDecoder,
    attention: DenseAttention,
    prefill_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    semantic_logits: torch.Tensor,
) -> tuple[dict[tuple[int, str], list[int]], torch.Tensor]:
    """Prefill all requests and decode once; returns cache positions and frames."""
    device = prefill_hidden.device
    request_ids, positions, input_frames, next_frames = prefill_rows()
    attention.bind(request_ids, positions)
    decoder.prefill(
        prefill_hidden,
        torch.tensor(positions, device=device),
        torch.tensor(input_frames, device=device),
        torch.tensor(next_frames, device=device),
        attention,
    )
    positions_after_prefill = {
        cache_key: [entry[0] for entry in entries]
        for cache_key, entries in attention.entries.items()
    }
    request_ids, positions, input_frames = decode_rows()
    attention.bind(request_ids, positions)
    row_count = len(request_ids)
    frames = decoder.decode_step(
        semantic_logits,
        decode_hidden,
        torch.tensor(positions, device=device),
        torch.tensor(input_frames, device=device),
        torch.full((row_count, WINDOW_FRAMES), EMPTY_WINDOW_TOKEN, device=device),
        torch.ones(row_count, device=device),
        torch.ones(row_count, dtype=torch.int64, device=device),
        torch.ones(row_count, device=device),
        attention,
    )
    return positions_after_prefill, frames


def run_reference(
    checkpoint: dict[str, torch.Tensor],
    prefill_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    semantic_logits: torch.Tensor,
) -> tuple[ReferenceTower, list[list[int]]]:
    reference = ReferenceTower(checkpoint)
    request_ids, positions, input_frames, next_frames = prefill_rows()
    for row, request_id in enumerate(request_ids):
        reference.run(
            request_id,
            prefill_hidden[row],
            positions[row],
            input_frames[row],
            next_frames[row],
        )
    request_ids, positions, input_frames = decode_rows()
    frames = [
        reference.decode(
            request_id,
            decode_hidden[row],
            positions[row],
            input_frames[row],
            semantic_logits[row],
        )
        for row, request_id in enumerate(request_ids)
    ]
    return reference, frames


def random_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backbone hiddens and semantic logits; request "b" is steered to end of speech."""
    generator = torch.Generator().manual_seed(seed)
    prefill_row_count = len(prefill_rows()[0])
    semantic_logits = (
        torch.randn(len(REQUEST_FRAMES), TEXT_VOCAB_SIZE, generator=generator) * 3.0
    )
    semantic_logits[list(REQUEST_FRAMES).index("b"), EOS_TOKEN_ID] = 100.0
    return (
        torch.randn(prefill_row_count, DIM, generator=generator),
        torch.randn(len(REQUEST_FRAMES), DIM, generator=generator),
        semantic_logits,
    )


@torch_fused_op_backend()
def test_tower_matches_the_hand_written_formulas() -> None:
    """Grouped heads, a partial rotary width and every frame transition, in fp32.

    The config marks rope as not interleaved and q/k norm as off; the tower
    must ignore both fields, and the reference never reads them.
    """
    checkpoint = random_checkpoint(seed=0)
    prefill_hidden, decode_hidden, semantic_logits = random_inputs(seed=1)
    decoder = loaded_decoder(checkpoint, torch.float32, "cpu")
    attention = DenseAttention(
        head_count=HEAD_COUNT, kv_head_count=KV_HEAD_COUNT, head_dim=HEAD_DIM
    )

    positions_after_prefill, frames = run_decoder(
        decoder, attention, prefill_hidden, decode_hidden, semantic_logits
    )
    reference, reference_frames = run_reference(
        checkpoint, prefill_hidden, decode_hidden, semantic_logits
    )

    assert frames.tolist() == reference_frames
    assert frames[list(REQUEST_FRAMES).index("b"), 0] == EOS_TOKEN_ID
    assert frames.dtype == torch.int64
    for layer_index in range(SHARED_LAYER_COUNT + RESIDUAL_LAYER_COUNT):
        assert positions_after_prefill[(layer_index, "a")] == [3, 4, 5, 6, 7]
        assert positions_after_prefill[(layer_index, "c")] == [0]
        assert (layer_index, "b") not in positions_after_prefill
        for request_id in REQUEST_FRAMES:
            entries = attention.entries[(layer_index, request_id)]
            reference_entries = reference.caches[(layer_index, request_id)]
            assert [entry[0] for entry in entries] == [
                entry[0] for entry in reference_entries
            ]
            for entry, reference_entry in zip(entries, reference_entries):
                torch.testing.assert_close(
                    entry[1], reference_entry[1], rtol=1e-4, atol=1e-4
                )
                torch.testing.assert_close(
                    entry[2], reference_entry[2], rtol=1e-4, atol=1e-4
                )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
def test_bfloat16_tower_tracks_the_fp32_formulas(device: DeviceType) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    checkpoint = random_checkpoint(seed=0)
    prefill_hidden, decode_hidden, semantic_logits = random_inputs(seed=1)
    decoder = loaded_decoder(checkpoint, torch.bfloat16, device)
    attention = DenseAttention(
        head_count=HEAD_COUNT, kv_head_count=KV_HEAD_COUNT, head_dim=HEAD_DIM
    )

    with torch_fused_op_backend() if device == "cpu" else nullcontext():
        _, frames = run_decoder(
            decoder,
            attention,
            prefill_hidden.to(device=device, dtype=torch.bfloat16),
            decode_hidden.to(device=device, dtype=torch.bfloat16),
            semantic_logits.to(device),
        )
    reference, reference_frames = run_reference(
        checkpoint, prefill_hidden, decode_hidden, semantic_logits
    )

    assert frames.dtype == torch.int64
    assert frames.device.type == device
    # The main token and code 0 come from the same fp32 logits; later codes
    # may flip on bfloat16 rounding, so only the shared layers' decode rows
    # and every prefill row are compared.
    assert frames[:, :2].tolist() == [frame[:2] for frame in reference_frames]
    for (layer_index, request_id), entries in attention.entries.items():
        reference_entries = reference.caches[(layer_index, request_id)]
        compared = entries if layer_index < SHARED_LAYER_COUNT else entries[:-1]
        for entry, reference_entry in zip(compared, reference_entries):
            for cached, expected in (
                (entry[1], reference_entry[1]),
                (entry[2], reference_entry[2]),
            ):
                assert cached.dtype == torch.bfloat16
                assert cached.device.type == device
                # bfloat16 keeps 8 mantissa bits; five layers deep the cached
                # rows drift up to ~4% in norm (measured on this fixture).
                drift = (cached.float().cpu() - expected).norm() / expected.norm()
                assert drift < BFLOAT16_CACHE_DRIFT


def test_frame_embeddings() -> None:
    checkpoint = random_checkpoint(seed=2)
    decoder = loaded_decoder(checkpoint, torch.float32, "cpu")
    frames = [REQUEST_FRAMES["a"][0], REQUEST_FRAMES["a"][1], REQUEST_FRAMES["a"][3]]
    text_embeddings = torch.randn(3, DIM, generator=torch.Generator().manual_seed(3))

    backbone_input = decoder.embed_backbone(torch.tensor(frames), text_embeddings)
    shared_input = decoder.shared_input(torch.tensor(frames))

    mixed = (
        text_embeddings[1] + reference_codebook_sum(checkpoint, frames[1], range(1, 2))
    ) / math.sqrt(2.0)
    torch.testing.assert_close(backbone_input[0], text_embeddings[0])
    torch.testing.assert_close(backbone_input[1], mixed)
    torch.testing.assert_close(backbone_input[2], text_embeddings[2])
    torch.testing.assert_close(shared_input[0], torch.zeros(DIM))
    torch.testing.assert_close(
        shared_input[1], reference_codebook_sum(checkpoint, frames[1], range(2, 4))
    )
    torch.testing.assert_close(shared_input[2], torch.zeros(DIM))


@pytest.mark.parametrize(
    ("placement", "backbone_codebook_count"),
    [("backbone_partial", NUM_CODEBOOKS - 1), ("mtp_shared_pre", 0)],
)
def test_frame_embeddings_of_an_empty_codebook_range_are_zero(
    placement: Literal["backbone_partial", "mtp_shared_pre"],
    backbone_codebook_count: int,
) -> None:
    config_dict = fish_config_dict()
    decoder_config = config_dict["audio_decoder_config"]
    assert isinstance(decoder_config, dict)
    decoder_config["vq_feature_placement"] = placement
    decoder_config["vq_backbone_codebooks"] = backbone_codebook_count
    checkpoint = random_checkpoint(seed=2)
    decoder = AudioDecoder(FishOmniConfig(**config_dict))
    load_audio_decoder_weights(decoder, checkpoint.items())
    frames = [REQUEST_FRAMES["a"][0], REQUEST_FRAMES["a"][1], REQUEST_FRAMES["a"][3]]
    text_embeddings = torch.randn(3, DIM, generator=torch.Generator().manual_seed(3))

    backbone_input = decoder.embed_backbone(torch.tensor(frames), text_embeddings)
    shared_input = decoder.shared_input(torch.tensor(frames))

    mixed = (
        text_embeddings[1]
        + reference_codebook_sum(
            checkpoint, frames[1], range(1, 1 + backbone_codebook_count)
        )
    ) / math.sqrt(1.0 + backbone_codebook_count)
    torch.testing.assert_close(backbone_input[0], text_embeddings[0])
    torch.testing.assert_close(backbone_input[1], mixed)
    torch.testing.assert_close(backbone_input[2], text_embeddings[2])
    torch.testing.assert_close(
        shared_input[1],
        reference_codebook_sum(
            checkpoint, frames[1], range(1 + backbone_codebook_count, NUM_CODEBOOKS)
        ),
    )
    torch.testing.assert_close(shared_input[0], torch.zeros(DIM))
    torch.testing.assert_close(shared_input[2], torch.zeros(DIM))


@pytest.mark.parametrize(
    ("eos_token_id", "candidate_token_ids"),
    [
        (EOS_TOKEN_ID, [EOS_TOKEN_ID, *range(SEMANTIC_BEGIN_ID, SEMANTIC_END_ID + 1)]),
        (
            TEXT_VOCAB_SIZE - 1,
            [*range(SEMANTIC_BEGIN_ID, SEMANTIC_END_ID + 1), TEXT_VOCAB_SIZE - 1],
        ),
    ],
)
def test_main_token_candidates_are_sorted_and_build_on_the_meta_device(
    eos_token_id: int, candidate_token_ids: list[int]
) -> None:
    config_dict = fish_config_dict()
    config_dict["eos_token_id"] = eos_token_id
    with torch.device("meta"):
        meta_decoder = AudioDecoder(FishOmniConfig(**config_dict))
    decoder = AudioDecoder(FishOmniConfig(**config_dict))

    assert meta_decoder.main_token_candidate_ids.is_meta
    assert meta_decoder.main_token_candidate_ids.shape == (len(candidate_token_ids),)
    assert decoder.main_token_candidate_ids.dtype == torch.int64
    assert decoder.main_token_candidate_ids.tolist() == candidate_token_ids


def test_wqkv_rows_are_regrouped_per_head() -> None:
    # Two heads of width 2, one kv head: rows 0-3 query, 4-7 gate, 8-9 key, 10-11 value.
    flat_weight = torch.arange(12.0).unsqueeze(-1)

    regrouped = interleave_query_and_gate(flat_weight, head_count=2, head_dim=2)

    assert regrouped.squeeze(-1).tolist() == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11]


def broken_checkpoint(problem: CheckpointProblem) -> list[tuple[str, torch.Tensor]]:
    checkpoint = random_checkpoint(seed=0)
    up_name = "audio_decoder_mtp.layers.1.transformer_layer.feed_forward.w3.weight"
    # A whole gate_up tensor under the module's name instead of w1 / w3 halves.
    fused_gate_up = (
        "audio_decoder_mtp.layers.0.mlp.gate_up_proj.weight",
        torch.zeros(2 * INTERMEDIATE_SIZE, DIM),
    )
    if problem == "missing-up-half":
        checkpoint.pop(up_name)
        return list(checkpoint.items())
    elif problem == "duplicate":
        return [*checkpoint.items(), (up_name, checkpoint[up_name])]
    elif problem == "unknown-key":
        unknown_name = (
            "audio_decoder_mtp.layers.0.transformer_layer.attention.extra.weight"
        )
        return [*checkpoint.items(), (unknown_name, torch.zeros(1))]
    elif problem == "unknown-namespace":
        return [
            *checkpoint.items(),
            ("audio_decoder.speaker_table.weight", torch.zeros(1)),
        ]
    elif problem == "fused-gate-up-after-shards":
        return [*checkpoint.items(), fused_gate_up]
    elif problem == "fused-gate-up-before-shards":
        return [fused_gate_up, *checkpoint.items()]
    else:
        checkpoint[up_name] = torch.zeros(INTERMEDIATE_SIZE + 1, DIM)
        return list(checkpoint.items())


LOADING_ERRORS: dict[CheckpointProblem, str] = {
    "missing-up-half": r"gate_up_proj\.weight\[up\]",
    "duplicate": "second time",
    "unknown-key": "not a decoder parameter shard",
    "unknown-namespace": "not an audio decoder weight",
    "fused-gate-up-after-shards": "not a decoder parameter shard",
    "fused-gate-up-before-shards": "not a decoder parameter shard",
    "wrong-shape": "has shape",
}


@pytest.mark.parametrize(("problem", "error"), LOADING_ERRORS.items())
def test_loading_rejects_incomplete_or_foreign_weights(
    problem: CheckpointProblem, error: str
) -> None:
    decoder = AudioDecoder(small_config())

    with pytest.raises(ValueError, match=error):
        load_audio_decoder_weights(decoder, broken_checkpoint(problem))


def test_checkpoint_keys_route_by_namespace() -> None:
    assert (
        checkpoint_part(
            "model.language_model.layers.0.mlp.experts.7.up_proj.weight_scale_inv"
        )
        == "backbone"
    )
    assert checkpoint_part("lm_head.weight") == "backbone"
    assert (
        checkpoint_part("audio_decoder.codebook_embeddings.weight") == "audio_decoder"
    )
    assert checkpoint_part("audio_decoder_mtp.heads.0.weight") == "audio_decoder"
    for foreign_name in (
        "mtp.layers.0.eh_proj.weight",
        "visual.blocks.0.weight",
        "model.layers.0.x",
    ):
        with pytest.raises(ValueError, match="outside the Drama 3 namespaces"):
            checkpoint_part(foreign_name)


@pytest.fixture(scope="module")
def checkpoint_directory() -> Path:
    """FISH_DRAMA3_CHECKPOINT_DIR, when it has config.json and the weight index."""
    configured_directory = os.environ.get("FISH_DRAMA3_CHECKPOINT_DIR")
    if configured_directory is None:
        pytest.skip("FISH_DRAMA3_CHECKPOINT_DIR is not set")
    else:
        pass
    directory = Path(configured_directory).expanduser()
    for file_name in ("config.json", "model.safetensors.index.json"):
        if not (directory / file_name).is_file():
            pytest.skip(f"no {file_name} in {directory}")
        else:
            pass
    return directory


def test_checkpoint_index_covers_the_decoder_and_routes_the_backbone(
    checkpoint_directory: Path,
) -> None:
    """Decoder keys fill each shard once; backbone keys are checked for naming only."""
    config = FishOmniConfig.from_pretrained(checkpoint_directory)
    index = json.loads(
        (checkpoint_directory / "model.safetensors.index.json").read_text()
    )
    with torch.device("meta"):
        decoder = AudioDecoder(config)
    text_config = config.text_config

    names = list(index["weight_map"])
    decoder_shards = [
        parameter_shard(name)
        for name in names
        if checkpoint_part(name) == "audio_decoder"
    ]
    assert len(decoder_shards) == len(set(decoder_shards))
    assert set(decoder_shards) == expected_parameter_shards(decoder)
    for name in names:
        if checkpoint_part(name) == "backbone" and name.startswith(
            "model.language_model.layers."
        ):
            fields = name.split(".")
            layer_index = int(fields[3])
            assert layer_index < text_config.num_hidden_layers
            layer_type = text_config.layers_block_type[layer_index]
            assert (fields[4] == "self_attn") <= (layer_type == "attention")
            assert (fields[4] == "linear_attn") <= (layer_type == "linear_attention")
            if fields[5] == "experts":
                assert int(fields[6]) < text_config.num_experts
            else:
                pass
        else:
            pass
