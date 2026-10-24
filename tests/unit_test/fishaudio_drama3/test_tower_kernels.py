# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import math
import statistics
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import get_context
from torch import nn

from sglang_omni.models.fishaudio_drama3 import tower_kernels
from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder, DecoderLayer
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    VALIDATED_COMPUTE_CAPABILITY,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    MAX_FUSED_ROWS,
    fused_tower_layers_unsupported_reason,
    gate_o_proj_add,
    mix_eh_proj,
    norm_gate_up_silu,
    norm_qkv_rope_store,
)
from sglang_omni.models.fishaudio_drama3.weights import load_audio_decoder_weights
from sglang_omni.platforms import current_platform
from tests.unit_test.fishaudio_drama3.graph_fixtures import bits
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    CHECKPOINT_CONFIG,
    decoder_checkpoint,
)

STEP_COUNT = 3
SLOT_COUNT = 1 + MAX_FUSED_ROWS * (STEP_COUNT + 2)
FIRST_POSITION = 40
SENTINEL = 7.0
ERROR_RATIO = 1.25
ERROR_SLACK = 1e-4
TESTED_ROW_COUNTS = [1, 3, 8, 12, MAX_FUSED_ROWS]
STACKED_SEEDS = range(1000, 1016)
STACKED_ROW_COUNT = MAX_FUSED_ROWS
GRAPH_ROW_COUNTS = [1, 3, 8, MAX_FUSED_ROWS]
accelerator = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="the fused tower layers run on CUDA only",
)


@pytest.fixture
def published_server_args() -> Iterator[None]:
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        yield


def require_validated_gpu() -> None:
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability != VALIDATED_COMPUTE_CAPABILITY:
        pytest.skip(
            f"the fused layers are validated only on compute capability "
            f"{VALIDATED_COMPUTE_CAPABILITY}, this GPU is {compute_capability}"
        )
    else:
        pass


class PagedTestAttention:
    def __init__(
        self,
        *,
        decoder: AudioDecoder,
        device: torch.device,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        cache_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        attention = decoder.shared_layers[0].self_attn
        layer_count = len(decoder.shared_layers) + len(decoder.residual_layers)
        self.head_count = attention.head_count
        self.kv_head_count = attention.kv_head_count
        self.head_dim = attention.head_dim
        self.key_caches = torch.zeros(
            (layer_count, SLOT_COUNT, self.kv_head_count, self.head_dim),
            dtype=cache_dtype,
            device=device,
        )
        self.value_caches = torch.zeros_like(self.key_caches)
        self.mode = forward_mode
        self.slots = torch.zeros(0, dtype=torch.int64, device=device)
        self.history = torch.zeros((0, 1), dtype=torch.int64, device=device)
        self.history_lengths = torch.zeros(0, dtype=torch.int64, device=device)

    def bind(
        self, slots: torch.Tensor, history: torch.Tensor, history_lengths: torch.Tensor
    ) -> None:
        self.slots = slots
        self.history = history
        self.history_lengths = history_lengths

    def __call__(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        key_cache = self.key_caches[layer_index]
        value_cache = self.value_caches[layer_index]
        padding_key = key_cache[0].clone()
        padding_value = value_cache[0].clone()
        key_cache[self.slots] = key.reshape(-1, self.kv_head_count, self.head_dim).to(
            key_cache.dtype
        )
        value_cache[self.slots] = value.reshape(
            -1, self.kv_head_count, self.head_dim
        ).to(value_cache.dtype)
        key_cache[0] = padding_key
        value_cache[0] = padding_value
        return self.attend(layer_index, query)

    def forward_mode(self) -> ForwardMode:
        return self.mode

    def cache_slots(self) -> torch.Tensor:
        return self.slots

    def key_buffer(self, layer_index: int) -> torch.Tensor:
        return self.key_caches[layer_index]

    def value_buffer(self, layer_index: int) -> torch.Tensor:
        return self.value_caches[layer_index]

    def attend(self, layer_index: int, query: torch.Tensor) -> torch.Tensor:
        row_count = query.shape[0]
        group_size = self.head_count // self.kv_head_count
        keys = self.key_caches[layer_index][self.history].float()
        values = self.value_caches[layer_index][self.history].float()
        scores = torch.einsum(
            "rkgd,rnkd->rkgn",
            query.view(
                row_count, self.kv_head_count, group_size, self.head_dim
            ).float(),
            keys,
        ) / (self.head_dim**0.5)
        is_visible = (
            torch.arange(self.history.shape[1], device=query.device)[None, :]
            < self.history_lengths[:, None]
        )
        scores = scores.masked_fill(~is_visible[:, None, None, :], float("-inf"))
        attended = torch.einsum(
            "rkgn,rnkd->rkgd", torch.softmax(scores, dim=-1), values
        )
        return attended.reshape(row_count, -1).to(query.dtype)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TowerStep:
    hidden_states: torch.Tensor
    code_ids: torch.Tensor
    label_is_semantic: torch.Tensor
    shared_input: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    history: torch.Tensor
    history_lengths: torch.Tensor

    def rows(self, row_count: int) -> TowerStep:
        return TowerStep(
            hidden_states=self.hidden_states[:row_count].contiguous(),
            code_ids=self.code_ids[:, :row_count].contiguous(),
            label_is_semantic=self.label_is_semantic[:row_count].contiguous(),
            shared_input=self.shared_input[:row_count].contiguous(),
            positions=self.positions[:row_count].contiguous(),
            slots=self.slots[:row_count].contiguous(),
            history=self.history[:row_count].contiguous(),
            history_lengths=self.history_lengths[:row_count].contiguous(),
        )


def tower_layers(decoder: AudioDecoder) -> list[DecoderLayer]:
    return [*decoder.shared_layers, *decoder.residual_layers]


def build_decoder(device: torch.device) -> AudioDecoder:
    with set_default_torch_dtype(torch.bfloat16):
        decoder = AudioDecoder(FishOmniConfig(**CHECKPOINT_CONFIG))
    load_audio_decoder_weights(decoder, decoder_checkpoint().items())
    return decoder.to(device)


def fuse(decoder: AudioDecoder, device: torch.device) -> AudioDecoder:
    decoder.enable_fused_layers(device)
    return decoder


def tower_steps(
    decoder: AudioDecoder, row_count: int, seed: int, device: torch.device
) -> list[TowerStep]:
    generator = torch.Generator(device=device).manual_seed(seed)
    layer_count = len(tower_layers(decoder))
    dim = decoder.codebook_embeddings.embedding_dim
    codebook_size = decoder.codebook_size
    rows = torch.arange(row_count, device=device)
    steps = []
    for step_index in range(STEP_COUNT):
        history = (
            1
            + rows[:, None] * (STEP_COUNT + 1)
            + torch.arange(STEP_COUNT, device=device)[None, :]
        )
        steps.append(
            TowerStep(
                hidden_states=torch.randn(
                    (row_count, dim), device=device, generator=generator
                ).bfloat16(),
                code_ids=torch.randint(
                    -3,
                    codebook_size + 3,
                    (layer_count, row_count),
                    device=device,
                    generator=generator,
                ),
                label_is_semantic=torch.rand(
                    row_count, device=device, generator=generator
                )
                < 0.8,
                shared_input=torch.randn(
                    (row_count, dim), device=device, generator=generator
                ).bfloat16(),
                positions=FIRST_POSITION + step_index + rows,
                slots=history[:, step_index].contiguous(),
                history=history,
                history_lengths=torch.full(
                    (row_count,), step_index + 1, dtype=torch.int64, device=device
                ),
            )
        )
    return steps


def run_tower_step(
    decoder: AudioDecoder, attention: PagedTestAttention, step: TowerStep
) -> list[torch.Tensor]:
    attention.bind(step.slots, step.history, step.history_lengths)
    hidden_states = step.hidden_states
    outputs = []
    for layer_index, layer in enumerate(tower_layers(decoder)):
        hidden_states = decoder.run_layer(
            layer,
            layer_index,
            hidden_states,
            step.code_ids[layer_index],
            step.label_is_semantic,
            step.shared_input if layer_index == 0 else None,
            step.positions,
            attention,
            True,
        )
        outputs.append(hidden_states.clone())
    return outputs


def gemma_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (
        values
        * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
        * (1.0 + weight.float())
    )


def interleaved_rope(
    heads: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    rotary_dim = cos_sin.shape[1]
    cosine, sine = cos_sin[positions][:, None, :].chunk(2, dim=-1)
    even = heads[..., 0:rotary_dim:2]
    odd = heads[..., 1:rotary_dim:2]
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
    ).flatten(-2)
    return torch.cat((rotated, heads[..., rotary_dim:]), dim=-1)


class ReferenceTower:
    def __init__(self, decoder: AudioDecoder) -> None:
        self.decoder = decoder
        self.attention = PagedTestAttention(
            decoder=decoder,
            device=decoder.codebook_embeddings.weight.device,
            cache_dtype=torch.float32,
        )

    def step(self, step: TowerStep) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.attention.bind(step.slots, step.history, step.history_lengths)
        hidden = step.hidden_states.float()
        inputs = []
        outputs = []
        for layer_index, layer in enumerate(tower_layers(self.decoder)):
            inputs.append(hidden)
            hidden = self.layer(layer_index, layer, hidden, step)
            outputs.append(hidden)
        return inputs, outputs

    def layer(
        self,
        layer_index: int,
        layer: DecoderLayer,
        hidden: torch.Tensor,
        step: TowerStep,
    ) -> torch.Tensor:
        weights = layer.tower_layer_weights()
        attention = layer.self_attn
        row_count = hidden.shape[0]
        codebook_size = weights.token_embedding.shape[0]
        embeddings = torch.where(
            step.label_is_semantic[:, None],
            weights.token_embedding[
                step.code_ids[layer_index].clamp(0, codebook_size - 1)
            ].float(),
            0.0,
        )
        if layer_index == 0:
            embeddings = embeddings + step.shared_input.float()
        else:
            pass
        mixed = (
            torch.cat(
                (
                    gemma_norm(embeddings, weights.enorm.weight, weights.enorm.eps),
                    gemma_norm(hidden, weights.hnorm.weight, weights.hnorm.eps),
                ),
                dim=-1,
            )
            @ weights.eh_proj.weight.float().t()
        )
        query_width = attention.head_count * attention.head_dim
        kv_width = attention.kv_head_count * attention.head_dim
        query_and_gate, key, value = (
            gemma_norm(mixed, weights.input_norm.weight, weights.input_norm.eps)
            @ weights.qkv_proj.weight.float().t()
        ).split([2 * query_width, kv_width, kv_width], dim=-1)
        query, gate = query_and_gate.view(
            row_count, attention.head_count, 2 * attention.head_dim
        ).chunk(2, dim=-1)
        cos_sin = attention.rotary_embedding.cos_sin_cache
        query = interleaved_rope(
            gemma_norm(query, weights.q_norm.weight, weights.q_norm.eps),
            step.positions,
            cos_sin,
        )
        key = interleaved_rope(
            gemma_norm(
                key.view(row_count, attention.kv_head_count, attention.head_dim),
                weights.k_norm.weight,
                weights.k_norm.eps,
            ),
            step.positions,
            cos_sin,
        )
        attended = self.attention(
            layer_index, query.reshape(row_count, query_width), key, value
        ).float()
        summed = mixed + (
            attended * torch.sigmoid(gate.reshape(row_count, query_width))
        ) @ (weights.o_proj.weight.float().t())
        gate_up = (
            gemma_norm(summed, weights.post_norm.weight, weights.post_norm.eps)
            @ weights.gate_up_proj.weight.float().t()
        )
        gate_part, up_part = gate_up.chunk(2, dim=-1)
        summed = (
            summed
            + (torch.nn.functional.silu(gate_part) * up_part)
            @ layer.mlp.down_proj.weight.float().t()
        )
        return gemma_norm(
            summed, layer.final_layernorm.weight, layer.final_layernorm.variance_epsilon
        )


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference).norm() / reference.norm()).item()


def assert_tracks_like_the_plain_path(
    name: str, fused: torch.Tensor, plain: torch.Tensor, reference: torch.Tensor
) -> None:
    plain_error = relative_error(plain, reference)
    fused_error = relative_error(fused, reference)
    assert fused_error <= ERROR_RATIO * plain_error + ERROR_SLACK, (
        name,
        plain_error,
        fused_error,
    )


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", TESTED_ROW_COUNTS)
def test_each_kernel_tracks_the_fp32_reference_like_the_plain_path(
    row_count: int,
) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    fused_layers = decoder.fused_layers
    assert fused_layers is not None
    shape = fused_layers.shape
    generator = torch.Generator(device=device).manual_seed(row_count)

    def random_rows(width: int) -> torch.Tensor:
        return torch.randn(
            (row_count, width), device=device, generator=generator
        ).bfloat16()

    step = tower_steps(decoder, row_count, seed=row_count, device=device)[0]
    for layer_index in (0, len(decoder.shared_layers) + 1):
        layer = tower_layers(decoder)[layer_index]
        weights = layer.tower_layer_weights()
        attention = layer.self_attn
        hidden = random_rows(shape.dim)
        shared = step.shared_input if layer_index == 0 else None
        code_ids = step.code_ids[layer_index]
        embeddings = torch.where(
            step.label_is_semantic[:, None],
            layer.token_embedding(code_ids.clamp(0, decoder.codebook_size - 1)),
            0.0,
        )
        reference_embeddings = embeddings.float()
        if shared is not None:
            embeddings = embeddings + shared
            reference_embeddings = reference_embeddings + shared.float()
        else:
            pass
        mix_eh_proj(
            weights=weights,
            hidden_states=hidden,
            code_ids=code_ids,
            label_is_semantic=step.label_is_semantic,
            shared_input=shared,
            mixed=fused_layers.mixed[:row_count],
            scaled=fused_layers.scaled[:row_count],
            scratch=fused_layers.scratch,
            shape=shape,
        )
        assert_tracks_like_the_plain_path(
            "mix_eh_proj",
            fused_layers.mixed[:row_count],
            layer.eh_proj(
                torch.cat((layer.enorm(embeddings), layer.hnorm(hidden)), dim=-1)
            ),
            torch.cat(
                (
                    gemma_norm(
                        reference_embeddings, weights.enorm.weight, weights.enorm.eps
                    ),
                    gemma_norm(hidden.float(), weights.hnorm.weight, weights.hnorm.eps),
                ),
                dim=-1,
            )
            @ weights.eh_proj.weight.float().t(),
        )

        mixed = fused_layers.mixed[:row_count].clone()
        key_cache = torch.zeros(
            (SLOT_COUNT, shape.kv_head_count, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        value_cache = torch.zeros_like(key_cache)
        norm_qkv_rope_store(
            weights=weights,
            scaled=fused_layers.scaled[:row_count],
            rotary_cos_sin=fused_layers.rotary_cos_sin,
            positions=step.positions,
            cache_slots=step.slots,
            query=fused_layers.query[:row_count],
            gate_scale=fused_layers.gate_scale[:row_count],
            key_cache=key_cache,
            value_cache=value_cache,
            scratch=fused_layers.scratch,
            shape=shape,
        )
        query_width = shape.query_width
        kv_width = shape.kv_head_count * shape.head_dim
        query_and_gate, key, value = attention.qkv_proj(
            layer.input_layernorm(mixed)
        ).split([2 * query_width, kv_width, kv_width], dim=-1)
        query, gate = query_and_gate.view(
            row_count, shape.head_count, 2 * shape.head_dim
        ).chunk(2, dim=-1)
        query, key = attention.rotary_embedding(
            step.positions,
            attention.q_norm(query.reshape(-1, shape.head_dim)).view(
                row_count, query_width
            ),
            attention.k_norm(key.reshape(-1, shape.head_dim)).view(row_count, kv_width),
        )
        reference_query_and_gate, reference_key, reference_value = (
            gemma_norm(mixed.float(), weights.input_norm.weight, weights.input_norm.eps)
            @ weights.qkv_proj.weight.float().t()
        ).split([2 * query_width, kv_width, kv_width], dim=-1)
        reference_query, reference_gate = reference_query_and_gate.view(
            row_count, shape.head_count, 2 * shape.head_dim
        ).chunk(2, dim=-1)
        cos_sin = fused_layers.rotary_cos_sin
        assert_tracks_like_the_plain_path(
            "query",
            fused_layers.query[:row_count],
            query,
            interleaved_rope(
                gemma_norm(reference_query, weights.q_norm.weight, weights.q_norm.eps),
                step.positions,
                cos_sin,
            ).reshape(row_count, query_width),
        )
        assert_tracks_like_the_plain_path(
            "gate scale",
            fused_layers.gate_scale[:row_count],
            torch.sigmoid(gate.reshape(row_count, query_width)),
            torch.sigmoid(reference_gate.reshape(row_count, query_width)),
        )
        assert_tracks_like_the_plain_path(
            "key",
            key_cache[step.slots].reshape(row_count, kv_width),
            key,
            interleaved_rope(
                gemma_norm(
                    reference_key.view(row_count, shape.kv_head_count, shape.head_dim),
                    weights.k_norm.weight,
                    weights.k_norm.eps,
                ),
                step.positions,
                cos_sin,
            ).reshape(row_count, kv_width),
        )
        assert_tracks_like_the_plain_path(
            "value",
            value_cache[step.slots].reshape(row_count, kv_width),
            value,
            reference_value,
        )

        attended = random_rows(query_width)
        gate_values = random_rows(query_width)
        mixed = random_rows(shape.dim)
        gate_o_proj_add(
            weights=weights,
            attended=attended,
            gate_scale=torch.sigmoid(gate_values),
            residual_in=mixed,
            residual_out=fused_layers.residual[:row_count],
            scaled_out=fused_layers.scaled[:row_count],
            scratch=fused_layers.scratch,
            shape=shape,
        )
        norm_gate_up_silu(
            weights=weights,
            scaled=fused_layers.scaled[:row_count],
            activated=fused_layers.activated[:row_count],
            scratch=fused_layers.scratch,
            shape=shape,
        )
        projected = attention.o_proj(attended * torch.sigmoid(gate_values))
        mlp_input, plain_residual = layer.post_attention_layernorm(
            projected.clone(), mixed.clone()
        )
        reference_summed = (
            mixed.float()
            + (attended.float() * torch.sigmoid(gate_values.float()))
            @ weights.o_proj.weight.float().t()
        )
        reference_gate_up = (
            gemma_norm(
                reference_summed, weights.post_norm.weight, weights.post_norm.eps
            )
            @ weights.gate_up_proj.weight.float().t()
        )
        reference_gate_part, reference_up_part = reference_gate_up.chunk(2, dim=-1)
        assert_tracks_like_the_plain_path(
            "residual after attention",
            fused_layers.residual[:row_count],
            plain_residual,
            reference_summed,
        )
        assert_tracks_like_the_plain_path(
            "activated",
            fused_layers.activated[:row_count],
            layer.mlp.silu_and_mul(layer.mlp.gate_up_proj(mlp_input)),
            torch.nn.functional.silu(reference_gate_part) * reference_up_part,
        )


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", [5, MAX_FUSED_ROWS])
def test_every_fused_layer_tracks_the_fp32_reference_like_the_plain_path(
    row_count: int,
) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    fused_layers = decoder.fused_layers
    reference_tower = ReferenceTower(decoder)
    reference_caches = reference_tower.attention
    attention = PagedTestAttention(decoder=decoder, device=device)
    shared_layer_count = len(decoder.shared_layers)
    with torch.no_grad():
        for step_index, step in enumerate(
            tower_steps(decoder, row_count, seed=7, device=device)
        ):
            history_keys = reference_caches.key_caches.clone()
            history_values = reference_caches.value_caches.clone()
            inputs, outputs = reference_tower.step(step)
            for layer_index, layer in enumerate(tower_layers(decoder)):
                results = []
                for layer_fused_layers in (None, fused_layers):
                    decoder.fused_layers = layer_fused_layers
                    attention.key_caches.copy_(history_keys)
                    attention.value_caches.copy_(history_values)
                    attention.bind(step.slots, step.history, step.history_lengths)
                    output = decoder.run_layer(
                        layer,
                        layer_index,
                        inputs[layer_index].bfloat16(),
                        step.code_ids[layer_index],
                        step.label_is_semantic,
                        step.shared_input if layer_index == 0 else None,
                        step.positions,
                        attention,
                        True,
                    )
                    results.append(
                        (
                            output.clone(),
                            attention.key_caches[layer_index, step.slots].clone(),
                            attention.value_caches[layer_index, step.slots].clone(),
                        )
                    )
                name = f"step {step_index} layer {layer_index}"
                for part, plain, fused, reference in zip(
                    ("output", "keys", "values"),
                    results[0],
                    results[1],
                    (
                        outputs[layer_index],
                        reference_caches.key_caches[layer_index, step.slots],
                        reference_caches.value_caches[layer_index, step.slots],
                    ),
                    strict=True,
                ):
                    assert_tracks_like_the_plain_path(
                        f"{name} {part}", fused, plain, reference
                    )
                if layer_index >= shared_layer_count:
                    head = decoder.codebook_heads[layer_index - shared_layer_count]
                    assert_tracks_like_the_plain_path(
                        f"{name} head",
                        head(results[1][0]),
                        head(results[0][0]),
                        outputs[layer_index] @ head.weight.float().t(),
                    )
                else:
                    pass


@accelerator
@pytest.mark.accelerator
def test_the_stacked_fused_tower_tracks_the_fp32_reference_like_the_plain_path() -> (
    None
):
    require_validated_gpu()
    device = torch.device("cuda")
    plain_decoder = build_decoder(device)
    decoder = fuse(build_decoder(device), device)
    layer_ratio_logs = []
    head_ratio_logs = []
    head = decoder.codebook_heads[-1]
    with torch.no_grad():
        for seed in STACKED_SEEDS:
            step = tower_steps(decoder, STACKED_ROW_COUNT, seed=seed, device=device)[0]
            _, reference = ReferenceTower(decoder).step(step)
            plain = run_tower_step(
                plain_decoder,
                PagedTestAttention(decoder=plain_decoder, device=device),
                step,
            )[-1]
            fused = run_tower_step(
                decoder, PagedTestAttention(decoder=decoder, device=device), step
            )[-1]
            reference_logits = reference[-1] @ head.weight.float().t()
            layer_ratio_logs.append(
                math.log(
                    relative_error(fused, reference[-1])
                    / relative_error(plain, reference[-1])
                )
            )
            head_ratio_logs.append(
                math.log(
                    relative_error(head(fused), reference_logits)
                    / relative_error(head(plain), reference_logits)
                )
            )
    assert math.exp(statistics.mean(layer_ratio_logs)) <= ERROR_RATIO
    assert math.exp(statistics.mean(head_ratio_logs)) <= ERROR_RATIO


@accelerator
@pytest.mark.accelerator
def test_a_row_gets_the_same_bits_at_every_row_count_and_every_run() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    first_step, second_step, _ = tower_steps(
        decoder, MAX_FUSED_ROWS, seed=11, device=device
    )
    attention = PagedTestAttention(decoder=decoder, device=device)
    with torch.no_grad():
        run_tower_step(decoder, attention, first_step)
        history_keys = attention.key_caches.clone()
        history_values = attention.value_caches.clone()
        full = run_tower_step(decoder, attention, second_step)
        full_keys = attention.key_caches.clone()
        full_values = attention.value_caches.clone()
        for row_count in [*range(1, MAX_FUSED_ROWS + 1), MAX_FUSED_ROWS]:
            attention.key_caches.copy_(history_keys)
            attention.value_caches.copy_(history_values)
            outputs = run_tower_step(decoder, attention, second_step.rows(row_count))
            slots = second_step.slots[:row_count]
            for layer_index, (output, full_output) in enumerate(zip(outputs, full)):
                assert torch.equal(bits(output), bits(full_output[:row_count])), (
                    row_count,
                    layer_index,
                )
            assert torch.equal(
                bits(attention.key_caches[:, slots]), bits(full_keys[:, slots])
            ), row_count
            assert torch.equal(
                bits(attention.value_caches[:, slots]), bits(full_values[:, slots])
            ), row_count


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("row_count", GRAPH_ROW_COUNTS)
def test_a_captured_tower_step_replays_the_eager_bits(row_count: int) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    first_step, second_step, third_step = tower_steps(
        decoder, row_count, seed=13, device=device
    )
    attention = PagedTestAttention(decoder=decoder, device=device)
    static_step = TowerStep(
        **{
            field.name: getattr(first_step, field.name).clone()
            for field in dataclasses.fields(first_step)
        }
    )
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_tower_step(decoder, attention, static_step)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run_tower_step(decoder, attention, static_step)
        for step in (second_step, third_step):
            attention.key_caches.zero_()
            attention.value_caches.zero_()
            for field in dataclasses.fields(step):
                getattr(static_step, field.name).copy_(getattr(step, field.name))
            graph.replay()
            torch.cuda.synchronize()
            replayed = [output.clone() for output in captured]
            replayed_keys = attention.key_caches.clone()
            replayed_values = attention.value_caches.clone()
            attention.key_caches.zero_()
            attention.value_caches.zero_()
            eager = run_tower_step(decoder, attention, step)
            for replayed_output, eager_output in zip(replayed, eager):
                assert torch.equal(bits(replayed_output), bits(eager_output))
            assert torch.equal(bits(replayed_keys), bits(attention.key_caches))
            assert torch.equal(bits(replayed_values), bits(attention.value_caches))


@accelerator
@pytest.mark.accelerator
def test_a_step_writes_only_its_rows_slots_and_skips_slot_zero() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    step = tower_steps(decoder, 4, seed=17, device=device)[0]
    slots = torch.tensor([5, 0, 9, 0], device=device)
    padded_step = dataclasses.replace(step, slots=slots)
    attention = PagedTestAttention(decoder=decoder, device=device)
    attention.key_caches.fill_(SENTINEL)
    attention.value_caches.fill_(SENTINEL)
    with torch.no_grad():
        run_tower_step(decoder, attention, padded_step)
    untouched = torch.ones(SLOT_COUNT, dtype=torch.bool, device=device)
    untouched[[5, 9]] = False
    for caches in (attention.key_caches, attention.value_caches):
        assert torch.all(caches[:, untouched] == SENTINEL)
        assert not torch.any(caches[:, [5, 9]] == SENTINEL)


@accelerator
@pytest.mark.accelerator
def test_launches_reset_their_counters_and_leave_rows_past_the_pass_alone() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    decoder = fuse(build_decoder(device), device)
    fused_layers = decoder.fused_layers
    assert fused_layers is not None
    shape = fused_layers.shape
    scratch = fused_layers.scratch
    buffers = (
        fused_layers.mixed,
        fused_layers.query,
        fused_layers.gate_scale,
        fused_layers.residual,
        fused_layers.scaled,
        fused_layers.activated,
    )
    for buffer in buffers:
        buffer.fill_(SENTINEL)
    row_count = 3
    step = tower_steps(decoder, row_count, seed=19, device=device)[0]
    attention = PagedTestAttention(decoder=decoder, device=device)
    attention.bind(step.slots, step.history, step.history_lengths)
    weights = decoder.shared_layers[0].tower_layer_weights()
    launches = (
        lambda: mix_eh_proj(
            weights=weights,
            hidden_states=step.hidden_states,
            code_ids=step.code_ids[0],
            label_is_semantic=step.label_is_semantic,
            shared_input=step.shared_input,
            mixed=fused_layers.mixed[:row_count],
            scaled=fused_layers.scaled[:row_count],
            scratch=scratch,
            shape=shape,
        ),
        lambda: norm_qkv_rope_store(
            weights=weights,
            scaled=fused_layers.scaled[:row_count],
            rotary_cos_sin=fused_layers.rotary_cos_sin,
            positions=step.positions,
            cache_slots=step.slots,
            query=fused_layers.query[:row_count],
            gate_scale=fused_layers.gate_scale[:row_count],
            key_cache=attention.key_caches[0],
            value_cache=attention.value_caches[0],
            scratch=scratch,
            shape=shape,
        ),
        lambda: gate_o_proj_add(
            weights=weights,
            attended=attention.attend(0, fused_layers.query[:row_count]),
            gate_scale=fused_layers.gate_scale[:row_count],
            residual_in=fused_layers.mixed[:row_count],
            residual_out=fused_layers.residual[:row_count],
            scaled_out=fused_layers.scaled[:row_count],
            scratch=scratch,
            shape=shape,
        ),
        lambda: norm_gate_up_silu(
            weights=weights,
            scaled=fused_layers.scaled[:row_count],
            activated=fused_layers.activated[:row_count],
            scratch=scratch,
            shape=shape,
        ),
    )
    with torch.no_grad():
        for launch_index, launch in enumerate(launches):
            launch()
            torch.cuda.synchronize()
            assert not torch.any(scratch.counters), launch_index
        run_tower_step(decoder, attention, step)
    assert not torch.any(scratch.counters)
    for buffer in buffers:
        assert torch.all(buffer[row_count:] == SENTINEL)
        assert not torch.any(buffer[:row_count] == SENTINEL)


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.EXTEND,
        ForwardMode.IDLE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.MIXED,
    ],
)
def test_only_decode_passes_take_the_fused_layers(forward_mode: ForwardMode) -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    plain_decoder = build_decoder(device)
    decoder = fuse(build_decoder(device), device)
    fused_layers = decoder.fused_layers
    assert fused_layers is not None
    forward_layer = Mock(wraps=fused_layers.forward_layer)
    fused_layers.forward_layer = forward_layer
    row_count = 6
    generator = torch.Generator(device=device).manual_seed(23)
    dim = decoder.codebook_embeddings.embedding_dim
    frame_width = 1 + decoder.num_codebooks
    hidden = torch.randn(
        (row_count, dim), device=device, generator=generator
    ).bfloat16()
    frames = torch.randint(
        0,
        decoder.codebook_size,
        (row_count, frame_width),
        device=device,
        generator=generator,
    )
    frames[:, 0] = decoder.semantic_begin_id + frames[:, 1]
    positions = torch.arange(row_count, device=device)
    slots = 1 + positions
    history = slots[None, :].expand(row_count, row_count).contiguous()
    history_lengths = positions + 1
    caches = []
    with torch.no_grad():
        for tower_decoder in (plain_decoder, decoder):
            attention = PagedTestAttention(
                decoder=tower_decoder, device=device, forward_mode=ForwardMode.EXTEND
            )
            attention.bind(slots, history, history_lengths)
            tower_decoder.prefill(
                hidden, positions, frames, frames.roll(-1, 0), attention
            )
            caches.append(
                (attention.key_caches.clone(), attention.value_caches.clone())
            )
        assert forward_layer.call_count == 0
        assert torch.equal(bits(caches[0][0]), bits(caches[1][0]))
        assert torch.equal(bits(caches[0][1]), bits(caches[1][1]))
        step = tower_steps(decoder, row_count, seed=29, device=device)[0]
        attention = PagedTestAttention(
            decoder=decoder, device=device, forward_mode=forward_mode
        )
        run_tower_step(decoder, attention, step)
        assert forward_layer.call_count == 0
        attention.mode = ForwardMode.DECODE
        run_tower_step(decoder, attention, step)
        assert forward_layer.call_count == len(tower_layers(decoder))


@accelerator
@pytest.mark.accelerator
def test_passes_above_the_fused_rows_keep_the_plain_path_bit_for_bit() -> None:
    require_validated_gpu()
    device = torch.device("cuda")
    plain_decoder = build_decoder(device)
    decoder = fuse(build_decoder(device), device)
    for row_count, is_fused in ((MAX_FUSED_ROWS + 1, False), (MAX_FUSED_ROWS, True)):
        step = tower_steps(decoder, row_count, seed=31, device=device)[0]
        plain_attention = PagedTestAttention(decoder=plain_decoder, device=device)
        attention = PagedTestAttention(decoder=decoder, device=device)
        with torch.no_grad():
            plain = run_tower_step(plain_decoder, plain_attention, step)
            outputs = run_tower_step(decoder, attention, step)
        is_same = all(
            torch.equal(bits(output), bits(plain_output))
            for output, plain_output in zip(outputs, plain)
        ) and torch.equal(bits(attention.key_caches), bits(plain_attention.key_caches))
        assert is_same != is_fused, row_count


def tower_kv_pool(
    decoder: AudioDecoder, device: torch.device, kv_cache_layout: str | None = None
) -> tuple[Mock, list[int]]:
    attention = decoder.shared_layers[0].self_attn
    layer_count = len(tower_layers(decoder))
    full_pool = MHATokenToKVPool(
        size=SLOT_COUNT,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=attention.kv_head_count,
        head_dim=attention.head_dim,
        layer_num=layer_count,
        device=str(device),
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    layer_ids = [2 + layer_index for layer_index in range(layer_count)]
    mapping = {layer_id: index for index, layer_id in enumerate(layer_ids)}
    kv_pool = Mock(spec=HybridLinearKVPool)
    kv_pool.full_kv_pool = full_pool
    kv_pool.full_attention_layer_id_mapping = mapping
    kv_pool.get_key_buffer.side_effect = lambda layer_id: full_pool.get_key_buffer(
        mapping[layer_id]
    )
    kv_pool.get_value_buffer.side_effect = lambda layer_id: full_pool.get_value_buffer(
        mapping[layer_id]
    )
    return kv_pool, layer_ids


def unsupported_reason(
    decoder: AudioDecoder, kv_pool: Mock, layer_ids: list[int], device: torch.device
) -> str | None:
    return decoder.fused_layers_unsupported_reason(
        kv_pool=kv_pool,
        tower_layer_ids=layer_ids,
        context_length=FIRST_POSITION,
        device=device,
    )


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_the_resolver_takes_only_plain_bf16_layers_and_a_plain_kv_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = build_decoder(device)
    kv_pool, layer_ids = tower_kv_pool(decoder, device)
    with monkeypatch.context() as untested_gpu:
        untested_gpu.setattr(
            torch.cuda, "get_device_capability", lambda device=None: (12, 0)
        )
        assert (
            unsupported_reason(decoder, kv_pool, layer_ids, device)
            == "untested GPU sm_120"
        )
    with monkeypatch.context() as validated_gpu:
        validated_gpu.setattr(
            torch.cuda,
            "get_device_capability",
            lambda device=None: VALIDATED_COMPUTE_CAPABILITY,
        )
        assert unsupported_reason(decoder, kv_pool, layer_ids, device) is None
        assert "not a CUDA device" in unsupported_reason(
            decoder, kv_pool, layer_ids, torch.device("cpu")
        )
        assert "has no full-attention KV buffer" in unsupported_reason(
            decoder, kv_pool, [*layer_ids, 99], device
        )
        rotary = decoder.shared_layers[0].self_attn.rotary_embedding
        assert "rotary table" in decoder.fused_layers_unsupported_reason(
            kv_pool=kv_pool,
            tower_layer_ids=layer_ids,
            context_length=rotary.cos_sin_cache.shape[0] + 1,
            device=device,
        )
        full_pool = kv_pool.full_kv_pool
        for attribute, value in (
            ("use_hnd", True),
            ("kv_cache_layout", "vectorized_5d"),
            ("store_dtype", torch.uint8),
            ("page_size", 16),
        ):
            original = getattr(full_pool, attribute)
            setattr(full_pool, attribute, value)
            assert "KV pool" in unsupported_reason(decoder, kv_pool, layer_ids, device)
            setattr(full_pool, attribute, original)
        kv_pool.full_kv_pool = Mock(spec=MHATokenToKVPool)
        assert "KV pool" in unsupported_reason(decoder, kv_pool, layer_ids, device)
        kv_pool.full_kv_pool = full_pool
        with monkeypatch.context() as deterministic_inference:
            deterministic_inference.setattr(
                tower_kernels,
                "get_exec",
                lambda: SimpleNamespace(
                    deterministic=SimpleNamespace(enable_deterministic_inference=True)
                ),
            )
            assert "deterministic" in unsupported_reason(
                decoder, kv_pool, layer_ids, device
            )
        layer = decoder.residual_layers[2]
        original_gate_up = layer.mlp.gate_up_proj
        in_features = original_gate_up.in_features
        out_features = original_gate_up.out_features
        for replacement in (
            nn.Linear(in_features, out_features, bias=True).to(device, torch.bfloat16),
            nn.Linear(in_features, out_features, bias=False).to(device, torch.float16),
            type("QuantizedLinear", (nn.Linear,), {})(
                in_features, out_features, bias=False
            ).to(device, torch.bfloat16),
        ):
            layer.mlp.gate_up_proj = replacement
            assert "gate_up_proj" in unsupported_reason(
                decoder, kv_pool, layer_ids, device
            )
        layer.mlp.gate_up_proj = original_gate_up
        original_table = rotary.cos_sin_cache
        rotary.cos_sin_cache = original_table.half()
        assert "rotary table" in unsupported_reason(decoder, kv_pool, layer_ids, device)
        rotary.cos_sin_cache = original_table
        assert unsupported_reason(decoder, kv_pool, layer_ids, device) is None


@accelerator
@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_this_gpu_takes_the_fused_layers_only_if_validated() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    decoder = build_decoder(device)
    kv_pool, layer_ids = tower_kv_pool(decoder, device)
    major, minor = torch.cuda.get_device_capability(device)

    reason = unsupported_reason(decoder, kv_pool, layer_ids, device)

    if (major, minor) == VALIDATED_COMPUTE_CAPABILITY:
        assert reason is None
    else:
        assert reason == f"untested GPU sm_{major}{minor}"


def test_the_resolver_needs_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tower_kernels, "HAS_TRITON", False)
    reason = fused_tower_layers_unsupported_reason(
        layers=[],
        head_count=4,
        kv_head_count=2,
        rotary_dim=32,
        rotary_cos_sin=torch.zeros((8, 32)),
        context_length=8,
        kv_pool=Mock(spec=HybridLinearKVPool),
        tower_layer_ids=[],
        device=torch.device("cuda"),
    )
    assert reason == "Triton is not installed"


def resolver_reason_with_compute_capability(
    monkeypatch: pytest.MonkeyPatch, compute_capability: tuple[int, int]
) -> str | None:
    monkeypatch.setattr(tower_kernels, "HAS_TRITON", True)
    monkeypatch.setattr(
        tower_kernels, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: compute_capability
    )
    monkeypatch.setattr(
        tower_kernels,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=True)
        ),
    )
    return fused_tower_layers_unsupported_reason(
        layers=[],
        head_count=4,
        kv_head_count=2,
        rotary_dim=32,
        rotary_cos_sin=torch.zeros((8, 32)),
        context_length=8,
        kv_pool=Mock(spec=HybridLinearKVPool),
        tower_layer_ids=[],
        device=torch.device("cuda"),
    )


def test_compute_capability_9_0_goes_on_to_the_next_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = resolver_reason_with_compute_capability(monkeypatch, (9, 0))

    assert reason is not None and reason.startswith("deterministic inference is on")


@pytest.mark.parametrize(
    ("compute_capability", "expected_reason"),
    [
        ((8, 6), "untested GPU sm_86"),
        ((8, 9), "untested GPU sm_89"),
        ((10, 0), "untested GPU sm_100"),
        ((12, 0), "untested GPU sm_120"),
    ],
)
def test_other_compute_capabilities_keep_the_module_path(
    monkeypatch: pytest.MonkeyPatch,
    compute_capability: tuple[int, int],
    expected_reason: str,
) -> None:
    reason = resolver_reason_with_compute_capability(monkeypatch, compute_capability)

    assert reason == expected_reason


def test_tower_kernels_import_without_triton() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins

                original_import = builtins.__import__

                def without_triton(name, *args, **kwargs):
                    if name == "triton" or name.startswith("triton."):
                        raise ModuleNotFoundError(name, name=name)
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = without_triton

                import torch

                from sglang_omni.models.fishaudio_drama3 import tower_kernels

                assert not tower_kernels.HAS_TRITON
                assert (
                    tower_kernels.fused_tower_layers_unsupported_reason(
                        layers=[],
                        head_count=4,
                        kv_head_count=2,
                        rotary_dim=32,
                        rotary_cos_sin=torch.zeros((8, 32)),
                        context_length=8,
                        kv_pool=None,
                        tower_layer_ids=[],
                        device=torch.device("cpu"),
                    )
                    == "Triton is not installed"
                )
                """
            ),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
