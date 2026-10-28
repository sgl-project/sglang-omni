# SPDX-License-Identifier: Apache-2.0
"""The Drama 3 engine test's fixture: synthetic checkpoint, prompts, scenarios, records.

The checkpoint has the Drama 3 layout and names with random weights; the
records are what mtp_engines.py saves for test_mtp_engine.py to check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig as TransformersTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as TransformersBackbone,
)

TOTAL_GPU_MEMORY_FRACTION = 0.3
CONTEXT_LENGTH = 256
# Small enough that the allocator hands freed KV slots to later requests.
MAX_TOTAL_TOKENS = 1024
MAX_RUNNING_REQUESTS = 8
MAX_NEW_TOKENS = 24
ABORT_MAX_NEW_TOKENS = 150
ABORT_AFTER_FRAMES = 8
RETRACT_AFTER_FRAMES = 6
END_OF_SPEECH_FRAME = 5
# Synthetic weights: the backbone at its library init with a sharpened lm_head;
# tower q/k norm gains near 0.3 keep its attention soft, which keeps bfloat16
# rounding from growing as fast through the 13 tower layers.
LM_HEAD_SCALE = 30.0
TOWER_HEAD_STD = 0.2
TOWER_QK_NORM_GAIN = 0.3

TEXT_CONFIG: dict[str, object] = {
    "model_type": "fish_omni_text",
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 512,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "mlp_layer_types": ["moe"] * 4,
    "rope_parameters": {
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25,
        "rope_type": "default",
    },
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 64,
    "mamba_ssm_dtype": "float32",
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "shared_expert_gate": True,
    "intermediate_size": 128,
    "rotary_interleaved": False,
    "use_qk_norm": True,
    "attn_output_gate": True,
    "layernorm_zero_centered_gamma": True,
}
DECODER_CONFIG: dict[str, object] = {
    "model_type": "fish_qwen3_audio_decoder",
    "decoder_type": "codebook_mtp_shared",
    "dim": 256,
    "text_dim": 256,
    "n_head": 4,
    "n_local_heads": 2,
    "head_dim": 64,
    "n_layer": 4,
    "shared_enrichment_layers": 4,
    "intermediate_size": 256,
    "norm_eps": 1e-6,
    "num_codebooks": 10,
    "vocab_size": 64,
    "max_seq_len": 11,
    "rope_base": 10000.0,
    "rotary_percent": 0.5,
    "rotary_interleaved": True,
    "attention_qk_norm": True,
    "attention_output_gate": True,
    "layernorm_zero_centered_gamma": True,
    "vq_feature_scope": "all_vq",
    "vq_feature_placement": "backbone_partial",
    "vq_backbone_codebooks": 1,
    "mtp_spec": {"attention_variant": "full_attention"},
    "use_moe": False,
}
SEMANTIC_START_TOKEN_ID = 448
SEMANTIC_END_TOKEN_ID = 511
EOS_TOKEN_ID = 3
CHECKPOINT_CONFIG: dict[str, object] = {
    "model_type": "fish_omni",
    "architectures": ["FishOmniModel"],
    "text_config": TEXT_CONFIG,
    "audio_decoder_config": DECODER_CONFIG,
    "audio_encoder_config": None,
    "semantic_start_token_id": SEMANTIC_START_TOKEN_ID,
    "semantic_end_token_id": SEMANTIC_END_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_ID,
    "pad_token_id": 0,
    "speech_template_version": "reference-bracket-v1",
    "fish_schema_version": 1,
    "torch_dtype": "bfloat16",
}
PROMPT_LENGTHS = {
    "p2": 2,
    "p5": 5,
    "p9": 9,
    "p13": 13,
    "p23": 23,
    "p31": 31,
    "p40": 40,
    "p60": 60,
    "a100": 100,
    "b100": 100,
    "c100": 100,
}
WAVE_PROMPTS = ["p2", "p5", "p9", "p13", "p23", "p31", "p40", "p60"]
FP8_MAX = 448.0
FP8_BLOCK_SIZE = 128
# The offline quantizer floors a block's bfloat16 amax at bfloat16(1e-10).
FP8_BLOCK_AMAX_FLOOR = 1e-10
QUANTIZATION_CONFIG: dict[str, str | list[int]] = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE],
    "activation_scheme": "dynamic",
}
EXPERT_WEIGHT_SUFFIXES = (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
# How a checkpoint stores the routed experts: block-fp8 with fp32 dequant
# multipliers, or those values multiplied back out to bfloat16.
ExpertFormat = Literal["fp8", "dequantized"]
# The layers a quantization config chooses a method for.
QuantizableLayerKind = Literal["linear", "embedding", "routed_experts"]


class RequestSpec(BaseModel):
    prompt_name: str
    max_new_tokens: int = MAX_NEW_TOKENS
    end_at_frame: int | None = None


class Scenario(BaseModel):
    name: str
    waves: list[list[RequestSpec]]
    is_sequential: bool = False
    abort_target: RequestSpec | None = None
    abort_after_frames: int = ABORT_AFTER_FRAMES
    retract_after_frames: int | None = None


class EngineSpec(BaseModel):
    model_dir: str
    context_length: int
    max_running_requests: int
    max_prefill_tokens: int
    max_total_tokens: int
    total_gpu_memory_fraction: float
    prompts: dict[str, list[list[int]]]
    scenarios: list[Scenario]
    record_moe_calls: bool = False
    enable_async_decode: bool = False
    async_decode_min_batch_size: int = 1


@dataclass(frozen=True, kw_only=True)
class RequestStep:
    """One request's part of one forward, as the recording model saw it."""

    request_id: str
    is_extend: bool
    positions: list[int]
    hidden_states: torch.Tensor
    input_frames: torch.Tensor
    semantic_logits: torch.Tensor
    # [residual heads, codebook size]: head i scores code i + 1 of the frame.
    head_logits: torch.Tensor
    ras_window: list[int]
    sampled_frame: list[int]
    request_row: int
    written_slots: list[int]
    sequence_slots: list[int]
    # [tower layers, rows, 2, kv width] after the step; the whole sequence on
    # the request's last step.
    written_cache: torch.Tensor
    sequence_cache: torch.Tensor | None


@dataclass(frozen=True, kw_only=True)
class MoeCall:
    """One routed-expert (FusedMoE) call, copied before the shared expert is added."""

    layer_index: int
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class ForwardRecord:
    forward_index: int
    is_extend: bool
    steps: list[RequestStep]
    # Every FusedMoE call of the forward, when the spec records them.
    moe_calls: list[MoeCall] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class FreePoolCounts:
    request_row_count: int
    mamba_slot_count: int
    kv_token_count: int


@dataclass(kw_only=True)
class RequestOutcome:
    request_id: str
    request_spec: RequestSpec
    aborted: bool = False
    result_frames: list[list[list[int]]] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    stream_message_count: int = 0


@dataclass(frozen=True, kw_only=True)
class ScenarioOutcome:
    name: str
    request_ids: list[str]
    first_forward_index: int
    end_forward_index: int
    idle_free_counts: FreePoolCounts
    frames_at_abort: int | None
    results_at_abort: int | None
    retracted_request_count: int | None


@dataclass(frozen=True, kw_only=True)
class EngineFacts:
    full_attention_layer_ids: list[int]
    mamba_layer_ids: list[int]
    kv_layer_count: int
    kv_head_count: int
    kv_head_dim: int
    kv_element_bytes: int
    chunked_prefill_size: int | None
    max_prefill_tokens: int
    max_running_requests: int
    enable_async_decode: bool
    async_decode_min_batch_size: int
    async_decode_resolve_before_prefill: bool


@dataclass(frozen=True, kw_only=True)
class ExpertLayerFacts:
    """A FusedMoE layer as the engine runs it, after weight post-processing."""

    quant_method: str
    runner_backend: str
    uses_fp8_w8a8: bool
    block_shape: list[int] | None
    fused_shared_expert_count: int
    # The layer's w13 / w2 weights, and their scales when quantized.
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class LayerQuantMethod:
    layer_kind: QuantizableLayerKind
    quant_method: str


@dataclass(frozen=True, kw_only=True)
class LoadFacts:
    # Parameters still holding the NaN they were filled with before loading.
    unloaded_parameters: list[str]
    quant_methods: dict[str, LayerQuantMethod]
    parameter_dtypes: dict[str, torch.dtype]
    expert_layers: dict[int, ExpertLayerFacts]
    is_deep_gemm_enabled: bool
    device_capability: tuple[int, int]


@dataclass(frozen=True, kw_only=True)
class EngineRun:
    facts: EngineFacts
    load_facts: LoadFacts
    baseline_free_counts: FreePoolCounts
    scenarios: list[ScenarioOutcome]
    outcomes: dict[str, RequestOutcome]
    forwards: list[ForwardRecord]
    lookahead_forward_indices: list[int]
    discarded_steps: list[tuple[int, str]]


def prompt_rows(length: int, seed: int) -> list[list[int]]:
    """Text rows, semantic rows every third row and an end-of-turn row near the end."""
    generator = torch.Generator().manual_seed(seed)
    codebook_size = int(DECODER_CONFIG["vocab_size"])
    rows = []
    for row_index in range(length):
        if length >= 5 and row_index == length - 3:
            rows.append([EOS_TOKEN_ID] + [0] * 10)
        elif row_index % 3 == 1:
            codes = torch.randint(0, codebook_size, (10,), generator=generator).tolist()
            rows.append([SEMANTIC_START_TOKEN_ID + codes[0]] + codes)
        else:
            token = int(
                torch.randint(10, SEMANTIC_START_TOKEN_ID, (1,), generator=generator)
            )
            rows.append([token] + [0] * 10)
    return rows


PROMPTS = {
    name: prompt_rows(length, seed=index)
    for index, (name, length) in enumerate(PROMPT_LENGTHS.items())
}


def requests_for(*prompt_names: str) -> list[RequestSpec]:
    return [RequestSpec(prompt_name=prompt_name) for prompt_name in prompt_names]


MAIN_SCENARIOS = [
    # Every prompt also runs alone: the reference of the backbone comparison.
    Scenario(name="single", waves=[requests_for(*PROMPT_LENGTHS)], is_sequential=True),
    Scenario(name="concurrent", waves=[requests_for("p2", "p9", "p23", "p60")]),
    Scenario(name="full_wave_1", waves=[requests_for(*WAVE_PROMPTS)]),
    Scenario(name="full_wave_2", waves=[requests_for(*WAVE_PROMPTS)]),
    Scenario(
        name="abort",
        waves=[requests_for("p9", "p23")],
        abort_target=RequestSpec(
            prompt_name="p60", max_new_tokens=ABORT_MAX_NEW_TOKENS
        ),
    ),
    Scenario(name="after_abort", waves=[requests_for(*WAVE_PROMPTS)]),
    Scenario(
        name="end_of_speech",
        waves=[
            [
                RequestSpec(prompt_name="p23", end_at_frame=END_OF_SPEECH_FRAME),
                RequestSpec(prompt_name="p9", end_at_frame=0),
            ]
        ],
    ),
    Scenario(
        name="admin_retract",
        waves=[requests_for("p9", "p23", "p31")],
        retract_after_frames=RETRACT_AFTER_FRAMES,
    ),
    Scenario(name="admission", waves=[requests_for("a100", "b100", "c100")]),
]
TEST_RETRACT_SCENARIOS = [
    Scenario(name="test_retract", waves=[requests_for("p9", "p23", "p31")])
]


def write_synthetic_checkpoint(
    checkpoint_dir: Path, checkpoint_config: dict[str, object] | None = None
) -> None:
    """Drama 3 layout and names, random weights; the backbone as the library inits it.

    checkpoint_config defaults to CHECKPOINT_CONFIG.
    """
    config = CHECKPOINT_CONFIG if checkpoint_config is None else checkpoint_config
    text_config = config["text_config"]
    decoder_config = config["audio_decoder_config"]
    assert isinstance(text_config, dict) and isinstance(decoder_config, dict)
    torch.manual_seed(0)
    backbone = TransformersBackbone(
        TransformersTextConfig(
            **{
                key: value
                for key, value in text_config.items()
                if key not in ("model_type", "mlp_layer_types")
            },
            mlp_layer_types=["sparse"] * text_config["num_hidden_layers"],
        )
    )
    expert_width = int(text_config["moe_intermediate_size"])
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in backbone.state_dict().items():
        tensor = tensor.to(torch.bfloat16)
        if "rotary_emb" in name:
            continue
        elif name == "lm_head.weight":
            tensors[name] = (tensor.float() * LM_HEAD_SCALE).to(torch.bfloat16)
            continue
        else:
            pass
        name = "model.language_model." + name.removeprefix("model.")
        if name.endswith(".mlp.experts.gate_up_proj"):
            stem = name.removesuffix(".gate_up_proj")
            for expert in range(tensor.shape[0]):
                tensors[f"{stem}.{expert}.gate_proj.weight"] = tensor[
                    expert, :expert_width
                ].contiguous()
                tensors[f"{stem}.{expert}.up_proj.weight"] = tensor[
                    expert, expert_width:
                ].contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            stem = name.removesuffix(".down_proj")
            for expert in range(tensor.shape[0]):
                tensors[f"{stem}.{expert}.down_proj.weight"] = tensor[
                    expert
                ].contiguous()
        else:
            tensors[name] = tensor.contiguous()
    tensors.update(decoder_checkpoint(decoder_config))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, checkpoint_dir / "model.safetensors")
    (checkpoint_dir / "config.json").write_text(json.dumps(config))


def block_expanded(block_values: torch.Tensor) -> torch.Tensor:
    """Per-block values repeated over each 128x128 block of the weight."""
    return block_values.repeat_interleave(FP8_BLOCK_SIZE, dim=0).repeat_interleave(
        FP8_BLOCK_SIZE, dim=1
    )


def quantize_block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 values and fp32 per-128x128-block dequant multipliers, as quantized offline.

    The block amax is taken in bfloat16; the multiplier is 1 / (448 / amax), two
    fp32 divisions, so that value * multiplier dequantizes.
    """
    row_count, column_count = weight.shape
    if row_count % FP8_BLOCK_SIZE != 0 or column_count % FP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"block-fp8 needs dims divisible by {FP8_BLOCK_SIZE}; got "
            f"{tuple(weight.shape)}"
        )
    else:
        pass
    weight = weight.to(torch.bfloat16)
    blocks = weight.reshape(
        row_count // FP8_BLOCK_SIZE,
        FP8_BLOCK_SIZE,
        column_count // FP8_BLOCK_SIZE,
        FP8_BLOCK_SIZE,
    )
    amax_floor = torch.tensor(FP8_BLOCK_AMAX_FLOOR).to(torch.bfloat16)
    amax = torch.maximum(blocks.abs().amax(dim=(1, 3)), amax_floor).float()
    quant_scale = FP8_MAX / amax
    scale_inv = 1.0 / quant_scale
    values = (weight.float() * block_expanded(quant_scale)).clamp(-FP8_MAX, FP8_MAX)
    return values.to(torch.float8_e4m3fn), scale_inv


def dequantize_block_fp8(values: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """fp32 weight the block-fp8 values stand for."""
    return values.float() * block_expanded(scale_inv)


def is_routed_expert_weight(checkpoint_name: str) -> bool:
    return ".mlp.experts." in checkpoint_name and checkpoint_name.endswith(
        EXPERT_WEIGHT_SUFFIXES
    )


def write_expert_variant(
    source_dir: Path, target_dir: Path, expert_format: ExpertFormat
) -> dict[str, torch.Tensor]:
    """Rewrite the source checkpoint's routed experts in expert_format; copy the rest.

    Returns the expert tensors written: values and their .weight_scale_inv for
    fp8, bfloat16 weights for dequantized.
    """
    tensors = load_file(source_dir / "model.safetensors")
    expert_tensors = {}
    for name in [name for name in tensors if is_routed_expert_weight(name)]:
        values, scale_inv = quantize_block_fp8(tensors[name])
        if expert_format == "fp8":
            expert_tensors[name] = values
            expert_tensors[f"{name}_scale_inv"] = scale_inv
        else:
            dequantized = dequantize_block_fp8(values, scale_inv)
            expert_tensors[name] = dequantized.to(torch.bfloat16)
    tensors.update(expert_tensors)
    if expert_format == "fp8":
        config = {**CHECKPOINT_CONFIG, "quantization_config": QUANTIZATION_CONFIG}
    else:
        config = CHECKPOINT_CONFIG
    target_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, target_dir / "model.safetensors")
    (target_dir / "config.json").write_text(json.dumps(config))
    return expert_tensors


def decoder_checkpoint(
    decoder_config: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    """Random tower tensors under the checkpoint's names and layouts; decoder_config
    defaults to DECODER_CONFIG."""
    config = DECODER_CONFIG if decoder_config is None else decoder_config
    generator = torch.Generator().manual_seed(1)
    dim = int(config["dim"])
    head_count = int(config["n_head"])
    kv_head_count = int(config["n_local_heads"])
    head_dim = int(config["head_dim"])
    intermediate_size = int(config["intermediate_size"])
    codebook_size = int(config["vocab_size"])
    num_codebooks = int(config["num_codebooks"])

    def random_tensor(*shape: int, std: float) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * std).to(torch.bfloat16)

    tensors = {
        "audio_decoder.codebook_embeddings.weight": random_tensor(
            codebook_size * num_codebooks, dim, std=1.0
        )
    }
    layers = [
        (f"audio_decoder_mtp_shared.layers.{index}.", "semantic_embedding")
        for index in range(int(config["shared_enrichment_layers"]))
    ] + [
        (f"audio_decoder_mtp.layers.{index}.", "codebook_embedding")
        for index in range(num_codebooks - 1)
    ]
    for prefix, embedding_name in layers:
        tensors[f"{prefix}{embedding_name}.weight"] = random_tensor(
            codebook_size, dim, std=1.0
        )
        tensors[f"{prefix}eh_proj.weight"] = random_tensor(
            dim, 2 * dim, std=(2 * dim) ** -0.5
        )
        for norm_name in (
            "enorm",
            "hnorm",
            "final_layernorm",
            "transformer_layer.attention_norm",
            "transformer_layer.ffn_norm",
        ):
            tensors[f"{prefix}{norm_name}.weight"] = random_tensor(dim, std=0.1)
        for norm_name in ("q_norm", "k_norm"):
            # The checkpoint stores zero-centered gains as gain - 1.
            tensors[f"{prefix}transformer_layer.attention.{norm_name}.weight"] = (
                random_tensor(head_dim, std=0.02) + (TOWER_QK_NORM_GAIN - 1.0)
            )
        tensors[f"{prefix}transformer_layer.attention.wqkv.weight"] = random_tensor(
            (2 * head_count + 2 * kv_head_count) * head_dim, dim, std=dim**-0.5
        )
        tensors[f"{prefix}transformer_layer.attention.wo.weight"] = random_tensor(
            dim, head_count * head_dim, std=(head_count * head_dim) ** -0.5
        )
        for projection_name in ("w1", "w3"):
            tensors[
                f"{prefix}transformer_layer.feed_forward.{projection_name}.weight"
            ] = random_tensor(intermediate_size, dim, std=dim**-0.5)
        tensors[f"{prefix}transformer_layer.feed_forward.w2.weight"] = random_tensor(
            dim, intermediate_size, std=intermediate_size**-0.5
        )
    for index in range(num_codebooks - 1):
        tensors[f"audio_decoder_mtp.heads.{index}.weight"] = random_tensor(
            codebook_size, dim, std=TOWER_HEAD_STD
        )
    return tensors
