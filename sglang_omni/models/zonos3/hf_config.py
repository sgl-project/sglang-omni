# SPDX-License-Identifier: Apache-2.0
"""Zonos3 backbone config from params.json; a Qwen3.5 MoE text config so the runtime allocates the gated delta net state pools."""

from __future__ import annotations

import json
import os

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig

MODEL_TYPE = "zonos3"
ARCHITECTURE = "Zonos3ForCausalLM"
SGLANG_ARCHITECTURE = "Zonos3SGLangModel"
LANGUAGE_MODEL_DIRECTORY = os.path.join("lm", "zonos3_mini")
PARAMS_FILE_NAME = "params.json"
CHECKPOINT_FILE_NAME = "model.pth"
RESIDUAL_TYPE = "full_attnres"
# Every fourth layer is global attention, the rest are gated delta nets.
ATTENTION_PATTERN = "gdn3_global1"
FULL_ATTENTION_INTERVAL = 4
LINEAR_HEAD_COUNT = 4
LINEAR_HEAD_DIMENSION = 128
LINEAR_CONVOLUTION_KERNEL_SIZE = 4
# Loudness, fmax and speaking rate bucket counts of the released checkpoint.
DEFAULT_CONDITION_BUCKET_COUNTS = (10, 10, 30)
DEFAULT_CONDITION_NAMES = ("loudness", "fmax", "speaking_rate")


class Zonos3Config(Qwen3_5MoeTextConfig):
    """Hybrid backbone geometry plus the frame layout of the audio and text columns."""

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        codebook_count: int = 10,
        codebook_size: int = 4096,
        end_of_audio_id: int = 4096,
        audio_pad_id: int = 4097,
        text_pad_id: int = 8196,
        logit_softcap: float = 15.0,
        expert_latent_size: int = 256,
        dense_intermediate_size: int = 2048,
        condition_bucket_counts: list[int] | None = None,
        condition_names: list[str] | None = None,
        condition_edges: dict[str, list[float]] | None = None,
        **kwargs: object,
    ) -> None:
        self.codebook_count = codebook_count
        self.codebook_size = codebook_size
        self.end_of_audio_id = end_of_audio_id
        self.audio_pad_id = audio_pad_id
        self.text_pad_id = text_pad_id
        self.logit_softcap = logit_softcap
        self.expert_latent_size = expert_latent_size
        self.dense_intermediate_size = dense_intermediate_size
        self.condition_bucket_counts = condition_bucket_counts or list(
            DEFAULT_CONDITION_BUCKET_COUNTS
        )
        self.condition_names = condition_names or list(DEFAULT_CONDITION_NAMES)
        self.condition_edges = condition_edges or {}
        super().__init__(**kwargs)

    @property
    def audio_vocab_size(self) -> int:
        """Codes plus end-of-audio plus pad."""
        return self.codebook_size + 2

    @property
    def frame_width(self) -> int:
        """Audio codebook columns plus the text column."""
        return self.codebook_count + 1


def round_up(value: float, multiple: int) -> int:
    """Smallest multiple of multiple that is at least int(value)."""
    return multiple * ((int(value) + multiple - 1) // multiple)


def zonos3_config_from_params(params: dict[str, object]) -> Zonos3Config:
    """Translate the checkpoint's params.json model section into a Zonos3Config."""
    model = params["model"]
    assert isinstance(model, dict)
    if model["attention_pattern"] != ATTENTION_PATTERN:
        raise NotImplementedError(
            f"attention_pattern {model['attention_pattern']!r} is not supported; "
            f"expected {ATTENTION_PATTERN!r}"
        )
    elif model["residual_type"] != RESIDUAL_TYPE:
        raise NotImplementedError(
            f"residual_type {model['residual_type']!r} is not supported"
        )
    else:
        pass
    hidden_size = int(model["dim"])
    return Zonos3Config(
        architectures=[SGLANG_ARCHITECTURE],
        codebook_count=int(model["n_codebooks"]),
        codebook_size=int(model["codebook_size"]),
        end_of_audio_id=int(model["eoa_id"]),
        audio_pad_id=int(model["audio_pad_id"]),
        text_pad_id=int(model["text_vocab"]),
        logit_softcap=float(model["loss_softcap"]),
        expert_latent_size=int(hidden_size * float(model["moe_alpha"])),
        dense_intermediate_size=round_up(
            hidden_size * float(model["ffn_dim_multiplier"]), int(model["multiple_of"])
        ),
        condition_bucket_counts=[
            int(count) for count in model["condition_bucket_counts"]
        ],
        condition_names=[str(name) for name in model["condition_names"]],
        condition_edges={
            str(name): [float(edge) for edge in edges]
            for name, edges in model["condition_edges"].items()
        },
        vocab_size=int(model["text_vocab"]) + 1,
        hidden_size=hidden_size,
        num_hidden_layers=int(model["n_layers"]),
        num_attention_heads=int(model["n_heads"]),
        num_key_value_heads=int(model["n_kv_heads"]),
        head_dim=int(model["head_dim"]),
        max_position_embeddings=int(model["max_seqlen"]),
        rms_norm_eps=float(model["norm_eps"]),
        rope_theta=float(model["rope_theta"]),
        partial_rotary_factor=1.0,
        full_attention_interval=FULL_ATTENTION_INTERVAL,
        linear_conv_kernel_dim=LINEAR_CONVOLUTION_KERNEL_SIZE,
        linear_key_head_dim=LINEAR_HEAD_DIMENSION,
        linear_value_head_dim=LINEAR_HEAD_DIMENSION,
        linear_num_key_heads=LINEAR_HEAD_COUNT,
        linear_num_value_heads=LINEAR_HEAD_COUNT,
        num_experts=int(model["moe_n_experts"]),
        num_experts_per_tok=int(model["moe_router_topk"]),
        moe_intermediate_size=round_up(
            hidden_size * float(model["moe_ffn_dim_multiplier"]),
            int(model["moe_multiple_of"]),
        ),
        shared_expert_intermediate_size=0,
        norm_topk_prob=True,
        hidden_act="silu",
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
    )


def load_params(language_model_directory: str) -> dict[str, object]:
    """Decoded params.json of the language model directory."""
    with open(os.path.join(language_model_directory, PARAMS_FILE_NAME)) as params_file:
        return json.load(params_file)
