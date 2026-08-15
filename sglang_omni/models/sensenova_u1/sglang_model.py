# SPDX-License-Identifier: Apache-2.0
"""Native SGLang language tower for SenseNova U1.

This module intentionally does not import the official HF ``NEOChatModel``.
The classes here cover the U1 language tower with SGLang-native layers. Vision
and flow-matching modules remain separate work items; this file is the first
native load/forward milestone for the text/MoT backbone.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from torch import nn

from sglang_omni.models.sensenova_u1.hybrid_attention import (
    build_image_spans,
    build_image_token_tag_from_t_indexes,
    build_u1_hybrid_allowed_matrix,
    build_u1_hybrid_backend_mask,
)
from sglang_omni.models.weight_loader import default_weight_loader, resolve_dtype
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import get_tensor_model_parallel_world_size
from sglang_omni.vendor.sglang.layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RMSNorm,
    RadixAttention,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.utils import add_prefix


HF_MODELING_MODULE_PREFIXES: tuple[str, ...] = (
    "sensenova_u1.models.neo_unify.modeling_neo_chat",
    "sensenova_u1.models.neo_unify.modeling_qwen3",
    "sensenova_u1.models.neo_unify.modeling_qwen3_moe",
    "sensenova_u1.models.neo_unify.modeling_neo_vit",
    "sensenova_u1.models.neo_unify.modeling_fm_modules",
)


def _blocked_hf_modeling_modules() -> list[str]:
    candidates = (
        "sensenova_u1.models.neo_unify",
        *HF_MODELING_MODULE_PREFIXES,
    )
    # Importing any descendant necessarily installs its parent module in
    # sys.modules, so exact parent lookups preserve the guard while avoiding a
    # full scan of every loaded Python module on each native forward boundary.
    return sorted(name for name in candidates if name in sys.modules)


def assert_no_hf_modeling_imported(*, context: str) -> None:
    """Fail if the native serving path imported official U1 modeling code."""
    imported = _blocked_hf_modeling_modules()
    if imported:
        joined = ", ".join(imported)
        raise RuntimeError(f"{context}: HF U1 modeling modules imported: {joined}")


class _HFModelingBlocker:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        del path, target
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in HF_MODELING_MODULE_PREFIXES
        ):
            raise ImportError(
                "Native SenseNova U1 serving path must not import official HF "
                f"modeling module {fullname!r}"
            )
        return None


@contextmanager
def block_hf_modeling_imports() -> Iterator[None]:
    """Install a temporary import guard for official U1 HF modeling modules."""
    blocker = _HFModelingBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass


@dataclass(slots=True)
class SenseNovaU1LoadReport:
    loaded_language_keys: list[str] = field(default_factory=list)
    missing_language_keys: list[str] = field(default_factory=list)
    unexpected_language_keys: list[str] = field(default_factory=list)
    ignored_non_language_keys: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_language_keys)

    @property
    def missing_count(self) -> int:
        return len(self.missing_language_keys)

    @property
    def unexpected_count(self) -> int:
        return len(self.unexpected_language_keys)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def ok(self) -> bool:
        return (
            self.missing_count == 0
            and self.unexpected_count == 0
            and self.error_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "loaded_count": self.loaded_count,
                "missing_count": self.missing_count,
                "unexpected_count": self.unexpected_count,
                "error_count": self.error_count,
                "ok": self.ok,
            }
        )
        return data


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_u1_llm_config(model_path: str | Path) -> SimpleNamespace:
    """Read U1 ``llm_config`` from config.json without importing HF code."""
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    llm_config = raw.get("llm_config", raw)
    cfg = _to_namespace(llm_config)
    if not hasattr(cfg, "layer_types") or cfg.layer_types is None:
        cfg.layer_types = ["full_attention"] * int(cfg.num_hidden_layers)
    if not hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
    if not hasattr(cfg, "torch_dtype"):
        cfg.torch_dtype = raw.get("torch_dtype", "bfloat16")
    return cfg


def expected_language_weight_keys(config: Any) -> set[str]:
    cfg = _to_namespace(config)
    keys: set[str] = {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.norm_mot_gen.weight",
    }
    attn_suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_proj_mot_gen.weight",
        "k_proj_mot_gen.weight",
        "v_proj_mot_gen.weight",
        "o_proj_mot_gen.weight",
        "q_norm.weight",
        "q_norm_hw.weight",
        "k_norm.weight",
        "k_norm_hw.weight",
        "q_norm_mot_gen.weight",
        "q_norm_hw_mot_gen.weight",
        "k_norm_mot_gen.weight",
        "k_norm_hw_mot_gen.weight",
    )
    layer_suffixes = (
        "input_layernorm.weight",
        "input_layernorm_mot_gen.weight",
        "post_attention_layernorm.weight",
        "post_attention_layernorm_mot_gen.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_mot_gen.gate_proj.weight",
        "mlp_mot_gen.up_proj.weight",
        "mlp_mot_gen.down_proj.weight",
    )
    for idx in range(int(cfg.num_hidden_layers)):
        layer = f"model.layers.{idx}"
        keys.update(f"{layer}.{suffix}" for suffix in layer_suffixes)
        keys.update(f"{layer}.self_attn.{suffix}" for suffix in attn_suffixes)
    return keys


def scan_checkpoint_key_summary(model_path: str | Path) -> dict[str, int]:
    index_file = Path(model_path) / "model.safetensors.index.json"
    with index_file.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    summary: dict[str, int] = {}
    for key in weight_map:
        top = key.split(".", 1)[0]
        summary[top] = summary.get(top, 0) + 1
    return summary


def iter_language_safetensors(
    model_path: str | Path,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream language-model weights from sharded safetensors.

    Yields names with the leading ``language_model.`` stripped so they match
    this module's native parameter tree.
    """
    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"
    with index_file.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if key.startswith("language_model."):
            shards.setdefault(shard_name, []).append(key)

    from safetensors import safe_open

    for shard_name in sorted(shards):
        with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as f:
            for key in sorted(shards[shard_name]):
                yield key[len("language_model.") :], f.get_tensor(key)


def _load_parameter(
    params: dict[str, nn.Parameter],
    target_name: str,
    loaded_weight: torch.Tensor,
    shard_id: str | int | None = None,
) -> None:
    param = params[target_name]
    if loaded_weight.device != param.device or loaded_weight.dtype != param.dtype:
        loaded_weight = loaded_weight.to(device=param.device, dtype=param.dtype)
    loader = getattr(param, "weight_loader", default_weight_loader)
    if shard_id is None:
        loader(param, loaded_weight)
    else:
        loader(param, loaded_weight, shard_id)


def _direct_or_stacked_target(name: str) -> tuple[str, str | int | None] | None:
    if ".self_attn." in name:
        if ".q_proj_mot_gen.weight" in name:
            return name.replace(".q_proj_mot_gen.weight", ".qkv_proj_mot_gen.weight"), "q"
        if ".k_proj_mot_gen.weight" in name:
            return name.replace(".k_proj_mot_gen.weight", ".qkv_proj_mot_gen.weight"), "k"
        if ".v_proj_mot_gen.weight" in name:
            return name.replace(".v_proj_mot_gen.weight", ".qkv_proj_mot_gen.weight"), "v"
        if ".q_proj.weight" in name:
            return name.replace(".q_proj.weight", ".qkv_proj.weight"), "q"
        if ".k_proj.weight" in name:
            return name.replace(".k_proj.weight", ".qkv_proj.weight"), "k"
        if ".v_proj.weight" in name:
            return name.replace(".v_proj.weight", ".qkv_proj.weight"), "v"
        return name, None

    if ".mlp_mot_gen." in name:
        return name, None

    if ".mlp." in name:
        return name, None

    return name, None


def _rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + norm.variance_epsilon)
    return norm.weight.to(dtype=orig_dtype) * xf.to(orig_dtype)


def _eager_add_rms_impl(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = residual + update
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
    normed = hidden_f * torch.rsqrt(variance + eps)
    normed = weight.to(dtype=hidden.dtype) * normed.to(hidden.dtype)
    return hidden, normed


_compiled_eager_add_rms = torch.compile(
    _eager_add_rms_impl,
    fullgraph=True,
    dynamic=False,
)


def _repeat_kv(
    states: torch.Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if num_query_heads == num_kv_heads:
        return states
    repeat = num_query_heads // num_kv_heads
    return states.repeat_interleave(repeat, dim=1)


class _StandaloneRotaryEmbedding(nn.Module):
    """Minimal Neox-style RoPE for standalone smoke without server args."""

    def __init__(
        self,
        head_size: int,
        *,
        max_position: int,
        base: float,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.max_position = max_position
        self.base = base
        arange = torch.arange(0, head_size, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (arange / head_size))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(x.shape[0], -1, x.shape[-1])
        cos = cos.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        sin = sin.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten().to(device=self.cos_sin_cache.device)
        capture_active = bool(
            positions.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        if not capture_active and int(positions.max().item()) >= self.max_position:
            raise ValueError(
                f"standalone RoPE position {int(positions.max().item())} exceeds "
                f"max_position {self.max_position}"
            )
        cache = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cache.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        query = self._apply_rope(query, cos, sin).reshape(query_shape)
        key = self._apply_rope(key, cos, sin).reshape(key_shape)
        return query, key


class SenseNovaU1NativeMLP(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int | None = None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = None if layer_id is None else int(layer_id)
        self.tp_size = self._resolve_tp_size(tp_size)
        self.standalone = tp_rank == 0 and tp_size == 1
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=add_prefix("gate_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=add_prefix("up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    @staticmethod
    def _resolve_tp_size(tp_size: int | None) -> int:
        if tp_size is not None:
            return int(tp_size)
        try:
            return int(get_tensor_model_parallel_world_size())
        except Exception:
            return 1

    def _is_single_rank_diagnostic_path(self) -> bool:
        return int(self.tp_size) == 1

    @staticmethod
    def _use_fp32_down_projection() -> bool:
        return os.environ.get("SENSENOVA_U1_NATIVE_MLP_DOWN_FP32", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _fp32_residual_add_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_MLP_FP32_RESIDUAL_ADD", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _layer_selected_for_fp32_residual_add(self) -> bool:
        raw = os.environ.get(
            "SENSENOVA_U1_NATIVE_MLP_FP32_RESIDUAL_ADD_LAYERS", ""
        ).strip()
        if not raw:
            return True
        if self.layer_id is None:
            return False
        selected: set[int] = set()
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if item in {"*", "all", "ALL"}:
                return True
            if "-" in item:
                start, end = item.split("-", 1)
                selected.update(range(int(start), int(end) + 1))
            else:
                selected.add(int(item))
        return self.layer_id in selected

    def _use_fp32_residual_add(self) -> bool:
        return (
            self._fp32_residual_add_enabled()
            and self._layer_selected_for_fp32_residual_add()
        )

    def apply_down_proj(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_fp32_down = self._use_fp32_down_projection()
        use_fp32_add = self._use_fp32_residual_add()
        if use_fp32_down or use_fp32_add:
            if not self._is_single_rank_diagnostic_path():
                raise RuntimeError(
                    "SENSENOVA_U1_NATIVE_MLP_DOWN_FP32 and "
                    "SENSENOVA_U1_NATIVE_MLP_FP32_RESIDUAL_ADD are only supported "
                    "for single-rank diagnostic paths"
                )
            original_dtype = hidden_states.dtype
            down = torch.nn.functional.linear(
                hidden_states.float(),
                self.down_proj.weight.float(),
                (
                    self.down_proj.bias.float()
                    if self.down_proj.bias is not None
                    else None
                ),
            )
            if use_fp32_add:
                return down
            return down.to(dtype=original_dtype)
        if self.standalone:
            return torch.nn.functional.linear(
                hidden_states,
                self.down_proj.weight,
                self.down_proj.bias,
            )
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states

    def apply_residual_add(
        self,
        residual: torch.Tensor,
        mlp_output: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_fp32_residual_add():
            if not self._is_single_rank_diagnostic_path():
                raise RuntimeError(
                    "SENSENOVA_U1_NATIVE_MLP_FP32_RESIDUAL_ADD is only supported "
                    "for single-rank diagnostic paths"
                )
            return (residual.float() + mlp_output.float()).to(dtype=residual.dtype)
        return residual + mlp_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.standalone:
            gate = torch.nn.functional.linear(
                hidden_states,
                self.gate_proj.weight,
                self.gate_proj.bias,
            )
            up = torch.nn.functional.linear(
                hidden_states,
                self.up_proj.weight,
                self.up_proj.bias,
            )
        else:
            gate, _ = self.gate_proj(hidden_states)
            up, _ = self.up_proj(hidden_states)
        hidden_states = torch.nn.functional.silu(gate) * up
        return self.apply_down_proj(hidden_states)


class SenseNovaU1NativeAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.standalone = standalone
        self.tp_size = self._resolve_tp_size(tp_size)
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.total_num_heads))
        self.t_dim = self.head_dim // 2
        self.hw_dim = self.head_dim // 4
        self.q_size = self.total_num_heads * self.head_dim
        self.kv_size = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            params_dtype=params_dtype,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            params_dtype=params_dtype,
            prefix=add_prefix("qkv_proj_mot_gen", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj_mot_gen = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj_mot_gen", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        self.q_norm = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_hw = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_hw = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_hw_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)

        self.rotary_emb = _StandaloneRotaryEmbedding(
            self.t_dim,
            max_position=int(config.max_position_embeddings),
            base=float(config.rope_theta),
            dtype=params_dtype,
        )
        self.rotary_emb_hw = _StandaloneRotaryEmbedding(
            self.hw_dim,
            max_position=int(config.max_position_embeddings_hw),
            base=float(config.rope_theta_hw),
            dtype=params_dtype,
        )
        self.attn = RadixAttention(
            self.total_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.total_num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        qkv_proj = self.qkv_proj_mot_gen if use_mot_gen else self.qkv_proj
        if self._use_separate_qkv_projection():
            if not self._is_single_rank_diagnostic_path():
                raise RuntimeError(
                    "SENSENOVA_U1_NATIVE_SEPARATE_QKV_PROJ is only supported "
                    "for single-rank paths"
                )
            q_weight, k_weight, v_weight = qkv_proj.weight.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=0,
            )
            if qkv_proj.bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = qkv_proj.bias.split(
                    [self.q_size, self.kv_size, self.kv_size],
                    dim=0,
                )
            q = torch.nn.functional.linear(hidden_states, q_weight, q_bias)
            k = torch.nn.functional.linear(hidden_states, k_weight, k_bias)
            v = torch.nn.functional.linear(hidden_states, v_weight, v_bias)
        elif self.standalone:
            qkv = torch.nn.functional.linear(hidden_states, qkv_proj.weight, qkv_proj.bias)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            qkv, _ = qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.total_num_heads, self.head_dim)
        k = k.view(-1, self.total_num_kv_heads, self.head_dim)
        v = v.view(-1, self.total_num_kv_heads, self.head_dim)

        q_t, q_hw = q.split([self.t_dim, self.t_dim], dim=-1)
        k_t, k_hw = k.split([self.t_dim, self.t_dim], dim=-1)
        if use_mot_gen:
            q_t = _rms_norm(self.q_norm_mot_gen, q_t)
            q_hw = _rms_norm(self.q_norm_hw_mot_gen, q_hw)
            k_t = _rms_norm(self.k_norm_mot_gen, k_t)
            k_hw = _rms_norm(self.k_norm_hw_mot_gen, k_hw)
        else:
            q_t = _rms_norm(self.q_norm, q_t)
            q_hw = _rms_norm(self.q_norm_hw, q_hw)
            k_t = _rms_norm(self.k_norm, k_t)
            k_hw = _rms_norm(self.k_norm_hw, k_hw)
        q_h, q_w = q_hw.split([self.hw_dim, self.hw_dim], dim=-1)
        k_h, k_w = k_hw.split([self.hw_dim, self.hw_dim], dim=-1)
        return (
            q_t.reshape(q.shape[0], -1),
            k_t.reshape(k.shape[0], -1),
            (
                q_h.reshape(q.shape[0], -1),
                k_h.reshape(k.shape[0], -1),
                q_w.reshape(q.shape[0], -1),
                k_w.reshape(k.shape[0], -1),
            ),
            v.reshape(v.shape[0], -1),
        )

    @staticmethod
    def _resolve_tp_size(tp_size: int | None) -> int:
        if tp_size is not None:
            return int(tp_size)
        try:
            return int(get_tensor_model_parallel_world_size())
        except Exception:
            return 1

    def _is_single_rank_diagnostic_path(self) -> bool:
        return int(self.tp_size) == 1

    def _separate_qkv_projection_enabled(self) -> bool:
        raw = os.environ.get("SENSENOVA_U1_NATIVE_SEPARATE_QKV_PROJ")
        if raw is None:
            return self._is_single_rank_diagnostic_path()
        return raw.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _layer_selected_for_separate_qkv_projection(self) -> bool:
        raw = os.environ.get(
            "SENSENOVA_U1_NATIVE_SEPARATE_QKV_PROJ_LAYERS", ""
        ).strip()
        if not raw:
            return True
        selected: set[int] = set()
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if item in {"*", "all", "ALL"}:
                return True
            if "-" in item:
                start, end = item.split("-", 1)
                selected.update(range(int(start), int(end) + 1))
            else:
                selected.add(int(item))
        return self.layer_id in selected

    def _use_separate_qkv_projection(self) -> bool:
        return (
            self._separate_qkv_projection_enabled()
            and self._layer_selected_for_separate_qkv_projection()
        )

    def _apply_split_rope(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        hw_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_h, k_h, q_w, k_w = hw_parts
        q_t, k_t = self.rotary_emb(indexes[0], q_t, k_t)
        q_h, k_h = self.rotary_emb_hw(indexes[1], q_h, k_h)
        q_w, k_w = self.rotary_emb_hw(indexes[2], q_w, k_w)

        q_t = q_t.view(-1, self.total_num_heads, self.t_dim)
        k_t = k_t.view(-1, self.total_num_kv_heads, self.t_dim)
        q_h = q_h.view(-1, self.total_num_heads, self.hw_dim)
        k_h = k_h.view(-1, self.total_num_kv_heads, self.hw_dim)
        q_w = q_w.view(-1, self.total_num_heads, self.hw_dim)
        k_w = k_w.view(-1, self.total_num_kv_heads, self.hw_dim)
        q = torch.cat([q_t, q_h, q_w], dim=-1).reshape(-1, self.q_size)
        k = torch.cat([k_t, k_h, k_w], dim=-1).reshape(-1, self.kv_size)
        return q, k

    def _native_qkv(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_t, k_t, hw_parts, v = self._project_qkv(
            hidden_states, use_mot_gen=use_mot_gen
        )
        q, k = self._apply_split_rope(q_t, k_t, hw_parts, indexes)
        return q, k, v

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = q.shape[0]
        k_len = k.shape[0]
        q = q.view(q_len, self.total_num_heads, self.head_dim)
        k = k.view(k_len, self.total_num_kv_heads, self.head_dim)
        v = v.view(k_len, self.total_num_kv_heads, self.head_dim)
        k = _repeat_kv(
            k,
            num_query_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
        )
        v = _repeat_kv(
            v,
            num_query_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
        )
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * self.scaling
        if allowed is not None:
            allowed = allowed.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(
                ~allowed.unsqueeze(0),
                torch.finfo(scores.dtype).min,
            )
        elif causal:
            if q_len != k_len:
                raise ValueError("causal eager attention requires q_len == k_len")
            mask = torch.ones(
                (q_len, k_len), device=scores.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", probs, v)
        return out.reshape(q_len, self.q_size)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        allowed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = q.shape[0]
        k_len = k.shape[0]
        q = q.view(q_len, self.total_num_heads, self.head_dim)
        k = k.view(k_len, self.total_num_kv_heads, self.head_dim)
        v = v.view(k_len, self.total_num_kv_heads, self.head_dim)
        # Match the official U1 SDPA fallback: materialize GQA K/V repeats
        # before calling SDPA instead of relying on enable_gqa=True.
        k = _repeat_kv(
            k,
            num_query_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
        )
        v = _repeat_kv(
            v,
            num_query_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
        )
        q_bhsd = q.unsqueeze(0).transpose(1, 2)
        k_bhsd = k.unsqueeze(0).transpose(1, 2)
        v_bhsd = v.unsqueeze(0).transpose(1, 2)
        attn_mask = None
        if allowed is not None:
            attn_mask = allowed.to(device=q.device, dtype=torch.bool).view(
                1,
                1,
                q_len,
                k_len,
            )
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal and attn_mask is None,
            scale=self.scaling,
        )
        return out.transpose(1, 2).reshape(q_len, self.q_size)

    def _official_eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        allowed: torch.Tensor | None = None,
        kv_already_repeated: bool = False,
    ) -> torch.Tensor:
        """Match the public U1 eager attention math used by HF prefill."""

        q_len = q.shape[0]
        k_len = k.shape[0]
        q = q.view(q_len, self.total_num_heads, self.head_dim)
        cache_heads = (
            self.total_num_heads
            if kv_already_repeated
            else self.total_num_kv_heads
        )
        k = k.view(k_len, cache_heads, self.head_dim)
        v = v.view(k_len, cache_heads, self.head_dim)
        if not kv_already_repeated:
            k = _repeat_kv(
                k,
                num_query_heads=self.total_num_heads,
                num_kv_heads=self.total_num_kv_heads,
            )
            v = _repeat_kv(
                v,
                num_query_heads=self.total_num_heads,
                num_kv_heads=self.total_num_kv_heads,
            )
        q_bhsd = q.transpose(0, 1).unsqueeze(0)
        k_bhsd = k.transpose(0, 1).unsqueeze(0)
        v_bhsd = v.transpose(0, 1).unsqueeze(0)
        attn_weights = torch.matmul(q_bhsd, k_bhsd.transpose(2, 3)) * self.scaling

        mask = None
        if allowed is not None:
            allowed = allowed.to(device=attn_weights.device, dtype=torch.bool)
            mask = torch.where(
                allowed.view(1, 1, q_len, k_len),
                torch.zeros((), device=attn_weights.device, dtype=attn_weights.dtype),
                torch.full(
                    (),
                    float("-inf"),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                ),
            )
        elif causal:
            if q_len != k_len:
                raise ValueError("causal official eager attention requires q_len == k_len")
            allowed_causal = torch.ones(
                (q_len, k_len),
                device=attn_weights.device,
                dtype=torch.bool,
            ).tril()
            mask = torch.where(
                allowed_causal.view(1, 1, q_len, k_len),
                torch.zeros((), device=attn_weights.device, dtype=attn_weights.dtype),
                torch.full(
                    (),
                    float("-inf"),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                ),
            )
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = torch.nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(q_bhsd.dtype)
        out = torch.matmul(attn_weights, v_bhsd)
        return out.transpose(1, 2).contiguous().reshape(q_len, self.q_size)

    def repeat_eager_kv_cache(self, states: torch.Tensor) -> torch.Tensor:
        seq_len = int(states.shape[0])
        states = states.view(
            seq_len,
            self.total_num_kv_heads,
            self.head_dim,
        )
        return _repeat_kv(
            states,
            num_query_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
        ).reshape(seq_len, self.q_size)

    def _apply_o_proj(
        self,
        attn_output: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        o_proj = self.o_proj_mot_gen if use_mot_gen else self.o_proj
        if self.standalone:
            return torch.nn.functional.linear(
                attn_output,
                o_proj.weight,
                o_proj.bias,
            )
        attn_output, _ = o_proj(attn_output)
        return attn_output

    def _forward_one_path(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch | None,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        q, k, v = self._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=use_mot_gen,
        )
        if forward_batch is None:
            attn_output = self._eager_attention(q, k, v, causal=not use_mot_gen)
        else:
            attn_output = self.attn(q, k, v, forward_batch)
        return self._apply_o_proj(attn_output, use_mot_gen=use_mot_gen)

    def _forward_mixed_path(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch | None,
        image_gen_indicators: torch.Tensor,
    ) -> torch.Tensor:
        und_q, und_k, und_v = self._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        gen_q, gen_k, gen_v = self._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=True,
        )
        token_mask = image_gen_indicators[:, None].to(
            device=hidden_states.device,
            dtype=torch.bool,
        )
        q = torch.where(token_mask, gen_q, und_q)
        k = torch.where(token_mask, gen_k, und_k)
        v = torch.where(token_mask, gen_v, und_v)
        if forward_batch is None:
            allowed = build_u1_hybrid_allowed_matrix(indexes[0], image_gen_indicators)
            attn_output = self._eager_attention(
                q,
                k,
                v,
                causal=False,
                allowed=allowed,
            )
        else:
            attn_output = self.attn(q, k, v, forward_batch)

        und = self._apply_o_proj(attn_output, use_mot_gen=False)
        gen = self._apply_o_proj(attn_output, use_mot_gen=True)
        return torch.where(token_mask, gen, und)

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch | None,
        image_gen_indicators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_gen_indicators is None or not bool(image_gen_indicators.any()):
            return self._forward_one_path(
                hidden_states, indexes, forward_batch, use_mot_gen=False
            )
        if bool(image_gen_indicators.all()):
            return self._forward_one_path(
                hidden_states, indexes, forward_batch, use_mot_gen=True
            )

        return self._forward_mixed_path(
            hidden_states,
            indexes,
            forward_batch,
            image_gen_indicators,
        )


class SenseNovaU1NativeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_id: int,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.self_attn = SenseNovaU1NativeAttention(
            config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            params_dtype=params_dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
            standalone=standalone,
        )
        self.mlp = SenseNovaU1NativeMLP(
            config,
            layer_id=layer_id,
            prefix=add_prefix("mlp", prefix),
            params_dtype=params_dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.mlp_mot_gen = SenseNovaU1NativeMLP(
            config,
            layer_id=layer_id,
            prefix=add_prefix("mlp_mot_gen", prefix),
            params_dtype=params_dtype,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _use_compiled_eager_add_rms(self) -> bool:
        enabled = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return False
        raw = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS_LAYERS",
            "",
        ).strip()
        if not raw:
            return True
        selected: set[int] = set()
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if item in {"*", "all", "ALL"}:
                return True
            if "-" in item:
                start, end = item.split("-", 1)
                selected.update(range(int(start), int(end) + 1))
            else:
                selected.add(int(item))
        return self.layer_id in selected

    def _eager_post_attention_norm(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_compiled_eager_add_rms():
            return _compiled_eager_add_rms(
                residual,
                update,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden = residual + update
        return hidden, _rms_norm(self.post_attention_layernorm, hidden)

    def _forward_one_path(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch | None,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        norm = self.input_layernorm_mot_gen if use_mot_gen else self.input_layernorm
        hidden_states = _rms_norm(norm, hidden_states)
        hidden_states = self.self_attn._forward_one_path(
            hidden_states,
            indexes,
            forward_batch,
            use_mot_gen=use_mot_gen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        norm = (
            self.post_attention_layernorm_mot_gen
            if use_mot_gen
            else self.post_attention_layernorm
        )
        mlp = self.mlp_mot_gen if use_mot_gen else self.mlp
        hidden_states = mlp(_rms_norm(norm, hidden_states))
        return mlp.apply_residual_add(residual, hidden_states)

    def eager_text_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        allowed: torch.Tensor | None = None,
        repeat_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HF-style full prefill for understanding/text tokens plus raw KV."""

        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k, v = self.self_attn._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        attn_output = self.self_attn._official_eager_attention(
            q,
            k,
            v,
            causal=allowed is None,
            allowed=allowed,
        )
        attention_update = self.self_attn._apply_o_proj(
            attn_output,
            use_mot_gen=False,
        )

        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        hidden_states = self.mlp.apply_residual_add(residual, hidden_states)
        if repeat_kv_cache:
            k = self.self_attn.repeat_eager_kv_cache(k)
            v = self.self_attn.repeat_eager_kv_cache(v)
        return hidden_states, k.detach(), v.detach()

    def eager_text_decode_with_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HF-style single-token cached text decode plus updated raw KV."""

        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k_cur, v_cur = self.self_attn._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        k = torch.cat([past_k, k_cur], dim=0)
        v = torch.cat([past_v, v_cur], dim=0)
        attn_output = self.self_attn._official_eager_attention(
            q,
            k,
            v,
            causal=False,
        )
        attention_update = self.self_attn._apply_o_proj(
            attn_output,
            use_mot_gen=False,
        )

        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        hidden_states = self.mlp.apply_residual_add(residual, hidden_states)
        return hidden_states, k.detach(), v.detach()

    def eager_text_decode_with_repeated_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact eager decode with history already expanded to query heads."""

        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k_cur, v_cur = self.self_attn._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        k_cur = self.self_attn.repeat_eager_kv_cache(k_cur)
        v_cur = self.self_attn.repeat_eager_kv_cache(v_cur)
        k = torch.cat([past_k, k_cur], dim=0)
        v = torch.cat([past_v, v_cur], dim=0)
        attn_output = self.self_attn._official_eager_attention(
            q,
            k,
            v,
            causal=False,
            kv_already_repeated=True,
        )
        attention_update = self.self_attn._apply_o_proj(
            attn_output,
            use_mot_gen=False,
        )

        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        hidden_states = self.mlp.apply_residual_add(residual, hidden_states)
        return hidden_states, k.detach(), v.detach()

    def eager_text_decode_with_static_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        *,
        cache_position: int,
        repeat_kv_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact eager decode writing the current KV into fixed buffers."""

        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k_cur, v_cur = self.self_attn._native_qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        if repeat_kv_cache:
            k_cur = self.self_attn.repeat_eager_kv_cache(k_cur)
            v_cur = self.self_attn.repeat_eager_kv_cache(v_cur)
        cache_k[cache_position : cache_position + 1].copy_(k_cur)
        cache_v[cache_position : cache_position + 1].copy_(v_cur)
        k = cache_k[: cache_position + 1]
        v = cache_v[: cache_position + 1]
        attn_output = self.self_attn._official_eager_attention(
            q,
            k,
            v,
            causal=False,
            kv_already_repeated=repeat_kv_cache,
        )
        attention_update = self.self_attn._apply_o_proj(
            attn_output,
            use_mot_gen=False,
        )

        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        hidden_states = self.mlp.apply_residual_add(residual, hidden_states)
        return hidden_states, cache_k, cache_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch | None,
        image_gen_indicators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_gen_indicators is None or not bool(image_gen_indicators.any()):
            return self._forward_one_path(
                hidden_states, indexes, forward_batch, use_mot_gen=False
            )
        if bool(image_gen_indicators.all()):
            return self._forward_one_path(
                hidden_states, indexes, forward_batch, use_mot_gen=True
            )
        token_mask = image_gen_indicators[:, None].to(
            device=hidden_states.device,
            dtype=torch.bool,
        )

        residual = hidden_states
        und_normed = _rms_norm(self.input_layernorm, hidden_states)
        gen_normed = _rms_norm(self.input_layernorm_mot_gen, hidden_states)
        mixed_normed = torch.where(token_mask, gen_normed, und_normed)
        hidden_states = self.self_attn(
            mixed_normed,
            indexes,
            forward_batch,
            image_gen_indicators=image_gen_indicators,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        und_normed = _rms_norm(self.post_attention_layernorm, hidden_states)
        gen_normed = _rms_norm(self.post_attention_layernorm_mot_gen, hidden_states)
        mixed_normed = torch.where(token_mask, gen_normed, und_normed)
        und = self.mlp(mixed_normed)
        gen = self.mlp_mot_gen(mixed_normed)
        return self.mlp.apply_residual_add(
            residual,
            torch.where(token_mask, gen, und),
        )


class SenseNovaU1NativeTextModel(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
        standalone: bool = False,
    ) -> None:
        super().__init__()
        self.config = _to_namespace(config)
        self.standalone = standalone
        tp_rank = 0 if standalone else None
        tp_size = 1 if standalone else None
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=params_dtype,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not standalone,
        )
        self.layers = nn.ModuleList(
            [
                SenseNovaU1NativeDecoderLayer(
                    self.config,
                    layer_id=idx,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                    params_dtype=params_dtype,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    standalone=standalone,
                )
                for idx in range(int(self.config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.norm_mot_gen = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.force_mot_gen_for_prefill_graph_capture = False

    @staticmethod
    def _prepare_indexes(
        positions: torch.Tensor,
        indexes: torch.Tensor | None,
    ) -> torch.Tensor:
        if indexes is not None:
            return indexes.to(device=positions.device, dtype=torch.long)
        if positions.ndim == 2 and positions.shape[0] == 3:
            return positions.to(dtype=torch.long)
        flat_positions = positions.flatten().to(dtype=torch.long)
        zeros = torch.zeros_like(flat_positions)
        return torch.stack([flat_positions, zeros, zeros])

    @staticmethod
    def _use_direct_eager_embedding() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_DIRECT_EMBEDDING",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    def _eager_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.standalone or self._use_direct_eager_embedding():
            return torch.nn.functional.embedding(
                input_ids,
                self.embed_tokens.weight[: self.config.vocab_size],
            )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch | None,
        input_embeds: torch.Tensor | None = None,
        *,
        image_gen_indicators: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            if self.standalone:
                hidden_states = torch.nn.functional.embedding(
                    input_ids,
                    self.embed_tokens.weight[: self.config.vocab_size],
                )
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indexes = self._prepare_indexes(positions, indexes)
        if (
            self.force_mot_gen_for_prefill_graph_capture
            and image_gen_indicators is None
            and forward_batch is not None
        ):
            for layer in self.layers:
                hidden_states = layer._forward_one_path(
                    hidden_states,
                    indexes,
                    forward_batch,
                    use_mot_gen=True,
                )
            return _rms_norm(self.norm_mot_gen, hidden_states)
        if image_gen_indicators is not None:
            image_gen_indicators = image_gen_indicators.flatten().to(
                device=hidden_states.device, dtype=torch.bool
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                indexes,
                forward_batch,
                image_gen_indicators=image_gen_indicators,
            )

        if image_gen_indicators is None or not bool(image_gen_indicators.any()):
            return _rms_norm(self.norm, hidden_states)
        if bool(image_gen_indicators.all()):
            return _rms_norm(self.norm_mot_gen, hidden_states)
        und = _rms_norm(self.norm, hidden_states)
        gen = _rms_norm(self.norm_mot_gen, hidden_states)
        return torch.where(image_gen_indicators[:, None], gen, und)

    def eager_text_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        image_token_tag: torch.Tensor | None = None,
        repeat_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run native HF-style eager prefill and return per-layer KV cache."""

        if input_embeds is None:
            hidden_states = self._eager_embed(input_ids)
        else:
            hidden_states = input_embeds
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indexes = self._prepare_indexes(positions, indexes)
        allowed = None
        if image_token_tag is not None and bool(image_token_tag.reshape(-1).any().item()):
            allowed = build_u1_hybrid_allowed_matrix(
                indexes[0],
                image_token_tag.reshape(-1).to(
                    device=indexes.device,
                    dtype=torch.bool,
                ),
            )

        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden_states, k, v = layer.eager_text_prefill_with_cache(
                hidden_states,
                indexes,
                allowed=allowed,
                repeat_kv_cache=repeat_kv_cache,
            )
            caches.append((k, v))
        hidden_states = _rms_norm(self.norm, hidden_states)
        return hidden_states, caches

    def eager_text_decode_with_cache(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        indexes: torch.Tensor | None = None,
        capture_layer_outputs: list[torch.Tensor] | None = None,
        repeat_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one HF-style cached text decode step and update KV cache."""

        hidden_states = self._eager_embed(input_ids)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indexes = self._prepare_indexes(positions, indexes)
        if len(caches) != len(self.layers):
            raise ValueError(
                "eager text decode cache layer count mismatch: "
                f"caches={len(caches)} layers={len(self.layers)}"
            )

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (past_k, past_v) in zip(self.layers, caches):
            if repeat_kv_cache:
                hidden_states, k, v = (
                    layer.eager_text_decode_with_repeated_cache(
                        hidden_states,
                        indexes,
                        past_k,
                        past_v,
                    )
                )
            else:
                hidden_states, k, v = layer.eager_text_decode_with_cache(
                    hidden_states,
                    indexes,
                    past_k,
                    past_v,
                )
            if capture_layer_outputs is not None:
                capture_layer_outputs.append(hidden_states.detach().cpu())
            new_caches.append((k, v))
        hidden_states = _rms_norm(self.norm, hidden_states)
        if capture_layer_outputs is not None:
            capture_layer_outputs.append(hidden_states.detach().cpu())
        return hidden_states, new_caches

    def eager_text_decode_with_static_cache(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        cache_position: int,
        indexes: torch.Tensor | None = None,
        repeat_kv_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self._eager_embed(input_ids)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        indexes = self._prepare_indexes(positions, indexes)
        if len(caches) != len(self.layers):
            raise ValueError(
                "eager static cache layer count mismatch: "
                f"caches={len(caches)} layers={len(self.layers)}"
            )

        for layer, (cache_k, cache_v) in zip(self.layers, caches):
            hidden_states, _, _ = layer.eager_text_decode_with_static_cache(
                hidden_states,
                indexes,
                cache_k,
                cache_v,
                cache_position=cache_position,
                repeat_kv_cache=repeat_kv_cache,
            )
        return _rms_norm(self.norm, hidden_states), caches


class SenseNovaU1NativeForCausalLM(nn.Module):
    """SGLang-native U1 language tower with MoT dual-tower weights."""

    def __init__(
        self,
        config: Any,
        quant_config: Any | None = None,
        prefix: str = "",
        params_dtype: torch.dtype | str | None = None,
        standalone: bool = False,
    ) -> None:
        del quant_config
        super().__init__()
        if _get_attr(config, "llm_config") is not None:
            config = _get_attr(config, "llm_config")
        self.config = _to_namespace(config)
        dtype = resolve_dtype(params_dtype or _get_attr(self.config, "torch_dtype", None))
        self.standalone = standalone
        self.is_mrope_enabled = True
        self.model = SenseNovaU1NativeTextModel(
            self.config,
            params_dtype=dtype,
            prefix=add_prefix("model", prefix),
            standalone=standalone,
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            bias=False,
            params_dtype=dtype,
            prefix=add_prefix("lm_head", prefix),
            enable_tp=not standalone,
        )
        self.logits_processor = None if standalone else LogitsProcessor(self.config)
        self._expected_language_keys = expected_language_weight_keys(self.config)
        self._last_load_report: SenseNovaU1LoadReport | None = None
        self._last_forward_batch_prepare: dict[str, Any] | None = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @property
    def last_load_report(self) -> SenseNovaU1LoadReport | None:
        return self._last_load_report

    @property
    def last_forward_batch_prepare(self) -> dict[str, Any] | None:
        return self._last_forward_batch_prepare

    @staticmethod
    def _lens_to_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [int(v) for v in value.detach().cpu().tolist()]
        return [int(v) for v in value]

    @staticmethod
    def _slice_mm_tensor(
        value: Any,
        *,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value)
        if tensor.ndim == 2 and tensor.shape[0] == 3:
            tensor = tensor[:, prefix_len : prefix_len + extend_len]
        else:
            tensor = tensor.reshape(-1)[prefix_len : prefix_len + extend_len]
        return tensor.to(device=device, dtype=dtype, non_blocking=True)

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Attach U1 per-token metadata before SGLang attention planning."""

        if not forward_batch.forward_mode.is_extend():
            self._last_forward_batch_prepare = {
                "forward_mode": str(forward_batch.forward_mode),
                "metadata_attached": False,
                "reason": "non_extend",
            }
            return

        indexes = getattr(forward_batch, "mrope_positions", None)
        if indexes is None:
            indexes = SenseNovaU1NativeTextModel._prepare_indexes(
                forward_batch.positions,
                None,
            )
        indexes = indexes.to(device=forward_batch.input_ids.device, dtype=torch.long)

        extend_lens = self._lens_to_list(forward_batch.extend_seq_lens_cpu)
        prefix_lens = self._lens_to_list(forward_batch.extend_prefix_lens_cpu)
        if not extend_lens and forward_batch.extend_seq_lens is not None:
            extend_lens = self._lens_to_list(forward_batch.extend_seq_lens)
        if not prefix_lens:
            prefix_lens = [0] * len(extend_lens)

        mm_inputs = getattr(forward_batch, "mm_inputs", None) or []
        image_tags: list[torch.Tensor] = []
        image_gen_flags: list[torch.Tensor] = []
        for idx, extend_len in enumerate(extend_lens):
            prefix_len = prefix_lens[idx] if idx < len(prefix_lens) else 0
            mm_input = mm_inputs[idx] if idx < len(mm_inputs) else None
            tag = None
            gen = None
            if mm_input is not None:
                tag = self._slice_mm_tensor(
                    getattr(mm_input, "u1_image_token_tag", None),
                    prefix_len=prefix_len,
                    extend_len=extend_len,
                    device=indexes.device,
                    dtype=torch.bool,
                )
                gen = self._slice_mm_tensor(
                    getattr(mm_input, "u1_image_gen_indicators", None),
                    prefix_len=prefix_len,
                    extend_len=extend_len,
                    device=indexes.device,
                    dtype=torch.bool,
                )
            if tag is None:
                start = sum(extend_lens[:idx])
                stop = start + extend_len
                tag = build_image_token_tag_from_t_indexes(indexes[:, start:stop])
            if gen is None:
                gen = torch.zeros(extend_len, device=indexes.device, dtype=torch.bool)
            image_tags.append(tag.reshape(-1))
            image_gen_flags.append(gen.reshape(-1))

        if image_tags:
            image_token_tag = torch.cat(image_tags).to(device=indexes.device)
        else:
            image_token_tag = torch.zeros(
                indexes.shape[1], device=indexes.device, dtype=torch.bool
            )
        if image_gen_flags:
            image_gen_indicators = torch.cat(image_gen_flags).to(device=indexes.device)
        else:
            image_gen_indicators = torch.zeros_like(image_token_tag)

        custom_mask, mask_indptr = build_u1_hybrid_backend_mask(
            indexes,
            image_token_tag,
            extend_lens,
            prefix_lens,
        )
        forward_batch.cross_attention_custom_mask = custom_mask
        states = dict(forward_batch.model_specific_states or {})
        states["sensenova_u1"] = {
            "indexes": indexes,
            "image_token_tag": image_token_tag,
            "image_gen_indicators": image_gen_indicators,
            "custom_mask_indptr": mask_indptr,
        }
        forward_batch.model_specific_states = states

        self._last_forward_batch_prepare = {
            "forward_mode": str(forward_batch.forward_mode),
            "metadata_attached": True,
            "num_tokens": int(indexes.shape[1]),
            "extend_seq_lens": extend_lens,
            "extend_prefix_lens": prefix_lens,
            "image_token_count": int(image_token_tag.sum().item()),
            "image_gen_count": int(image_gen_indicators.sum().item()),
            "custom_mask_numel": 0 if custom_mask is None else int(custom_mask.numel()),
            "custom_mask_dtype": None if custom_mask is None else str(custom_mask.dtype),
            "custom_mask_device": None if custom_mask is None else str(custom_mask.device),
            "mask_indptr": [int(x) for x in mask_indptr.detach().cpu().tolist()],
            "image_spans": [
                span.as_dict()
                for span in build_image_spans(indexes[0], image_token_tag)
            ],
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch | None,
        *,
        input_embeds: torch.Tensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        omni_prefill_rids: Any | None = None,
    ) -> LogitsProcessorOutput:
        del omni_prefill_rids
        states = (
            None
            if forward_batch is None
            else (forward_batch.model_specific_states or {}).get("sensenova_u1")
        )
        if states is not None:
            if indexes is None:
                indexes = states.get("indexes")
            if image_gen_indicators is None:
                image_gen_indicators = states.get("image_gen_indicators")
        if indexes is None and forward_batch is not None:
            mrope_positions = getattr(forward_batch, "mrope_positions", None)
            if mrope_positions is not None:
                indexes = mrope_positions
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            image_gen_indicators=image_gen_indicators,
            indexes=indexes,
        )
        if forward_batch is not None:
            if self.logits_processor is None:
                raise RuntimeError(
                    "SenseNova U1 standalone native model cannot process a "
                    "ForwardBatch; instantiate under SGLang runtime instead."
                )
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )

        last_hidden = hidden_states[-1:]
        logits = torch.matmul(
            last_hidden.to(self.lm_head.weight.dtype),
            self.lm_head.weight.T,
        )
        logits = logits[..., : self.config.vocab_size].to(last_hidden.dtype)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def eager_text_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        last_hidden = hidden_states[-1:].to(self.lm_head.weight.dtype)
        if os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_LM_HEAD_LINEAR",
            "",
        ).lower() in {"1", "true", "yes", "on"}:
            logits = torch.nn.functional.linear(
                last_hidden,
                self.lm_head.weight,
            )
        else:
            logits = torch.matmul(
                last_hidden,
                self.lm_head.weight.T,
            )
        return logits[..., : self.config.vocab_size].to(hidden_states.dtype)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> SenseNovaU1LoadReport:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        unexpected: list[str] = []
        ignored: dict[str, int] = {}
        errors: list[str] = []

        for raw_name, loaded_weight in weights:
            if raw_name.startswith("language_model."):
                name = raw_name[len("language_model.") :]
            elif raw_name in self._expected_language_keys:
                name = raw_name
            else:
                top = raw_name.split(".", 1)[0]
                ignored[top] = ignored.get(top, 0) + 1
                continue
            if name not in self._expected_language_keys:
                unexpected.append(name)
                continue
            target = _direct_or_stacked_target(name)
            if target is None:
                unexpected.append(name)
                continue
            target_name, shard_id = target
            if target_name not in params:
                errors.append(f"{name} -> {target_name}: target parameter not found")
                continue
            try:
                _load_parameter(params, target_name, loaded_weight, shard_id)
            except Exception as exc:  # pragma: no cover - recorded in evidence
                errors.append(f"{name} -> {target_name}: {type(exc).__name__}: {exc}")
                continue
            loaded.add(name)

        report = SenseNovaU1LoadReport(
            loaded_language_keys=sorted(loaded),
            missing_language_keys=sorted(self._expected_language_keys - loaded),
            unexpected_language_keys=sorted(unexpected),
            ignored_non_language_keys=ignored,
            errors=errors,
        )
        self._last_load_report = report
        return report


class SenseNovaU1NativeLoadExecutor:
    """Minimal native executor used for import-guard and load smoke checks."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        dtype: str | torch.dtype = "bfloat16",
        load_weights: bool = False,
        standalone: bool = True,
    ) -> None:
        assert_no_hf_modeling_imported(context="before native U1 executor init")
        self.model_path = model_path
        self.config = load_u1_llm_config(model_path)
        self.model = SenseNovaU1NativeForCausalLM(
            self.config,
            params_dtype=dtype,
            standalone=standalone,
        )
        if load_weights:
            report = self.model.load_weights(iter_language_safetensors(model_path))
            if not report.ok:
                raise RuntimeError(f"native U1 load failed: {report.to_dict()}")
        self.model.to(device=device, dtype=resolve_dtype(dtype))
        self.model.eval()
        assert_no_hf_modeling_imported(context="after native U1 executor init")

    def complete_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        raise NotImplementedError(
            "SenseNova U1 native serving scheduler is not complete yet; this "
            "executor exists only for M6 native model-load/import-guard smoke."
        )


__all__ = [
    "HF_MODELING_MODULE_PREFIXES",
    "SenseNovaU1LoadReport",
    "SenseNovaU1NativeForCausalLM",
    "SenseNovaU1NativeLoadExecutor",
    "assert_no_hf_modeling_imported",
    "block_hf_modeling_imports",
    "expected_language_weight_keys",
    "iter_language_safetensors",
    "load_u1_llm_config",
    "scan_checkpoint_key_summary",
]
