# SPDX-License-Identifier: Apache-2.0
"""Bounded native SGLang serving probe for SenseNova U1."""

from __future__ import annotations

import hashlib
import os
import time
from array import array
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.sensenova_u1.native_vision import (
    SenseNovaU1NativeVisionModel,
)
from sglang_omni.models.sensenova_u1.sglang_model import (
    assert_no_hf_modeling_imported,
    block_hf_modeling_imports,
    load_u1_llm_config,
)
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)

_CACHE_EXTRA_KEY_NOT_SET = object()


@dataclass(slots=True)
class NativeServingResult:
    request_id: str
    next_token_id: int
    logits_shape: list[int]
    logits_dtype: str
    logits_device: str
    logits_all_finite: bool
    forward_elapsed_s: float
    backend_name: str
    prepare_metadata: dict[str, Any] | None
    forward_batch_log: dict[str, Any]
    input_embeds_used: bool = False
    input_embeds_shape: list[int] | None = None
    batch_index: int = 0
    batch_size: int = 1
    cache_inserted: bool = False
    cache_extra_key: str | None = None
    next_token_logits: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "request_id": self.request_id,
            "next_token_id": self.next_token_id,
            "logits_shape": self.logits_shape,
            "logits_dtype": self.logits_dtype,
            "logits_device": self.logits_device,
            "logits_all_finite": self.logits_all_finite,
            "forward_elapsed_s": self.forward_elapsed_s,
            "backend_name": self.backend_name,
            "prepare_metadata": self.prepare_metadata,
            "forward_batch_log": self.forward_batch_log,
            "input_embeds_used": self.input_embeds_used,
            "input_embeds_shape": self.input_embeds_shape,
            "batch_index": self.batch_index,
            "batch_size": self.batch_size,
            "cache_inserted": self.cache_inserted,
            "cache_extra_key": self.cache_extra_key,
        }
        if self.next_token_logits is not None:
            logits = self.next_token_logits.float().reshape(-1)
            top_values, top_indices = torch.topk(logits, k=min(5, logits.numel()))
            data.update(
                {
                    "logits_min": float(logits.min().item()),
                    "logits_max": float(logits.max().item()),
                    "logits_l2": float(torch.linalg.vector_norm(logits).item()),
                    "logits_top5_token_ids": [
                        int(x) for x in top_indices.detach().cpu().tolist()
                    ],
                    "logits_top5_values": [
                        float(x) for x in top_values.detach().cpu().tolist()
                    ],
                }
            )
        return data


@dataclass(slots=True)
class NativeHiddenPrefillResult:
    request_id: str
    hidden_states: torch.Tensor
    forward_elapsed_s: float
    backend_name: str
    prepare_metadata: dict[str, Any] | None
    forward_batch_log: dict[str, Any]
    input_embeds_used: bool
    input_embeds_shape: list[int] | None
    cache_inserted: bool
    cache_extra_key: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "hidden_states_shape": list(self.hidden_states.shape),
            "hidden_states_dtype": str(self.hidden_states.dtype),
            "hidden_states_device": str(self.hidden_states.device),
            "hidden_states_all_finite": bool(
                torch.isfinite(self.hidden_states).all().item()
            ),
            "forward_elapsed_s": self.forward_elapsed_s,
            "backend_name": self.backend_name,
            "prepare_metadata": self.prepare_metadata,
            "forward_batch_log": self.forward_batch_log,
            "input_embeds_used": self.input_embeds_used,
            "input_embeds_shape": self.input_embeds_shape,
            "cache_inserted": bool(self.cache_inserted),
            "cache_extra_key": self.cache_extra_key,
        }


@dataclass(slots=True)
class NativeDecodeBenchmarkResult:
    request_count: int
    decode_steps: int
    prompt_tokens: int
    generated_tokens: int
    prefill_elapsed_s: float
    decode_elapsed_s: float
    total_elapsed_s: float
    generated_token_ids: list[list[int]]
    per_step_elapsed_s: list[float]
    prefill_forward_batch_log: dict[str, Any]
    decode_forward_batch_logs: list[dict[str, Any]]
    release_log: dict[str, Any] | None
    logits_all_finite: bool

    def to_dict(self) -> dict[str, Any]:
        decode_generated_tokens = self.request_count * max(self.decode_steps - 1, 0)
        total_tps = (
            self.generated_tokens / self.total_elapsed_s
            if self.total_elapsed_s > 0
            else float("inf")
        )
        decode_tps = (
            decode_generated_tokens / self.decode_elapsed_s
            if self.decode_elapsed_s > 0
            else float("inf")
        )
        return {
            "request_count": self.request_count,
            "decode_steps": self.decode_steps,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "decode_generated_tokens": decode_generated_tokens,
            "prefill_elapsed_s": self.prefill_elapsed_s,
            "decode_elapsed_s": self.decode_elapsed_s,
            "total_elapsed_s": self.total_elapsed_s,
            "generated_tokens_per_s_total": total_tps,
            "generated_tokens_per_s_decode_only": decode_tps,
            "generated_token_ids": self.generated_token_ids,
            "per_step_elapsed_s": self.per_step_elapsed_s,
            "prefill_forward_batch_log": self.prefill_forward_batch_log,
            "decode_forward_batch_logs": self.decode_forward_batch_logs,
            "release_log": self.release_log,
            "logits_all_finite": self.logits_all_finite,
        }


@dataclass(slots=True)
class NativeEagerTextDecodeResult:
    decode_steps: int
    prompt_tokens: int
    generated_token_ids: list[int]
    prefill_elapsed_s: float
    decode_elapsed_s: float
    total_elapsed_s: float
    per_step_elapsed_s: list[float]
    step_logs: list[dict[str, Any]]
    logits_all_finite: bool
    next_token_logits: torch.Tensor | None = None
    captured_layer_hidden_states: list[torch.Tensor] | None = None
    full_loop_cuda_graph_used: bool = False
    full_loop_cuda_graph_created: bool = False
    full_loop_cuda_graph_key: str | None = None
    eager_prefix_cache_hit: bool = False
    eager_prefix_cache_key: str | None = None
    full_loop_initial_cache_copy_skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        suppress_hits = [
            int(log["suppressed_argmax_token_id"])
            for log in self.step_logs
            if log.get("suppressed_argmax_token_id") is not None
        ]
        return {
            "decode_steps": self.decode_steps,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": len(self.generated_token_ids),
            "generated_token_ids": self.generated_token_ids,
            "prefill_elapsed_s": self.prefill_elapsed_s,
            "decode_elapsed_s": self.decode_elapsed_s,
            "total_elapsed_s": self.total_elapsed_s,
            "generated_tokens_per_s_total": (
                len(self.generated_token_ids) / self.total_elapsed_s
                if self.total_elapsed_s > 0
                else float("inf")
            ),
            "generated_tokens_per_s_decode_only": (
                max(len(self.generated_token_ids) - 1, 0) / self.decode_elapsed_s
                if self.decode_elapsed_s > 0
                else float("inf")
            ),
            "per_step_elapsed_s": self.per_step_elapsed_s,
            "step_logs": self.step_logs,
            "logits_all_finite": self.logits_all_finite,
            "suppressed_token_hits": len(suppress_hits),
            "suppressed_argmax_token_ids": suppress_hits,
            "full_loop_cuda_graph_used": self.full_loop_cuda_graph_used,
            "full_loop_cuda_graph_created": self.full_loop_cuda_graph_created,
            "full_loop_cuda_graph_key": self.full_loop_cuda_graph_key,
            "eager_prefix_cache_hit": self.eager_prefix_cache_hit,
            "eager_prefix_cache_key": self.eager_prefix_cache_key,
            "full_loop_initial_cache_copy_skipped": (
                self.full_loop_initial_cache_copy_skipped
            ),
            "captured_layer_hidden_shapes": (
                None
                if self.captured_layer_hidden_states is None
                else [list(t.shape) for t in self.captured_layer_hidden_states]
            ),
        }


@dataclass(slots=True)
class _NativeEagerDecodeGraph:
    graphs: list[torch.cuda.CUDAGraph]
    graph_pool: Any
    initial_caches: list[tuple[torch.Tensor, torch.Tensor]]
    input_token: torch.Tensor
    decode_indexes: list[torch.Tensor]
    generated_tokens: list[torch.Tensor]
    last_logits: torch.Tensor
    final_caches: list[tuple[torch.Tensor, torch.Tensor]]
    cache_history: list[list[tuple[torch.Tensor, torch.Tensor]]]
    source_prefix_cache_key: str | None
    prefix_len: int
    verified_finite: bool


def _tensor_from_payload(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _update_hash_with_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(str(tuple(cpu.shape)).encode("utf-8"))
    hasher.update(str(cpu.dtype).encode("utf-8"))
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    elif cpu.dtype == torch.bool:
        cpu = cpu.to(torch.uint8)
    hasher.update(cpu.numpy().tobytes())


class SenseNovaU1NativeServingExecutor:
    """Small native SGLang executor for M6 serving-attention validation.

    The executor intentionally accepts already-expanded U1 prompt token ids and
    image metadata. It verifies the SGLang-native language/MoT tower,
    ForwardBatch metadata transport, and attention backend mask path without
    reusing the older HF NEOChatModel fallback.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_backend: str = "triton",
        mem_fraction_static: float = 0.65,
        max_total_tokens: int = 4096,
        max_running_requests: int = 2,
        enable_radix_cache: bool = True,
        disable_cuda_graph: bool = True,
        cuda_graph_bs: list[int] | None = None,
        enable_deterministic_inference: bool = False,
        prefill_cuda_graph_backend: str = "disabled",
        prefill_cuda_graph_bs: list[int] | None = None,
    ) -> None:
        assert_no_hf_modeling_imported(context="before native serving executor")
        from sglang_omni.models.sensenova_u1.hf_config import (
            register_sensenova_u1_native_config_parser,
        )

        register_sensenova_u1_native_config_parser()
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        self.model_path = model_path
        self.config = load_u1_llm_config(model_path)
        self.vocab_size = int(self.config.vocab_size)
        self.max_running_requests = max(int(max_running_requests), 1)
        self.enable_radix_cache = bool(enable_radix_cache)
        if cuda_graph_bs is None:
            cuda_graph_bs = [self.max_running_requests]
        cuda_graph_bs = sorted({int(value) for value in cuda_graph_bs})
        prefill_cuda_graph_backend = str(prefill_cuda_graph_backend).lower()
        prefill_cuda_graph_enabled = prefill_cuda_graph_backend != "disabled"
        if prefill_cuda_graph_bs is None:
            prefill_cuda_graph_bs = [64]
        prefill_cuda_graph_bs = sorted(
            {int(value) for value in prefill_cuda_graph_bs}
        )
        cuda_graph_overrides = (
            {
                "cuda_graph_bs": cuda_graph_bs,
                "cuda_graph_max_bs": max(cuda_graph_bs),
            }
            if not disable_cuda_graph
            else {}
        )
        if prefill_cuda_graph_enabled:
            cuda_graph_overrides.update(
                {
                    "cuda_graph_backend_prefill": prefill_cuda_graph_backend,
                    "cuda_graph_bs_prefill": prefill_cuda_graph_bs,
                    "cuda_graph_max_bs_prefill": max(prefill_cuda_graph_bs),
                    "enable_return_hidden_states": True,
                }
            )
            if disable_cuda_graph:
                cuda_graph_overrides["cuda_graph_backend_decode"] = "disabled"
        server_args = build_sglang_server_args(
            model_path,
            context_length=min(int(self.config.max_position_embeddings), max_total_tokens),
            max_prefill_tokens=max_total_tokens,
            max_running_requests=self.max_running_requests,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            model_config_parser="sensenova_u1_native",
            attention_backend=attention_backend,
            sampling_backend="pytorch",
            disable_cuda_graph=(
                disable_cuda_graph and not prefill_cuda_graph_enabled
            ),
            disable_overlap_schedule=True,
            enable_deterministic_inference=enable_deterministic_inference,
            disable_radix_cache=not self.enable_radix_cache,
            chunked_prefill_size=None,
            max_total_tokens=max_total_tokens,
            **cuda_graph_overrides,
        )
        with block_hf_modeling_imports():
            from sglang_omni.scheduling import bootstrap as scheduling_bootstrap

            (
                self.model_worker,
                self.tree_cache,
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
                self.prefill_manager,
                self.decode_manager,
                self.model_config,
            ) = scheduling_bootstrap.create_sglang_infrastructure(
                server_args,
                gpu_id,
                model_arch_override="SenseNovaU1NativeForCausalLM",
                total_gpu_memory_fraction=mem_fraction_static,
                defer_cuda_graph_capture=prefill_cuda_graph_enabled,
                enable_prefill_input_embeds=True,
            )
            from sglang_omni.models.sensenova_u1.attention_backend import (
                install_sensenova_u1_triton_attention_adapter,
            )

            install_sensenova_u1_triton_attention_adapter(
                self.model_worker.model_runner
            )
            if prefill_cuda_graph_enabled:
                text_model = self.model_worker.model_runner.model.model
                text_model.force_mot_gen_for_prefill_graph_capture = True
                try:
                    scheduling_bootstrap.init_sglang_cuda_graphs(
                        self.model_worker
                    )
                finally:
                    text_model.force_mot_gen_for_prefill_graph_capture = False
        self.server_args = server_args
        self.prefill_cuda_graph_enabled = prefill_cuda_graph_enabled
        self._eager_text_decode_graphs: dict[
            tuple[int, int, int, str],
            _NativeEagerDecodeGraph,
        ] = {}
        self._eager_text_prefill_cache: dict[
            str,
            tuple[
                torch.Tensor,
                list[tuple[torch.Tensor, torch.Tensor]],
                bool,
            ],
        ] = {}
        dtype_obj = next(self.model_worker.model_runner.model.parameters()).dtype
        device_obj = torch.device(self.model_worker.device)
        self.vision_model = SenseNovaU1NativeVisionModel.from_model_path(
            model_path,
            params_dtype=dtype_obj,
        ).to(device=device_obj, dtype=dtype_obj)
        self.vision_model.eval()
        self.vision_load_report = self.vision_model.load_weights(model_path)
        if not self.vision_load_report.ok:
            raise RuntimeError(
                "native U1 vision load failed: "
                f"{self.vision_load_report.to_dict()}"
            )
        assert_no_hf_modeling_imported(context="after native serving executor")

    def _make_req(
        self,
        *,
        request_id: str,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        image_gen_indicators: torch.Tensor | None,
        max_new_tokens: int = 1,
        cache_extra_key: str | None = None,
        has_projected_input_embeds: bool = False,
    ) -> Req:
        sampling_params = SamplingParams(
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        sampling_params.normalize(None)
        sampling_params.verify(self.vocab_size)
        input_list = [int(x) for x in input_ids.detach().cpu().tolist()]
        req = Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=array("q", input_list),
            sampling_params=sampling_params,
            vocab_size=self.vocab_size,
            extra_key=cache_extra_key,
        )
        req.tokenizer = None
        req._codec_suppress_tokens = None
        req._input_embeds_are_projected = bool(has_projected_input_embeds)
        req.omni_model_inputs = None
        mm_inputs = MultimodalInputs(mm_items=[])
        mm_inputs.mrope_positions = indexes.detach().cpu().to(dtype=torch.long)
        mm_inputs.mrope_position_delta = torch.zeros(1, dtype=torch.long)
        mm_inputs.u1_image_token_tag = image_token_tag.detach().cpu().to(dtype=torch.bool)
        if image_gen_indicators is not None:
            mm_inputs.u1_image_gen_indicators = (
                image_gen_indicators.detach().cpu().to(dtype=torch.bool)
            )
        req.multimodal_inputs = mm_inputs
        return req

    @staticmethod
    def _cache_extra_key(
        *,
        image_token_tag: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_hw: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
    ) -> str | None:
        if not bool(image_token_tag.reshape(-1).bool().any().item()):
            return None
        hasher = hashlib.sha256()
        hasher.update(b"sensenova-u1-native-image-cache-v1")
        _update_hash_with_tensor(hasher, image_token_tag.reshape(-1).bool())
        if grid_hw is not None:
            _update_hash_with_tensor(hasher, grid_hw.to(dtype=torch.long))
        if pixel_values is not None:
            _update_hash_with_tensor(hasher, pixel_values)
        elif input_embeds is not None:
            tag = image_token_tag.reshape(-1).to(dtype=torch.bool, device=input_embeds.device)
            image_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])[tag]
            _update_hash_with_tensor(hasher, image_embeds)
        return "u1-image:" + hasher.hexdigest()[:32]

    def compose_input_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        image_token_tag: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_hw: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build text embeddings with native vision features pasted into image spans."""

        if pixel_values is None and grid_hw is None:
            return None
        if pixel_values is None or grid_hw is None:
            raise ValueError("pixel_values and grid_hw must be provided together.")
        assert_no_hf_modeling_imported(context="before native input embedding compose")
        model = self.model_worker.model_runner.model
        device = torch.device(self.model_worker.device)
        dtype = next(model.parameters()).dtype
        flat_ids = input_ids.reshape(-1).to(device=device, dtype=torch.long)
        tag = image_token_tag.reshape(-1).to(device=device, dtype=torch.bool)
        with torch.inference_mode(), block_hf_modeling_imports():
            input_embeds = model.get_input_embeddings()(flat_ids).to(dtype=dtype)
            vit_embeds = self.vision_model(
                pixel_values.to(device=device, dtype=dtype),
                grid_hw.to(device=device, dtype=torch.long),
            ).to(device=device, dtype=input_embeds.dtype)
        image_token_count = int(tag.sum().item())
        if image_token_count != int(vit_embeds.shape[0]):
            raise ValueError(
                "image token count does not match native vision embeddings: "
                f"tokens={image_token_count} embeds={vit_embeds.shape[0]}"
            )
        input_embeds = input_embeds.clone()
        input_embeds[tag] = vit_embeds.reshape(image_token_count, -1)
        assert_no_hf_modeling_imported(context="after native input embedding compose")
        return input_embeds

    def _pool_snapshot(self) -> dict[str, int | None]:
        token_available = None
        try:
            token_available = int(self.token_to_kv_pool_allocator.available_size())
        except Exception:
            token_available = None
        return {
            "req_available": int(self.req_to_token_pool.available_size()),
            "token_available": token_available,
        }

    def _cache_snapshot(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "cache_type": type(self.tree_cache).__name__,
            "radix_cache_enabled": self.enable_radix_cache,
            "pool": self._pool_snapshot(),
        }
        for name in ("total_size", "evictable_size", "full_evictable_size"):
            value = getattr(self.tree_cache, name, None)
            if callable(value):
                try:
                    data[name] = int(value())
                except Exception:
                    data[name] = None
        for name in ("disable", "protected_size_", "evictable_size_"):
            if hasattr(self.tree_cache, name):
                try:
                    data[name] = getattr(self.tree_cache, name)
                except Exception:
                    data[name] = None
        return data

    def cached_prefix_length(
        self,
        *,
        input_ids: torch.Tensor,
        cache_extra_key: str | None,
    ) -> int:
        """Return the current Radix match length without scheduling a forward."""

        from sglang.srt.mem_cache.base_prefix_cache import (
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey

        token_ids = array(
            "q",
            [int(x) for x in input_ids.detach().cpu().reshape(-1).tolist()],
        )
        result = self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=token_ids,
                    extra_key=cache_extra_key,
                )
            )
        )
        return int(len(result.device_indices))

    def _finish_prefill_batch(
        self,
        batch: Any,
        *,
        cache_insert: bool,
    ) -> dict[str, Any]:
        from sglang.srt.mem_cache.common import release_kv_cache

        before = self._pool_snapshot()
        before_cache = self._cache_snapshot()
        finished: list[dict[str, Any]] = []
        for req in list(batch.reqs):
            if getattr(req, "req_pool_idx", None) is None:
                continue
            committed_len = int(req.effective_kv_committed_len())
            prefix_len = int(len(getattr(req, "prefix_indices", [])))
            release_kv_cache(req, self.tree_cache, is_insert=cache_insert)
            finished.append(
                {
                    "rid": str(req.rid),
                    "committed_len": committed_len,
                    "prefix_len": prefix_len,
                    "cache_insert": bool(cache_insert),
                    "cache_extra_key": getattr(req, "extra_key", None),
                }
            )
        return {
            "released_reqs": len(finished),
            "finished_reqs": finished,
            "before": before,
            "before_cache": before_cache,
            "after": self._pool_snapshot(),
            "after_cache": self._cache_snapshot(),
        }

    def _prepare_prefill_request(
        self,
        *,
        request_id: str,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        image_gen_indicators: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
        max_new_tokens: int = 1,
        cache_extra_key: str | None | object = _CACHE_EXTRA_KEY_NOT_SET,
    ) -> dict[str, Any]:
        device = torch.device(self.model_worker.device)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        indexes = indexes.to(device=device, dtype=torch.long)
        image_token_tag = image_token_tag.to(device=device, dtype=torch.bool)
        if image_gen_indicators is not None:
            image_gen_indicators = image_gen_indicators.to(device=device, dtype=torch.bool)
        if indexes.shape != (3, input_ids.numel()):
            raise ValueError("indexes must have shape (3, len(input_ids)).")
        if image_token_tag.numel() != input_ids.numel():
            raise ValueError("image_token_tag length must match input_ids.")
        if input_embeds is not None:
            input_embeds = input_embeds.to(
                device=device,
                dtype=next(self.model_worker.model_runner.model.parameters()).dtype,
            )
            if input_embeds.ndim == 3:
                input_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
            if input_embeds.shape[0] != input_ids.numel():
                raise ValueError("input_embeds rows must match input_ids length.")
        else:
            input_embeds = self.compose_input_embeds(
                input_ids=input_ids,
                image_token_tag=image_token_tag,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
            )

        req = self._make_req(
            request_id=request_id,
            input_ids=input_ids,
            indexes=indexes,
            image_token_tag=image_token_tag,
            image_gen_indicators=image_gen_indicators,
            max_new_tokens=max_new_tokens,
            cache_extra_key=(
                self._cache_extra_key(
                    image_token_tag=image_token_tag,
                    pixel_values=pixel_values,
                    grid_hw=grid_hw,
                    input_embeds=input_embeds,
                )
                if cache_extra_key is _CACHE_EXTRA_KEY_NOT_SET
                else cache_extra_key
            ),
            has_projected_input_embeds=input_embeds is not None,
        )
        return {
            "request_id": request_id,
            "input_ids": input_ids,
            "indexes": indexes,
            "image_token_tag": image_token_tag,
            "image_gen_indicators": image_gen_indicators,
            "input_embeds": input_embeds,
            "req": req,
            "cache_extra_key": req.extra_key,
        }

    def run_hidden_prefill(
        self,
        request: dict[str, Any],
        *,
        cache_insert: bool = False,
    ) -> NativeHiddenPrefillResult:
        """Run one native extend prefill and return full extend-token hidden states."""

        assert_no_hf_modeling_imported(context="before native hidden prefill")
        device = torch.device(self.model_worker.device)
        prepared = [self._prepare_prefill_request(**request)]
        for item in prepared:
            self.prefill_manager.add_one_request(item["req"])
        batch = self.prefill_manager.schedule_next_batch(
            self.decode_manager.running_batch,
            num_allocatable_reqs=1,
        )
        if batch is None or len(batch.reqs) != 1:
            raise RuntimeError(
                "native hidden prefill failed to schedule one request: "
                f"waiting_queue={len(getattr(self.prefill_manager, 'waiting_queue', []))} "
                f"chunked_req={getattr(getattr(self.prefill_manager, 'chunked_req', None), 'rid', None)} "
                f"running_batch_size={self.decode_manager.running_batch.batch_size()} "
                f"running_batch_full={getattr(self.decode_manager.running_batch, 'batch_is_full', None)} "
                f"request_tokens={int(request['input_ids'].numel())} "
                f"image_tokens={int(request['image_token_tag'].reshape(-1).sum().item())} "
                f"pool={self._pool_snapshot()} cache={self._cache_snapshot()}"
            )

        try:
            from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs

            resolve_deferred_prefill_inputs(
                batch,
                torch.device(self.model_worker.device),
            )
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_worker.model_runner,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
            forward_batch_log = {
                "forward_mode": str(forward_batch.forward_mode),
                "batch_size": int(forward_batch.batch_size),
                "input_ids_shape": list(forward_batch.input_ids.shape),
                "positions_shape": list(forward_batch.positions.shape),
                "mrope_positions_shape": (
                    None
                    if forward_batch.mrope_positions is None
                    else list(forward_batch.mrope_positions.shape)
                ),
                "extend_seq_lens_cpu": list(forward_batch.extend_seq_lens_cpu or []),
                "extend_prefix_lens_cpu": list(
                    forward_batch.extend_prefix_lens_cpu or []
                ),
                "rids": list(forward_batch.rids or []),
                "requested_batch_size": 1,
                "scheduled_batch_size": int(forward_batch.batch_size),
                "radix_cache_enabled": self.enable_radix_cache,
            }
            sidecar_input_embeds = self._attach_prefill_sidecar(
                prepared=prepared,
                forward_batch=forward_batch,
                forward_batch_log=forward_batch_log,
            )
            if self.prefill_cuda_graph_enabled:
                self.model_worker.model_runner.model.prepare_forward_batch(
                    forward_batch
                )

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            start = time.perf_counter()
            with torch.inference_mode(), block_hf_modeling_imports():
                batch_result = self.model_worker.forward_batch_generation(
                    forward_batch,
                    batch=batch,
                )
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            elapsed = time.perf_counter() - start
            forward_batch_log["can_run_cuda_graph"] = bool(
                getattr(batch_result, "can_run_cuda_graph", False)
            )

            logits_output = batch_result.logits_output
            hidden_states = None if logits_output is None else logits_output.hidden_states
            if hidden_states is None:
                raise RuntimeError("native hidden prefill did not return hidden states")
            if isinstance(hidden_states, dict):
                raise RuntimeError("native hidden prefill returned dict hidden states")
            hidden_states = hidden_states.detach()

            prepare = getattr(
                self.model_worker.model_runner.model,
                "last_forward_batch_prepare",
                None,
            )
            attn_metadata = getattr(
                self.model_worker.model_runner.attn_backend,
                "forward_metadata",
                None,
            )
            if attn_metadata is not None:
                backend_mask = getattr(attn_metadata, "custom_mask", None)
                backend_mask_indptr = getattr(attn_metadata, "mask_indptr", None)
                forward_batch_log["backend_forward_metadata"] = {
                    "class": type(attn_metadata).__name__,
                    "custom_mask_present": backend_mask is not None,
                    "custom_mask_numel": (
                        None if backend_mask is None else int(backend_mask.numel())
                    ),
                    "mask_indptr": (
                        None
                        if backend_mask_indptr is None
                        else [
                            int(x)
                            for x in backend_mask_indptr.detach().cpu().tolist()
                        ]
                    ),
                }
            states = forward_batch.model_specific_states or {}
            forward_batch_log["model_specific_state_keys"] = sorted(states.keys())
            forward_batch_log["resource_release"] = self._finish_prefill_batch(
                batch,
                cache_insert=cache_insert,
            )
            batch = None
            assert_no_hf_modeling_imported(context="after native hidden prefill")
            return NativeHiddenPrefillResult(
                request_id=str(prepared[0]["request_id"]),
                hidden_states=hidden_states,
                forward_elapsed_s=elapsed,
                backend_name=type(self.model_worker.model_runner.attn_backend).__name__,
                prepare_metadata=prepare,
                forward_batch_log=forward_batch_log,
                input_embeds_used=sidecar_input_embeds is not None,
                input_embeds_shape=(
                    None
                    if sidecar_input_embeds is None
                    else list(sidecar_input_embeds.shape)
                ),
                cache_inserted=bool(cache_insert),
                cache_extra_key=prepared[0]["cache_extra_key"],
            )
        finally:
            if batch is not None:
                self._finish_prefill_batch(batch, cache_insert=False)

    @staticmethod
    def _sample_next_token_ids(
        batch_result: Any,
        *,
        suppress_token_ids: list[int] | None = None,
    ) -> list[int]:
        suppress_ids = sorted({int(x) for x in (suppress_token_ids or [])})
        if suppress_ids:
            logits = batch_result.logits_output.next_token_logits
            if logits is None:
                raise RuntimeError("native decode benchmark produced no logits")
            filtered = logits.float().clone()
            filtered[:, suppress_ids] = torch.finfo(filtered.dtype).min
            return [
                int(x)
                for x in torch.argmax(filtered, dim=-1).detach().cpu().tolist()
            ]

        next_token_ids = getattr(batch_result, "next_token_ids", None)
        if next_token_ids is not None:
            if isinstance(next_token_ids, torch.Tensor):
                values = next_token_ids.detach().reshape(-1).cpu().tolist()
                return [int(x) for x in values]
            if isinstance(next_token_ids, list):
                values: list[int] = []
                for item in next_token_ids:
                    if isinstance(item, torch.Tensor):
                        flat = item.detach().reshape(-1).cpu().tolist()
                        values.append(int(flat[0]))
                    elif isinstance(item, (list, tuple)):
                        values.append(int(item[0]))
                    else:
                        values.append(int(item))
                return values

        logits = batch_result.logits_output.next_token_logits
        if logits is None:
            raise RuntimeError("native decode benchmark produced no logits")
        return [int(x) for x in torch.argmax(logits.float(), dim=-1).cpu().tolist()]

    @staticmethod
    def _zero_text_decode_hw_axes(forward_batch: ForwardBatch) -> None:
        """Match U1 official AR text decode indexes: [t_index, 0, 0]."""

        mrope_positions = getattr(forward_batch, "mrope_positions", None)
        if mrope_positions is not None and mrope_positions.ndim == 2:
            mrope_positions[1:].zero_()

    def _attach_prefill_sidecar(
        self,
        *,
        prepared: list[dict[str, Any]],
        forward_batch: ForwardBatch,
        forward_batch_log: dict[str, Any],
    ) -> torch.Tensor | None:
        extend_lens = list(forward_batch.extend_seq_lens_cpu or [])
        prefix_lens = list(forward_batch.extend_prefix_lens_cpu or [])
        sidecar_input_embeds = None
        if any(item["input_embeds"] is not None for item in prepared):
            if not all(item["input_embeds"] is not None for item in prepared):
                raise RuntimeError(
                    "native serving batch cannot mix projected input_embeds and "
                    "plain token embeddings yet"
                )
            if not extend_lens:
                extend_lens = [int(forward_batch.input_ids.numel())]
            if not prefix_lens:
                prefix_lens = [0] * len(extend_lens)
            if len(extend_lens) != len(prepared):
                raise RuntimeError(
                    "native serving batch lens mismatch: "
                    f"lens={len(extend_lens)} requests={len(prepared)}"
                )
            parts = []
            for item, prefix_len, extend_len in zip(prepared, prefix_lens, extend_lens):
                embeds = item["input_embeds"]
                if embeds is None:
                    raise RuntimeError("missing projected input_embeds")
                parts.append(
                    embeds[int(prefix_len) : int(prefix_len) + int(extend_len)]
                    .contiguous()
                )
            sidecar_input_embeds = torch.cat(parts, dim=0)
            attach_omni_prefill_inputs(
                forward_batch,
                OmniPrefillInputs(input_embeds=sidecar_input_embeds),
            )
            forward_batch_log["omni_prefill_input_embeds_shape"] = list(
                sidecar_input_embeds.shape
            )
        return sidecar_input_embeds

    def run_prefill_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        cache_insert: bool = False,
    ) -> list[NativeServingResult]:
        assert_no_hf_modeling_imported(context="before native prefill batch")
        if not requests:
            return []
        device = torch.device(self.model_worker.device)
        if len(requests) > self.max_running_requests:
            raise ValueError(
                "native prefill batch exceeds max_running_requests: "
                f"batch={len(requests)} max={self.max_running_requests}"
            )
        prepared = [self._prepare_prefill_request(**item) for item in requests]
        cache_before_schedule = self._cache_snapshot()
        for item in prepared:
            self.prefill_manager.add_one_request(item["req"])
        batch = self.prefill_manager.schedule_next_batch(
            self.decode_manager.running_batch,
            num_allocatable_reqs=len(prepared),
        )
        if batch is None:
            raise RuntimeError("native serving prefill manager returned no batch")
        if len(batch.reqs) != len(prepared):
            raise RuntimeError(
                "native serving prefill manager did not schedule the full batch: "
                f"scheduled={len(batch.reqs)} requested={len(prepared)}"
            )

        from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs

        resolve_deferred_prefill_inputs(
            batch,
            torch.device(self.model_worker.device),
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.model_worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        forward_batch_log = {
            "forward_mode": str(forward_batch.forward_mode),
            "batch_size": int(forward_batch.batch_size),
            "input_ids_shape": list(forward_batch.input_ids.shape),
            "positions_shape": list(forward_batch.positions.shape),
            "mrope_positions_shape": (
                None
                if forward_batch.mrope_positions is None
                else list(forward_batch.mrope_positions.shape)
            ),
            "extend_seq_lens_cpu": list(forward_batch.extend_seq_lens_cpu or []),
            "extend_prefix_lens_cpu": list(forward_batch.extend_prefix_lens_cpu or []),
            "rids": list(forward_batch.rids or []),
            "requested_batch_size": len(prepared),
            "scheduled_batch_size": int(forward_batch.batch_size),
            "radix_cache_enabled": self.enable_radix_cache,
            "cache_before_schedule": cache_before_schedule,
        }
        extend_lens = list(forward_batch.extend_seq_lens_cpu or [])
        prefix_lens = list(forward_batch.extend_prefix_lens_cpu or [])
        sidecar_input_embeds = self._attach_prefill_sidecar(
            prepared=prepared,
            forward_batch=forward_batch,
            forward_batch_log=forward_batch_log,
        )
        if self.prefill_cuda_graph_enabled:
            self.model_worker.model_runner.model.prepare_forward_batch(
                forward_batch
            )
        per_request_log = []
        for idx, (item, req) in enumerate(zip(prepared, batch.reqs)):
            prefix_len = int(prefix_lens[idx]) if idx < len(prefix_lens) else 0
            extend_len = int(extend_lens[idx]) if idx < len(extend_lens) else 0
            per_request_log.append(
                {
                    "batch_index": idx,
                    "rid": str(req.rid),
                    "input_tokens": int(item["input_ids"].numel()),
                    "prefix_len": prefix_len,
                    "extend_len": extend_len,
                    "cache_hit": prefix_len > 0,
                    "cache_extra_key": item["cache_extra_key"],
                    "image_token_count": int(item["image_token_tag"].sum().item()),
                }
            )
        forward_batch_log["per_request"] = per_request_log
        forward_batch_log["cache_hit_requests"] = sum(
            1 for item in per_request_log if item["cache_hit"]
        )
        forward_batch_log["cache_hit_tokens"] = sum(
            int(item["prefix_len"]) for item in per_request_log
        )

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        dense_calls_before = None
        dense_calls_after = None
        try:
            from sglang_omni.models.sensenova_u1 import (
                attention_backend as _u1_attn,
            )

            dense_calls_before = int(
                getattr(_u1_attn, "CUSTOM_MASK_DENSE_ATTENTION_CALLS", 0)
            )
        except Exception:
            _u1_attn = None
        start = time.perf_counter()
        with torch.inference_mode(), block_hf_modeling_imports():
            batch_result = self.model_worker.forward_batch_generation(
                forward_batch,
                batch=batch,
            )
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        if _u1_attn is not None:
            dense_calls_after = int(
                getattr(_u1_attn, "CUSTOM_MASK_DENSE_ATTENTION_CALLS", 0)
            )
            forward_batch_log["custom_mask_dense_attention_calls_delta"] = (
                dense_calls_after - (dense_calls_before or 0)
            )
        elapsed = time.perf_counter() - start
        forward_batch_log["can_run_cuda_graph"] = bool(
            getattr(batch_result, "can_run_cuda_graph", False)
        )
        logits = batch_result.logits_output.next_token_logits
        if logits is None:
            raise RuntimeError("native serving prefill produced no logits")
        if int(logits.shape[0]) != len(prepared):
            raise RuntimeError(
                "native serving logits batch size mismatch: "
                f"logits={list(logits.shape)} requests={len(prepared)}"
            )
        prepare = getattr(
            self.model_worker.model_runner.model,
            "last_forward_batch_prepare",
            None,
        )
        attn_metadata = getattr(
            self.model_worker.model_runner.attn_backend,
            "forward_metadata",
            None,
        )
        if attn_metadata is not None:
            backend_mask = getattr(attn_metadata, "custom_mask", None)
            backend_mask_indptr = getattr(attn_metadata, "mask_indptr", None)
            forward_batch_log["backend_forward_metadata"] = {
                "class": type(attn_metadata).__name__,
                "custom_mask_present": backend_mask is not None,
                "custom_mask_numel": (
                    None if backend_mask is None else int(backend_mask.numel())
                ),
                "mask_indptr": (
                    None
                    if backend_mask_indptr is None
                    else [int(x) for x in backend_mask_indptr.detach().cpu().tolist()]
                ),
            }
        if forward_batch.cross_attention_custom_mask is not None:
            forward_batch_log["custom_mask_shape"] = [
                int(forward_batch.cross_attention_custom_mask.numel())
            ]
            forward_batch_log["custom_mask_dtype"] = str(
                forward_batch.cross_attention_custom_mask.dtype
            )
        states = forward_batch.model_specific_states or {}
        forward_batch_log["model_specific_state_keys"] = sorted(states.keys())
        forward_batch_log["resource_release"] = self._finish_prefill_batch(
            batch,
            cache_insert=cache_insert,
        )

        assert_no_hf_modeling_imported(context="after native prefill batch")
        results: list[NativeServingResult] = []
        for idx, item in enumerate(prepared):
            row_logits = logits[idx : idx + 1]
            next_token_id = int(torch.argmax(row_logits[0].float()).item())
            item_log = dict(forward_batch_log)
            item_log["this_request"] = per_request_log[idx]
            results.append(
                NativeServingResult(
                    request_id=str(item["request_id"]),
                    next_token_id=next_token_id,
                    logits_shape=list(row_logits.shape),
                    logits_dtype=str(row_logits.dtype),
                    logits_device=str(row_logits.device),
                    logits_all_finite=bool(torch.isfinite(row_logits).all().item()),
                    forward_elapsed_s=elapsed,
                    backend_name=type(self.model_worker.model_runner.attn_backend).__name__,
                    prepare_metadata=prepare,
                    forward_batch_log=item_log,
                    input_embeds_used=sidecar_input_embeds is not None,
                    input_embeds_shape=(
                        None
                        if sidecar_input_embeds is None
                        else list(sidecar_input_embeds.shape)
                    ),
                    batch_index=idx,
                    batch_size=len(prepared),
                    cache_inserted=bool(cache_insert),
                    cache_extra_key=item["cache_extra_key"],
                    next_token_logits=row_logits.detach().float().cpu(),
                )
            )
        return results

    def run_greedy_decode_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        decode_steps: int,
        suppress_token_ids: list[int] | None = None,
    ) -> NativeDecodeBenchmarkResult:
        """Run a fixed-step greedy decode benchmark on the native SGLang path.

        This intentionally bypasses tokenizer streaming and stop handling so the
        benchmark measures the same number of generated tokens for every batch.
        It still uses SGLang's ScheduleBatch, ForwardBatch, RadixAttention, and
        KV-cache allocation path.
        """

        assert_no_hf_modeling_imported(context="before native greedy decode batch")
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if not requests:
            raise ValueError("requests must be non-empty")
        if len(requests) > self.max_running_requests:
            raise ValueError(
                "native decode batch exceeds max_running_requests: "
                f"batch={len(requests)} max={self.max_running_requests}"
            )

        device = torch.device(self.model_worker.device)
        prepared = [
            self._prepare_prefill_request(**item, max_new_tokens=decode_steps)
            for item in requests
        ]
        batch = None
        release_log = None
        generated: list[list[int]] = [[] for _ in prepared]
        per_step_elapsed_s: list[float] = []
        decode_logs: list[dict[str, Any]] = []
        logits_all_finite = True
        total_start = time.perf_counter()
        try:
            for item in prepared:
                self.prefill_manager.add_one_request(item["req"])
            batch = self.prefill_manager.schedule_next_batch(
                self.decode_manager.running_batch,
                num_allocatable_reqs=len(prepared),
            )
            if batch is None:
                raise RuntimeError("native decode prefill manager returned no batch")
            if len(batch.reqs) != len(prepared):
                raise RuntimeError(
                    "native decode benchmark did not schedule the full batch: "
                    f"scheduled={len(batch.reqs)} requested={len(prepared)}"
                )

            from sglang_omni.model_runner.base import resolve_deferred_prefill_inputs

            resolve_deferred_prefill_inputs(
                batch,
                torch.device(self.model_worker.device),
            )
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_worker.model_runner,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )
            prefill_log = {
                "forward_mode": str(forward_batch.forward_mode),
                "batch_size": int(forward_batch.batch_size),
                "input_ids_shape": list(forward_batch.input_ids.shape),
                "positions_shape": list(forward_batch.positions.shape),
                "mrope_positions_shape": (
                    None
                    if forward_batch.mrope_positions is None
                    else list(forward_batch.mrope_positions.shape)
                ),
                "extend_seq_lens_cpu": list(forward_batch.extend_seq_lens_cpu or []),
                "extend_prefix_lens_cpu": list(forward_batch.extend_prefix_lens_cpu or []),
                "rids": list(forward_batch.rids or []),
                "requested_batch_size": len(prepared),
                "scheduled_batch_size": int(forward_batch.batch_size),
                "radix_cache_enabled": self.enable_radix_cache,
            }
            sidecar_input_embeds = self._attach_prefill_sidecar(
                prepared=prepared,
                forward_batch=forward_batch,
                forward_batch_log=prefill_log,
            )

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            prefill_start = time.perf_counter()
            with torch.inference_mode(), block_hf_modeling_imports():
                batch_result = self.model_worker.forward_batch_generation(
                    forward_batch,
                    batch=batch,
                )
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            prefill_elapsed = time.perf_counter() - prefill_start
            prefill_logits = batch_result.logits_output.next_token_logits
            logits_all_finite = bool(
                logits_all_finite
                and prefill_logits is not None
                and torch.isfinite(prefill_logits).all().item()
            )
            next_ids = self._sample_next_token_ids(
                batch_result,
                suppress_token_ids=suppress_token_ids,
            )
            if len(next_ids) != len(prepared):
                raise RuntimeError(
                    f"prefill sampled {len(next_ids)} ids for {len(prepared)} requests"
                )
            for req, token_id, out in zip(batch.reqs, next_ids, generated):
                req.output_ids.append(int(token_id))
                out.append(int(token_id))
            batch.input_ids = torch.tensor(
                next_ids,
                dtype=torch.int64,
                device=device,
            )

            decode_elapsed = 0.0
            for step_idx in range(1, decode_steps):
                if not batch.check_decode_mem():
                    raise RuntimeError("native decode benchmark ran out of KV memory")
                batch.prepare_for_decode()
                forward_batch = ForwardBatch.init_new(
                    batch,
                    self.model_worker.model_runner,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    return_hidden_states_before_norm=False,
                )
                self._zero_text_decode_hw_axes(forward_batch)
                step_log = {
                    "step": step_idx,
                    "forward_mode": str(forward_batch.forward_mode),
                    "batch_size": int(forward_batch.batch_size),
                    "input_ids_shape": list(forward_batch.input_ids.shape),
                    "positions_shape": list(forward_batch.positions.shape),
                    "mrope_positions_shape": (
                        None
                        if forward_batch.mrope_positions is None
                        else list(forward_batch.mrope_positions.shape)
                    ),
                    "mrope_positions": (
                        None
                        if forward_batch.mrope_positions is None
                        else [
                            [int(v) for v in row]
                            for row in forward_batch.mrope_positions.detach()
                            .cpu()
                            .tolist()
                        ]
                    ),
                    "seq_lens_cpu": (
                        None
                        if forward_batch.seq_lens_cpu is None
                        else [int(x) for x in forward_batch.seq_lens_cpu.tolist()]
                    ),
                    "rids": list(forward_batch.rids or []),
                }
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                step_start = time.perf_counter()
                with torch.inference_mode(), block_hf_modeling_imports():
                    batch_result = self.model_worker.forward_batch_generation(
                        forward_batch,
                        batch=batch,
                    )
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                step_elapsed = time.perf_counter() - step_start
                step_log["can_run_cuda_graph"] = bool(
                    getattr(batch_result, "can_run_cuda_graph", False)
                )
                decode_elapsed += step_elapsed
                per_step_elapsed_s.append(step_elapsed)
                logits = batch_result.logits_output.next_token_logits
                logits_all_finite = bool(
                    logits_all_finite
                    and logits is not None
                    and torch.isfinite(logits).all().item()
                )
                next_ids = self._sample_next_token_ids(
                    batch_result,
                    suppress_token_ids=suppress_token_ids,
                )
                if len(next_ids) != len(prepared):
                    raise RuntimeError(
                        f"decode step {step_idx} sampled {len(next_ids)} ids "
                        f"for {len(prepared)} requests"
                    )
                for req, token_id, out in zip(batch.reqs, next_ids, generated):
                    req.output_ids.append(int(token_id))
                    out.append(int(token_id))
                batch.input_ids = torch.tensor(
                    next_ids,
                    dtype=torch.int64,
                    device=device,
                )
                decode_logs.append(step_log)

            total_elapsed = time.perf_counter() - total_start
            release_log = self._finish_prefill_batch(batch, cache_insert=False)
            batch = None
            assert_no_hf_modeling_imported(context="after native greedy decode batch")
            prompt_tokens = sum(int(item["input_ids"].numel()) for item in prepared)
            return NativeDecodeBenchmarkResult(
                request_count=len(prepared),
                decode_steps=decode_steps,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(prepared) * decode_steps,
                prefill_elapsed_s=prefill_elapsed,
                decode_elapsed_s=decode_elapsed,
                total_elapsed_s=total_elapsed,
                generated_token_ids=generated,
                per_step_elapsed_s=per_step_elapsed_s,
                prefill_forward_batch_log=prefill_log,
                decode_forward_batch_logs=decode_logs,
                release_log=release_log,
                logits_all_finite=logits_all_finite,
            )
        finally:
            if batch is not None:
                self._finish_prefill_batch(batch, cache_insert=False)

    def run_eager_text_decode(
        self,
        *,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        decode_steps: int,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
        forced_token_ids: list[int] | None = None,
        suppress_token_ids: list[int] | None = None,
        capture_step: int | None = None,
    ) -> NativeEagerTextDecodeResult:
        """Run HF-style eager cached text decode with native weights.

        This path is scoped to U1 interleave text-state validation: it does not
        import official U1 modeling code and it keeps the SGLang-loaded native
        weights, but it mirrors HF DynamicCache/eager attention numerics for
        single-request text continuation.
        """

        assert_no_hf_modeling_imported(context="before native eager text decode")
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        device = torch.device(self.model_worker.device)
        model = self.model_worker.model_runner.model
        dtype = next(model.parameters()).dtype
        input_ids = input_ids.to(device=device, dtype=torch.long).reshape(-1)
        indexes = indexes.to(device=device, dtype=torch.long)
        image_token_tag = image_token_tag.to(device=device, dtype=torch.bool).reshape(-1)
        if input_embeds is not None:
            input_embeds = input_embeds.to(device=device, dtype=dtype)
            if input_embeds.ndim == 3:
                input_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
        else:
            input_embeds = self.compose_input_embeds(
                input_ids=input_ids,
                image_token_tag=image_token_tag,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
            )
        if indexes.shape != (3, input_ids.numel()):
            raise ValueError("indexes must have shape (3, len(input_ids)).")
        if image_token_tag.numel() != input_ids.numel():
            raise ValueError("image_token_tag length must match input_ids.")
        if input_embeds is not None and input_embeds.shape[0] != input_ids.numel():
            raise ValueError("input_embeds rows must match input_ids length.")
        suppress_ids = sorted({int(x) for x in (suppress_token_ids or [])})
        bf16_argmax = self._native_eager_bf16_argmax_enabled()
        lm_head_linear = self._native_eager_lm_head_linear_enabled()
        direct_embedding = self._native_eager_direct_embedding_enabled()
        compiled_add_rms = self._native_eager_compiled_add_rms_enabled()
        compiled_add_rms_layers = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS_LAYERS",
            "",
        )

        def _select_next_token(logits_tensor: torch.Tensor) -> tuple[int, int | None]:
            if bf16_argmax and not suppress_ids:
                return int(torch.argmax(logits_tensor[0]).item()), None
            scores = logits_tensor[0].float()
            raw_argmax = int(torch.argmax(scores).item())
            if suppress_ids and raw_argmax in suppress_ids:
                scores = scores.clone()
                scores[suppress_ids] = torch.finfo(scores.dtype).min
                return int(torch.argmax(scores).item()), raw_argmax
            return raw_argmax, None

        generated: list[int] = []
        step_logs: list[dict[str, Any]] = []
        per_step_elapsed_s: list[float] = []
        logits_all_finite = True
        captured_layer_hidden_states: list[torch.Tensor] | None = None
        total_start = time.perf_counter()
        repeat_kv_cache = self._native_eager_repeated_kv_cache_enabled()
        use_static_kv_cache = self._native_eager_static_kv_cache_enabled()
        eager_prefix_cache_key = (
            self._eager_text_prefix_cache_key(
                input_ids=input_ids,
                indexes=indexes,
                image_token_tag=image_token_tag,
                input_embeds=input_embeds,
                repeat_kv_cache=repeat_kv_cache,
            )
            if self._native_eager_prefix_cache_enabled()
            else None
        )
        eager_prefix_cache_hit = bool(
            eager_prefix_cache_key is not None
            and eager_prefix_cache_key in self._eager_text_prefill_cache
        )
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        prefill_start = time.perf_counter()
        if eager_prefix_cache_hit:
            logits, caches, prefill_logits_finite = self._eager_text_prefill_cache[
                eager_prefix_cache_key
            ]
        else:
            with torch.inference_mode(), block_hf_modeling_imports():
                hidden_states, caches = model.model.eager_text_prefill_with_cache(
                    input_ids,
                    torch.arange(
                        input_ids.numel(),
                        device=device,
                        dtype=torch.long,
                    ),
                    input_embeds=input_embeds,
                    indexes=indexes,
                    image_token_tag=image_token_tag,
                    repeat_kv_cache=repeat_kv_cache,
                )
                logits = model.eager_text_logits(hidden_states)
            prefill_logits_finite = bool(torch.isfinite(logits).all().item())
            if eager_prefix_cache_key is not None:
                self._eager_text_prefill_cache[eager_prefix_cache_key] = (
                    logits.detach(),
                    caches,
                    prefill_logits_finite,
                )
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        prefill_elapsed = time.perf_counter() - prefill_start
        logits_all_finite = prefill_logits_finite
        next_token_id, suppressed_argmax = _select_next_token(logits)
        generated.append(next_token_id)
        step_logs.append(
            {
                "step": 0,
                "mode": "prefill",
                "predicts_token_index": 0,
                "predicted_token_id": next_token_id,
                "suppressed_argmax_token_id": suppressed_argmax,
                "suppress_token_ids": suppress_ids,
                "cache_seq_len_after": int(caches[0][0].shape[0]) if caches else 0,
                "indexes": [
                    [int(v) for v in row[-5:]]
                    for row in indexes.detach().cpu().tolist()
                ],
            }
        )

        decode_elapsed = 0.0
        last_logits = logits.detach().float().cpu()
        full_loop_graph_used = bool(
            device.type == "cuda"
            and self._native_eager_full_loop_cuda_graph_enabled()
            and decode_steps > 1
            and forced_token_ids is None
            and not suppress_ids
            and capture_step is None
        )
        if full_loop_graph_used:
            graph_key = (
                int(caches[0][0].shape[0]),
                int(indexes[0].max().item()),
                int(decode_steps),
                repeat_kv_cache,
                use_static_kv_cache,
                bf16_argmax,
                lm_head_linear,
                direct_embedding,
                compiled_add_rms,
                compiled_add_rms_layers,
                self._native_eager_graph_mode(),
            )
            graph_created = graph_key not in self._eager_text_decode_graphs
            if graph_created:
                self._eager_text_decode_graphs[graph_key] = (
                    self._capture_eager_text_decode_graph(
                        model=model,
                        caches=caches,
                        input_token_id=next_token_id,
                        start_t_index=int(graph_key[1]),
                        decode_steps=decode_steps,
                        device=device,
                        source_prefix_cache_key=eager_prefix_cache_key,
                        graph_mode=str(graph_key[10]),
                        repeat_kv_cache=bool(graph_key[3]),
                        use_static_kv_cache=bool(graph_key[4]),
                        bf16_argmax=bool(graph_key[5]),
                    )
                )
            graph_runner = self._eager_text_decode_graphs[graph_key]
            torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
            skip_initial_copy = bool(
                eager_prefix_cache_hit
                and eager_prefix_cache_key is not None
                and graph_runner.source_prefix_cache_key
                == eager_prefix_cache_key
            )
            if not skip_initial_copy:
                for (static_k, static_v), (live_k, live_v) in zip(
                    graph_runner.initial_caches,
                    caches,
                ):
                    static_k[: live_k.shape[0]].copy_(live_k)
                    static_v[: live_v.shape[0]].copy_(live_v)
                graph_runner.input_token.fill_(next_token_id)
                graph_runner.source_prefix_cache_key = eager_prefix_cache_key
            for graph in graph_runner.graphs:
                graph.replay()
            torch.cuda.synchronize(device)
            decode_elapsed = time.perf_counter() - decode_start
            tail_ids = [
                int(x)
                for x in torch.cat(
                    graph_runner.generated_tokens,
                ).detach().cpu().tolist()
            ]
            generated.extend(tail_ids)
            fast_result = bool(
                skip_initial_copy
                and self._native_eager_fast_result_enabled()
            )
            if fast_result:
                last_logits = None
                logits_all_finite = bool(
                    logits_all_finite and graph_runner.verified_finite
                )
            else:
                last_logits = graph_runner.last_logits.detach().float().cpu()
                logits_all_finite = bool(
                    logits_all_finite
                    and torch.isfinite(graph_runner.last_logits).all().item()
                )
            average_step_s = decode_elapsed / max(len(tail_ids), 1)
            per_step_elapsed_s.extend([average_step_s] * len(tail_ids))
            if not fast_result:
                for step_idx, token_id in enumerate(tail_ids, start=1):
                    step_logs.append(
                        {
                            "step": step_idx,
                            "mode": "decode_full_loop_cuda_graph",
                            "predicts_token_index": step_idx,
                            "predicted_token_id": token_id,
                            "suppressed_argmax_token_id": None,
                            "suppress_token_ids": [],
                            "decode_indexes": [
                                [int(v) for v in row]
                                for row in graph_runner.decode_indexes[
                                    step_idx - 1
                                ].detach().cpu().tolist()
                            ],
                            "cache_seq_len_after": graph_key[0] + step_idx,
                            "elapsed_s": average_step_s,
                        }
                    )
            total_elapsed = time.perf_counter() - total_start
            assert_no_hf_modeling_imported(
                context="after native eager text decode"
            )
            return NativeEagerTextDecodeResult(
                decode_steps=decode_steps,
                prompt_tokens=int(input_ids.numel()),
                generated_token_ids=generated,
                prefill_elapsed_s=prefill_elapsed,
                decode_elapsed_s=decode_elapsed,
                total_elapsed_s=total_elapsed,
                per_step_elapsed_s=per_step_elapsed_s,
                step_logs=step_logs,
                logits_all_finite=logits_all_finite,
                next_token_logits=last_logits,
                captured_layer_hidden_states=None,
                full_loop_cuda_graph_used=True,
                full_loop_cuda_graph_created=graph_created,
                full_loop_cuda_graph_key=":".join(str(x) for x in graph_key),
                eager_prefix_cache_hit=eager_prefix_cache_hit,
                eager_prefix_cache_key=eager_prefix_cache_key,
                full_loop_initial_cache_copy_skipped=skip_initial_copy,
            )

        for step_idx in range(1, decode_steps):
            if forced_token_ids is not None and step_idx - 1 < len(forced_token_ids):
                input_token_id = int(forced_token_ids[step_idx - 1])
            else:
                input_token_id = int(generated[-1])
            past_t_index = int(indexes[0].max().item()) + step_idx - 1
            decode_indexes = torch.tensor(
                [[past_t_index + 1], [0], [0]],
                device=device,
                dtype=torch.long,
            )
            input_token = torch.tensor([input_token_id], device=device, dtype=torch.long)
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            step_start = time.perf_counter()
            capture_layer_outputs: list[torch.Tensor] | None = (
                [] if capture_step is not None and step_idx == int(capture_step) else None
            )
            with torch.inference_mode(), block_hf_modeling_imports():
                hidden_states, caches = model.model.eager_text_decode_with_cache(
                    input_token,
                    decode_indexes[0],
                    caches,
                    indexes=decode_indexes,
                    capture_layer_outputs=capture_layer_outputs,
                    repeat_kv_cache=repeat_kv_cache,
                )
                logits = model.eager_text_logits(hidden_states)
            if capture_layer_outputs is not None:
                captured_layer_hidden_states = capture_layer_outputs
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            step_elapsed = time.perf_counter() - step_start
            decode_elapsed += step_elapsed
            per_step_elapsed_s.append(step_elapsed)
            logits_all_finite = bool(
                logits_all_finite and torch.isfinite(logits).all().item()
            )
            next_token_id, suppressed_argmax = _select_next_token(logits)
            generated.append(next_token_id)
            last_logits = logits.detach().float().cpu()
            step_logs.append(
                {
                    "step": step_idx,
                    "mode": "decode",
                    "input_token_id": input_token_id,
                    "predicts_token_index": step_idx,
                    "predicted_token_id": next_token_id,
                    "suppressed_argmax_token_id": suppressed_argmax,
                    "suppress_token_ids": suppress_ids,
                    "decode_indexes": [
                        [int(v) for v in row]
                        for row in decode_indexes.detach().cpu().tolist()
                    ],
                    "cache_seq_len_after": int(caches[0][0].shape[0]) if caches else 0,
                    "elapsed_s": step_elapsed,
                }
            )

        total_elapsed = time.perf_counter() - total_start
        assert_no_hf_modeling_imported(context="after native eager text decode")
        return NativeEagerTextDecodeResult(
            decode_steps=decode_steps,
            prompt_tokens=int(input_ids.numel()),
            generated_token_ids=generated,
            prefill_elapsed_s=prefill_elapsed,
            decode_elapsed_s=decode_elapsed,
            total_elapsed_s=total_elapsed,
            per_step_elapsed_s=per_step_elapsed_s,
            step_logs=step_logs,
            logits_all_finite=logits_all_finite,
            next_token_logits=last_logits,
            captured_layer_hidden_states=captured_layer_hidden_states,
            eager_prefix_cache_hit=eager_prefix_cache_hit,
            eager_prefix_cache_key=eager_prefix_cache_key,
        )

    @staticmethod
    def _native_eager_full_loop_cuda_graph_enabled() -> bool:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_TEXT_FULL_LOOP_CUDA_GRAPH",
            "",
        ).lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _native_eager_prefix_cache_enabled() -> bool:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_TEXT_PREFIX_CACHE",
            "",
        ).lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _native_eager_repeated_kv_cache_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_REPEATED_KV_CACHE",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_static_kv_cache_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_STATIC_KV_CACHE",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_bf16_argmax_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_BF16_ARGMAX",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_lm_head_linear_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_LM_HEAD_LINEAR",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_fast_result_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_FAST_RESULT",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_direct_embedding_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_DIRECT_EMBEDDING",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_compiled_add_rms_enabled() -> bool:
        return os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_COMPILED_ADD_RMS",
            "",
        ).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _native_eager_graph_mode() -> str:
        value = os.environ.get(
            "SENSENOVA_U1_NATIVE_EAGER_TEXT_GRAPH_MODE",
            "segmented",
        ).lower()
        if value not in {"segmented", "monolithic"}:
            raise ValueError(
                "SENSENOVA_U1_NATIVE_EAGER_TEXT_GRAPH_MODE must be "
                "'segmented' or 'monolithic'"
            )
        return value

    @staticmethod
    def _eager_text_prefix_cache_key(
        *,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        input_embeds: torch.Tensor | None,
        repeat_kv_cache: bool,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"sensenova-u1-native-eager-prefix-v1")
        hasher.update(
            b":repeated-kv" if repeat_kv_cache else b":compact-kv"
        )
        _update_hash_with_tensor(hasher, input_ids.to(dtype=torch.long))
        _update_hash_with_tensor(hasher, indexes.to(dtype=torch.long))
        tag = image_token_tag.to(dtype=torch.bool)
        _update_hash_with_tensor(hasher, tag)
        if bool(tag.any().item()) and input_embeds is not None:
            _update_hash_with_tensor(hasher, input_embeds)
        return "u1-eager-prefix:" + hasher.hexdigest()[:32]

    @staticmethod
    def _run_eager_text_decode_graph_step(
        *,
        model: Any,
        input_token: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        indexes: torch.Tensor,
        repeat_kv_cache: bool,
        use_static_kv_cache: bool,
        cache_position: int,
        bf16_argmax: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        if use_static_kv_cache:
            hidden_states, caches = (
                model.model.eager_text_decode_with_static_cache(
                    input_token,
                    indexes[0],
                    caches,
                    cache_position=cache_position,
                    indexes=indexes,
                    repeat_kv_cache=repeat_kv_cache,
                )
            )
        else:
            hidden_states, caches = model.model.eager_text_decode_with_cache(
                input_token,
                indexes[0],
                caches,
                indexes=indexes,
                repeat_kv_cache=repeat_kv_cache,
            )
        logits = model.eager_text_logits(hidden_states)
        next_token = torch.argmax(
            logits if bf16_argmax else logits.float(),
            dim=-1,
        ).to(dtype=torch.long)
        return next_token, logits, caches

    @classmethod
    def _run_eager_text_decode_graph_body(
        cls,
        *,
        model: Any,
        input_token: torch.Tensor,
        initial_caches: list[tuple[torch.Tensor, torch.Tensor]],
        decode_indexes: list[torch.Tensor],
        repeat_kv_cache: bool,
        use_static_kv_cache: bool,
        prefix_len: int,
        bf16_argmax: bool,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        list[list[tuple[torch.Tensor, torch.Tensor]]],
    ]:
        current_token = input_token
        current_caches = initial_caches
        generated_tokens: list[torch.Tensor] = []
        cache_history: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        last_logits = None
        for step_offset, indexes in enumerate(decode_indexes):
            current_token, logits, current_caches = (
                cls._run_eager_text_decode_graph_step(
                    model=model,
                    input_token=current_token,
                    caches=current_caches,
                    indexes=indexes,
                    repeat_kv_cache=repeat_kv_cache,
                    use_static_kv_cache=use_static_kv_cache,
                    cache_position=prefix_len + step_offset,
                    bf16_argmax=bf16_argmax,
                )
            )
            generated_tokens.append(current_token)
            cache_history.append(current_caches)
            last_logits = logits
        if last_logits is None:
            raise RuntimeError("monolithic CUDA graph requires a decode step")
        return generated_tokens, last_logits, current_caches, cache_history

    def _capture_eager_text_decode_graph(
        self,
        *,
        model: Any,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        input_token_id: int,
        start_t_index: int,
        decode_steps: int,
        device: torch.device,
        source_prefix_cache_key: str | None,
        graph_mode: str,
        repeat_kv_cache: bool,
        use_static_kv_cache: bool,
        bf16_argmax: bool,
    ) -> _NativeEagerDecodeGraph:
        prefix_len = int(caches[0][0].shape[0])
        if use_static_kv_cache:
            capacity = prefix_len + decode_steps - 1
            initial_caches = [
                (
                    torch.empty(
                        (capacity, *k.shape[1:]),
                        device=k.device,
                        dtype=k.dtype,
                    ),
                    torch.empty(
                        (capacity, *v.shape[1:]),
                        device=v.device,
                        dtype=v.dtype,
                    ),
                )
                for k, v in caches
            ]
        else:
            initial_caches = [
                (torch.empty_like(k), torch.empty_like(v))
                for k, v in caches
            ]
        for (static_k, static_v), (live_k, live_v) in zip(
            initial_caches,
            caches,
        ):
            static_k[: live_k.shape[0]].copy_(live_k)
            static_v[: live_v.shape[0]].copy_(live_v)
        input_token = torch.full(
            (1,),
            int(input_token_id),
            device=device,
            dtype=torch.long,
        )
        decode_indexes = [
            torch.tensor(
                [[start_t_index + step_idx], [0], [0]],
                device=device,
                dtype=torch.long,
            )
            for step_idx in range(1, decode_steps)
        ]

        capture_stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        if graph_mode == "monolithic":
            with torch.cuda.stream(capture_stream):
                warmup_outputs = self._run_eager_text_decode_graph_body(
                    model=model,
                    input_token=input_token,
                    initial_caches=initial_caches,
                    decode_indexes=decode_indexes,
                    repeat_kv_cache=repeat_kv_cache,
                    use_static_kv_cache=use_static_kv_cache,
                    prefix_len=prefix_len,
                    bf16_argmax=bf16_argmax,
                )
            capture_stream.synchronize()
            del warmup_outputs

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=graph_pool,
                stream=capture_stream,
            ):
                (
                    generated_tokens,
                    last_logits,
                    final_caches,
                    cache_history,
                ) = self._run_eager_text_decode_graph_body(
                    model=model,
                    input_token=input_token,
                    initial_caches=initial_caches,
                    decode_indexes=decode_indexes,
                    repeat_kv_cache=repeat_kv_cache,
                    use_static_kv_cache=use_static_kv_cache,
                    prefix_len=prefix_len,
                    bf16_argmax=bf16_argmax,
                )
            capture_stream.synchronize()
            verified_finite = bool(torch.isfinite(last_logits).all().item())
            torch.cuda.current_stream(device).wait_stream(capture_stream)
            return _NativeEagerDecodeGraph(
                graphs=[graph],
                graph_pool=graph_pool,
                initial_caches=initial_caches,
                input_token=input_token,
                decode_indexes=decode_indexes,
                generated_tokens=generated_tokens,
                last_logits=last_logits,
                final_caches=final_caches,
                cache_history=cache_history,
                source_prefix_cache_key=source_prefix_cache_key,
                prefix_len=prefix_len,
                verified_finite=verified_finite,
            )

        graphs: list[torch.cuda.CUDAGraph] = []
        generated_tokens: list[torch.Tensor] = []
        cache_history: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        current_token = input_token
        current_caches = initial_caches
        last_logits = None
        for step_offset, indexes in enumerate(decode_indexes):
            with torch.cuda.stream(capture_stream):
                warmup_outputs = self._run_eager_text_decode_graph_step(
                    model=model,
                    input_token=current_token,
                    caches=current_caches,
                    indexes=indexes,
                    repeat_kv_cache=repeat_kv_cache,
                    use_static_kv_cache=use_static_kv_cache,
                    cache_position=prefix_len + step_offset,
                    bf16_argmax=bf16_argmax,
                )
            capture_stream.synchronize()
            del warmup_outputs

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=graph_pool,
                stream=capture_stream,
            ):
                next_token, logits, next_caches = (
                    self._run_eager_text_decode_graph_step(
                        model=model,
                        input_token=current_token,
                        caches=current_caches,
                        indexes=indexes,
                        repeat_kv_cache=repeat_kv_cache,
                        use_static_kv_cache=use_static_kv_cache,
                        cache_position=prefix_len + step_offset,
                        bf16_argmax=bf16_argmax,
                    )
                )
            graphs.append(graph)
            generated_tokens.append(next_token)
            cache_history.append(next_caches)
            current_token = next_token
            current_caches = next_caches
            last_logits = logits
        if last_logits is None:
            raise RuntimeError("segmented CUDA graph requires a decode step")
        capture_stream.synchronize()
        verified_finite = bool(torch.isfinite(last_logits).all().item())
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        return _NativeEagerDecodeGraph(
            graphs=graphs,
            graph_pool=graph_pool,
            initial_caches=initial_caches,
            input_token=input_token,
            decode_indexes=decode_indexes,
            generated_tokens=generated_tokens,
            last_logits=last_logits,
            final_caches=current_caches,
            cache_history=cache_history,
            source_prefix_cache_key=source_prefix_cache_key,
            prefix_len=prefix_len,
            verified_finite=verified_finite,
        )

    def run_prefill(
        self,
        *,
        request_id: str,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        image_gen_indicators: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
        cache_extra_key: str | None | object = _CACHE_EXTRA_KEY_NOT_SET,
        cache_insert: bool = False,
    ) -> NativeServingResult:
        return self.run_prefill_batch(
            [
                {
                    "request_id": request_id,
                    "input_ids": input_ids,
                    "indexes": indexes,
                    "image_token_tag": image_token_tag,
                    "image_gen_indicators": image_gen_indicators,
                    "input_embeds": input_embeds,
                    "pixel_values": pixel_values,
                    "grid_hw": grid_hw,
                    "cache_extra_key": cache_extra_key,
                }
            ],
            cache_insert=cache_insert,
        )[0]

    def complete_payload(self, payload: Any) -> dict[str, Any]:
        data = dict(getattr(payload, "data", None) or payload or {})
        input_ids = _tensor_from_payload(data["input_ids"], dtype=torch.long)
        indexes = _tensor_from_payload(data["indexes"], dtype=torch.long)
        image_token_tag = _tensor_from_payload(data["image_token_tag"], dtype=torch.bool)
        image_gen_indicators = data.get("image_gen_indicators")
        if image_gen_indicators is not None:
            image_gen_indicators = _tensor_from_payload(
                image_gen_indicators,
                dtype=torch.bool,
            )
        result = self.run_prefill(
            request_id=str(getattr(payload, "request_id", "native-u1-req")),
            input_ids=input_ids,
            indexes=indexes,
            image_token_tag=image_token_tag,
            image_gen_indicators=image_gen_indicators,
            cache_insert=True,
        )
        out = result.to_dict()
        out["text"] = str(result.next_token_id)
        out["finish_reason"] = "length"
        out["usage"] = {
            "prompt_tokens": int(input_ids.numel()),
            "completion_tokens": 1,
            "total_tokens": int(input_ids.numel()) + 1,
        }
        return out

    def complete_payload_batch(self, payloads: list[Any]) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        token_counts: list[int] = []
        for payload in payloads:
            data = dict(getattr(payload, "data", None) or payload or {})
            input_ids = _tensor_from_payload(data["input_ids"], dtype=torch.long)
            indexes = _tensor_from_payload(data["indexes"], dtype=torch.long)
            image_token_tag = _tensor_from_payload(
                data["image_token_tag"],
                dtype=torch.bool,
            )
            image_gen_indicators = data.get("image_gen_indicators")
            if image_gen_indicators is not None:
                image_gen_indicators = _tensor_from_payload(
                    image_gen_indicators,
                    dtype=torch.bool,
                )
            token_counts.append(int(input_ids.numel()))
            requests.append(
                {
                    "request_id": str(getattr(payload, "request_id", "native-u1-req")),
                    "input_ids": input_ids,
                    "indexes": indexes,
                    "image_token_tag": image_token_tag,
                    "image_gen_indicators": image_gen_indicators,
                }
            )
        results = self.run_prefill_batch(requests, cache_insert=True)
        outs: list[dict[str, Any]] = []
        for result, prompt_tokens in zip(results, token_counts):
            out = result.to_dict()
            out["text"] = str(result.next_token_id)
            out["finish_reason"] = "length"
            out["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            }
            outs.append(out)
        return outs


__all__ = [
    "NativeEagerTextDecodeResult",
    "NativeHiddenPrefillResult",
    "NativeServingResult",
    "SenseNovaU1NativeServingExecutor",
]
