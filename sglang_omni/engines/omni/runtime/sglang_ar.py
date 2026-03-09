# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache

from ..types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)
from .ar import ARRequestData

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

logger = logging.getLogger(__name__)


@dataclass
class SGLangARRequestData(ARRequestData):
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None


class SGLangBatchPlanner:
    def __init__(
        self,
        prefill_manager: "PrefillManager",
        decode_manager: "DecodeManager",
        server_args: "ServerArgs",
    ):
        self.prefill_manager = prefill_manager
        self.decode_manager = decode_manager
        self.server_args = server_args
        self.last_batch: Any | None = None
        self.forward_ct: int = 0
        self.req_id_map: dict[str, SchedulerRequest] = {}
        self._cached_schedule_batch: Any | None = None
        self._last_schedule_source: str | None = None
        self._last_schedule_waiting: list[str] = []
        self._last_schedule_running: list[str] = []
        self._last_schedule_summaries: list[dict[str, Any]] = []

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
        active_request_ids.update(
            sched_req.request_id
            for sched_req in self.req_id_map.values()
            if sched_req.status == SchedulerStatus.WAITING_FEEDBACK
        )
        self._prune_inactive_state(active_request_ids)

        for sched_req in waiting:
            data: SGLangARRequestData = sched_req.data
            if not data.synced:
                self.prefill_manager.add_one_request(data.req)
                data.synced = True
                self.req_id_map[data.req.rid] = sched_req

        running_batch = self.decode_manager.running_batch
        running_bs = running_batch.batch_size()
        num_allocatable_reqs = max(
            self.server_args.max_running_requests - running_bs, 0
        )

        running_batch_for_prefill = self.decode_manager.running_batch
        if (
            running_batch_for_prefill is not None
            and running_batch_for_prefill.is_empty()
        ):
            running_batch_for_prefill = None

        schedule_source: str | None = None
        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )
        if schedule_batch is not None:
            schedule_source = "prefill"

        if (
            schedule_batch is None
            and self.decode_manager.runnable
            and not self._decode_waiting_feedback_blocked()
        ):
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)
            if schedule_batch is not None:
                schedule_source = "decode"

        if schedule_batch is None:
            self._cached_schedule_batch = None
            self._last_schedule_source = None
            self._last_schedule_waiting = []
            self._last_schedule_running = []
            self._last_schedule_summaries = []
            return []

        self._cached_schedule_batch = schedule_batch
        self._last_schedule_source = schedule_source
        self._last_schedule_waiting = [req.request_id for req in waiting]
        self._last_schedule_running = [req.request_id for req in running]
        self.forward_ct += 1

        selected: list[SchedulerRequest] = []
        keep_indices: list[int] = []
        for i, req in enumerate(schedule_batch.reqs):
            sched_req = self.req_id_map.get(req.rid)
            if sched_req is not None and sched_req.status not in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                keep_indices.append(i)
                selected.append(sched_req)
            elif sched_req is None:
                logger.warning("SGLang req %s not found in req_id_map", req.rid)

        if len(keep_indices) != len(schedule_batch.reqs):
            if keep_indices:
                schedule_batch.filter_batch(keep_indices=keep_indices)
                self._cached_schedule_batch = schedule_batch
            else:
                self._cached_schedule_batch = None
                return []

        self._last_schedule_summaries = [
            {
                "rid": sched_req.request_id,
                "status": sched_req.status.name,
                "generation_steps": int(
                    getattr(sched_req.data, "generation_steps", -1)
                ),
                "data_projected": bool(
                    getattr(sched_req.data, "input_embeds_are_projected", False)
                ),
                "req_projected": bool(
                    getattr(
                        getattr(sched_req.data, "req", None),
                        "_input_embeds_are_projected",
                        False,
                    )
                ),
                "req_embed_len": (
                    len(
                        getattr(
                            getattr(sched_req.data, "req", None), "input_embeds", None
                        )
                    )
                    if isinstance(
                        getattr(
                            getattr(sched_req.data, "req", None), "input_embeds", None
                        ),
                        list,
                    )
                    else None
                ),
                "req_output_len": len(
                    getattr(getattr(sched_req.data, "req", None), "output_ids", [])
                ),
                "req_is_chunked": int(
                    getattr(getattr(sched_req.data, "req", None), "is_chunked", -1)
                ),
                "batch_forward_mode": getattr(
                    schedule_batch.forward_mode,
                    "name",
                    str(schedule_batch.forward_mode),
                ),
                "extend_input_len": int(
                    getattr(
                        getattr(sched_req.data, "req", None), "extend_input_len", -1
                    )
                ),
                "decode_batch_idx": int(
                    getattr(
                        getattr(sched_req.data, "req", None), "decode_batch_idx", -1
                    )
                ),
                "extend_batch_idx": int(
                    getattr(
                        getattr(sched_req.data, "req", None), "extend_batch_idx", -1
                    )
                ),
            }
            for sched_req in selected
        ]
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return self._cached_schedule_batch

    def _decode_waiting_feedback_blocked(self) -> bool:
        running_batch = self.decode_manager.running_batch
        if running_batch is None or running_batch.is_empty():
            return False
        for req in running_batch.reqs:
            sched_req = self.req_id_map.get(req.rid)
            if sched_req is None:
                continue
            if sched_req.status == SchedulerStatus.WAITING_FEEDBACK:
                return True
        return False

    def _post_step_operations(self):
        chunked_req_to_exclude = set()
        active_chunked_req = self.prefill_manager.chunked_req
        if active_chunked_req is not None:
            chunked_req_to_exclude.add(active_chunked_req)
            self.prefill_manager.tree_cache.cache_unfinished_req(
                active_chunked_req, chunked=True
            )
            if active_chunked_req.req_pool_idx is not None:
                self.prefill_manager.req_to_token_pool.free(
                    active_chunked_req.req_pool_idx
                )

        if self.last_batch is None:
            return

        if self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.decode_manager.running_batch.batch_is_full = False

            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.decode_manager.running_batch.is_empty():
                    self.decode_manager.running_batch = self.last_batch
                else:
                    self.decode_manager.running_batch.merge_batch(self.last_batch)

        if not self.decode_manager.running_batch.is_empty():
            finished_indices = []
            for i, req in enumerate(self.decode_manager.running_batch.reqs):
                sched_req = self.req_id_map.get(req.rid)
                if req.finished() or (
                    sched_req is not None
                    and sched_req.status
                    in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED)
                ):
                    finished_indices.append(i)

            if finished_indices:
                keep = [
                    i
                    for i in range(len(self.decode_manager.running_batch.reqs))
                    if i not in finished_indices
                ]
                if keep:
                    self.decode_manager.running_batch.filter_batch(keep_indices=keep)
                else:
                    from sglang.srt.managers.schedule_batch import ScheduleBatch

                    self.decode_manager.running_batch = ScheduleBatch(
                        reqs=[], batch_is_full=False
                    )

        self.last_batch = None

    def record_last_batch(self, schedule_batch: Any):
        self.last_batch = schedule_batch

    def _prune_inactive_state(self, active_request_ids: set[str]) -> None:
        inactive_rids: set[str] = set()
        for rid, sched_req in list(self.req_id_map.items()):
            if sched_req.request_id not in active_request_ids or sched_req.status in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                inactive_rids.add(rid)
                del self.req_id_map[rid]

        if not inactive_rids:
            return

        if self.prefill_manager.waiting_queue:
            self.prefill_manager.waiting_queue = [
                req
                for req in self.prefill_manager.waiting_queue
                if req.rid not in inactive_rids
            ]

        running_batch = self.decode_manager.running_batch
        if running_batch is None or running_batch.is_empty():
            return
        keep_indices = [
            i
            for i, req in enumerate(running_batch.reqs)
            if req.rid not in inactive_rids
        ]
        if len(keep_indices) == len(running_batch.reqs):
            return
        if keep_indices:
            running_batch.filter_batch(keep_indices=keep_indices)
        else:
            from sglang.srt.managers.schedule_batch import ScheduleBatch

            self.decode_manager.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=False
            )


class SGLangResourceManager:
    def __init__(self, token_to_kv_pool_allocator, req_to_token_pool, tree_cache):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return True

    def allocate(self, request: SchedulerRequest) -> None:
        pass

    def free(self, request: SchedulerRequest) -> None:
        data: SGLangARRequestData = request.data
        if data.req is not None:
            release_kv_cache(data.req, self.tree_cache)


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if model_output.next_token_ids is not None
            else []
        )

        # Extract hidden states if configured and available
        hidden_states_dict = None
        if self._capture_hidden:
            hidden_states_dict = self._extract_hidden_states(model_output)

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = None
            if hidden_states_dict is not None:
                if "_single" in hidden_states_dict:
                    extra = {"hidden_states": hidden_states_dict["_single"][i]}
                else:
                    per_req = {}
                    for key, tensor in hidden_states_dict.items():
                        per_req[key] = tensor[i] if tensor.ndim >= 2 else tensor
                    extra = {"hidden_states": per_req}
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra,
            )
        return outputs

    def _extract_hidden_states(
        self, model_output: Any
    ) -> dict[str, torch.Tensor] | None:
        """Extract hidden states from model output or side-channel.

        Priority:
        1. Side-channel (_captured_aux_hidden_states) from hidden capture hooks
        2. logits_output.hidden_states (legacy single-tensor path)
        """
        # Check side-channel first (set by _hidden_capture hooks)
        if self._model is not None and self._capture_hidden_layers:
            aux = getattr(self._model, "_captured_aux_hidden_states", None)
            if aux is not None:
                # aux is a list of tensors from layers_to_capture, one per layer
                self._model._captured_aux_hidden_states = None  # consume
                result = {}
                for layer_id, tensor in zip(self._capture_hidden_layers, aux):
                    key = "embed" if layer_id == 0 else layer_id
                    result[key] = tensor
                return result

        # Fallback: logits_output.hidden_states
        logits_output = getattr(model_output, "logits_output", None)
        if logits_output is None:
            return None
        raw_hidden = getattr(logits_output, "hidden_states", None)
        if raw_hidden is None:
            return None

        if isinstance(raw_hidden, dict):
            return raw_hidden
        elif isinstance(raw_hidden, torch.Tensor):
            return {"_single": raw_hidden}
        return None


class SGLangIterationController:
    """Handles per-request state updates with chunked prefill semantics.

    Chunked prefill: req.is_chunked > 0 means the request is still in
    chunked prefill — decrement counter, do NOT append token or check finish.
    """

    def __init__(self, tree_cache, feedback_enabled: bool = False):
        self.tree_cache = tree_cache
        self._feedback_enabled = feedback_enabled

    def needs_feedback(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        """Check if request needs Code Predictor feedback before next step."""
        if not self._feedback_enabled:
            return False
        data = request.data
        if data.req.is_chunked > 0:
            return False
        if data.req.finished():
            return False
        # Decode steps need feedback (not prefill)
        return data.generation_steps > 0

    def apply_feedback(
        self, request: SchedulerRequest, feedback_embeds: torch.Tensor
    ) -> None:
        """Apply Code Predictor feedback to request's next input."""
        request.data.feedback_embeds = feedback_embeds

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: SGLangARRequestData = request.data
        req = data.req
        projected = bool(
            getattr(data, "input_embeds_are_projected", False)
            or getattr(req, "_input_embeds_are_projected", False)
        )
        if projected and logger.isEnabledFor(logging.INFO):
            logger.info(
                "SGLang update_request before rid=%s token=%s generation_steps=%s "
                "req_is_chunked=%s req_output_len=%s finished=%s",
                request.request_id,
                output.data,
                data.generation_steps,
                req.is_chunked,
                len(req.output_ids),
                req.finished(),
            )

        if (
            getattr(data, "capture_prompt_hidden", False)
            and data.generation_steps == 0
            and output.extra is not None
        ):
            prefill_hidden = output.extra.get("prefill_hidden_states")
            if prefill_hidden is not None:
                store = data.extra_model_outputs.setdefault(
                    "prompt_prefill_hidden_states", {}
                )
                if isinstance(prefill_hidden, torch.Tensor):
                    existing = store.get("_single")
                    tensor = prefill_hidden.detach().cpu()
                    store["_single"] = (
                        tensor
                        if existing is None
                        else torch.cat([existing, tensor], dim=0)
                    )
                elif isinstance(prefill_hidden, dict):
                    for key, value in prefill_hidden.items():
                        if not isinstance(value, torch.Tensor):
                            continue
                        existing = store.get(key)
                        tensor = value.detach().cpu()
                        store[key] = (
                            tensor
                            if existing is None
                            else torch.cat([existing, tensor], dim=0)
                        )

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        token_id = output.data
        if token_id is not None:
            req.output_ids.append(token_id)
            data.generation_steps += 1
            req.check_finished()
            if not req.finished() and req.decode_batch_idx == 0:
                self.tree_cache.cache_unfinished_req(req)
        if projected and logger.isEnabledFor(logging.INFO):
            logger.info(
                "SGLang update_request after rid=%s token=%s generation_steps=%s "
                "req_is_chunked=%s req_output_len=%s finished=%s",
                request.request_id,
                output.data,
                data.generation_steps,
                req.is_chunked,
                len(req.output_ids),
                req.finished(),
            )

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return request.data.req.finished()


class SGLangModelRunner:
    """Model runner that uses SGLang's ModelWorker for execution.

    Replaces the generic ModelRunner — handles ScheduleBatch → ForwardBatch
    conversion, forward pass, and conditional sampling.
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        output_processor: SGLangOutputProcessor,
        batch_planner: SGLangBatchPlanner | None = None,
    ):
        self.model_worker = model_worker
        self.output_processor = output_processor
        self.batch_planner = batch_planner
        self.device = torch.device(f"cuda:{model_worker.gpu_id}")

        model = model_worker.model_runner.model
        self._embed_tokens, self._inner_model = self._get_inner_model_components(model)
        self._is_talker_model = hasattr(self._inner_model, "prepare_input_embeds")

        hf_config = model_worker.model_runner.model_config.hf_config
        thinker_cfg = (
            hf_config.thinker_config
            if hasattr(hf_config, "thinker_config")
            else hf_config
        )
        self._image_token_id = thinker_cfg.image_token_id
        self._video_token_id = thinker_cfg.video_token_id
        self._audio_token_id = thinker_cfg.audio_token_id

    @staticmethod
    def _get_inner_model_components(model):
        outer = model.thinker if hasattr(model, "thinker") else model
        inner = getattr(outer, "model", outer)

        embed_tokens = getattr(inner, "embed_tokens", None)
        if embed_tokens is None:
            get_input_embeddings = getattr(inner, "get_input_embeddings", None)
            if callable(get_input_embeddings):
                embed_tokens = get_input_embeddings()
        if embed_tokens is None:
            embed_tokens = getattr(inner, "codec_embedding", None)

        return embed_tokens, outer

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> tuple[torch.Tensor | None, list | None, torch.Tensor | None]:

        if not any(req.omni_model_inputs is not None for req in schedule_batch.reqs):
            return None, None, None

        device = forward_batch.input_ids.device
        image_token_id = self._image_token_id
        video_token_id = self._video_token_id
        audio_token_id = self._audio_token_id

        input_embeds = self._embed_tokens(forward_batch.input_ids)

        extend_lens = forward_batch.extend_seq_lens_cpu
        offsets = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += length

        deepstack_visual_embeds_list = []
        visual_pos_masks_list = []
        has_deepstack = False

        for i, req in enumerate(schedule_batch.reqs):
            omni_inputs = req.omni_model_inputs
            if omni_inputs is None:
                continue

            start = offsets[i]
            end = start + extend_lens[i]
            req_input_ids = forward_batch.input_ids[start:end]
            consumed = req._omni_consumed or {}

            for modality, token_id in [
                ("image", image_token_id),
                ("video", video_token_id),
                ("audio", audio_token_id),
            ]:
                embeds = omni_inputs.get(f"{modality}_embeds")
                if embeds is None:
                    continue
                mask = req_input_ids == token_id
                if not mask.any():
                    continue
                n_tokens = int(mask.sum().item())
                offset = consumed.get(modality, 0)
                chunk_embeds = embeds[offset : offset + n_tokens].to(
                    device=device, dtype=input_embeds.dtype
                )
                input_embeds[torch.where(mask)[0] + start] = chunk_embeds
                consumed[modality] = offset + n_tokens

            req._omni_consumed = consumed

            ds_embeds = omni_inputs.get("deepstack_visual_embeds")
            image_ds = omni_inputs.get("image_deepstack_visual_embeds")
            video_ds = omni_inputs.get("video_deepstack_visual_embeds")

            if ds_embeds is not None or image_ds is not None or video_ds is not None:
                has_deepstack = True
                img_mask = req_input_ids == image_token_id
                vid_mask = req_input_ids == video_token_id
                visual_mask = img_mask | vid_mask

                if ds_embeds is None:
                    if image_ds and video_ds:
                        merged = []
                        for img_e, vid_e in zip(image_ds, video_ds):
                            num_visual = int(visual_mask.sum().item())
                            joint = img_e.new_zeros(num_visual, img_e.shape[-1])
                            img_in_visual = img_mask[visual_mask]
                            vid_in_visual = vid_mask[visual_mask]
                            if img_in_visual.any():
                                joint[img_in_visual] = img_e.to(device=device)
                            if vid_in_visual.any():
                                joint[vid_in_visual] = vid_e.to(device=device)
                            merged.append(joint)
                        ds_embeds = merged
                    elif image_ds:
                        ds_embeds = image_ds
                    elif video_ds:
                        ds_embeds = video_ds

                if ds_embeds is not None:
                    global_mask = torch.zeros(
                        len(forward_batch.input_ids),
                        dtype=torch.bool,
                        device=device,
                    )
                    global_mask[start:end] = visual_mask
                    deepstack_visual_embeds_list.append(ds_embeds)
                    visual_pos_masks_list.append(global_mask)

            if req.is_chunked == 0:
                req.omni_model_inputs = None
                req._omni_consumed = None

        ds_embeds_out = None
        visual_masks_out = None
        if has_deepstack and deepstack_visual_embeds_list:
            if len(deepstack_visual_embeds_list) == 1:
                ds_embeds_out = deepstack_visual_embeds_list[0]
                visual_masks_out = visual_pos_masks_list[0]
            else:
                combined_mask = torch.zeros(
                    len(forward_batch.input_ids), dtype=torch.bool, device=device
                )
                for m in visual_pos_masks_list:
                    combined_mask |= m
                num_layers = len(deepstack_visual_embeds_list[0])
                merged_ds = []
                for layer_idx in range(num_layers):
                    parts = [
                        req_ds[layer_idx].to(device=device, dtype=input_embeds.dtype)
                        for req_ds in deepstack_visual_embeds_list
                    ]
                    merged_ds.append(torch.cat(parts, dim=0))
                ds_embeds_out = merged_ds
                visual_masks_out = combined_mask

        return input_embeds, ds_embeds_out, visual_masks_out

    def _forward_with_omni_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
        deepstack_visual_embeds: list | None = None,
        visual_pos_masks: torch.Tensor | None = None,
    ) -> Any:

        model_runner = self.model_worker.model_runner
        outer = self._inner_model

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        ds_input = None
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            device = input_embeds.device
            dtype = input_embeds.dtype
            layer_tensors = [
                t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds
            ]
            ds_input = torch.cat(layer_tensors, dim=-1)

            full_ds = torch.zeros(
                input_embeds.shape[0],
                ds_input.shape[-1],
                device=device,
                dtype=dtype,
            )
            full_ds[visual_pos_masks] = ds_input
            ds_input = full_ds

        hidden_states = outer.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=ds_input,
        )

        logits_output = outer.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            outer.lm_head,
            forward_batch,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    def _forward_talker(
        self,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        input_deepstack_embeds: torch.Tensor | None = None,
        input_deepstack_mask: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
    ) -> GenerationBatchResult:
        model_runner = self.model_worker.model_runner
        outer = self._inner_model
        model_dtype = next(outer.parameters()).dtype

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        if input_deepstack_embeds is not None:
            input_deepstack_embeds = input_deepstack_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=model_dtype,
            )

        logits_output = outer(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            input_deepstack_embeds=input_deepstack_embeds,
            input_deepstack_mask=input_deepstack_mask,
            input_embeds_are_projected=input_embeds_are_projected,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    def _build_feedback_input_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> torch.Tensor | None:
        if not self._is_talker_model or schedule_batch.forward_mode.is_extend():
            return None
        if self._embed_tokens is None:
            return None

        input_embeds = self._embed_tokens(forward_batch.input_ids)
        has_feedback = False

        for idx, req in enumerate(schedule_batch.reqs):
            sched_req = None
            if self.batch_planner is not None:
                sched_req = self.batch_planner.req_id_map.get(req.rid)
            data = getattr(sched_req, "data", None)
            if data is None:
                continue

            feedback = getattr(data, "feedback_embeds", None)
            if feedback is None:
                continue
            combined = feedback.to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            ).reshape(-1)
            step_index = max(int(getattr(data, "generation_steps", 0)) - 1, 0)
            trailing = getattr(data, "trailing_text_hidden", None)
            tts_pad_embed = getattr(data, "tts_pad_embed", None)
            thinker_chunks_done = bool(getattr(data, "thinker_chunks_done", True))

            trailing_value = None
            if isinstance(trailing, list):
                if step_index < len(trailing):
                    trailing_value = trailing[step_index]
            elif isinstance(trailing, torch.Tensor):
                if step_index < trailing.shape[0]:
                    trailing_value = trailing[step_index]

            if trailing_value is not None:
                combined = combined + trailing_value.to(
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                ).reshape(-1)
            elif thinker_chunks_done and tts_pad_embed is not None:
                combined = combined + tts_pad_embed.to(
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                ).reshape(-1)

            if os.environ.get("SGLANG_OMNI_DUMP_TALKER_FEEDBACK_INPUTS") == "1":
                req = getattr(data, "req", None)
                trailing_len = 0
                if isinstance(trailing, list):
                    trailing_len = len(trailing)
                elif isinstance(trailing, torch.Tensor):
                    trailing_len = int(trailing.shape[0])
                try:
                    dump_path = (
                        Path("/tmp")
                        / f"talker_feedback_input_{getattr(req, 'rid', idx)}_"
                        f"step{int(getattr(data, 'generation_steps', -1))}.pt"
                    )
                    torch.save(
                        {
                            "request_id": getattr(req, "rid", None),
                            "generation_steps": int(
                                getattr(data, "generation_steps", -1)
                            ),
                            "decode_batch_idx": int(
                                getattr(req, "decode_batch_idx", -1)
                            ),
                            "input_token": (
                                int(forward_batch.input_ids[idx].item())
                                if getattr(forward_batch, "input_ids", None) is not None
                                and idx < len(forward_batch.input_ids)
                                else None
                            ),
                            "raw_feedback_embeds": feedback.detach().cpu(),
                            "trailing_len": trailing_len,
                            "thinker_chunks_done": thinker_chunks_done,
                            "used_trailing_value": (
                                trailing_value.detach().cpu()
                                if isinstance(trailing_value, torch.Tensor)
                                else trailing_value
                            ),
                            "tts_pad_embed": (
                                tts_pad_embed.detach().cpu()
                                if isinstance(tts_pad_embed, torch.Tensor)
                                else tts_pad_embed
                            ),
                            "combined_feedback_input_embeds": combined.detach().cpu(),
                        },
                        dump_path,
                    )
                    logger.info(
                        "Talker feedback input dump saved rid=%s path=%s",
                        getattr(req, "rid", None),
                        dump_path,
                    )
                except Exception:
                    logger.exception(
                        "Failed to dump talker feedback input for %s",
                        getattr(req, "rid", None),
                    )

            input_embeds[idx] = combined
            data.feedback_embeds = None
            has_feedback = True

        return input_embeds if has_feedback else None

    def _request_uses_projected_prefill(self, sched_req: Any) -> bool:
        data = getattr(sched_req, "data", None)
        if data is None:
            return False
        if bool(getattr(data, "input_embeds_are_projected", False)):
            return True
        req = getattr(data, "req", None)
        return bool(getattr(req, "_input_embeds_are_projected", False))

    def _rebuild_prefill_input_embeds(
        self,
        requests: list[Any],
    ) -> torch.Tensor | None:
        rows: list[Any] = []
        for sched_req in requests:
            data = getattr(sched_req, "data", None)
            req = getattr(data, "req", None)
            input_embeds = getattr(req, "input_embeds", None)
            if input_embeds:
                rows.extend(input_embeds)
        if not rows:
            return None
        return torch.as_tensor(rows, device=self.device, dtype=torch.float32)

    def _apply_codec_suppress_tokens(
        self,
        logits_output: Any,
        requests: list[Any],
    ) -> None:
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None or logits.ndim != 2:
            return

        for row_idx, sched_req in enumerate(requests):
            data = getattr(sched_req, "data", None)
            suppress_tokens = getattr(data, "suppress_tokens", None)
            if not suppress_tokens:
                req = getattr(data, "req", None)
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if not suppress_tokens:
                continue
            for token_id in suppress_tokens:
                if 0 <= int(token_id) < logits.shape[1]:
                    logits[row_idx, int(token_id)] = float("-inf")

    def _log_talker_prefill_debug(
        self,
        *,
        scheduler_output: SchedulerOutput,
        forward_batch: Any,
        request_prefill_input_embeds: torch.Tensor | None,
        has_projected_prefill: bool,
        projected_prefill: bool,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        summaries = []
        for sched_req in scheduler_output.requests:
            data = getattr(sched_req, "data", None)
            req = getattr(data, "req", None)
            req_embeds = getattr(req, "input_embeds", None)
            req_embed_len = len(req_embeds) if isinstance(req_embeds, list) else None
            req_mm = getattr(req, "multimodal_inputs", None)
            summaries.append(
                {
                    "rid": sched_req.request_id,
                    "generation_steps": int(getattr(data, "generation_steps", -1)),
                    "is_chunked": int(getattr(req, "is_chunked", -1)),
                    "data_projected": bool(
                        getattr(data, "input_embeds_are_projected", False)
                    ),
                    "req_projected": bool(
                        getattr(req, "_input_embeds_are_projected", False)
                    ),
                    "req_embed_len": req_embed_len,
                    "req_has_mm": req_mm is not None,
                    "req_mrope_shape": (
                        tuple(req_mm.mrope_positions.shape)
                        if req_mm is not None
                        and getattr(req_mm, "mrope_positions", None) is not None
                        else None
                    ),
                    "req_mrope_delta_shape": (
                        tuple(req_mm.mrope_position_delta.shape)
                        if req_mm is not None
                        and getattr(req_mm, "mrope_position_delta", None) is not None
                        else None
                    ),
                    "data_type": type(data).__name__,
                }
            )
        fb_shape = (
            tuple(forward_batch.input_embeds.shape)
            if getattr(forward_batch, "input_embeds", None) is not None
            else None
        )
        fb_mrope_shape = (
            tuple(forward_batch.mrope_positions.shape)
            if getattr(forward_batch, "mrope_positions", None) is not None
            else None
        )
        rebuilt_shape = (
            tuple(request_prefill_input_embeds.shape)
            if request_prefill_input_embeds is not None
            else None
        )
        schedule_source = None
        schedule_waiting = None
        schedule_running = None
        schedule_summaries = None
        if self.batch_planner is not None:
            schedule_source = getattr(self.batch_planner, "_last_schedule_source", None)
            schedule_waiting = getattr(
                self.batch_planner, "_last_schedule_waiting", None
            )
            schedule_running = getattr(
                self.batch_planner, "_last_schedule_running", None
            )
            schedule_summaries = getattr(
                self.batch_planner, "_last_schedule_summaries", None
            )
        logger.info(
            "Talker prefill debug: forward_mode=%s projected=%s has_projected=%s "
            "forward_batch_input_embeds=%s forward_batch_mrope=%s rebuilt_input_embeds=%s "
            "schedule_source=%s schedule_waiting=%s schedule_running=%s "
            "schedule_summaries=%s requests=%s",
            getattr(forward_batch, "forward_mode", None),
            projected_prefill,
            has_projected_prefill,
            fb_shape,
            fb_mrope_shape,
            rebuilt_shape,
            schedule_source,
            schedule_waiting,
            schedule_running,
            schedule_summaries,
            summaries,
        )

    def _log_talker_prefill_topk(
        self,
        *,
        scheduler_output: SchedulerOutput,
        logits_output: Any,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        logits = getattr(logits_output, "next_token_logits", None)
        if logits is None or logits.ndim != 2 or logits.shape[0] == 0:
            return
        top_k = min(10, logits.shape[1])
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            values, ids = torch.topk(logits[row_idx].float(), k=top_k)
            logger.info(
                "Talker prefill topk rid=%s ids=%s scores=%s",
                sched_req.request_id,
                ids.tolist(),
                [float(v) for v in values.tolist()],
            )
            try:
                dump_path = (
                    Path("/tmp") / f"talker_prefill_logits_{sched_req.request_id}.pt"
                )
                torch.save(
                    {
                        "request_id": sched_req.request_id,
                        "logits": logits[row_idx].detach().cpu(),
                    },
                    dump_path,
                )
                logger.info(
                    "Talker prefill logits dump saved rid=%s path=%s",
                    sched_req.request_id,
                    dump_path,
                )
            except Exception:
                logger.exception(
                    "Failed to dump talker prefill logits for %s",
                    sched_req.request_id,
                )

    def _log_talker_decode_debug(
        self,
        *,
        scheduler_output: SchedulerOutput,
        forward_batch: Any,
        feedback_input_embeds: torch.Tensor | None,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        request_summaries = []
        should_log = False
        for sched_req in scheduler_output.requests:
            data = getattr(sched_req, "data", None)
            req = getattr(data, "req", None)
            generation_steps = int(getattr(data, "generation_steps", -1))
            decode_batch_idx = int(getattr(req, "decode_batch_idx", -1))
            seq_len = None
            if getattr(forward_batch, "seq_lens_cpu", None) is not None:
                req_index = len(request_summaries)
                if req_index < len(forward_batch.seq_lens_cpu):
                    seq_len = int(forward_batch.seq_lens_cpu[req_index])
            request_summaries.append(
                {
                    "rid": sched_req.request_id,
                    "generation_steps": generation_steps,
                    "decode_batch_idx": decode_batch_idx,
                    "seq_len": seq_len,
                    "input_token": (
                        int(forward_batch.input_ids[len(request_summaries)].item())
                        if getattr(forward_batch, "input_ids", None) is not None
                        and len(request_summaries) < len(forward_batch.input_ids)
                        else None
                    ),
                }
            )
            if generation_steps <= 1 or decode_batch_idx <= 1:
                should_log = True

        if not should_log:
            return

        positions = getattr(forward_batch, "positions", None)
        mrope_positions = getattr(forward_batch, "mrope_positions", None)
        logger.info(
            "Talker decode debug: forward_mode=%s input_ids=%s positions=%s "
            "mrope_shape=%s mrope_last=%s feedback_shape=%s requests=%s",
            getattr(forward_batch, "forward_mode", None),
            (
                forward_batch.input_ids.detach().cpu().tolist()
                if getattr(forward_batch, "input_ids", None) is not None
                else None
            ),
            (
                positions.detach().cpu().tolist()
                if isinstance(positions, torch.Tensor)
                else None
            ),
            (
                tuple(mrope_positions.shape)
                if isinstance(mrope_positions, torch.Tensor)
                else None
            ),
            (
                mrope_positions[:, -1].detach().cpu().tolist()
                if isinstance(mrope_positions, torch.Tensor)
                and mrope_positions.ndim == 2
                and mrope_positions.shape[1] > 0
                else None
            ),
            (
                tuple(feedback_input_embeds.shape)
                if feedback_input_embeds is not None
                else None
            ),
            request_summaries,
        )

    def _dump_talker_decode_layer0_qk_debug(
        self,
        *,
        scheduler_output: SchedulerOutput,
        forward_batch: Any,
        feedback_input_embeds: torch.Tensor | None,
    ) -> None:
        if os.environ.get("SGLANG_OMNI_DUMP_TALKER_QK") != "1":
            return
        if feedback_input_embeds is None or not self._is_talker_model:
            return

        active = [
            sched_req
            for sched_req in scheduler_output.requests
            if int(getattr(getattr(sched_req, "data", None), "generation_steps", -1))
            <= 1
        ]
        if not active:
            return

        talker = self._inner_model
        text_model = getattr(talker, "model", None)
        if text_model is None or not getattr(text_model, "layers", None):
            return

        try:
            from sglang_omni.models.qwen3_omni.talker import apply_qk_norm

            layer0 = text_model.layers[0]
            attn = layer0.self_attn
            model_dtype = next(text_model.parameters()).dtype
            positions = getattr(forward_batch, "mrope_positions", None)
            if positions is None:
                positions = getattr(forward_batch, "positions", None)
            if positions is None:
                return

            hidden_states = feedback_input_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=model_dtype,
            )
            layer0_input_ln = layer0.input_layernorm(hidden_states)
            qkv, _ = attn.qkv_proj(layer0_input_ln)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=attn.q_norm,
                k_norm=attn.k_norm,
                head_dim=attn.head_dim,
                alt_stream=attn.alt_stream,
            )
            q, k = attn.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=None,
            )

            for row_idx, sched_req in enumerate(scheduler_output.requests):
                data = getattr(sched_req, "data", None)
                generation_steps = int(getattr(data, "generation_steps", -1))
                if generation_steps > 1:
                    continue

                req = getattr(data, "req", None)
                row_positions = positions
                if isinstance(positions, torch.Tensor):
                    if positions.ndim == 1:
                        row_positions = positions[row_idx : row_idx + 1]
                    elif positions.ndim == 2:
                        row_positions = positions[:, row_idx : row_idx + 1]

                dump = {
                    "request_id": sched_req.request_id,
                    "generation_steps": generation_steps,
                    "decode_batch_idx": int(getattr(req, "decode_batch_idx", -1)),
                    "input_token": int(forward_batch.input_ids[row_idx].item()),
                    "positions": (
                        row_positions.detach().cpu()
                        if isinstance(row_positions, torch.Tensor)
                        else row_positions
                    ),
                    "feedback_input_embeds": hidden_states[row_idx].detach().cpu(),
                    "layer0_input_ln": layer0_input_ln[row_idx].detach().cpu(),
                    "layer0_q_after_rope": q[row_idx].detach().cpu(),
                    "layer0_k_after_rope": k[row_idx].detach().cpu(),
                    "layer0_v": v[row_idx].detach().cpu(),
                }
                dump_path = (
                    Path("/tmp") / f"talker_decode_layer0_qk_{sched_req.request_id}.pt"
                )
                torch.save(dump, dump_path)
                logger.info(
                    "Talker decode layer0 qk dump saved rid=%s path=%s",
                    sched_req.request_id,
                    dump_path,
                )
        except Exception:
            logger.exception("Failed to dump talker decode layer0 qk debug")

    def _set_talker_layer0_attn_debug_context(
        self,
        *,
        scheduler_output: SchedulerOutput,
        forward_batch: Any,
    ) -> None:
        if os.environ.get("SGLANG_OMNI_DUMP_TALKER_ATTN") != "1":
            return
        if not self._is_talker_model:
            return

        talker = self._inner_model
        text_model = getattr(talker, "model", None)
        if text_model is None or not getattr(text_model, "layers", None):
            return

        layer0_attn = text_model.layers[0].self_attn
        row_indices: list[int] = []
        request_ids: list[str] = []
        generation_steps: list[int] = []
        decode_batch_indices: list[int] = []
        input_tokens: list[int | None] = []
        positions: list[Any] = []
        dump_paths: list[str] = []

        pos_tensor = getattr(forward_batch, "mrope_positions", None)
        if pos_tensor is None:
            pos_tensor = getattr(forward_batch, "positions", None)

        for row_idx, sched_req in enumerate(scheduler_output.requests):
            data = getattr(sched_req, "data", None)
            req = getattr(data, "req", None)
            step = int(getattr(data, "generation_steps", -1))
            if step > 1:
                continue
            row_indices.append(row_idx)
            request_ids.append(sched_req.request_id)
            generation_steps.append(step)
            decode_batch_indices.append(int(getattr(req, "decode_batch_idx", -1)))
            input_tokens.append(
                int(forward_batch.input_ids[row_idx].item())
                if getattr(forward_batch, "input_ids", None) is not None
                and row_idx < len(forward_batch.input_ids)
                else None
            )
            if isinstance(pos_tensor, torch.Tensor):
                if pos_tensor.ndim == 1:
                    pos_value = pos_tensor[row_idx : row_idx + 1].detach().cpu()
                elif pos_tensor.ndim == 2:
                    pos_value = pos_tensor[:, row_idx : row_idx + 1].detach().cpu()
                else:
                    pos_value = pos_tensor.detach().cpu()
            else:
                pos_value = None
            positions.append(pos_value)
            dump_paths.append(
                str(
                    Path("/tmp")
                    / f"talker_decode_layer0_attn_{sched_req.request_id}.pt"
                )
            )

        if not row_indices:
            return

        setattr(
            layer0_attn,
            "_sglang_omni_debug_capture_attn",
            {
                "row_indices": row_indices,
                "request_ids": request_ids,
                "generation_steps": generation_steps,
                "decode_batch_indices": decode_batch_indices,
                "input_tokens": input_tokens,
                "positions": positions,
                "dump_paths": dump_paths,
            },
        )

    def _clear_talker_layer0_attn_debug_context(self) -> None:
        if not self._is_talker_model:
            return
        talker = self._inner_model
        text_model = getattr(talker, "model", None)
        if text_model is None or not getattr(text_model, "layers", None):
            return
        layer0_attn = text_model.layers[0].self_attn
        if hasattr(layer0_attn, "_sglang_omni_debug_capture_attn"):
            setattr(layer0_attn, "_sglang_omni_debug_capture_attn", None)

    def _set_talker_disable_fused_set_kv(self, disabled: bool) -> None:
        if os.environ.get("SGLANG_OMNI_DISABLE_TALKER_FUSED_SET_KV") != "1":
            return
        if not self._is_talker_model:
            return
        talker = self._inner_model
        text_model = getattr(talker, "model", None)
        if text_model is None or not getattr(text_model, "layers", None):
            return
        for layer in text_model.layers:
            setattr(layer.self_attn, "_sglang_omni_disable_fused_set_kv", disabled)

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        # Ensure correct CUDA device context when running in thread pool
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data

        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()

        # Enable hidden state capture if output processor needs it
        if self.output_processor._capture_hidden:
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        omni_embeds = None
        if schedule_batch.forward_mode.is_extend():
            omni_embeds = self._inject_multimodal_embeds(forward_batch, schedule_batch)
        feedback_input_embeds = self._build_feedback_input_embeds(
            forward_batch, schedule_batch
        )
        request_prefill_input_embeds = (
            self._rebuild_prefill_input_embeds(scheduler_output.requests)
            if schedule_batch.forward_mode.is_extend()
            else None
        )
        has_projected_prefill = (
            any(
                self._request_uses_projected_prefill(req)
                for req in scheduler_output.requests
            )
            or request_prefill_input_embeds is not None
        )
        projected_prefill = (
            self._is_talker_model
            and schedule_batch.forward_mode.is_extend()
            and has_projected_prefill
        )
        if self._is_talker_model and schedule_batch.forward_mode.is_extend():
            self._log_talker_prefill_debug(
                scheduler_output=scheduler_output,
                forward_batch=forward_batch,
                request_prefill_input_embeds=request_prefill_input_embeds,
                has_projected_prefill=has_projected_prefill,
                projected_prefill=projected_prefill,
            )
        if self._is_talker_model and schedule_batch.forward_mode.is_decode():
            self._log_talker_decode_debug(
                scheduler_output=scheduler_output,
                forward_batch=forward_batch,
                feedback_input_embeds=feedback_input_embeds,
            )
            self._dump_talker_decode_layer0_qk_debug(
                scheduler_output=scheduler_output,
                forward_batch=forward_batch,
                feedback_input_embeds=feedback_input_embeds,
            )

        if omni_embeds is not None and omni_embeds[0] is not None:
            input_embeds, ds_embeds, vis_masks = omni_embeds
            batch_result = self._forward_with_omni_embeds(
                forward_batch, input_embeds, ds_embeds, vis_masks
            )
        elif projected_prefill:
            projected_input_embeds = forward_batch.input_embeds
            if projected_input_embeds is None:
                projected_input_embeds = request_prefill_input_embeds
            if projected_input_embeds is None:
                raise RuntimeError(
                    "Projected talker prefill requested without input_embeds"
                )
            batch_result = self._forward_talker(
                forward_batch,
                input_embeds=projected_input_embeds,
                input_embeds_are_projected=True,
            )
        elif feedback_input_embeds is not None:
            self._set_talker_layer0_attn_debug_context(
                scheduler_output=scheduler_output,
                forward_batch=forward_batch,
            )
            self._set_talker_disable_fused_set_kv(True)
            try:
                batch_result = self._forward_talker(
                    forward_batch,
                    input_embeds=feedback_input_embeds,
                    input_embeds_are_projected=True,
                )
            finally:
                self._set_talker_disable_fused_set_kv(False)
                self._clear_talker_layer0_attn_debug_context()
        else:
            batch_result = self.model_worker.forward_batch_generation(forward_batch)

        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )
        else:
            self._apply_codec_suppress_tokens(
                batch_result.logits_output, scheduler_output.requests
            )
            if projected_prefill:
                self._log_talker_prefill_topk(
                    scheduler_output=scheduler_output,
                    logits_output=batch_result.logits_output,
                )
            batch_result.next_token_ids = self.model_worker.model_runner.sample(
                batch_result.logits_output, forward_batch
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        if self.batch_planner is not None:
            self.batch_planner.record_last_batch(schedule_batch)

        outputs = self.output_processor.process(batch_result, scheduler_output)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
