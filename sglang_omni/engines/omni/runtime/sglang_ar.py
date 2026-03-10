# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass
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

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
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

        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )

        if schedule_batch is None and self.decode_manager.runnable:
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)

        if schedule_batch is None:
            self._cached_schedule_batch = None
            return []

        self._cached_schedule_batch = schedule_batch
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

        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return self._cached_schedule_batch

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

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
            )
        return outputs


class SGLangIterationController:
    def __init__(self, tree_cache):
        self.tree_cache = tree_cache

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: SGLangARRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        token_id = output.data
        if token_id is not None:
            req.output_ids.append(token_id)
            req.check_finished()
            if not req.finished() and req.decode_batch_idx == 0:
                self.tree_cache.cache_unfinished_req(req)

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

        model = model_worker.model_runner.model
        self._embed_tokens, self._inner_model = self._get_inner_model_components(model)

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
        return outer.model.embed_tokens, outer

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

            visual_pad_value = omni_inputs.get("_visual_pad_value")

            for modality, token_id in [
                ("image", image_token_id),
                ("video", video_token_id),
                ("audio", audio_token_id),
            ]:
                embeds = omni_inputs.get(f"{modality}_embeds")
                if embeds is None:
                    continue
                detect_id = (
                    visual_pad_value if visual_pad_value is not None else token_id
                )
                mask = req_input_ids == detect_id
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
                if visual_pad_value is not None:
                    visual_mask = req_input_ids == visual_pad_value
                    img_mask = visual_mask
                    vid_mask = visual_mask
                else:
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

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data

        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        omni_embeds = None
        if schedule_batch.forward_mode.is_extend():
            omni_embeds = self._inject_multimodal_embeds(forward_batch, schedule_batch)

        if omni_embeds is not None and omni_embeds[0] is not None:
            input_embeds, ds_embeds, vis_masks = omni_embeds
            batch_result = self._forward_with_omni_embeds(
                forward_batch, input_embeds, ds_embeds, vis_masks
            )
        else:
            batch_result = self.model_worker.forward_batch_generation(forward_batch)

        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )
        else:
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
