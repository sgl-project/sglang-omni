# SPDX-License-Identifier: Apache-2.0
"""SGLang AR backend integration into OmniEngine's Protocol system.

Provides BatchPlanner, ResourceManager, OutputProcessor, IterationController,
and ModelRunner implementations that delegate to SGLang's PrefillManager,
DecodeManager, and ModelWorker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
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


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class SGLangARRequestData(ARRequestData):
    """SGLang-specific AR request data.

    Subclasses ARRequestData so isinstance(result, ARRequestData) passes
    and all existing field accesses (.output_ids, .extra_model_outputs) work.
    """

    req: Any = None
    synced: bool = False


# -----------------------------------------------------------------------------
# BatchPlanner
# -----------------------------------------------------------------------------


class SGLangBatchPlanner:
    """Batch planner that delegates to SGLang's PrefillManager and DecodeManager.

    Implements the BatchPlanner protocol (select_requests, build_batch).
    """

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
        # Maps SGLang req.rid -> OmniEngine SchedulerRequest
        self.req_id_map: dict[str, SchedulerRequest] = {}
        self._cached_schedule_batch: Any | None = None

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        # 1. Post-step from previous batch
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
        self._prune_inactive_state(active_request_ids)

        # 2. Sync new SchedulerRequests to PrefillManager
        for sched_req in waiting:
            data: SGLangARRequestData = sched_req.data
            if not data.synced:
                self.prefill_manager.add_one_request(data.req)
                data.synced = True
                self.req_id_map[data.req.rid] = sched_req

        # 3. Try prefill first
        running_batch = self.decode_manager.running_batch
        if hasattr(running_batch, "batch_size"):
            running_bs = running_batch.batch_size()
        else:
            running_bs = len(getattr(running_batch, "reqs", []))
        num_allocatable_reqs = max(
            self.server_args.max_running_requests - running_bs, 0
        )

        running_batch_for_prefill = self.decode_manager.running_batch
        if (
            running_batch_for_prefill is not None
            and hasattr(running_batch_for_prefill, "is_empty")
            and running_batch_for_prefill.is_empty()
        ):
            running_batch_for_prefill = None

        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )

        # 4. Fallback to decode
        if schedule_batch is None and self.decode_manager.runnable:
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)

        if schedule_batch is None:
            self._cached_schedule_batch = None
            return []

        self._cached_schedule_batch = schedule_batch
        self.forward_ct += 1

        # Map ScheduleBatch reqs back to SchedulerRequests
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
        """Handle post-step merging from last prefill batch into decode batch."""
        chunked_req_to_exclude = set()
        active_chunked_req = self.prefill_manager.chunked_req
        if active_chunked_req is not None:
            chunked_req_to_exclude.add(active_chunked_req)
            self.prefill_manager.tree_cache.cache_unfinished_req(
                active_chunked_req, chunked=True
            )
            # Reuse req_to_token slot across chunk rounds; otherwise long chunked
            # prompts can exhaust req_to_token_pool entries.
            if getattr(active_chunked_req, "req_pool_idx", None) is not None:
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

        # Filter finished requests from decode running_batch
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
        """Called after model execution to record the last batch for post-step."""
        self.last_batch = schedule_batch

    def _prune_inactive_state(self, active_request_ids: set[str]) -> None:
        """Prune stale request mappings and remove inactive reqs from SGLang queues."""
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


# -----------------------------------------------------------------------------
# ResourceManager
# -----------------------------------------------------------------------------


class SGLangResourceManager:
    """Resource manager that delegates KV cache management to SGLang internals.

    PrefillAdder handles real allocation checks internally, so can_allocate/allocate
    are pass-throughs. KV release happens only in free().
    """

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


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

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


# -----------------------------------------------------------------------------
# IterationController
# -----------------------------------------------------------------------------


class SGLangIterationController:
    """Handles per-request state updates with chunked prefill semantics.

    Chunked prefill: req.is_chunked > 0 means the request is still in
    chunked prefill — decrement counter, do NOT append token or check finish.
    """

    def __init__(self, tree_cache):
        self.tree_cache = tree_cache

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: SGLangARRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            # Suppress stream emission for this chunk. Scheduler emits stream
            # after update_request(), so marking output None here avoids races.
            output.data = None
            req.is_chunked -= 1
            return

        token_id = output.data
        if token_id is not None:
            # output_ids is shared: req.output_ids IS data.output_ids
            req.output_ids.append(token_id)
            req.check_finished()
            # Mirror upstream prefill behavior: unfinished prefill requests
            # should insert/update prefix cache after extend step.
            if not req.finished() and getattr(req, "decode_batch_idx", 0) == 0:
                self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return request.data.req.finished()


# -----------------------------------------------------------------------------
# SGLangModelRunner (duck-typed replacement for ModelRunner)
# -----------------------------------------------------------------------------


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

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> None:
        """Merge image/video/audio embeddings into forward_batch.input_embeds.

        During prefill, reads omni_model_inputs from each Req, computes text
        embeddings via embed_tokens, and scatters multimodal embeddings into
        the correct positions. Also attaches deepstack data to forward_batch.
        """
        # Collect omni_model_inputs from requests that have them
        has_multimodal = False
        for req in schedule_batch.reqs:
            if getattr(req, "omni_model_inputs", None):
                has_multimodal = True
                break

        if not has_multimodal:
            return

        model_runner = self.model_worker.model_runner
        model = model_runner.model
        # Navigate model hierarchy to find embed_tokens:
        # Qwen3OmniMoeForConditionalGeneration.thinker.model.embed_tokens
        # or Qwen3OmniMoeThinkerTextModel.embed_tokens (custom model)
        if hasattr(model, "embed_tokens"):
            embed_tokens = model.embed_tokens
        elif hasattr(model, "thinker") and hasattr(model.thinker, "model"):
            embed_tokens = model.thinker.model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
        else:
            logger.warning(
                "Cannot find embed_tokens on model, skipping multimodal inject"
            )
            return
        device = forward_batch.input_ids.device

        # Get token IDs from SGLang's model_config.hf_config (thinker config level)
        hf_config = model_runner.model_config.hf_config
        image_token_id = getattr(hf_config, "image_token_id", 151655)
        video_token_id = getattr(hf_config, "video_token_id", 151656)
        audio_token_id = getattr(hf_config, "audio_token_id", 151646)

        # Compute text embeddings for the full batch
        input_embeds = embed_tokens(forward_batch.input_ids)

        # Build per-request offset mapping using extend_seq_lens
        # (input_ids only contains extend tokens, not cached prefix)
        extend_lens = forward_batch.extend_seq_lens_cpu
        offsets = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += length

        # Scatter multimodal embeddings and collect deepstack data
        deepstack_visual_embeds_list = []
        visual_pos_masks_list = []
        has_deepstack = False

        for i, req in enumerate(schedule_batch.reqs):
            omni_inputs = getattr(req, "omni_model_inputs", None)
            if omni_inputs is None:
                continue

            start = offsets[i]
            end = start + extend_lens[i]
            req_input_ids = forward_batch.input_ids[start:end]

            # For chunked prefill, we need to know which subset of the
            # full embeddings corresponds to the current chunk's tokens.
            # Track consumed offsets per modality using a counter on the req.
            consumed = getattr(req, "_omni_consumed", None) or {}

            # Scatter image embeddings
            image_embeds = omni_inputs.get("image_embeds")
            if image_embeds is not None:
                mask = req_input_ids == image_token_id
                if mask.any():
                    n_tokens = int(mask.sum().item())
                    offset = consumed.get("image", 0)
                    chunk_embeds = image_embeds[offset : offset + n_tokens].to(
                        device=device, dtype=input_embeds.dtype
                    )
                    global_indices = torch.where(mask)[0] + start
                    input_embeds[global_indices] = chunk_embeds
                    consumed["image"] = offset + n_tokens

            # Scatter video embeddings
            video_embeds = omni_inputs.get("video_embeds")
            if video_embeds is not None:
                mask = req_input_ids == video_token_id
                if mask.any():
                    n_tokens = int(mask.sum().item())
                    offset = consumed.get("video", 0)
                    chunk_embeds = video_embeds[offset : offset + n_tokens].to(
                        device=device, dtype=input_embeds.dtype
                    )
                    global_indices = torch.where(mask)[0] + start
                    input_embeds[global_indices] = chunk_embeds
                    consumed["video"] = offset + n_tokens

            # Scatter audio embeddings
            audio_embeds = omni_inputs.get("audio_embeds")
            if audio_embeds is not None:
                mask = req_input_ids == audio_token_id
                if mask.any():
                    n_tokens = int(mask.sum().item())
                    offset = consumed.get("audio", 0)
                    chunk_embeds = audio_embeds[offset : offset + n_tokens].to(
                        device=device, dtype=input_embeds.dtype
                    )
                    global_indices = torch.where(mask)[0] + start
                    input_embeds[global_indices] = chunk_embeds
                    consumed["audio"] = offset + n_tokens

            req._omni_consumed = consumed

            # Collect deepstack visual embeddings
            ds_embeds = omni_inputs.get("deepstack_visual_embeds")
            image_ds = omni_inputs.get("image_deepstack_visual_embeds")
            video_ds = omni_inputs.get("video_deepstack_visual_embeds")

            if ds_embeds is not None or image_ds is not None or video_ds is not None:
                has_deepstack = True
                # Compute visual_pos_masks for this request
                img_mask = req_input_ids == image_token_id
                vid_mask = req_input_ids == video_token_id
                visual_mask = img_mask | vid_mask

                if ds_embeds is None:
                    if image_ds and video_ds:
                        # Merge image and video deepstack embeds
                        merged = []
                        for img_e, vid_e in zip(image_ds, video_ds):
                            num_visual = int(visual_mask.sum().item())
                            joint = img_e.new_zeros(num_visual, img_e.shape[-1])
                            # Image positions within the visual set
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
                    # Build a global visual_pos_mask (indexed into the batch)
                    global_mask = torch.zeros(
                        len(forward_batch.input_ids),
                        dtype=torch.bool,
                        device=device,
                    )
                    global_mask[start:end] = visual_mask
                    deepstack_visual_embeds_list.append(ds_embeds)
                    visual_pos_masks_list.append(global_mask)

            # Clean up: free model_inputs after final prefill chunk to save memory
            if req.is_chunked == 0:
                req.omni_model_inputs = None
                req._omni_consumed = None

        # Store on a custom attribute — NOT forward_batch.input_embeds, because
        # that triggers SGLang's forward_extend to pass input_embeds as a kwarg
        # to model.forward(), which the upstream Qwen3VL model doesn't accept.
        # The inner Qwen3OmniMoeThinkerTextModel reads this in its forward().
        forward_batch.omni_input_embeds = input_embeds

        # Attach deepstack data to forward_batch for the model to use
        if has_deepstack and deepstack_visual_embeds_list:
            # For single request (common case), attach directly
            # For multi-request, merge visual_pos_masks
            if len(deepstack_visual_embeds_list) == 1:
                forward_batch.omni_deepstack_visual_embeds = (
                    deepstack_visual_embeds_list[0]
                )
                forward_batch.omni_visual_pos_masks = visual_pos_masks_list[0]
            else:
                # Merge: combine visual masks and deepstack embeds
                combined_mask = torch.zeros(
                    len(forward_batch.input_ids), dtype=torch.bool, device=device
                )
                for m in visual_pos_masks_list:
                    combined_mask |= m
                # Merge deepstack embeds per layer
                num_layers = len(deepstack_visual_embeds_list[0])
                merged_ds = []
                for layer_idx in range(num_layers):
                    parts = []
                    for req_ds, req_mask in zip(
                        deepstack_visual_embeds_list, visual_pos_masks_list
                    ):
                        parts.append(
                            req_ds[layer_idx].to(
                                device=device, dtype=input_embeds.dtype
                            )
                        )
                    merged_ds.append(torch.cat(parts, dim=0))
                forward_batch.omni_deepstack_visual_embeds = merged_ds
                forward_batch.omni_visual_pos_masks = combined_mask

    def _forward_with_omni_embeds(self, forward_batch: Any) -> Any:
        """Run forward pass with pre-merged multimodal embeddings.

        Bypasses the outer Qwen3VLForConditionalGeneration.forward() (which
        doesn't accept input_embeds) and calls the inner language model directly.
        """
        from sglang.srt.managers.scheduler import GenerationBatchResult

        model_runner = self.model_worker.model_runner
        model = model_runner.model

        # Navigate to the inner components
        # Qwen3OmniMoeForConditionalGeneration.thinker (Qwen3OmniMoeThinkerForConditionalGeneration)
        #   extends Qwen3VLForConditionalGeneration which has .model, .lm_head, .logits_processor
        outer = model
        if hasattr(model, "thinker"):
            outer = model.thinker
        language_model = outer.model
        lm_head = outer.lm_head
        logits_processor = outer.logits_processor

        # Init attention backend metadata (normally done inside forward_extend)
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        # Compute positions (use mrope if enabled)
        positions = forward_batch.positions
        if getattr(outer, "is_mrope_enabled", False):
            positions = forward_batch.mrope_positions

        # Get pre-merged embeddings
        input_embeds = forward_batch.omni_input_embeds

        # Build deepstack kwarg if present (convert list-of-tensors to concatenated format)
        kwargs = {}
        ds_embeds_list = getattr(forward_batch, "omni_deepstack_visual_embeds", None)
        if ds_embeds_list is not None and isinstance(ds_embeds_list, list):
            # Upstream expects [num_visual_tokens, hidden_size * num_layers]
            ds_on_device = [
                e.to(device=input_embeds.device, dtype=input_embeds.dtype)
                for e in ds_embeds_list
            ]
            kwargs["input_deepstack_embeds"] = torch.cat(ds_on_device, dim=-1)

        # Call inner language model directly with input_embeds
        hidden_states = language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

        # Compute logits
        logits_output = logits_processor(
            forward_batch.input_ids,
            hidden_states,
            lm_head,
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

        # 1. Get ModelWorkerBatch from ScheduleBatch
        model_worker_batch = schedule_batch.get_model_worker_batch()

        # 2. Create ForwardBatch
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        # 2.5. Inject image/video/audio embeddings into input_embeds for prefill
        has_omni_embeds = False
        if schedule_batch.forward_mode.is_extend():
            self._inject_multimodal_embeds(forward_batch, schedule_batch)
            has_omni_embeds = (
                getattr(forward_batch, "omni_input_embeds", None) is not None
            )

        # 3. Forward pass
        if has_omni_embeds:
            # Custom forward: the upstream Qwen3VLForConditionalGeneration.forward()
            # doesn't accept input_embeds kwarg. We bypass it and call the inner
            # language model directly with our pre-merged embeddings.
            batch_result = self._forward_with_omni_embeds(forward_batch)
        else:
            batch_result = self.model_worker.forward_batch_generation(forward_batch)

        # 4. Produce per-request next tokens and store them on ScheduleBatch.
        # Decode preparation relies on schedule_batch.output_ids from the
        # previous step; without this, the next decode step cannot run.
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

        # 5. Record last batch for post-step operations
        if self.batch_planner is not None:
            self.batch_planner.record_last_batch(schedule_batch)

        # 6. Process outputs
        outputs = self.output_processor.process(batch_result, scheduler_output)

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
