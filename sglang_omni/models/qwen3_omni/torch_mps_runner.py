# SPDX-License-Identifier: Apache-2.0
"""Scheduler model runners for the eager Torch MPS Qwen3-Omni thinker and talker.

The Task 2 external-forward worker gives the scheduler real request/KV
bookkeeping while holding SGLang's zero-weight stub model; these runners own
every real forward. That has three consequences the code below makes explicit:

* **No SGLang forward machinery.** ``_build_forward_batch`` returns the schedule
  batch as-is. It does not select a device (``torch.mps.set_device`` is not the
  CUDA-style selector the base runner assumes), does not initialise an attention
  backend, and never builds a ``ForwardBatch``: the stub worker has no attention
  state from which one could be constructed.
* **No SGLang sampler or CUDA graph.** The stage runs under the Apple profile
  (one request, greedy, no radix cache, no chunked prefill), so the next token is
  a plain ``argmax`` and ``can_run_cuda_graph`` is always ``False``.
* **Runner-owned KV cache and captures.** Transformers' own ``past_key_values``
  is kept per request and released on completion and on abort. Hidden captures
  are attached to ``logits_output.hidden_states`` as a ``{"embed": ..., N: ...}``
  dictionary, which is the sole source for ``SGLangOutputProcessor`` (constructed
  with ``model=None`` on this backend, exactly as on MLX).

Capture convention, identical to the MLX and CUDA paths: requested layer ``0``
becomes ``"embed"`` from ``hidden_states[0]`` (the input of layer 0) and a
requested layer ``N > 0`` is ``hidden_states[N]`` (the input of layer ``N``), not
``N + 1``. Only the batch dimension is removed, so a prefill keeps every prompt
row and the existing stream normaliser selects the same first prompt row as CUDA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_omni.components.apple_adapter import (
    emit_talker_step,
    projected_prefill_rows,
    release_talker_host_queues,
    require_single_request,
    validate_capture_layers,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.torch_mps import (
    build_deepstack_visual_inputs,
    build_suppress_mask,
    load_torch_mps_talker,
    load_torch_mps_thinker,
    merge_multimodal_rows,
    restore_placeholder_token_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Qwen3OmniTorchMpsTalkerRunner",
    "Qwen3OmniTorchMpsThinkerRunner",
    "TorchMpsTalkerState",
    "build_qwen3_omni_torch_mps_talker_runner",
    "build_qwen3_omni_torch_mps_thinker_runner",
]

_MODALITIES = ("image", "video", "audio")
_CPU = torch.device("cpu")


class Qwen3OmniTorchMpsThinkerRunner(ModelRunner):
    """Run one Qwen3-Omni thinker request eagerly through Torch on MPS."""

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        *,
        thinker: Any,
        thinker_config: Any,
        capture_hidden_layers: tuple[int, ...] | list[int] | None = None,
        accept_hidden_layer: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._thinker = thinker
        self._device = torch.device(device) if device is not None else self.device
        self._past_key_values: dict[str, Any] = {}
        self._mrope_deltas: dict[str, int] = {}
        self._capture_layers = self._validate_capture_layers(
            capture_hidden_layers, accept_hidden_layer=accept_hidden_layer
        )
        self._placeholder_token_ids = {
            modality: int(getattr(thinker_config, f"{modality}_token_id"))
            for modality in _MODALITIES
        }
        self._vocab_size = int(thinker_config.text_config.vocab_size)

    # -- configuration -----------------------------------------------------

    @staticmethod
    def _validate_capture_layers(
        capture_hidden_layers: tuple[int, ...] | list[int] | None,
        *,
        accept_hidden_layer: int | None,
    ) -> tuple[int, ...]:
        return validate_capture_layers(
            capture_hidden_layers,
            accept_hidden_layer=accept_hidden_layer,
        )

    # -- scheduler contract ------------------------------------------------

    def lookahead_eligible(self, batch: Any) -> bool:
        """Eager Torch MPS decodes synchronously; there is nothing to overlap."""

        del batch
        return False

    def _build_forward_batch(self, scheduler_output: Any):
        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None
        # This runner owns the forward, so there is no ForwardBatch to build and
        # no device to select: MPS exposes a single process-global Metal device.
        return None, schedule_batch, bool(schedule_batch.forward_mode.is_extend())

    @staticmethod
    def _one_request(requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                "Apple Qwen3-Omni Torch MPS thinker serves one request at a time, "
                f"got {len(requests)}"
            )
        return requests[0]

    # -- prefill inputs ----------------------------------------------------

    def _prefill_positions(
        self, req: Any, length: int, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """``[3, 1, sequence]`` M-RoPE rows plus the decode position delta."""

        multimodal_inputs = getattr(req, "multimodal_inputs", None)
        positions = getattr(multimodal_inputs, "mrope_positions", None)
        if positions is None:
            rows = (
                torch.arange(length, dtype=torch.long, device=device)
                .view(1, 1, length)
                .expand(3, 1, length)
                .contiguous()
            )
            return rows, 0
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                "Qwen3-Omni M-RoPE positions must be [3, sequence], got "
                f"{tuple(positions.shape)}"
            )
        if int(positions.shape[1]) != length:
            raise ValueError(
                f"Qwen3-Omni M-RoPE positions cover {int(positions.shape[1])} tokens "
                f"but the prefill holds {length}"
            )
        delta_tensor = getattr(multimodal_inputs, "mrope_position_delta", None)
        delta = (
            0
            if delta_tensor is None
            else int(torch.as_tensor(delta_tensor).reshape(-1)[0])
        )
        rows = positions.to(device=device, dtype=torch.long).unsqueeze(1)
        return rows, delta

    def _modality_rows(
        self, req: Any, model_inputs: dict[str, Any]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        positions = getattr(req, "_omni_mm_positions", None) or {}
        rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality in _MODALITIES:
            embeds = model_inputs.get(f"{modality}_embeds")
            if embeds is None:
                continue
            if isinstance(embeds, (list, tuple)):
                embeds = torch.cat([part for part in embeds], dim=0)
            rows[modality] = (positions.get(modality), embeds)
        return rows

    def _build_prefill_inputs(
        self, sched_req: Any, schedule_batch: Any
    ) -> dict[str, Any]:
        req = sched_req.data.req
        token_ids = [int(token) for token in schedule_batch.input_ids.tolist()]
        prompt_length = len(req.origin_input_ids)
        if len(token_ids) != prompt_length:
            raise NotImplementedError(
                "Apple Qwen3-Omni Torch MPS thinker runs with chunked prefill and "
                f"the radix cache disabled; got {len(token_ids)} of {prompt_length} "
                "prompt tokens"
            )

        model_inputs = getattr(req, "omni_model_inputs", None) or {}
        restored = restore_placeholder_token_ids(
            token_ids,
            positions=getattr(req, "_omni_mm_positions", None),
            pad_values=model_inputs.get("pad_values"),
            placeholder_token_ids=self._placeholder_token_ids,
            vocab_size=self._vocab_size,
        )
        input_ids = torch.tensor([restored], dtype=torch.long, device=self._device)
        inputs_embeds = self._thinker.embed_tokens(input_ids)

        modality_rows = self._modality_rows(req, model_inputs)
        if modality_rows:
            inputs_embeds = merge_multimodal_rows(
                inputs_embeds, modality_rows=modality_rows
            )

        mm_positions = getattr(req, "_omni_mm_positions", None) or {}
        deepstack, visual_pos_masks = build_deepstack_visual_inputs(
            sequence_length=len(restored),
            image_positions=mm_positions.get("image"),
            video_positions=mm_positions.get("video"),
            image_layers=model_inputs.get("image_deepstack_visual_embeds"),
            video_layers=model_inputs.get("video_deepstack_visual_embeds"),
            merged_layers=model_inputs.get("deepstack_visual_embeds"),
        )

        positions, delta = self._prefill_positions(req, len(restored), self._device)
        self._mrope_deltas[sched_req.request_id] = delta
        return {
            "inputs_embeds": inputs_embeds,
            "position_ids": positions,
            "deepstack_visual_embeds": deepstack,
            "visual_pos_masks": visual_pos_masks,
        }

    def _decode_positions(self, request_id: str, past_key_values: Any) -> torch.Tensor:
        """M-RoPE row for the next token.

        SGLang derives a decode position as ``seq_len - 1 + mrope_position_delta``
        on all three axes; the cache length is exactly that ``seq_len - 1`` (the
        tokens already resident).
        """

        position = int(past_key_values.get_seq_length()) + int(
            self._mrope_deltas.get(request_id, 0)
        )
        return torch.full((3, 1, 1), position, dtype=torch.long, device=self._device)

    # -- hidden capture ----------------------------------------------------

    def _wants_speech(self, sched_req: Any) -> bool:
        from sglang_omni.models.qwen3_omni.request_builders import (
            should_generate_audio_output,
        )

        if not self._capture_layers:
            return False
        return bool(should_generate_audio_output(sched_req.data.stage_payload))

    def _capture_hidden(
        self,
        hidden_states: Any,
        *,
        wants_speech: bool,
    ) -> dict[Any, torch.Tensor] | None:
        """Materialise the requested capture rows as CPU tensors.

        Requested layer ``0`` maps to ``"embed"`` (``hidden_states[0]``) and a
        requested layer ``N > 0`` maps to ``hidden_states[N]``, matching the
        repository's layer-input capture convention. Only the batch dimension is
        removed; every prompt row is retained.
        """

        if not wants_speech or not self._capture_layers or hidden_states is None:
            return None
        captured: dict[Any, torch.Tensor] = {}
        for layer in self._capture_layers:
            if layer >= len(hidden_states):
                raise ValueError(
                    f"requested thinker capture layer {layer} is outside the "
                    f"{len(hidden_states)} captured hidden states"
                )
            key = "embed" if layer == 0 else layer
            captured[key] = hidden_states[layer][0].detach().to(_CPU)
        return captured

    def _generation_result(
        self,
        *,
        logits: torch.Tensor,
        hidden_states: dict[Any, torch.Tensor] | None,
    ) -> Any:
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.scheduler import GenerationBatchResult

        next_token_ids = logits.argmax(dim=-1).reshape(-1).to(device=_CPU)
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=hidden_states,
            ),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    # -- forward -----------------------------------------------------------

    @torch.inference_mode()
    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch
        sched_req = self._one_request(requests)
        wants_speech = self._wants_speech(sched_req)
        inputs = self._build_prefill_inputs(sched_req, schedule_batch)
        output = self._thinker(
            **inputs,
            use_cache=True,
            output_hidden_states=wants_speech,
        )
        self._past_key_values[sched_req.request_id] = output.past_key_values
        self._release_prefill_inputs(sched_req)
        return self._generation_result(
            logits=output.logits[:, -1, :],
            hidden_states=self._capture_hidden(
                output.hidden_states, wants_speech=wants_speech
            ),
        )

    @torch.inference_mode()
    def custom_decode_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch
        sched_req = self._one_request(requests)
        request_id = sched_req.request_id
        try:
            past_key_values = self._past_key_values[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Qwen3-Omni Torch MPS decode has no cache for {request_id!r}"
            ) from exc

        wants_speech = self._wants_speech(sched_req)
        output = self._thinker(
            input_ids=schedule_batch.input_ids.reshape(1, 1).to(
                device=self._device, dtype=torch.long
            ),
            position_ids=self._decode_positions(request_id, past_key_values),
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=wants_speech,
        )
        self._past_key_values[request_id] = output.past_key_values
        return self._generation_result(
            logits=output.logits[:, -1, :],
            hidden_states=self._capture_hidden(
                output.hidden_states, wants_speech=wants_speech
            ),
        )

    @staticmethod
    def _release_prefill_inputs(sched_req: Any) -> None:
        """Drop the encoder rows once they have been merged into the prompt."""

        req = sched_req.data.req
        if getattr(req, "inflight_middle_chunks", 0):
            return
        req.omni_model_inputs = None
        req._omni_consumed = None

    # -- per-request lifecycle --------------------------------------------

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Normal completion: drop this request's Transformers KV cache."""

        del req_data
        self._release_request(request_id)

    def abort_request(self, request_id: str) -> None:
        """Scheduler abort callback: drop this request's Transformers KV cache."""

        self._release_request(request_id)

    def _release_request(self, request_id: str) -> None:
        self._past_key_values.pop(request_id, None)
        self._mrope_deltas.pop(request_id, None)

    def _finalize(
        self,
        batch_result,
        forward_batch,
        schedule_batch,
        scheduler_output,
        skip_rids: set[str] | None = None,
    ):
        logits_output = getattr(batch_result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        output = super()._finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
            skip_rids=skip_rids,
        )
        if hidden is not None and logits_output is not None:
            # Nothing downstream may consume the captures destructively: the
            # thinker stream builder reads them again off the request output.
            logits_output.hidden_states = hidden
        return output


def build_qwen3_omni_torch_mps_thinker_runner(
    *,
    tp_worker: Any,
    output_processor: Any,
    model_path: str,
    thinker_config: Any,
    capture_hidden_layers: tuple[int, ...] | list[int] | None = None,
    accept_hidden_layer: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    thinker: Any = None,
) -> Qwen3OmniTorchMpsThinkerRunner:
    """Load the eager Torch thinker and wrap it in its scheduler runner."""

    from sglang_omni.platforms import current_platform

    target_device = device or current_platform.get_device(tp_worker.gpu_id)
    if thinker is None:
        thinker = load_torch_mps_thinker(
            model_path,
            dtype=dtype or torch.bfloat16,
            device=target_device,
        )
    return Qwen3OmniTorchMpsThinkerRunner(
        tp_worker,
        output_processor,
        thinker=thinker,
        thinker_config=thinker_config,
        capture_hidden_layers=capture_hidden_layers,
        accept_hidden_layer=accept_hidden_layer,
        device=target_device,
    )


# ---------------------------------------------------------------------------
# Talker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TorchMpsTalkerState:
    """Everything one talker request owns on the runner.

    ``past_key_values`` is the *outer* talker cache -- the only tensor state that
    survives a step; the code predictor allocates and drops its own cache inside
    every expansion. ``generated_steps`` and ``mrope_delta`` are the rope
    bookkeeping, and ``codes``/``feedback`` stage the step the forward computed
    until ``post_prefill``/``post_decode`` emits it.
    """

    past_key_values: Any = None
    generated_steps: int = 0
    mrope_delta: int = 0
    suppress_mask: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    feedback: torch.Tensor | None = None


class Qwen3OmniTorchMpsTalkerRunner(ModelRunner):
    """Run one Qwen3-Omni talker request eagerly through Torch on MPS.

    Ownership split, identical to the reviewed MLX talker runner:

    * **Device side (this runner).** One outer talker KV cache per request, the
      per-request codec suppression mask, and the M-RoPE row for each step. The
      code equations -- greedy layer-0 selection after suppression, the residual
      RVQ group expansion, and the summed feedback row -- live in
      :class:`~sglang_omni.models.qwen3_omni.torch_mps.Qwen3OmniTorchMpsTalker`.
    * **Host side (unchanged).** The pending text FIFO and the feedback deque
      stay CPU float32 and are consumed through ``QwenTalkerModelRunner``'s own
      helpers, so decode readiness, thinker-done padding, and row ownership are
      the *same* code as the CUDA path. Exactly one feedback row and one
      text/thinker row cross to Metal per decode.

    Apple policy: batch-1, greedy, no chunked prefill, no radix prefix, and no
    async decode lookahead.
    """

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        talker: Any,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        if talker is None:
            raise ValueError(
                "Qwen3OmniTorchMpsTalkerRunner requires a loaded Torch MPS talker"
            )
        self._outbox = outbox
        self._talker = talker
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)
        self._device = torch.device(device) if device is not None else talker.device
        self._states: dict[str, TorchMpsTalkerState] = {}
        # Diagnostics the unit tests read to pin position progression.
        self.last_positions: torch.Tensor | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def talker(self) -> Any:
        return self._talker

    @property
    def num_code_groups(self) -> int:
        return int(self._talker.num_code_groups)

    @property
    def codec_vocab_size(self) -> int:
        return int(self._talker.codec_vocab_size)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._states

    # -- lifecycle ---------------------------------------------------------

    def abort_request(self, request_id: str) -> None:
        """Scheduler abort callback: drop every per-request resource."""

        self._release_request(request_id)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Normal completion: drop every per-request resource."""

        self._release_request(request_id, req_data)

    def _release_request(self, request_id: str, req_data: Any = None) -> None:
        self._states.pop(request_id, None)
        if req_data is None:
            return
        release_talker_host_queues(req_data)

    def clear(self) -> None:
        self._states.clear()

    # -- no async lookahead ------------------------------------------------

    def lookahead_eligible(self, batch: Any) -> bool:
        """The Apple talker never runs a speculative decode step.

        A lookahead launch would have to sample -- and therefore emit a code
        frame and queue a feedback row -- before its predecessor was resolved,
        which the single feedback/text row-per-step contract cannot express.
        """

        del batch
        return False

    def execute_launch(self, scheduler_output: Any):
        raise NotImplementedError(
            "Apple Qwen3-Omni Torch MPS talker does not support async decode "
            "lookahead; every step is resolved synchronously"
        )

    def execute_resolve(self, pending: Any):
        if pending is None:
            return None
        raise NotImplementedError(
            "Apple Qwen3-Omni Torch MPS talker does not support async decode "
            "lookahead; every step is resolved synchronously"
        )

    # -- SGLang execution contract ----------------------------------------

    def _build_forward_batch(self, scheduler_output: Any):
        """No ``ForwardBatch``: this runner consumes talker embeddings directly.

        The bookkeeping stub carries no Torch attention backend state, and MPS
        exposes a single process-global Metal device, so there is neither a batch
        to build nor a device to select.
        """

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None
        return None, schedule_batch, bool(schedule_batch.forward_mode.is_extend())

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def is_decode_batch_ready(self, schedule_batch: Any) -> bool:
        """Identical readiness rule to the CUDA and MLX talker runners."""

        if not self._feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        return all(
            QwenTalkerModelRunner._data_has_next_decode_input(
                getattr(req, "_omni_data", None)
            )
            for req in schedule_batch.reqs
        )

    # -- forward -----------------------------------------------------------

    @torch.inference_mode()
    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch, schedule_batch
        sched_req = self._single_request(requests)
        request_id = sched_req.request_id
        if request_id in self._states:
            raise RuntimeError(
                "Apple Qwen3-Omni Torch MPS talker does not support re-prefilling "
                f"a resident request ({request_id!r})"
            )
        rows = self._prefill_rows(sched_req)
        state = TorchMpsTalkerState()
        positions = self._prefill_positions(sched_req, int(rows.shape[0]), state)
        step = self._talker.step(
            rows,
            mrope_positions=positions,
            past_key_values=None,
            suppress_tokens=self._suppress_mask(sched_req, state),
        )
        # Registered only once the forward succeeded, so a failed prefill leaves
        # no half-initialised cache behind for a later decode to find.
        result = self._record_step(state, step, positions)
        self._states[request_id] = state
        return result

    @torch.inference_mode()
    def custom_decode_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any:
        del forward_batch, schedule_batch
        sched_req = self._single_request(requests)
        request_id = sched_req.request_id
        state = self._states.get(request_id)
        if state is None or state.past_key_values is None:
            raise RuntimeError(
                "Apple Qwen3-Omni Torch MPS talker decode has no prefilled cache "
                f"for {request_id!r}"
            )
        row = self._take_next_decode_row(sched_req)
        if row is None:
            raise RuntimeError(
                "Torch MPS talker decode requires feedback and text input; the "
                "scheduler must defer the batch until both rows are ready"
            )
        positions = self._decode_positions(state)
        step = self._talker.step(
            row,
            mrope_positions=positions,
            past_key_values=state.past_key_values,
            suppress_tokens=self._suppress_mask(sched_req, state),
        )
        return self._record_step(state, step, positions)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        self._emit_codes_and_queue_feedback(
            schedule_batch=schedule_batch, requests=requests
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        self._emit_codes_and_queue_feedback(
            schedule_batch=schedule_batch, requests=requests
        )

    # -- step bookkeeping --------------------------------------------------

    def _record_step(
        self, state: TorchMpsTalkerState, step: Any, positions: torch.Tensor
    ) -> Any:
        """Materialise one device step into CPU Torch and stage it for emission.

        Only the emitted codes, the feedback row, and the published token leave
        the Metal device; the outer cache and the hidden row stay on it.
        """

        from sglang.srt.managers.utils import GenerationBatchResult

        codes = step.codes.detach().reshape(-1).to(device=_CPU, dtype=torch.long)
        if int(codes.numel()) != self.num_code_groups:
            raise RuntimeError(
                f"Torch MPS talker produced {int(codes.numel())} codes but the "
                f"config declares {self.num_code_groups} code groups"
            )
        state.past_key_values = step.past_key_values
        state.generated_steps += 1
        state.codes = codes
        state.feedback = (
            step.feedback.detach().reshape(-1).to(device=_CPU, dtype=torch.float32)
        )
        self.last_positions = positions
        return GenerationBatchResult(
            next_token_ids=codes[:1].clone(),
            can_run_cuda_graph=False,
        )

    def _emit_codes_and_queue_feedback(
        self, *, schedule_batch: Any, requests: list
    ) -> None:
        """Emit this step's code row once and queue its feedback row.

        Matches ``QwenTalkerModelRunner._emit_code_chunks_and_feedback``: one
        ``OutgoingMessage(target="code2wav", type="stream")`` per step carrying
        ``[num_code_groups]`` codes, and one feedback row appended to the
        request's pending queue. The codec-EOS frame is emitted too -- exactly as
        the CUDA and MLX paths do, and code2wav drops it -- so a terminal step
        neither duplicates nor silently swallows a frame.
        """

        if not self._feedback_enabled:
            for sched_req in requests:
                state = self._states.get(sched_req.request_id)
                if state is not None:
                    state.codes = state.feedback = None
            return
        for index, sched_req in enumerate(requests):
            state = self._states.get(sched_req.request_id)
            if state is None or state.codes is None or state.feedback is None:
                raise RuntimeError(
                    "Torch MPS talker has no computed step to emit for "
                    f"{sched_req.request_id!r}"
                )
            codes, feedback = state.codes, state.feedback
            state.codes = state.feedback = None
            emit_talker_step(
                outbox=self._outbox,
                target=self._code2wav_target,
                request_id=schedule_batch.reqs[index].rid,
                data=sched_req.data,
                codes=codes,
                feedback=feedback,
            )

    # -- inputs ------------------------------------------------------------

    @staticmethod
    def _single_request(requests: list) -> Any:
        return require_single_request(requests, backend_name="Torch MPS talker")

    @staticmethod
    def _prefill_rows(sched_req: Any) -> torch.Tensor:
        """The already-projected CPU float32 prompt rows for this prefill."""

        return projected_prefill_rows(sched_req, backend_name="Torch MPS talker")

    @staticmethod
    def _take_next_decode_row(sched_req: Any) -> torch.Tensor | None:
        """Consume exactly one feedback row and one text row, FIFO.

        This is ``QwenTalkerModelRunner``'s own helper, not a copy of it: the
        readiness rule, the thinker-done TTS pad fallback, the CPU float32
        ownership assertion (``_decode_row``), and the pop order are the CUDA
        path's. The summed row is also recorded in the request's replay history,
        exactly as ``_write_feedback_buffers`` does.
        """

        combined = QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=sched_req,
            device=_CPU,
            dtype=torch.float32,
        )
        if combined is None:
            return None
        QwenTalkerModelRunner._append_decode_input_history(sched_req.data, combined)
        return combined

    # -- positions ---------------------------------------------------------

    def _prefill_positions(
        self, sched_req: Any, length: int, state: TorchMpsTalkerState
    ) -> torch.Tensor:
        """The ``[3, 1, length]`` M-RoPE rows for the talker prompt.

        The request builder attaches the rows it computed (``linear_mrope_positions``
        for a prompt with no emitted multimodal segment, the full multimodal
        computation otherwise); a request built without ``talker_model_inputs``
        falls back to the same shared linear helper rather than a re-derivation.
        """

        from sglang_omni.models.qwen3_omni.mrope_positions import linear_mrope_positions

        multimodal_inputs = getattr(sched_req.data.req, "multimodal_inputs", None)
        positions = getattr(multimodal_inputs, "mrope_positions", None)
        if positions is None:
            rows, delta = linear_mrope_positions(length)
        else:
            rows = torch.as_tensor(positions)
            if rows.ndim != 2 or int(rows.shape[0]) != 3:
                raise ValueError(
                    "Qwen3-Omni talker M-RoPE positions must be [3, sequence], "
                    f"got {tuple(rows.shape)}"
                )
            if int(rows.shape[1]) != length:
                raise ValueError(
                    f"Qwen3-Omni talker M-RoPE positions cover {int(rows.shape[1])} "
                    f"tokens but the prefill holds {length}"
                )
            delta = getattr(multimodal_inputs, "mrope_position_delta", None)
        state.mrope_delta = (
            0 if delta is None else int(torch.as_tensor(delta).reshape(-1)[0])
        )
        return rows.detach().to(device=self._device, dtype=torch.long).unsqueeze(1)

    def _decode_positions(self, state: TorchMpsTalkerState) -> torch.Tensor:
        """The ``[3, 1, 1]`` M-RoPE row for the next talker token.

        SGLang derives a decode position as ``seq_len - 1 + mrope_position_delta``
        on all three axes; the cache length is exactly that ``seq_len - 1`` (the
        rows already resident in the talker cache).
        """

        position = int(state.past_key_values.get_seq_length()) + int(state.mrope_delta)
        return torch.full((3, 1, 1), position, dtype=torch.long, device=self._device)

    # -- suppression -------------------------------------------------------

    def _suppress_mask(
        self, sched_req: Any, state: TorchMpsTalkerState
    ) -> torch.Tensor | None:
        """The request's additive codec suppression mask, built once."""

        if state.suppress_mask is not None:
            return state.suppress_mask
        suppress_tokens = getattr(sched_req.data, "suppress_tokens", None)
        if not suppress_tokens:
            return None
        state.suppress_mask = build_suppress_mask(
            self.codec_vocab_size,
            tuple(suppress_tokens),
            device=self._device,
            dtype=self._talker.dtype,
        )
        return state.suppress_mask


def build_qwen3_omni_torch_mps_talker_runner(
    *,
    tp_worker: Any,
    output_processor: Any,
    outbox: Any,
    talker: Any = None,
    model_path: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    code2wav_target: str = "code2wav",
    feedback_enabled: bool = True,
) -> Qwen3OmniTorchMpsTalkerRunner:
    """Wrap an eager Torch talker (loading it when needed) in its runner."""

    from sglang_omni.platforms import current_platform

    target_device = device or current_platform.get_device(tp_worker.gpu_id)
    if talker is None:
        if model_path is None:
            raise ValueError(
                "build_qwen3_omni_torch_mps_talker_runner needs either a loaded "
                "talker or a model_path to load one from"
            )
        talker = load_torch_mps_talker(
            model_path,
            dtype=dtype or torch.bfloat16,
            device=target_device,
        )
    return Qwen3OmniTorchMpsTalkerRunner(
        tp_worker,
        output_processor,
        outbox,
        talker=talker,
        code2wav_target=code2wav_target,
        feedback_enabled=feedback_enabled,
        device=target_device,
    )
