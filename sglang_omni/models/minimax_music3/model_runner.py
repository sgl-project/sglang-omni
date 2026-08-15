# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for the MiniMax Music 3 AR stage."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.sampling.seed import derive_sampling_seed

from .chunking import ChunkWindow, chunk_windows
from .constants import AR_CHUNK_FRAMES, AR_CHUNK_HOP_FRAMES
from .rvq_decoder import sample_topk_seeded
from .sglang_model import apply_cfg, depth_decode, embed_audio_frames, select_c0_logits

logger = logging.getLogger(__name__)

_PROGRESS_LOG_DIVISOR = 10
_SAMPLING_NAMESPACE = "minimax-ttm-ar"
_FORCED_CODES_DIR_ENV = "MINIMAX_MUSIC3_FORCED_CODES"
_HIDDEN_DUMP_DIR_ENV = "MINIMAX_MUSIC3_HIDDEN_DUMP"


class _HiddenFrameBuffer:
    """Rolling GPU buffer addressed by absolute AR frame indexes."""

    def __init__(self) -> None:
        self.base_frame = 0
        self.frames: list[torch.Tensor] = []
        self.peak_frames = 0

    @property
    def end_frame(self) -> int:
        return self.base_frame + len(self.frames)

    def append(self, frame_hidden: torch.Tensor) -> None:
        self.frames.append(frame_hidden.unsqueeze(1))
        self.peak_frames = max(self.peak_frames, len(self.frames))

    def concatenate(self, start: int, end: int) -> torch.Tensor:
        relative_start = start - self.base_frame
        relative_end = end - self.base_frame
        if relative_start < 0 or relative_end > len(self.frames):
            raise RuntimeError(
                f"MiniMax Music 3 hidden buffer range [{start}, {end}) is unavailable; "
                f"the buffer holds [{self.base_frame}, {self.end_frame})"
            )
        return torch.cat(self.frames[relative_start:relative_end], dim=1)

    def discard_before(self, frame: int) -> None:
        discard = min(max(frame - self.base_frame, 0), len(self.frames))
        if discard:
            del self.frames[:discard]
            self.base_frame += discard


@dataclass
class _ARState:
    """Per-request AR state owned by the runner for one generation."""

    sampling_seed: int
    seed: int
    decode_limit: int
    sampling_position: int = 0
    forced_codes: torch.Tensor | None = None
    forced_step: int = 0
    reference_ranks: list[torch.Tensor] = field(default_factory=list)
    frames: _HiddenFrameBuffer = field(default_factory=_HiddenFrameBuffer)
    feedback_codes: torch.Tensor | None = None
    emitted_windows: int = 0
    generated_frames: int = 0
    finish_reason: str = "length"
    started_s: float = 0.0
    pending_chunks: list[tuple[torch.Tensor, dict[str, Any]]] = field(
        default_factory=list
    )


class MiniMaxMusic3ModelRunner(ModelRunner):
    """Own c0 sampling, RVQ depth decode and FM8 chunk streaming."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._request_data: dict[str, Any] = {}
        self._dump_dir = os.environ.get(_HIDDEN_DUMP_DIR_ENV) or None
        self._forced_codes_dir = os.environ.get(_FORCED_CODES_DIR_ENV) or None
        if self._forced_codes_dir is not None:
            logger.warning(
                f"MiniMax Music 3 is replaying reference code trajectories from {self._forced_codes_dir}; generated audio is the reference's, not this model's"
            )

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.LAST

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.LAST

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        if not requests:
            return
        device = forward_batch.input_ids.device
        embed_tokens = self.model.get_input_embeddings()
        rows = []
        for request in requests:
            data = request.data
            if not data.is_cfg_uncond:
                self._start_request(request.request_id, data, device)
            prompt_ids = data.prompt_token_ids.to(device=device)
            extend_length = int(data.req.extend_range.length)
            if len(data.req.prefix_indices) or extend_length != prompt_ids.numel():
                raise RuntimeError(
                    "MiniMax Music 3 prefill must cover the whole prompt exactly "
                    f"once (prefix={len(data.req.prefix_indices)} "
                    f"extend={extend_length} prompt={prompt_ids.numel()}); "
                    "radix reuse and retract/replay are not supported"
                )
            rows.append(embed_tokens(prompt_ids))
        forward_batch.input_embeds = torch.cat(rows, dim=0)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)) or not requests:
            return
        self._advance(result, requests, emit=False)

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        codes = torch.cat(
            [self._ar_state(request).feedback_codes for request in requests[0::2]],
            dim=0,
        )
        embeds = embed_audio_frames(self.model, codes).repeat_interleave(2, dim=0)
        buffer = self.model.graph_feedback_buffer
        if buffer is None:
            forward_batch.input_embeds = embeds
            return
        buffer[: embeds.shape[0]].copy_(embeds)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        self._advance(result, requests, emit=True)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        ar_state = req_data.ar_state
        self._request_data.pop(request_id, None)
        if ar_state is None:
            return
        if ar_state.generated_frames == 0:
            raise ValueError(
                f"MiniMax Music 3 generated zero audio frames for request {request_id}"
            )
        for window in chunk_windows(ar_state.generated_frames):
            if window.index >= ar_state.emitted_windows:
                self._emit_window(request_id, ar_state, window, is_final=window.is_last)
        self._dump_reference_ranks(ar_state)
        state = req_data.minimax_state
        state.generated_frames = ar_state.generated_frames
        state.finish_reason = ar_state.finish_reason
        state.prompt_tokens = req_data.prompt_tokens
        state.completion_tokens = ar_state.generated_frames
        logger.info(
            f"MiniMax Music 3 AR done request={request_id} frames={ar_state.generated_frames} prompt_tokens={req_data.prompt_tokens} finish_reason={ar_state.finish_reason} peak_buffer_frames={ar_state.frames.peak_frames} elapsed={time.perf_counter() - ar_state.started_s:.1f}s"
        )

    def reset_request(self, request_id: str) -> None:
        data = self._request_data.pop(request_id, None)
        if data is not None:
            data.ar_state = None

    def _advance(self, result: Any, requests: list, *, emit: bool) -> None:
        """Sample one frame per row and queue whatever windows it completes."""
        model = self.model
        logits_output = result.logits_output
        hidden = logits_output.hidden_states
        logits = logits_output.next_token_logits
        if hidden is None or logits is None:
            raise RuntimeError(
                "MiniMax Music 3 forward returned no hidden states or logits"
            )
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        if hidden.shape[0] != len(requests) or logits.shape[0] != len(requests):
            raise RuntimeError(
                "MiniMax Music 3 expected one row per request: "
                f"hidden={hidden.shape[0]} logits={logits.shape[0]} "
                f"requests={len(requests)}"
            )

        if len(requests) % 2:
            raise RuntimeError(
                "MiniMax Music 3 decodes conditioned and unconditioned rows as "
                f"pairs; this batch has {len(requests)} rows"
            )
        cond_requests = requests[0::2]
        for cond, uncond in zip(cond_requests, requests[1::2], strict=True):
            if cond.data.cfg_uncond is not uncond.data:
                raise RuntimeError(
                    "MiniMax Music 3 CFG rows are not adjacent pairs: "
                    f"{cond.request_id} is followed by {uncond.request_id}"
                )
        ar_states = [self._ar_state(request) for request in cond_requests]
        seeds = torch.tensor(
            [ar_state.sampling_seed for ar_state in ar_states],
            dtype=torch.int64,
            device=hidden.device,
        )
        frame_positions = torch.tensor(
            [ar_state.sampling_position for ar_state in ar_states],
            dtype=torch.int64,
            device=hidden.device,
        )
        for ar_state in ar_states:
            ar_state.sampling_position += model.num_codebooks

        narrowed = select_c0_logits(model, logits)
        c0_logits = apply_cfg(narrowed[0::2], narrowed[1::2])
        sampled = sample_topk_seeded(c0_logits, seeds, frame_positions)
        forced = self._forced_codes_for(ar_states, hidden.device)

        stopped = sampled == 0
        c0 = (sampled - 1).clamp_min(0)
        if forced is not None:
            stopped = torch.zeros_like(stopped)
            c0 = forced[:, 0]
        paired_codes, paired_hidden, paired_ranks = depth_decode(
            model,
            hidden,
            c0.repeat_interleave(2),
            seeds.repeat_interleave(2),
            frame_positions.repeat_interleave(2),
            None if forced is None else forced.repeat_interleave(2, dim=0),
        )
        codes = paired_codes[0::2]
        frame_hidden = torch.cat((hidden[0::2], paired_hidden[0::2]), dim=-1)
        if forced is not None:
            self._record_reference_ranks(
                ar_states, c0_logits, forced, paired_ranks[0::2]
            )

        for index, stop in enumerate(stopped.tolist()):
            ar_state = ar_states[index]
            if stop:
                ar_state.finish_reason = "stop"
                continue
            ar_state.feedback_codes = codes[index : index + 1]
            if not emit:
                continue
            ar_state.frames.append(frame_hidden[index : index + 1])
            ar_state.generated_frames += 1
            self._log_progress(cond_requests[index].request_id, ar_state)
            self._emit_ready_window(cond_requests[index].request_id, ar_state)
        result.next_token_ids = model.c0_logit_ids[sampled.repeat_interleave(2)]

    def _start_request(self, request_id: str, data: Any, device: torch.device) -> None:
        del device
        state = data.minimax_state
        decode_limit = int(state.max_audio_frames)
        data.ar_state = _ARState(
            sampling_seed=derive_sampling_seed(_SAMPLING_NAMESPACE, state.seed),
            seed=state.seed,
            decode_limit=decode_limit,
            started_s=time.perf_counter(),
            forced_codes=self._load_forced_codes(state.seed),
        )
        self._request_data[request_id] = data
        logger.info(
            f"MiniMax Music 3 AR start request={request_id} seed={state.seed} max_frames={decode_limit} prompt_tokens={data.prompt_tokens}"
        )

    @staticmethod
    def _record_reference_ranks(
        ar_states: list[_ARState],
        c0_logits: torch.Tensor,
        forced: torch.Tensor,
        depth_ranks: torch.Tensor,
    ) -> None:
        """Rank every replayed reference code inside this model's own logits.

        The reference drew each code from the top of logits this model should
        reproduce to within a few ULP, so a faithful implementation still ranks
        them near the front. Nothing else checks code selection: the replay
        overwrites the sampled codes, so a wrong id mapping or logit slice would
        leave every latent untouched and slip through silently.
        """
        chosen = c0_logits.gather(1, (forced[:, 0] + 1).unsqueeze(1))
        c0_ranks = (c0_logits > chosen).sum(dim=1, keepdim=True)
        ranks = torch.cat((c0_ranks, depth_ranks), dim=1)
        for index, ar_state in enumerate(ar_states):
            ar_state.reference_ranks.append(ranks[index])

    def _load_forced_codes(self, seed: int) -> torch.Tensor | None:
        """Reference codes for this request seed, or None when not replaying."""
        if self._forced_codes_dir is None:
            return None
        path = Path(self._forced_codes_dir) / f"codes_seed{int(seed)}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"MiniMax Music 3 reference replay has no trajectory for seed {seed}: "
                f"{path}"
            )
        return torch.load(path, weights_only=True)

    def _forced_codes_for(
        self, ar_states: list[_ARState], device: torch.device
    ) -> torch.Tensor | None:
        """Stack this step's reference codes, one row per request."""
        if self._forced_codes_dir is None:
            return None
        rows = []
        for ar_state in ar_states:
            trajectory = ar_state.forced_codes
            if trajectory is None or ar_state.forced_step >= trajectory.shape[0]:
                raise RuntimeError(
                    "MiniMax Music 3 reference replay ran past the trajectory at step "
                    f"{ar_state.forced_step}"
                )
            rows.append(trajectory[ar_state.forced_step])
            ar_state.forced_step += 1
        return torch.stack(rows).to(device=device)

    @staticmethod
    def _ar_state(request: Any) -> _ARState:
        ar_state = request.data.ar_state
        if ar_state is None:
            raise RuntimeError(
                f"MiniMax Music 3 request {request.request_id} has no AR state"
            )
        return ar_state

    def _log_progress(self, request_id: str, ar_state: _ARState) -> None:
        interval = max(
            1, min(250, max(1, ar_state.decode_limit // _PROGRESS_LOG_DIVISOR))
        )
        if ar_state.generated_frames % interval:
            return
        logger.info(
            f"MiniMax Music 3 AR progress request={request_id} frames={ar_state.generated_frames}/{ar_state.decode_limit} elapsed={time.perf_counter() - ar_state.started_s:.1f}s"
        )

    def _emit_ready_window(self, request_id: str, ar_state: _ARState) -> None:
        """Emit the next window once one frame of lookahead proves it is not
        the final one; the tail is flushed in on_request_finished."""
        start = ar_state.emitted_windows * AR_CHUNK_HOP_FRAMES
        if ar_state.frames.end_frame <= start + AR_CHUNK_FRAMES:
            return
        self._emit_window(
            request_id,
            ar_state,
            ChunkWindow(
                index=ar_state.emitted_windows,
                start=start,
                end=start + AR_CHUNK_FRAMES,
                is_first=ar_state.emitted_windows == 0,
                is_last=False,
            ),
            is_final=False,
        )

    def _emit_window(
        self,
        request_id: str,
        ar_state: _ARState,
        window: ChunkWindow,
        *,
        is_final: bool,
    ) -> None:
        chunk = ar_state.frames.concatenate(window.start, window.end)
        self._dump_chunk(chunk, ar_state.seed, window)
        torch.cuda.current_stream(chunk.device).synchronize()
        ar_state.pending_chunks.append(
            (
                chunk,
                {
                    "stream": True,
                    "modality": "ttm_hidden",
                    "chunk_idx": window.index,
                    "start_frame": window.start,
                    "end_frame": window.end,
                    "is_final": is_final,
                    "seed": ar_state.seed,
                },
            )
        )
        ar_state.emitted_windows = max(ar_state.emitted_windows, window.index + 1)
        if not is_final:
            ar_state.frames.discard_before(window.start + AR_CHUNK_HOP_FRAMES)
        logger.info(
            f"MiniMax Music 3 AR emitted request={request_id} chunk={window.index} frames={window.length} frame_range=[{window.start},{window.end}) final={is_final}"
        )

    def _dump_reference_ranks(self, ar_state: _ARState) -> None:
        if self._dump_dir is None or not ar_state.reference_ranks:
            return
        os.makedirs(self._dump_dir, exist_ok=True)
        torch.save(
            torch.stack(ar_state.reference_ranks).cpu(),
            os.path.join(self._dump_dir, f"seed{int(ar_state.seed)}_ranks.pt"),
        )

    def _dump_chunk(self, chunk: torch.Tensor, seed: int, window: ChunkWindow) -> None:
        """Write one window for the latent comparison."""
        if self._dump_dir is None:
            return
        os.makedirs(self._dump_dir, exist_ok=True)
        path = os.path.join(
            self._dump_dir, f"seed{int(seed)}_chunk{window.index:03d}_hidden.pt"
        )
        torch.save(chunk.cpu(), path)


__all__ = ["MiniMaxMusic3ModelRunner"]
