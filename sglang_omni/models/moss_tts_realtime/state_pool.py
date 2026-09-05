# SPDX-License-Identifier: Apache-2.0
"""Fixed per-request eager decode state for MOSS-TTS-Realtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.sampling.seed import new_random_sampling_seed

_MAX_WIRE_INT = (1 << 63) - 1
_DEFAULT_MAX_HISTORY_FRAMES = 1000


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if value > _MAX_WIRE_INT:
        raise ValueError(f"{name} exceeds the signed 64-bit range")
    return int(value)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeSamplingParams:
    """Immutable HF sampling contract owned by one state-pool row."""

    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    repetition_penalty: float
    repetition_window: int
    seed: int

    @classmethod
    def from_request_data(cls, data: Any) -> "MossTTSRealtimeSamplingParams":
        try:
            state = data.state
        except AttributeError:
            state = None
        try:
            values = state.generation_kwargs
        except AttributeError:
            values = None
        values = dict(values or {})

        temperature = _require_float(values.get("temperature", 0.8), "temperature")
        top_p = _require_float(values.get("top_p", 0.6), "top_p")
        top_k = _require_int(values.get("top_k", 30), "top_k", minimum=1)
        do_sample = values.get("do_sample", True)
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        repetition_penalty = _require_float(
            values.get("repetition_penalty", 1.1),
            "repetition_penalty",
        )
        repetition_window = _require_int(
            values.get("repetition_window", 50),
            "repetition_window",
            minimum=1,
        )

        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

        explicit_seed = getattr(data, "sampling_seed", None)
        public_seed = values.get("seed")
        if explicit_seed is not None and public_seed is not None:
            if _require_int(explicit_seed, "sampling_seed") != _require_int(
                public_seed, "seed"
            ):
                raise ValueError("sampling_seed does not match generation seed")
        seed = explicit_seed if explicit_seed is not None else public_seed
        if seed is None:
            seed = new_random_sampling_seed()
        seed = _require_int(seed, "sampling_seed")
        try:
            data.sampling_seed = seed
        except AttributeError:
            pass

        return cls(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            seed=seed,
        )


@dataclass(slots=True)
class MossTTSRealtimeDecodeJournal:
    """Step-private frame record kept aligned with one scheduler batch."""

    rids: list[str]
    pool_rows: list[int]
    sample_positions: list[int]
    frames: torch.Tensor
    eos_mask: torch.Tensor
    generator_states_before: tuple[torch.Tensor, ...]
    model_config: Any

    def __post_init__(self) -> None:
        batch_size = len(self.rids)
        if len(self.pool_rows) != batch_size:
            raise ValueError("journal pool row count does not match request count")
        if len(self.sample_positions) != batch_size:
            raise ValueError("journal sample position count does not match requests")
        if len(self.generator_states_before) != batch_size:
            raise ValueError("journal generator state count does not match requests")
        if self.frames.ndim != 2 or tuple(self.frames.shape) != (
            batch_size,
            int(self.model_config.rvq),
        ):
            raise ValueError(
                "journal frames must have shape "
                f"[{batch_size}, {int(self.model_config.rvq)}]"
            )
        if self.eos_mask.ndim != 1 or int(self.eos_mask.shape[0]) != batch_size:
            raise ValueError("journal EOS mask must have shape [batch]")
        if self.eos_mask.dtype is not torch.bool:
            raise TypeError("journal EOS mask must be boolean")


class MossTTSRealtimeDecodeStatePool:
    """Fixed request rows for exact eager HF sampling and frame feedback."""

    def __init__(
        self,
        model: Any,
        *,
        max_running_requests: int,
        max_history_frames: int = _DEFAULT_MAX_HISTORY_FRAMES,
    ) -> None:
        max_running_requests = _require_int(
            max_running_requests,
            "max_running_requests",
            minimum=1,
        )
        max_history_frames = _require_int(
            max_history_frames,
            "max_history_frames",
            minimum=1,
        )
        self.model = model
        self.model_config = model.config
        self.max_running_requests = max_running_requests
        self.num_rows = max_running_requests + 1
        self.padding_row = max_running_requests
        self.max_history_frames = max_history_frames

        local_transformer = model.local_transformer
        weight = local_transformer.local_lm_heads[0].weight
        self.device = weight.device
        self.dtype = weight.dtype
        self.hidden_size = int(weight.shape[1])

        self.previous_audio_frames = torch.full(
            (self.num_rows, int(self.model_config.rvq)),
            int(self.model_config.audio_pad_token),
            dtype=torch.long,
            device=self.device,
        )
        self.has_previous_frame = torch.zeros(
            self.num_rows,
            dtype=torch.bool,
            device=self.device,
        )
        self.feedback_embeds = torch.zeros(
            self.num_rows,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.feedback_valid = torch.zeros(
            self.num_rows,
            dtype=torch.bool,
            device=self.device,
        )
        self.history_codes = torch.full(
            (
                self.num_rows,
                self.max_history_frames,
                int(self.model_config.rvq),
            ),
            int(self.model_config.audio_pad_token),
            dtype=torch.long,
            device=self.device,
        )
        self.history_lengths = torch.zeros(
            self.num_rows,
            dtype=torch.long,
            device=self.device,
        )
        self.frame_positions = torch.zeros(
            self.num_rows,
            dtype=torch.long,
            device=self.device,
        )
        self.provisional = torch.zeros(
            self.num_rows,
            dtype=torch.bool,
            device=self.device,
        )
        self.audio_eos_seen = torch.zeros(
            self.num_rows,
            dtype=torch.bool,
            device=self.device,
        )

        self._rid_to_row: dict[str, int] = {}
        self._row_to_rid: list[str | None] = [None] * self.num_rows
        self._free_rows: list[int] = list(range(self.padding_row))
        self._sampling_params: list[MossTTSRealtimeSamplingParams | None] = [
            None
        ] * self.num_rows
        self._generators: list[torch.Generator | None] = [None] * self.num_rows
        self._history_lengths_host: list[int] = [0] * self.num_rows
        self._frame_positions_host: list[int] = [0] * self.num_rows
        self._max_active_rows_observed = 0

    def resource_snapshot(self) -> dict[str, int]:
        """Return fixed-pool ownership gauges without exposing mutable state."""

        active_rows = len(self._rid_to_row)
        return {
            "model_state_capacity": self.padding_row,
            "model_state_active_rows": active_rows,
            "model_state_free_rows": len(self._free_rows),
            "model_state_max_active_rows_observed": self._max_active_rows_observed,
        }

    def row_for(self, rid: str) -> int | None:
        return self._rid_to_row.get(rid)

    def rid_for(self, row_idx: int) -> str | None:
        self._validate_real_row(row_idx)
        return self._row_to_rid[row_idx]

    def _validate_real_row(self, row_idx: int) -> int:
        row_idx = _require_int(row_idx, "row_idx")
        if row_idx >= self.padding_row:
            raise ValueError(
                f"row_idx must address a real pool row below {self.padding_row}"
            )
        return row_idx

    def _turn_state(self, data: Any) -> Any | None:
        return getattr(data, "turn_state", None)

    def _choose_row(self, data: Any | None) -> tuple[int, bool]:
        turn_state = self._turn_state(data) if data is not None else None
        desired_row = getattr(turn_state, "model_state_slot_id", None)
        if desired_row is not None:
            desired_row = self._validate_real_row(desired_row)
            if desired_row not in self._free_rows:
                raise RuntimeError(
                    f"model-state row {desired_row} is already owned by another request"
                )
            self._free_rows.remove(desired_row)
            return desired_row, False
        if not self._free_rows:
            raise RuntimeError(
                "MOSS-TTS-Realtime decode-state pool exhausted "
                f"({self.padding_row} rows, all held); raise max_running_requests"
            )
        return self._free_rows.pop(), True

    def acquire_row(self, rid: str, data: Any | None = None) -> int:
        if not isinstance(rid, str) or not rid:
            raise ValueError("request id must be a non-empty string")
        params = (
            MossTTSRealtimeSamplingParams.from_request_data(data)
            if data is not None
            else None
        )
        existing = self._rid_to_row.get(rid)
        if existing is not None:
            if params is not None:
                self._ensure_sampling_params(existing, params)
            self._verify_turn_slot(existing, data)
            return existing

        row_idx, assign_turn_slot = self._choose_row(data)
        turn_state = self._turn_state(data) if data is not None else None
        try:
            self._rid_to_row[rid] = row_idx
            self._row_to_rid[row_idx] = rid
            if params is not None:
                self._install_sampling_params(row_idx, params)
            if assign_turn_slot and turn_state is not None:
                turn_state.assign_model_state_slot(row_idx)
            self._verify_turn_slot(row_idx, data)
            self._max_active_rows_observed = max(
                self._max_active_rows_observed,
                len(self._rid_to_row),
            )
        except BaseException:
            self._rid_to_row.pop(rid, None)
            self._row_to_rid[row_idx] = None
            self.reset_row(row_idx)
            self._free_rows.append(row_idx)
            raise
        return row_idx

    def _verify_turn_slot(self, row_idx: int, data: Any | None) -> None:
        if data is None:
            return
        turn_state = self._turn_state(data)
        if turn_state is None:
            return
        slot_id = getattr(turn_state, "model_state_slot_id", None)
        if slot_id != row_idx:
            raise RuntimeError(
                f"model-state slot ownership mismatch: {slot_id} != {row_idx}"
            )

    def _install_sampling_params(
        self,
        row_idx: int,
        params: MossTTSRealtimeSamplingParams,
    ) -> None:
        if params.repetition_window > self.max_history_frames:
            raise ValueError(
                "repetition_window exceeds the fixed history capacity "
                f"({params.repetition_window} > {self.max_history_frames})"
            )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(params.seed)
        self._sampling_params[row_idx] = params
        self._generators[row_idx] = generator

    def _ensure_sampling_params(
        self,
        row_idx: int,
        params: MossTTSRealtimeSamplingParams,
    ) -> None:
        installed = self._sampling_params[row_idx]
        if installed is None:
            self._install_sampling_params(row_idx, params)
            return
        if installed != params:
            raise RuntimeError(
                "sampling parameters cannot change while a realtime turn owns "
                f"state row {row_idx}"
            )

    def release_row(self, rid: str, turn_state: Any | None = None) -> int | None:
        row_idx = self._rid_to_row.get(rid)
        if row_idx is None:
            if (
                turn_state is not None
                and getattr(turn_state, "model_state_slot_id", None) is not None
            ):
                raise RuntimeError(
                    "host turn retains a model-state slot after pool release"
                )
            return None
        if turn_state is not None:
            slot_id = getattr(turn_state, "model_state_slot_id", None)
            if slot_id != row_idx:
                raise RuntimeError(
                    f"model-state slot ownership mismatch: {slot_id} != {row_idx}"
                )

        self._rid_to_row.pop(rid)
        self._row_to_rid[row_idx] = None
        self.reset_row(row_idx)
        self._free_rows.append(row_idx)
        if turn_state is not None:
            turn_state.release_model_state_slot(expected_slot_id=row_idx)
        return row_idx

    def reset_row(self, row_idx: int) -> None:
        row_idx = self._validate_real_row(row_idx)
        self.previous_audio_frames[row_idx].fill_(
            int(self.model_config.audio_pad_token)
        )
        self.has_previous_frame[row_idx] = False
        self.feedback_embeds[row_idx].zero_()
        self.feedback_valid[row_idx] = False
        history_len = self._history_lengths_host[row_idx]
        if history_len:
            self.history_codes[row_idx, :history_len].fill_(
                int(self.model_config.audio_pad_token)
            )
        self.history_lengths[row_idx] = 0
        self.frame_positions[row_idx] = 0
        self.provisional[row_idx] = False
        self.audio_eos_seen[row_idx] = False
        self._sampling_params[row_idx] = None
        self._generators[row_idx] = None
        self._history_lengths_host[row_idx] = 0
        self._frame_positions_host[row_idx] = 0

    def prepare_active_rows(
        self, requests: list[Any]
    ) -> tuple[torch.Tensor, list[int]]:
        pool_rows: list[int] = []
        newly_acquired: list[tuple[str, Any | None]] = []
        try:
            for request in requests:
                rid = request.request_id
                data = request.data
                existed = self.row_for(rid) is not None
                row_idx = self.acquire_row(rid, data)
                pool_rows.append(row_idx)
                if not existed:
                    newly_acquired.append((rid, self._turn_state(data)))
        except BaseException:
            for acquired_rid, turn_state in reversed(newly_acquired):
                self.release_row(acquired_rid, turn_state)
            raise
        return (
            torch.tensor(pool_rows, dtype=torch.long, device=self.device),
            pool_rows,
        )

    def sampling_params_for(self, row_idx: int) -> MossTTSRealtimeSamplingParams:
        row_idx = self._validate_real_row(row_idx)
        params = self._sampling_params[row_idx]
        if params is None:
            raise RuntimeError(f"state row {row_idx} has no sampling parameters")
        return params

    def generator_for(self, row_idx: int) -> torch.Generator:
        row_idx = self._validate_real_row(row_idx)
        generator = self._generators[row_idx]
        if generator is None:
            raise RuntimeError(f"state row {row_idx} has no sampling generator")
        return generator

    def sample_positions_for(self, pool_rows: list[int]) -> list[int]:
        return [
            self._frame_positions_host[self._validate_real_row(r)] for r in pool_rows
        ]

    def snapshot_generator_states(
        self, pool_rows: list[int]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.generator_for(row).get_state().clone() for row in pool_rows)

    def restore_generator_states(
        self,
        pool_rows: list[int],
        states: tuple[torch.Tensor, ...],
    ) -> None:
        if len(pool_rows) != len(states):
            raise ValueError("generator restore state count mismatch")
        for row_idx, state in zip(pool_rows, states, strict=True):
            self.generator_for(row_idx).set_state(state)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        *,
        row_idx: int,
        codebook: int,
        params: MossTTSRealtimeSamplingParams,
    ) -> torch.Tensor:
        history_len = self._history_lengths_host[row_idx]
        if params.repetition_penalty == 1.0 or history_len == 0:
            return logits
        start = max(0, history_len - params.repetition_window)
        history = self.history_codes[row_idx, start:history_len, codebook]
        scores = logits.clone()
        indices = history.unsqueeze(0)
        current = scores.gather(1, indices)
        updated = torch.where(
            current < 0,
            current * params.repetition_penalty,
            current / params.repetition_penalty,
        )
        scores.scatter_(1, indices, updated)
        return scores

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        top_k = min(max(int(top_k), 1), int(logits.shape[-1]))
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - float(top_p))
        sorted_indices_to_remove[..., -1:] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter(
            1,
            sorted_indices,
            sorted_indices_to_remove,
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def sample_audio(
        self,
        logits: torch.Tensor,
        codebook: int,
        pool_rows: list[int],
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("audio logits must have shape [batch, vocab]")
        if int(logits.shape[0]) != len(pool_rows):
            raise ValueError("audio logits batch size does not match pool rows")
        if int(logits.shape[1]) != int(self.model_config.audio_vocab_size):
            raise ValueError(
                "audio logits vocab size must be "
                f"{int(self.model_config.audio_vocab_size)}"
            )
        codebook = _require_int(codebook, "codebook")
        if codebook >= int(self.model_config.rvq):
            raise ValueError("audio codebook index is out of range")

        sampled: list[torch.Tensor] = []
        for batch_idx, row_idx in enumerate(pool_rows):
            row_idx = self._validate_real_row(row_idx)
            if self._row_to_rid[row_idx] is None:
                raise RuntimeError(f"state row {row_idx} is not owned")
            params = self.sampling_params_for(row_idx)
            row_logits = logits[batch_idx : batch_idx + 1]
            row_logits = self._apply_repetition_penalty(
                row_logits,
                row_idx=row_idx,
                codebook=codebook,
                params=params,
            )
            if not params.do_sample or params.temperature == 0:
                sampled.append(torch.argmax(row_logits, dim=-1))
                continue
            filtered = row_logits / params.temperature
            filtered = self._apply_top_k(filtered, params.top_k)
            filtered = self._apply_top_p(filtered, params.top_p)
            probs = F.softmax(filtered, dim=-1)
            token = torch.multinomial(
                probs,
                num_samples=1,
                generator=self.generator_for(row_idx),
            ).view(-1)
            sampled.append(token)
        return torch.cat(sampled, dim=0)

    def commit_frames(
        self,
        *,
        rids: list[str],
        pool_rows: list[int],
        sample_positions: list[int],
        frames: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(rids)
        if len(pool_rows) != batch_size or len(sample_positions) != batch_size:
            raise ValueError("frame commit metadata length mismatch")
        if frames.ndim != 2 or tuple(frames.shape) != (
            batch_size,
            int(self.model_config.rvq),
        ):
            raise ValueError(
                "committed frames must have shape "
                f"[{batch_size}, {int(self.model_config.rvq)}]"
            )
        if (
            frames.dtype == torch.bool
            or torch.is_floating_point(frames)
            or torch.is_complex(frames)
        ):
            raise TypeError("committed frames must be integer token ids")
        if torch.any(frames < 0) or torch.any(
            frames >= int(self.model_config.audio_vocab_size)
        ):
            raise ValueError("committed frames contain an out-of-range audio id")

        history_positions: list[int] = []
        for rid, row_idx, sample_position in zip(
            rids,
            pool_rows,
            sample_positions,
            strict=True,
        ):
            row_idx = self._validate_real_row(row_idx)
            if self._rid_to_row.get(rid) != row_idx:
                raise RuntimeError(
                    f"frame commit ownership mismatch for {rid!r}: {row_idx}"
                )
            if bool(self.audio_eos_seen[row_idx].item()):
                raise RuntimeError(f"request {rid!r} already sampled audio EOS")
            if bool(self.provisional[row_idx].item()):
                raise RuntimeError(
                    f"request {rid!r} must materialize its previous frame before "
                    "sampling another"
                )
            current_position = self._frame_positions_host[row_idx]
            if current_position != sample_position:
                raise RuntimeError(
                    f"frame position mismatch for {rid!r}: "
                    f"{sample_position} != {current_position}"
                )
            history_len = self._history_lengths_host[row_idx]
            if history_len >= self.max_history_frames:
                raise RuntimeError(
                    f"realtime history capacity exhausted for {rid!r} "
                    f"({self.max_history_frames} frames)"
                )
            history_positions.append(history_len)

        row_t = torch.tensor(pool_rows, dtype=torch.long, device=self.device)
        history_t = torch.tensor(
            history_positions,
            dtype=torch.long,
            device=self.device,
        )
        frames = frames.to(device=self.device, dtype=torch.long)
        eos_mask = frames[:, 0].eq(int(self.model_config.audio_eos_token))
        self.history_codes[row_t, history_t] = frames
        self.history_lengths[row_t] = history_t + 1
        self.frame_positions[row_t] = torch.tensor(
            [position + 1 for position in sample_positions],
            dtype=torch.long,
            device=self.device,
        )
        self.previous_audio_frames[row_t] = torch.where(
            eos_mask.unsqueeze(-1),
            torch.full_like(frames, int(self.model_config.audio_pad_token)),
            frames,
        )
        self.has_previous_frame[row_t] = ~eos_mask
        self.provisional[row_t] = ~eos_mask
        self.audio_eos_seen[row_t] |= eos_mask
        self.feedback_embeds[row_t].zero_()
        self.feedback_valid[row_t] = False

        for row_idx, history_len, sample_position in zip(
            pool_rows,
            history_positions,
            sample_positions,
            strict=True,
        ):
            self._history_lengths_host[row_idx] = history_len + 1
            self._frame_positions_host[row_idx] = sample_position + 1
        return eos_mask

    def mark_materialized(self, rid: str, row: Any) -> int:
        row_idx = self._rid_to_row.get(rid)
        if row_idx is None:
            raise RuntimeError(f"request {rid!r} does not own a state row")
        row_t = torch.as_tensor(row, dtype=torch.long, device=self.device)
        row_width = int(self.model_config.rvq) + 1
        if row_t.ndim != 1 or int(row_t.shape[0]) != row_width:
            raise ValueError(f"materialized row must have shape [{row_width}]")
        if not bool(self.has_previous_frame[row_idx].item()):
            raise RuntimeError("no previous audio frame is available to materialize")
        if not bool(self.provisional[row_idx].item()):
            raise RuntimeError("state row does not hold a provisional audio frame")
        if not torch.equal(row_t[1:], self.previous_audio_frames[row_idx]):
            raise ValueError("materialized row audio columns do not match pool state")
        self.provisional[row_idx] = False
        return row_idx

    def ensure_materialized(self, rid: str, row: Any) -> int:
        """Mark once in the scheduler, then validate again before model decode."""

        row_idx = self._rid_to_row.get(rid)
        if row_idx is None:
            raise RuntimeError(f"request {rid!r} does not own a state row")
        if bool(self.provisional[row_idx].item()):
            return self.mark_materialized(rid, row)

        row_t = torch.as_tensor(row, dtype=torch.long, device=self.device)
        row_width = int(self.model_config.rvq) + 1
        if row_t.ndim != 1 or int(row_t.shape[0]) != row_width:
            raise ValueError(f"materialized row must have shape [{row_width}]")
        if not bool(self.has_previous_frame[row_idx].item()):
            raise RuntimeError("no previous audio frame is available to materialize")
        if not torch.equal(row_t[1:], self.previous_audio_frames[row_idx]):
            raise ValueError("materialized row audio columns do not match pool state")
        return row_idx

    def stage_feedback(self, row_t: torch.Tensor, embeddings: torch.Tensor) -> None:
        if row_t.ndim != 1:
            raise ValueError("feedback row indices must have shape [batch]")
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != int(row_t.shape[0]):
            raise ValueError("feedback embeddings must have shape [batch, hidden]")
        if int(embeddings.shape[1]) != self.hidden_size:
            raise ValueError(
                f"feedback embedding hidden size must be {self.hidden_size}"
            )
        row_t = row_t.to(device=self.device, dtype=torch.long)
        if torch.any(row_t < 0) or torch.any(row_t >= self.padding_row):
            raise ValueError("feedback row index is outside the real pool rows")
        self.feedback_embeds[row_t] = embeddings.to(
            device=self.device,
            dtype=self.dtype,
        )
        self.feedback_valid[row_t] = True

    def feedback_for(self, row_t: torch.Tensor) -> torch.Tensor:
        row_t = row_t.to(device=self.device, dtype=torch.long)
        if not bool(torch.all(self.feedback_valid[row_t]).item()):
            raise RuntimeError("one or more feedback embeddings are not staged")
        return self.feedback_embeds[row_t]

    def frame_position_for(self, rid: str) -> int:
        row_idx = self._rid_to_row.get(rid)
        if row_idx is None:
            raise RuntimeError(f"request {rid!r} does not own a state row")
        return self._frame_positions_host[row_idx]

    def history_for(self, rid: str) -> torch.Tensor:
        row_idx = self._rid_to_row.get(rid)
        if row_idx is None:
            raise RuntimeError(f"request {rid!r} does not own a state row")
        history_len = self._history_lengths_host[row_idx]
        return self.history_codes[row_idx, :history_len]
