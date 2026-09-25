# SPDX-License-Identifier: Apache-2.0
"""Streaming talker scheduler for Ming-Omni V1.

Consumes text segments emitted by the segmenter stage and produces audio
chunks by driving ``MingOmniTalker.omni_audio_generation(stream=True, ...)``.
Each audio chunk is published on the outbox stream channel with ``target=None``
so the stage runtime forwards it directly to the coordinator (terminal).
"""

from __future__ import annotations

import json
import logging
import os
import queue as _queue_mod
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.streaming_text import uint8_tensor_to_text
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
        AudioVAE,
    )
    from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
        MingOmniTalker,
    )
else:
    pass

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "DB30"
DEFAULT_SAMPLE_RATE = 44100


class AudioChunkPayloadRequired(TypedDict):
    modality: str
    audio_waveform: bytes
    audio_waveform_shape: list[int]
    audio_waveform_dtype: str
    sample_rate: int
    stage_name: str
    segment_id: int


class AudioChunkPayload(AudioChunkPayloadRequired, total=False):
    talker_first_audio_ms: float


@dataclass
class RequestState:
    payload: StagePayload | None = None
    payload_arrived: bool = False
    stream_done: bool = False
    finalized: bool = False
    aborted: bool = False
    abort_event: threading.Event = field(default_factory=threading.Event)
    segment_count: int = 0
    audio_chunk_count: int = 0
    request_t_start_s: float = field(default_factory=time.perf_counter)
    first_audio_emit_ms: float | None = None


class MingStreamingTalkerScheduler:
    """Stream-aware scheduler that turns text segments into audio chunks.

    Same inbox/outbox/start/stop/abort contract as ``SimpleScheduler`` /
    ``MingStreamingSegmenterScheduler`` so the V1 stage runtime drives it
    without branching.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cuda",
        voice: str = DEFAULT_VOICE,
        talker: "MingOmniTalker | None" = None,
        audio_detokenizer: "AudioVAE | None" = None,
        sample_rate: int | None = None,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()

        self.model_path = model_path
        self.device = device
        self.voice = voice
        self.talker = talker
        self.audio_detokenizer = audio_detokenizer
        self.sample_rate = sample_rate

        self.running = False
        self.states: dict[str, RequestState] = {}
        self.states_lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.running = True
        if self.talker is None:
            self.load_models()
        else:
            pass
        if self.sample_rate is None:
            self.sample_rate = self.resolve_sample_rate()
        else:
            pass

        while self.running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            try:
                self.handle_message(msg)
            except Exception as exc:
                logger.exception(
                    "MingStreamingTalkerScheduler: failed handling %s for %s",
                    msg.type,
                    msg.request_id,
                )
                self.outbox.put(
                    OutgoingMessage(request_id=msg.request_id, type="error", data=exc)
                )
                self.discard_state(msg.request_id)

    def stop(self) -> None:
        self.running = False
        # Signal abort on every still-active request so any in-flight talker
        # generation unwinds in time.
        with self.states_lock:
            for state in self.states.values():
                state.abort_event.set()

    def abort(self, request_id: str) -> None:
        with self.states_lock:
            state = self.states.get(request_id)
            if state is None:
                return
            else:
                pass
            state.aborted = True
            state.abort_event.set()

    # ------------------------------------------------------------------ dispatch
    def handle_message(self, msg: IncomingMessage) -> None:
        if msg.type == "new_request":
            self.on_new_request(msg)
        elif msg.type == "stream_chunk":
            self.on_stream_chunk(msg)
        elif msg.type == "stream_done":
            self.on_stream_done(msg)
        else:
            logger.debug("MingStreamingTalker: ignored message type=%s", msg.type)

    # ------------------------------------------------------------------ handlers
    def on_new_request(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        with self.states_lock:
            state = self.states.get(request_id)
            if state is None:
                state = RequestState()
                self.states[request_id] = state
            else:
                pass
            state.payload = msg.data
            state.payload_arrived = True
            should_finalize = state.stream_done and not state.finalized
        if should_finalize:
            self.finalize(request_id)
        else:
            pass

    def on_stream_chunk(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        item = msg.data
        if not isinstance(item, StreamItem):
            return
        else:
            pass
        with self.states_lock:
            state = self.states.get(request_id)
            if state is None:
                state = RequestState()
                self.states[request_id] = state
            else:
                pass
            if state.aborted or state.finalized:
                return
            else:
                pass

        metadata = dict(item.metadata or {})
        is_final_segment = bool(metadata.get("is_final_segment", False))
        text = uint8_tensor_to_text(item.data)
        if text:
            self.generate_audio_for_segment(
                request_id=request_id,
                state=state,
                text=text,
                segment_id=int(metadata.get("segment_id", state.segment_count)),
            )
            state.segment_count += 1
        else:
            pass
        # is_final_segment is informational; we still wait for stream_done
        # to finalize the result payload.
        if is_final_segment:
            logger.debug(
                "[TALKER_STREAM] saw final segment for %s segment_count=%d",
                request_id,
                state.segment_count,
            )
        else:
            pass

    def on_stream_done(self, msg: IncomingMessage) -> None:
        request_id = msg.request_id
        with self.states_lock:
            state = self.states.get(request_id)
            if state is None:
                return
            else:
                pass
            state.stream_done = True
            should_finalize = state.payload_arrived and not state.finalized
        if should_finalize:
            self.finalize(request_id)
        else:
            pass

    # ------------------------------------------------------------------ generation
    def generate_audio_for_segment(
        self,
        *,
        request_id: str,
        state: RequestState,
        text: str,
        segment_id: int,
    ) -> None:
        if self.talker is None:
            raise RuntimeError("Talker model not loaded")
        else:
            pass
        t_start = time.perf_counter()
        generator = self.build_generation_iterator(text, state.abort_event)
        try:
            for item in generator:
                if state.abort_event.is_set():
                    break
                else:
                    pass
                waveform = self.extract_waveform(item)
                if waveform is None or self.waveform_numel(waveform) == 0:
                    continue
                else:
                    pass
                if state.first_audio_emit_ms is None:
                    state.first_audio_emit_ms = (
                        time.perf_counter() - state.request_t_start_s
                    ) * 1000.0
                else:
                    pass
                self.emit_audio_chunk(
                    request_id, state, waveform, segment_id=segment_id
                )
        except BaseException as exc:  # CancelledError from abort surfaces here
            import asyncio as _asyncio

            if isinstance(exc, _asyncio.CancelledError):
                logger.info(
                    "[TALKER_STREAM] segment %d aborted for %s", segment_id, request_id
                )
                return
            else:
                pass
            raise
        finally:
            logger.debug(
                "[TALKER_STREAM] segment %d for %s took %.2fs",
                segment_id,
                request_id,
                time.perf_counter() - t_start,
            )

    def build_generation_iterator(
        self, text: str, abort_event: threading.Event
    ) -> Generator[
        tuple[torch.Tensor, str | None, tuple[int, int] | None, float | None],
        None,
        None,
    ]:
        if hasattr(self.talker, "omni_audio_generation"):
            return self.talker.omni_audio_generation(
                tts_text=text,
                voice_name=self.voice,
                audio_detokenizer=self.audio_detokenizer,
                stream=True,
                abort_event=abort_event,
            )
        else:
            pass
        if hasattr(self.talker, "instruct_audio_generation"):
            return self.talker.instruct_audio_generation(
                prompt="Please generate speech based on the following description.\n",
                text=text,
                audio_detokenizer=self.audio_detokenizer,
                stream=True,
                abort_event=abort_event,
            )
        else:
            pass
        raise RuntimeError("Talker has no streaming generation method")

    # ------------------------------------------------------------------ outbox
    def emit_audio_chunk(
        self,
        request_id: str,
        state: RequestState,
        waveform: torch.Tensor,
        *,
        segment_id: int,
    ) -> None:
        audio_bytes, shape, dtype = self.serialize_waveform(waveform)
        payload: AudioChunkPayload = {
            "modality": "audio",
            "audio_waveform": audio_bytes,
            "audio_waveform_shape": shape,
            "audio_waveform_dtype": dtype,
            "sample_rate": self.resolve_sample_rate(),
            "stage_name": TALKER_STREAM_STAGE,
            "segment_id": segment_id,
        }
        if state.first_audio_emit_ms is not None:
            payload["talker_first_audio_ms"] = state.first_audio_emit_ms
        else:
            pass
        state.audio_chunk_count += 1
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data=payload,
                metadata={"modality": "audio", "segment_id": segment_id},
            )
        )

    def finalize(self, request_id: str) -> None:
        with self.states_lock:
            state = self.states.get(request_id)
            if state is None or state.finalized:
                return
            else:
                pass
            state.finalized = True

        # Build the final result payload as a fresh small dict — do not
        # inherit the upstream StagePayload.data which contains tensors
        # (prompt.input_ids, encoder_outs, etc.). The terminal result is
        # serialized via msgpack to the coordinator and cannot carry
        # torch.Tensor objects.
        payload = state.payload
        if payload is None:
            payload = StagePayload(request_id=request_id, request=None, data={})
        else:
            pass
        payload.data = {
            "modality": "audio",
            "audio_chunk_count": state.audio_chunk_count,
            "segment_count": state.segment_count,
            "first_audio_emit_ms": state.first_audio_emit_ms,
            "aborted": state.aborted,
        }
        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="result", data=payload)
        )
        self.discard_state(request_id)

    def discard_state(self, request_id: str) -> None:
        with self.states_lock:
            self.states.pop(request_id, None)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def extract_waveform(
        item: tuple[torch.Tensor, str | None, tuple[int, int] | None, float | None],
    ) -> torch.Tensor | None:
        if isinstance(item, tuple):
            return item[0] if item else None
        else:
            pass
        return item

    @staticmethod
    def waveform_numel(
        waveform: torch.Tensor,
    ) -> int:
        if isinstance(waveform, torch.Tensor):
            return int(waveform.numel())
        else:
            pass
        if isinstance(waveform, np.ndarray):
            return int(waveform.size)
        else:
            pass
        if isinstance(waveform, (bytes, bytearray, memoryview)):
            return len(waveform)
        else:
            pass
        return int(np.asarray(waveform).size)

    @staticmethod
    def serialize_waveform(
        waveform: torch.Tensor,
    ) -> tuple[bytes, list[int], str]:
        if isinstance(waveform, torch.Tensor):
            array = waveform.detach().cpu().float().numpy()
        elif isinstance(waveform, np.ndarray):
            array = waveform.astype(np.float32, copy=False)
        elif isinstance(waveform, (bytes, bytearray, memoryview)):
            raw = bytes(waveform)
            return raw, [len(raw)], "uint8"
        else:
            array = np.asarray(waveform, dtype=np.float32)
        array = np.asarray(array, dtype=np.float32)
        return array.tobytes(), list(array.shape), str(array.dtype)

    def resolve_sample_rate(self) -> int:
        if self.sample_rate is not None:
            return int(self.sample_rate)
        else:
            pass
        for owner in (self.audio_detokenizer, self.talker):
            sr = self.sample_rate_from(owner)
            if sr is not None:
                self.sample_rate = sr
                return sr
            else:
                pass
        self.sample_rate = DEFAULT_SAMPLE_RATE
        return self.sample_rate

    @staticmethod
    def sample_rate_from(owner: object) -> int | None:
        if owner is None:
            return None
        else:
            pass
        config = getattr(owner, "config", None)
        sr = getattr(config, "sample_rate", None)
        if sr is None:
            sr = getattr(owner, "sample_rate", None)
        else:
            pass
        return int(sr) if sr is not None else None

    def validate_voice_presets(
        self, voice_dict: dict, manifest_path: str, talker_dir: str
    ) -> None:
        """Resolve relative prompt-wav paths and validate the manifest.

        Mutates ``voice_dict`` in place so each entry's ``prompt_wav_path``
        becomes an absolute path on disk.
        """
        if self.voice is not None and self.voice not in voice_dict:
            raise ValueError(
                f"[TALKER_STREAM] default voice {self.voice!r} not found in "
                f"{manifest_path}; available presets: "
                f"{sorted(voice_dict.keys())}"
            )
        else:
            pass
        for name, entry in voice_dict.items():
            rel_path = entry.get("prompt_wav_path")
            if rel_path is None:
                raise ValueError(
                    f"[TALKER_STREAM] voice preset {name!r} in "
                    f"{manifest_path} is missing prompt_wav_path"
                )
            else:
                pass
            resolved = os.path.join(talker_dir, rel_path)
            if not os.path.isfile(resolved):
                raise FileNotFoundError(
                    f"[TALKER_STREAM] voice preset {name!r} references "
                    f"missing prompt wav {resolved}"
                )
            else:
                pass
            entry["prompt_wav_path"] = resolved

    # ------------------------------------------------------------------ model load
    def load_models(self) -> None:
        if self.model_path is None:
            raise RuntimeError(
                "MingStreamingTalkerScheduler needs model_path to load talker"
            )
        else:
            pass
        from transformers import AutoTokenizer

        from sglang_omni.models.ming_omni.talker import (
            MingOmniTalker,
            MingOmniTalkerConfig,
            SpkembExtractor,
        )
        from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
            AudioVAE,
        )
        from sglang_omni.models.weight_loader import load_weights_by_prefix

        t_start = time.perf_counter()
        talker_dir = str(Path(self.model_path) / "talker")
        logger.info(
            "[TALKER_STREAM] loading talker from %s device=%s",
            talker_dir,
            self.device,
        )
        config = MingOmniTalkerConfig.from_pretrained_dir(talker_dir)
        if torch.device(self.device).type == "npu":
            config.use_torch_attention()
        else:
            pass
        talker = MingOmniTalker(config)
        talker.eval()
        weights = load_weights_by_prefix(talker_dir, prefix="")
        talker.load_weights(weights.items())
        talker.to(device=self.device, dtype=torch.bfloat16)
        talker.set_tokenizer(
            AutoTokenizer.from_pretrained(str(Path(talker_dir) / "llm"))
        )

        voice_json = os.path.join(talker_dir, "data", "voice_name.json")
        if os.path.exists(voice_json):
            with open(voice_json) as f:
                voice_dict = json.load(f)
            self.validate_voice_presets(voice_dict, voice_json, talker_dir)
            talker.set_voice_presets(voice_dict)
        elif self.voice is not None:
            raise FileNotFoundError(
                f"[TALKER_STREAM] voice_name.json not found at {voice_json}; "
                f"default voice {self.voice!r} cannot be resolved"
            )
        else:
            logger.info(
                "[TALKER_STREAM] no voice_name.json at %s; presets disabled", voice_json
            )

        campplus = os.path.join(talker_dir, "campplus.onnx")
        try:
            talker.set_spkemb_extractor(SpkembExtractor(campplus))
        except (ImportError, Exception) as exc:
            logger.warning("[TALKER_STREAM] SpkembExtractor unavailable: %s", exc)

        try:
            from talker_tn.talker_tn import TalkerTN

            talker.set_normalizer(TalkerTN())
        except ImportError:
            logger.warning("[TALKER_STREAM] TalkerTN unavailable; identity normalizer")

        vae_dir = str(Path(talker_dir) / "vae")
        vae = None
        if Path(vae_dir).exists():
            vae = AudioVAE.from_pretrained(vae_dir, dtype=torch.bfloat16)
            vae.to(self.device)
            vae.eval()
        else:
            logger.warning("[TALKER_STREAM] AudioVAE missing at %s", vae_dir)

        logger.info("[TALKER_STREAM] initializing device graphs")
        talker.initial_graph()
        self.talker = talker
        self.audio_detokenizer = vae
        logger.info(
            "[TALKER_STREAM] talker loaded in %.2fs",
            time.perf_counter() - t_start,
        )
