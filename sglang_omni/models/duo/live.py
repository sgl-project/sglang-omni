"""Live observations to Intent plans, causal Realization blocks, and decoded media."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import httpx
import msgspec
import numpy as np
import torch
from sglang.multimodal_gen.configs.pipeline_configs.duo import (
    DUO_DECODE_SPECS,
    DuoDecodeMode,
    DuoDecodeSpec,
)
from sglang.multimodal_gen.runtime.models.dits.duo.geometry import (
    spatial_axis,
    video_rows,
)
from sglang.multimodal_gen.runtime.models.dits.duo.model import RealizationSegment
from sglang.multimodal_gen.runtime.models.dits.duo.sequence import AttentionSpan
from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.models.duo.live_intent import LiveIntentPlanner
from sglang_omni.models.duo.streaming import (
    LiveSessionInput,
    RealizationSessionHooks,
    discard_latents,
    segment_from_h50,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks
from sglang_omni.serve.protocol import TranscriptionResponse

logger = logging.getLogger(__name__)

from sglang_omni.models.duo.reference import DCPRealizationSessionHooks
from sglang_omni.models.duo.reference_parallel import ParallelDCPRealizationSessionHooks


@dataclass(kw_only=True)
class LiveAVInput:
    frames_rgb: bytes
    height: int
    width: int
    user_text: str = ""
    audio_pcm16: bytes = b""
    reset_realization_kv: bool = False


@dataclass(kw_only=True)
class LiveSessionState:
    reference: torch.Tensor
    reference_segment: RealizationSegment
    seed: int
    next_block: int = 0
    ended: bool = False
    shared: RealizationSegment | None = None
    previous: list[torch.Tensor] = field(default_factory=list)
    pending_audio: bytes = b""
    decode_spec: DuoDecodeSpec = DUO_DECODE_SPECS["duo_1_4"]


@dataclass(kw_only=True)
class GeneratedLiveBlock:
    chunk: TimedChunk
    payload: StagePayload
    context: SessionContext
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    metrics: dict[str, str | int | float]


class LiveWhisper:
    """Transcribe committed, non-overlapping 16 kHz mono audio windows."""

    def __init__(self, model_path: str, *, device: str) -> None:
        self.processor = WhisperProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float16,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    @torch.inference_mode()
    def transcribe(self, pcm16: bytes) -> str:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        inputs = self.processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.cuda.device(self.model.device):
            encoded = self.model.model.encoder(
                inputs.input_features.to(self.model.device, dtype=torch.float16),
                return_dict=True,
            )
            tokens = self.model.generate(
                encoder_outputs=encoded,
                attention_mask=inputs.attention_mask.to(self.model.device),
                task="transcribe",
                max_new_tokens=128,
            )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


class LiveQwenASR:
    """Transcribe through the resident SGLang-Omni Qwen3-ASR stage."""

    def __init__(self, base_url: str, model_name: str) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=60)
        self.model_name = model_name

    def transcribe(self, pcm16: bytes) -> str:
        response = self.client.post(
            "audio/transcriptions",
            data={"model": self.model_name, "response_format": "json"},
            files={"file": ("audio.wav", pcm16_wav(pcm16, 16000), "audio/wav")},
        )
        response.raise_for_status()
        return (
            TranscriptionResponse.model_validate_json(response.content)
            .text.strip()
            .rstrip("。．.")
        )

    def close(self) -> None:
        self.client.close()


class LiveDuoSessionHooks(SessionHooks):
    def __init__(
        self,
        planner: LiveIntentPlanner,
        realization: (
            RealizationSessionHooks
            | DCPRealizationSessionHooks
            | ParallelDCPRealizationSessionHooks
        ),
        codecs: DuoCodecs,
        *,
        max_blocks: int,
        asr: LiveWhisper | LiveQwenASR | None = None,
        release_intent_workspace: bool = False,
        reference_cache_entries: int = 2,
        input_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.planner = planner
        self.realization = realization
        self.codecs = codecs
        self.max_blocks = max_blocks
        self.asr = asr
        self.input_executor = input_executor
        self.release_intent_workspace = release_intent_workspace
        if reference_cache_entries < 0:
            raise ValueError("Reference cache capacity must be nonnegative")
        else:
            pass
        self.reference_cache_entries = reference_cache_entries
        self.reference_cache: OrderedDict[
            tuple[DuoDecodeMode, int, int, int, bytes], torch.Tensor
        ] = OrderedDict()
        self.device = realization.device
        self.states: dict[SessionIdentity, LiveSessionState] = {}

    @torch.inference_mode()
    def open(
        self,
        session_identity: SessionIdentity,
        request: OmniRequest,
        *,
        decode_mode: DuoDecodeMode = "duo_1_4",
    ) -> None:
        settings = msgspec.convert(request.inputs, type=LiveSessionInput, strict=True)
        if (
            min(settings.height, settings.width) <= 0
            or settings.height % 32
            or settings.width % 32
        ):
            raise ValueError("Reference dimensions must be positive multiples of 32")
        else:
            pass
        if len(settings.reference_rgb) != settings.height * settings.width * 3:
            raise ValueError("Reference must contain height*width*3 RGB bytes")
        else:
            pass
        if not {"appearance", "environment", "initial_pose"}.issubset(
            settings.description
        ):
            raise ValueError(
                "Description requires appearance, environment, and initial_pose"
            )
        else:
            pass
        opened_at = time.monotonic()
        reference_key = (
            decode_mode,
            settings.seed if decode_mode == "duo_1_4" else 0,
            settings.height,
            settings.width,
            hashlib.sha256(settings.reference_rgb).digest(),
        )
        cached_reference = self.reference_cache.pop(reference_key, None)
        reference_cache_hit = cached_reference is not None
        if cached_reference is not None:
            reference = cached_reference.clone()
            self.reference_cache[reference_key] = cached_reference
        else:
            pixels = torch.from_numpy(
                np.frombuffer(settings.reference_rgb, dtype=np.uint8).copy()
            ).reshape(settings.height, settings.width, 3)
            with torch.cuda.device(self.codecs.device):
                reference = self.codecs.encode_reference(
                    pixels, seed=settings.seed, mode=decode_mode
                )
            if self.reference_cache_entries:
                if len(self.reference_cache) == self.reference_cache_entries:
                    self.reference_cache.popitem(last=False)
                else:
                    pass
                self.reference_cache[reference_key] = reference.detach().clone()
            else:
                pass
        reference_done = time.monotonic()
        height, width = reference.shape[-2:]
        rows = height * width // 4
        positions = torch.zeros(rows, 3, dtype=torch.float64, device=self.device)
        # Match Realization's patch-center spatial coordinates, including aspect scaling.
        positions[:, 1:] = torch.cartesian_prod(*spatial_axis((height, width))).to(
            self.device
        )
        segment = RealizationSegment(
            span=AttentionSpan(kind="reference", token_count=rows),
            modality=0,
            tokens=video_rows(reference.to(self.device)),
            positions=positions,
            roles=torch.ones(rows, dtype=torch.long, device=self.device),
        )
        segments_done = time.monotonic()
        self.planner.open(str(session_identity), settings.description)
        planner_opened = time.monotonic()
        self.realization.open(session_identity, request)
        initialized = time.monotonic()
        logger.info(
            f"Realization session initialization: {dict(session=str(session_identity), reference_encode_seconds=reference_done - opened_at, reference_cache_hit=reference_cache_hit, reference_segment_seconds=segments_done - reference_done, planner_open_seconds=planner_opened - segments_done, realization_open_seconds=initialized - planner_opened, total_seconds=initialized - opened_at)}"
        )
        self.states[session_identity] = LiveSessionState(
            reference=reference,
            reference_segment=segment,
            seed=settings.seed,
            decode_spec=DUO_DECODE_SPECS[decode_mode],
        )

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        try:
            generated = self.generate_block(chunk, payload, context)
            return self.decode_block(generated)
        except BaseException:
            self.close(context.session_identity)
            raise

    @torch.inference_mode()
    def generate_block(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
        previous_decode: Future[StagePayload] | None = None,
    ) -> GeneratedLiveBlock:
        state = self.states[context.session_identity]
        spec = state.decode_spec
        if (
            state.ended
            or chunk.modality != "duplex_av"
            or chunk.seq != state.next_block
        ):
            raise ValueError("Expected the next duplex_av block of an open session")
        else:
            pass
        if state.next_block >= self.max_blocks:
            raise ValueError("Duplex session block limit reached")
        else:
            pass
        expected_start = state.next_block * spec.duration_ms
        if (
            abs(chunk.t_start_ms - expected_start) > 1
            or abs(chunk.duration_ms - spec.duration_ms) > 1
        ):
            raise ValueError(f"Duplex input uses {spec.input_frames}-frame blocks")
        else:
            pass
        observation = msgspec.convert(chunk.payload, type=LiveAVInput, strict=True)
        if (
            min(observation.height, observation.width) <= 0
            or len(observation.frames_rgb)
            != spec.input_frames * observation.height * observation.width * 3
        ):
            raise ValueError(
                f"Each input block requires exactly {spec.input_frames} RGB frames"
            )
        else:
            pass
        expected_audio_bytes = 2 * spec.input_audio_samples
        if observation.audio_pcm16 and (
            self.asr is None or len(observation.audio_pcm16) != expected_audio_bytes
        ):
            raise ValueError("PCM16 requires ASR and one matching 16 kHz mono block")
        else:
            pass
        started = time.monotonic()
        if context.cancelled.is_set():
            raise RuntimeError("Duplex session cancelled")
        else:
            pass
        user_text = observation.user_text
        state.pending_audio += observation.audio_pcm16
        frames = np.frombuffer(observation.frames_rgb, dtype=np.uint8).reshape(
            spec.input_frames, observation.height, observation.width, 3
        )
        asr_futures: list[Future[tuple[str, float, float]]] = []
        asr_intervals: list[tuple[float, float]] = []
        recognizer = self.asr
        # One second per ASR window; a final partial window is flushed on EOS.
        if recognizer is not None and (len(state.pending_audio) >= 32000 or chunk.eos):

            def transcribe_window(audio_pcm16: bytes) -> tuple[str, float, float]:
                transcribe_started = time.monotonic()
                with torch.profiler.record_function("duplex.asr"):
                    transcript = recognizer.transcribe(audio_pcm16)
                return transcript, transcribe_started, time.monotonic()

            while len(state.pending_audio) >= 32000 or (
                chunk.eos and state.pending_audio
            ):
                window, state.pending_audio = (
                    state.pending_audio[:32000],
                    state.pending_audio[32000:],
                )
                if self.input_executor is not None:
                    asr_futures.append(
                        self.input_executor.submit(transcribe_window, window)
                    )
                else:
                    transcript, transcribe_started, transcribe_done = transcribe_window(
                        window
                    )
                    user_text += transcript
                    asr_intervals.append((transcribe_started, transcribe_done))
        else:
            pass
        visual_started = time.monotonic()
        prepared_visual = self.planner.visual_patch(frames)
        visual_done = time.monotonic()
        for asr_future in asr_futures:
            transcript, transcribe_started, transcribe_done = asr_future.result()
            user_text += transcript
            asr_intervals.append((transcribe_started, transcribe_done))
        asr_done = time.monotonic()
        with (
            torch.profiler.record_function("duplex.intent"),
            torch.cuda.device(self.planner.device),
        ):
            plan = self.planner.append(
                str(context.session_identity),
                frames,
                user_text,
                state.next_block,
                context.cancelled,
                frames_per_block=spec.input_frames,
                block_duration_ms=spec.duration_ms,
                prepared_visual=prepared_visual,
            )
            intent_compute_done = time.monotonic()
            if self.release_intent_workspace:
                # A colocated Realization rank needs free workspace as its KV grows.
                torch.cuda.empty_cache()
            else:
                pass
        intent_done = time.monotonic()
        previous_decode_metrics = None
        if previous_decode is not None:
            previous_decode_metrics = previous_decode.result().data
        else:
            pass
        decode_barrier_done = time.monotonic()
        if context.cancelled.is_set():
            raise RuntimeError("Duplex session cancelled")
        else:
            pass
        with torch.cuda.device(self.device):
            if state.shared is None:
                shared_positions = torch.zeros(
                    plan.shared_h50.shape[0],
                    3,
                    device=self.device,
                    dtype=torch.float64,
                )
                shared_positions[:, 0] = -1
                state.shared = segment_from_h50(
                    plan.shared_h50.to(self.device),
                    kind="shared",
                    positions=shared_positions,
                )
            else:
                pass
            plan_positions = torch.zeros(
                plan.plan_h50.shape[0], 3, device=self.device, dtype=torch.float64
            )
            plan_positions[:, 0] = chunk.t_start_ms * 0.04
            plan_segment = segment_from_h50(
                plan.plan_h50.to(self.device),
                kind="plan",
                block_index=state.next_block,
                positions=plan_positions,
                roles=torch.ones(
                    plan.plan_h50.shape[0], device=self.device, dtype=torch.long
                ),
            )
            spatial_positions = state.reference_segment.positions
            video_positions = spatial_positions.repeat(spec.video_latents, 1)
            for latent_index in range(spec.video_latents):
                begin = latent_index * spatial_positions.shape[0]
                end = (latent_index + 1) * spatial_positions.shape[0]
                video_positions[begin:end, 0] = chunk.t_start_ms * 0.04 + latent_index
            audio_positions = torch.zeros(
                2 * spec.audio_latents_per_channel,
                3,
                device=self.device,
                dtype=torch.float64,
            )
            audio_positions[:, 0] = (
                chunk.t_start_ms * 0.04
                + torch.arange(spec.audio_latents_per_channel, device=self.device)
            ).repeat(2)
            audio_positions[: spec.audio_latents_per_channel, 2] = video_positions[
                :, 2
            ].min()
            audio_positions[spec.audio_latents_per_channel :, 2] = video_positions[
                :, 2
            ].max()
            generator = torch.Generator(device=self.device).manual_seed(
                state.seed + state.next_block
            )
            count = video_positions.shape[0]
            video = RealizationSegment(
                span=AttentionSpan(
                    kind="noisy", block_index=state.next_block, token_count=count
                ),
                modality=0,
                tokens=torch.randn(count, 96, device=self.device, generator=generator),
                positions=video_positions,
                roles=torch.ones(count, device=self.device, dtype=torch.long),
            )
            audio = RealizationSegment(
                span=AttentionSpan(
                    kind="noisy",
                    block_index=state.next_block,
                    token_count=2 * spec.audio_latents_per_channel,
                ),
                modality=2,
                tokens=torch.randn(
                    2 * spec.audio_latents_per_channel,
                    32,
                    device=self.device,
                    generator=generator,
                ),
                positions=audio_positions,
                roles=torch.ones(
                    2 * spec.audio_latents_per_channel,
                    device=self.device,
                    dtype=torch.long,
                ),
            )
            realization_payload = StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={
                    "reference": state.reference_segment,
                    "shared": state.shared,
                    "description_tokens": plan.description_tokens,
                    "plan": plan_segment,
                    "video": video,
                    "audio": audio,
                    "block_index": state.next_block,
                    "reset_realization_kv": observation.reset_realization_kv,
                },
            )
            with torch.profiler.record_function("duplex.realization"):
                result = self.realization.append(
                    replace(chunk, modality="realization_block"),
                    realization_payload,
                    replace(context, emit=discard_latents),
                )
        height, width = state.reference.shape[-2:]
        video_latent = (
            result.data["video_latent"]
            .reshape(
                spec.video_latents,
                height // 2,
                width // 2,
                24,
                2,
                2,
            )
            .permute(3, 0, 1, 4, 2, 5)
            .reshape(24, spec.video_latents, height, width)
            .cpu()
        )
        audio_latent = (
            result.data["audio_latent"]
            .reshape(2, spec.audio_latents_per_channel, 32)
            .permute(0, 2, 1)
            .cpu()
        )
        realization_done = time.monotonic()
        metrics = {
            "block": state.next_block,
            "realization_kv_reset": int(observation.reset_realization_kv),
            "plan": plan.text,
            "user_text": user_text,
            "intent_kv_tokens": plan.kv_tokens,
            "intent_reused_tokens": plan.reused_tokens,
            "realization_kv_tokens": self.realization.usage(
                context.session_identity
            ).kv_tokens,
            "realization_graph_captures": self.realization.graph_counts[0],
            "realization_graph_replays": self.realization.graph_counts[1],
            "asr_seconds": sum(stop - start for start, stop in asr_intervals),
            "input_prepare_seconds": asr_done - started,
            "input_visual_seconds": visual_done - visual_started,
            "input_asr_visual_overlap_seconds": sum(
                max(0.0, min(stop, visual_done) - max(start, visual_started))
                for start, stop in asr_intervals
            ),
            "intent_seconds": intent_done - asr_done,
            "intent_serialization_seconds": plan.serialization_seconds,
            "intent_position_seconds": plan.position_seconds,
            "intent_worker_seconds": plan.worker_seconds,
            "intent_prefill_submit_seconds": plan.prefill_submit_seconds,
            "intent_decode_wall_seconds": plan.decode_wall_seconds,
            "intent_h50_return_seconds": plan.h50_return_seconds,
            "intent_graph_replays": plan.graph_replays,
            "intent_workspace_release_seconds": intent_done - intent_compute_done,
            "realization_seconds": realization_done - decode_barrier_done,
            "decode_barrier_seconds": decode_barrier_done - intent_done,
            "block_started_monotonic": started,
            "intent_started_monotonic": asr_done,
            "intent_done_monotonic": intent_done,
            "realization_started_monotonic": decode_barrier_done,
            "realization_done_monotonic": realization_done,
            "intent_previous_decode_overlap_seconds": (
                max(
                    0.0,
                    min(intent_done, previous_decode_metrics["decode_done_monotonic"])
                    - max(
                        asr_done, previous_decode_metrics["decode_started_monotonic"]
                    ),
                )
                if previous_decode_metrics is not None
                else 0.0
            ),
        }
        state.next_block += 1
        state.ended = chunk.eos
        return GeneratedLiveBlock(
            chunk=chunk,
            payload=payload,
            context=context,
            video_latent=video_latent,
            audio_latent=audio_latent,
            metrics=metrics,
        )

    @torch.inference_mode()
    def decode_block(self, generated: GeneratedLiveBlock) -> StagePayload:
        chunk, payload, context = generated.chunk, generated.payload, generated.context
        state = self.states[context.session_identity]
        spec = state.decode_spec
        video_latent, audio_latent = generated.video_latent, generated.audio_latent
        metrics = generated.metrics
        decode_started = time.monotonic()
        audio_samples = spec.output_audio_samples
        with (
            torch.profiler.record_function("duplex.decode"),
            torch.cuda.device(self.codecs.device),
        ):
            frames, wave = self.codecs.decode(
                state.reference,
                state.previous,
                video_latent,
                audio_latent,
                session_id=str(context.session_identity),
                audio_samples=audio_samples,
                mode=spec.name,
            )
        state.previous = (state.previous + [video_latent])[-2:]
        if chunk.eos and spec.name == "duo_5_17":
            with (
                torch.profiler.record_function("duplex.decode_flush"),
                torch.cuda.device(self.codecs.device),
            ):
                tail = self.codecs.finish_5_17(state.previous)
            if tail is not None:
                frames = tail if frames is None else torch.cat((frames, tail), dim=0)
            else:
                pass
        else:
            pass
        decode_done = time.monotonic()
        metrics["decode_seconds"] = decode_done - decode_started
        metrics["decode_started_monotonic"] = decode_started
        metrics["decode_done_monotonic"] = decode_done
        metrics["block_wall_seconds"] = decode_done - metrics["block_started_monotonic"]
        logger.info(f"Realization realtime block: {metrics}")
        video_payload = (
            None
            if frames is None
            else {
                "data": frames.numpy().tobytes(),
                "shape": list(frames.shape),
                "dtype": "uint8",
                "fps": 24,
            }
        )
        audio_payload = {
            "data": wave.transpose(0, 1).contiguous().numpy().astype("<f4").tobytes(),
            "sample_rate": 32000,
            "channels": 2,
            "samples": audio_samples,
        }
        if video_payload is not None:
            context.emit(
                replace(
                    chunk,
                    modality="video",
                    payload=video_payload,
                    format="rgb24",
                    t_start_ms=(
                        max(0, chunk.seq - self.codecs.video_lookahead_blocks)
                        * spec.duration_ms
                        if spec.name == "duo_5_17"
                        else chunk.t_start_ms
                    ),
                    duration_ms=1000 * frames.shape[0] / 24,
                )
            )
        else:
            pass
        context.emit(
            replace(chunk, modality="audio", payload=audio_payload, format="f32le")
        )
        context.emit(replace(chunk, modality="metrics", payload=metrics, format="json"))
        payload.data = metrics
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)
        self.codecs.close_stream(str(session_identity))
        self.planner.close(str(session_identity))
        with torch.cuda.device(self.device):
            self.realization.close(session_identity)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        # Intent and Realization retain distinct caches with the same session lifetime.
        usage = self.realization.usage(session_identity)
        tokens, intent_bytes = self.planner.usage(str(session_identity))
        return ResourceUsage(
            kv_tokens=usage.kv_tokens + tokens, bytes=usage.bytes + intent_bytes
        )
