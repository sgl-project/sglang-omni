"""Duo Realization adapter for the shared /v1/realtime protocol."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from sglang.multimodal_gen.configs.pipeline_configs.duo import (
    DUO_DECODE_SPECS,
    resolve_duo_decode_mode,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend,
)
from sglang.multimodal_gen.runtime.models.dits.duo.checkpoint import (
    load_causal_realization,
)
from sglang.multimodal_gen.runtime.models.dits.duo.native_execution import (
    NativeDCPOptions,
)
from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs
from sglang.multimodal_gen.runtime.models.vaes.duo.parallel import ParallelDuoCodecs
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from torchtitan.components.inference_config_realization import parser, validate
from torchtitan.models.duo_realization.modules.intent import (
    load as load_reference_intent,
)

from sglang_omni.models.duo.conditioning import load_intent
from sglang_omni.models.duo.frame_intent import FrameIntentPlanner
from sglang_omni.models.duo.kernels import use_intent_kernels
from sglang_omni.models.duo.live import LiveDuoSessionHooks, LiveQwenASR, LiveWhisper
from sglang_omni.models.duo.live_intent import LiveIntentPlanner
from sglang_omni.models.duo.live_intent_srt import SRTFrameIntentPlanner
from sglang_omni.models.duo.reference import DCPRealizationSessionHooks
from sglang_omni.models.duo.reference_parallel import ParallelDCPRealizationSessionHooks
from sglang_omni.models.duo.streaming import RealizationSessionHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionLimits, TimedChunk
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TurnFailure,
    VideoDelta,
    VideoFinished,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    OutputSink,
    ProtocolError,
    RuntimeLimits,
    Unit,
)

logger = logging.getLogger(__name__)


def decode_frame(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def encode_jpeg(frame: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(output, format="JPEG", quality=85)
    return output.getvalue()


def pcm16_from_f32(data: bytes) -> bytes:
    wave = np.frombuffer(data, dtype="<f4")
    return (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()


@dataclass
class DuoLiveResources:
    hooks: LiveDuoSessionHooks
    max_blocks: int
    decode_modes: tuple[str, ...] = ("duo_1_4", "duo_5_17")
    inference_lock: threading.Lock = field(default_factory=threading.Lock)
    shutdown: Callable[[], None] | None = None
    supports_realization_kv_reset: bool = False
    realization_kv_reset_interval_blocks: int | None = None
    reference_size: tuple[int, int] | None = None
    block_executor: ThreadPoolExecutor | None = None
    decode_executor: ThreadPoolExecutor | None = None

    def configuration(self) -> dict[str, str | bool | int | None]:
        prefix, suffix = self.hooks.planner.system_prompt_fragments()
        return {
            "realization_kv_reset_supported": self.manual_reset_enabled,
            "realization_kv_reset_interval_blocks": self.realization_kv_reset_interval_blocks,
            "system_prompt_prefix": prefix,
            "system_prompt_suffix": suffix,
            "video_lookahead_blocks": self.hooks.codecs.video_lookahead_blocks,
        }

    @property
    def manual_reset_enabled(self) -> bool:
        return (
            self.supports_realization_kv_reset
            and self.realization_kv_reset_interval_blocks is None
        )


class DuoRealtimeAdapter(InteractionAdapter):
    """Serialize realtime units into causal Realization blocks selected per session.

    The first camera frame is the default reference when none is supplied.
    The server keeps Intent and Realization state inside the same adapter instance
    for the lifetime of the WebSocket session; the model weights are shared by
    all adapters created by the deployment.
    """

    def __init__(self, resources: DuoLiveResources) -> None:
        self.resources = resources
        self.hooks = resources.hooks
        self.session_id: SessionIdentity | None = None
        self.emit: OutputSink | None = None
        self.config: SessionConfiguration = {}
        self.cancelled = threading.Event()
        self.started = False
        self.response_id = ""
        self.applied_reset_generation = 0
        self.item_id = ""
        self.decode_mode = "duo_1_4"
        self.blocks_since_kv_reset = 0
        self.decode_future: Future[StagePayload] | None = None
        self.delivery_task: asyncio.Task[None] | None = None
        self.delivery_error: Exception | None = None

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.session_id = SessionIdentity(session_id)
        self.emit = emit
        self.config = config
        options = config.get("sglang", {}).get("realization", {})
        self.decode_mode = resolve_duo_decode_mode(
            options.get("decode", self.resources.decode_modes[0])
        )
        if self.decode_mode not in self.resources.decode_modes:
            raise ValueError(
                "The loaded Realization checkpoint does not support this decode mode"
            )
        else:
            pass
        self.cancelled.clear()
        self.started = False
        self.decode_future = None
        self.delivery_task = None
        self.delivery_error = None
        self.response_id = f"realization_{session_id}"
        self.applied_reset_generation = 0
        self.item_id = f"{self.response_id}_item"
        self.blocks_since_kv_reset = 0

    async def update(self, config: SessionConfiguration) -> None:
        previous = self.config.get("sglang", {}).get("realization", {})
        updated = config.get("sglang", {}).get("realization", {})
        if updated.get("kv_reset_generation", 0) < previous.get(
            "kv_reset_generation", 0
        ):
            raise ProtocolError(
                "invalid_request", "DiT reset generation cannot decrease"
            )
        else:
            pass
        self.config = config

    async def process(self, unit: Unit) -> int:
        if self.delivery_error is not None:
            raise self.delivery_error
        else:
            pass
        if self.session_id is None or self.emit is None:
            raise RuntimeError("Realization realtime session is not open")
        else:
            pass
        if unit.eos and unit.real_samples == 0 and not unit.images:
            if self.delivery_task is not None:
                await asyncio.shield(self.delivery_task)
            else:
                pass
            if self.delivery_error is not None:
                raise self.delivery_error
            elif self.started:
                await self.finish_response()
            else:
                pass
            return 0
        else:
            pass
        options = self.config.get("sglang", {}).get("realization", {})
        expected_frames = 17 if self.decode_mode == "duo_5_17" else 4
        if len(unit.images) != expected_frames:
            raise ValueError(
                f"Duo Realization {self.decode_mode} requires {expected_frames} image frames per unit"
            )
        else:
            pass
        if unit.index >= self.resources.max_blocks:
            raise ValueError("Realization realtime block limit reached")
        else:
            pass
        frames = np.stack([decode_frame(frame) for frame in unit.images])
        height, width = frames.shape[1:3]
        if height % 32 or width % 32:
            raise ValueError("Realization reference dimensions must be multiples of 32")
        else:
            pass
        request: OmniRequest | None = None
        if not self.started:
            description = options.get(
                "description",
                {
                    "appearance": "a person",
                    "environment": "a neutral room",
                    "initial_pose": "facing the camera",
                },
            )
            reference = (
                decode_frame(
                    base64.b64decode(options["reference_image"], validate=True)
                )
                if "reference_image" in options
                else frames[0]
            )
            if (
                self.resources.reference_size is not None
                and reference.shape[:2] != self.resources.reference_size
            ):
                raise ValueError(
                    f"Reference height and width must be {self.resources.reference_size}, got {reference.shape[:2]}"
                )
            else:
                pass
            request = OmniRequest(
                {
                    "reference_rgb": reference.tobytes(),
                    "height": reference.shape[0],
                    "width": reference.shape[1],
                    "description": description,
                    "seed": int(options.get("seed", 42)),
                    "sample_id": options.get("sample_id", "realtime"),
                }
            )
        else:
            pass

        spec = DUO_DECODE_SPECS[self.decode_mode]
        reset_interval = self.resources.realization_kv_reset_interval_blocks
        reset_generation = options.get("kv_reset_generation", 0)
        reset_realization_kv = reset_generation > self.applied_reset_generation or (
            self.resources.supports_realization_kv_reset
            and reset_interval is not None
            and self.blocks_since_kv_reset >= reset_interval
        )
        expected_samples = spec.input_audio_samples
        pcm = unit.pcm[: expected_samples * 2]
        pcm += b"\0" * max(0, expected_samples * 2 - len(pcm))
        collected: list[TimedChunk] = []
        context = SessionContext(
            session_identity=self.session_id,
            cancelled=self.cancelled,
            emit=collected.append,
        )
        chunk = TimedChunk(
            modality="duplex_av",
            t_start_ms=unit.index * spec.duration_ms,
            duration_ms=spec.duration_ms,
            seq=unit.index,
            payload={
                "frames_rgb": frames.tobytes(),
                "height": height,
                "width": width,
                "audio_pcm16": pcm,
                "reset_realization_kv": reset_realization_kv,
            },
            eos=unit.eos,
        )
        payload = StagePayload(
            request_id=f"realization_{self.session_id.id}",
            request=OmniRequest(None),
            data=None,
        )
        decoded = await asyncio.get_running_loop().run_in_executor(
            self.resources.block_executor,
            self.run_block,
            request,
            chunk,
            payload,
            context,
        )
        if reset_realization_kv:
            self.applied_reset_generation = reset_generation
            self.blocks_since_kv_reset = 0
        else:
            pass
        self.blocks_since_kv_reset += 1
        if self.cancelled.is_set():
            return unit.real_samples
        else:
            pass
        if decoded is None:
            await self.emit_block(unit, collected)
        else:
            self.delivery_task = asyncio.create_task(
                self.deliver_block(decoded, self.delivery_task, unit, collected),
                name=f"realization-delivery-{self.session_id.id}-{unit.index}",
            )
            if unit.eos:
                await asyncio.shield(self.delivery_task)
                if self.delivery_error is not None:
                    raise self.delivery_error
                else:
                    pass
            else:
                pass
        return unit.real_samples

    async def deliver_block(
        self,
        decoded: Future[StagePayload],
        previous_delivery: asyncio.Task[None] | None,
        unit: Unit,
        collected: list[TimedChunk],
    ) -> None:
        try:
            if previous_delivery is not None:
                await previous_delivery
            else:
                pass
            await asyncio.shield(asyncio.wrap_future(decoded))
            if not self.cancelled.is_set():
                await self.emit_block(unit, collected)
            else:
                pass
        except Exception as error:
            self.delivery_error = error
            self.cancelled.set()
            assert self.emit is not None
            await self.emit(
                TurnFailure("server_error", "internal_error", str(error)), unit
            )

    async def emit_block(self, unit: Unit, collected: list[TimedChunk]) -> None:
        assert self.emit is not None
        if not self.response_id:
            self.response_id = f"realization_{self.session_id.id}"
        else:
            pass
        if unit.index == 0:
            await self.emit(ResponseStarted(self.response_id))
        else:
            pass
        for output in collected:
            if output.modality == "video" and isinstance(output.payload, dict):
                shape = tuple(int(value) for value in output.payload["shape"])
                rgb = np.frombuffer(output.payload["data"], dtype=np.uint8).reshape(
                    shape
                )
                frame_duration = output.duration_ms / max(1, rgb.shape[0])
                for index, frame in enumerate(rgb):
                    await self.emit(
                        VideoDelta(
                            self.response_id,
                            self.item_id,
                            encode_jpeg(frame),
                            "image/jpeg",
                            output.t_start_ms + index * frame_duration,
                            frame_duration,
                        ),
                        unit,
                    )
                    # Let the transport drain before encoding the next frame.
                    await asyncio.sleep(0)
            elif output.modality == "audio" and isinstance(output.payload, dict):
                await self.emit(
                    AudioDelta(
                        self.response_id,
                        self.item_id,
                        pcm16_from_f32(output.payload["data"]),
                        output.t_start_ms,
                        output.duration_ms,
                    ),
                    unit,
                )
            else:
                pass
        if unit.eos:
            await self.finish_response()
        else:
            pass

    async def finish_response(self) -> None:
        assert self.emit is not None
        await self.emit(VideoFinished(self.response_id, self.item_id))
        await self.emit(AudioFinished(self.response_id, self.item_id))
        await self.emit(
            ResponseFinished(
                self.response_id,
                self.item_id,
                "",
                True,
                "completed",
                "stop",
                has_video=True,
            )
        )

    async def cancel(self) -> None:
        self.cancelled.set()

    async def close(self) -> None:
        self.cancelled.set()
        try:
            if self.delivery_task is not None:
                await asyncio.shield(self.delivery_task)
            else:
                pass
        finally:
            await asyncio.to_thread(self.close_session)

    def run_block(
        self,
        request: OmniRequest | None,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> Future[StagePayload] | None:
        # Model work must leave the event loop available for input and heartbeats.
        with self.resources.inference_lock:
            if self.cancelled.is_set():
                return
            else:
                pass
            if request is not None:
                self.hooks.open(
                    context.session_identity, request, decode_mode=self.decode_mode
                )
                self.started = True
            else:
                pass
            if self.resources.decode_executor is None:
                self.hooks.append(chunk, payload, context)
                return None
            else:
                generated = self.hooks.generate_block(
                    chunk, payload, context, self.decode_future
                )
                self.decode_future = self.resources.decode_executor.submit(
                    self.hooks.decode_block, generated
                )
                return self.decode_future

    def close_session(self) -> None:
        # A cancelled asyncio waiter cannot interrupt an in-flight GPU forward.
        with self.resources.inference_lock:
            if self.decode_future is not None:
                wait([self.decode_future])
            else:
                pass
            if self.session_id is not None:
                self.hooks.close(self.session_id)
            else:
                pass
            self.session_id = None
            self.emit = None
            self.started = False
            self.decode_future = None
            self.delivery_task = None


async def warmup_duo_realtime(
    resources: DuoLiveResources,
    *,
    sample_directory: Path,
    reference_image: Path,
    blocks: int,
    height: int,
    width: int,
    description: dict[str, str] | None = None,
) -> None:
    """Exercise real session setup, ASR, planning, generation and codec state."""
    if blocks < 1 or blocks > resources.max_blocks:
        raise ValueError("Warmup block count must fit the deployment session limit")
    else:
        pass
    frames = sorted((sample_directory / "frames").glob("*.jpg"))
    audio = (sample_directory / "audio.pcm").read_bytes()
    spec = DUO_DECODE_SPECS["duo_5_17"]
    audio_bytes = spec.input_audio_samples * 2
    source_blocks = min(len(frames) // spec.input_frames, len(audio) // audio_bytes)
    if source_blocks == 0:
        raise ValueError("Warmup sample must contain a complete 5-to-17 input block")
    else:
        pass
    with Image.open(reference_image) as image:
        reference = ImageOps.fit(image.convert("RGB"), (width, height))
        reference_buffer = io.BytesIO()
        reference.save(reference_buffer, format="JPEG", quality=95)

    async def discard_warmup_output(
        event: OutputEvent, unit: Unit | None = None
    ) -> None:
        pass

    configuration: SessionConfiguration = {
        "sglang": {
            "realization": {
                "decode": "duo_5_17",
                "seed": 42,
                "sample_id": "warmup",
                "reference_image": base64.b64encode(
                    reference_buffer.getvalue()
                ).decode(),
            }
        }
    }
    if description is not None:
        configuration["sglang"]["realization"]["description"] = description
    else:
        pass
    adapter = DuoRealtimeAdapter(resources)
    await adapter.open(
        "realization-startup-warmup", configuration, discard_warmup_output
    )
    logger.info(f"Duo startup warmup: {blocks} blocks before accepting connections")
    try:
        for block in range(blocks):
            source = block % source_blocks
            await adapter.process(
                Unit(
                    index=block,
                    start_sample=block * spec.input_audio_samples,
                    pcm=audio[source * audio_bytes : (source + 1) * audio_bytes],
                    real_samples=spec.input_audio_samples,
                    eos=block + 1 == blocks,
                    images=tuple(
                        frame.read_bytes()
                        for frame in frames[
                            source
                            * spec.input_frames : (source + 1)
                            * spec.input_frames
                        ]
                    ),
                )
            )
    finally:
        await adapter.close()
    logger.info(f"Duo startup warmup complete: {blocks} blocks; session state released")


def load_duo_live_resources(
    *,
    model_path: str,
    intent_model_path: str,
    intent_adapter_path: str,
    raven_lora_path: str,
    trainables_path: str,
    intent_device: str = "cuda:0",
    intent_tp_size: int = 1,
    intent_quantization: Literal["fp8"] | None = None,
    intent_worker_devices: list[int] | None = None,
    intent_vision_device: str | None = None,
    realization_device: str = "cuda:1",
    vae_device: str = "cuda:2",
    vae_worker_devices: list[int] | None = None,
    vae_decode_backend: Literal["native", "solarx"] = "native",
    vae_use_graph: bool = True,
    vae_attention_backend: Literal[
        "torch_sdpa", "fa", "torch_cudnn_sdpa"
    ] = "torch_sdpa",
    asr_device: str = "cuda:3",
    asr_model_path: str | None = None,
    asr_base_url: str | None = None,
    asr_model_name: str = "intent-asr",
    max_blocks: int = 360,
    overlap_intent_codec: bool = False,
    max_context_tokens: int = 32768,
    max_new_tokens: int = 32,
    intent_kv_window_tokens: int | None = None,
    realization_kv_window_tokens: int = 65536,
    realization_reference_window_chunks: int = 0,
    realization_kv_reset_interval_blocks: int | None = None,
    realization_kv_workspace_capacity_tokens: int | None = None,
    enable_cuda_graph: bool = True,
    realization_enable_cuda_graph: bool = False,
    realization_optimize_copy_sync: bool = False,
    realization_merge_fixed_lora: bool = False,
    realization_fp8_block_linears: bool = False,
    realization_modelopt_checkpoint: str | None = None,
    realization_fp8_direct_kernels: bool = False,
    realization_fp8_attention_calibration: str | None = None,
    realization_fp8_attention_library: str | None = None,
    realization_cuda_graph_min_history_tokens: int = 0,
    realization_cuda_graph_max_entries: int = 8,
    realization_checkpoint: str | None = None,
    realization_stage: Literal["causal", "dmd"] = "causal",
    realization_sp_size: int = 1,
    realization_tp_size: int = 1,
    realization_sp_mode: Literal["ulysses", "ring"] = "ulysses",
    realization_worker_devices: list[int] | None = None,
    kernel_backend: Literal["sglang", "torch"] = "sglang",
    realization_backend: Literal["reference", "native"] = "reference",
    intent_backend: Literal["reference", "native"] = "reference",
    height: int = 576,
    width: int = 768,
    steps: int = 50,
    flow_shift: float = 1.0,
) -> DuoLiveResources:

    if (
        realization_backend == "native" or intent_backend == "native"
    ) and realization_checkpoint is None:
        raise ValueError("Native backends require a DCP checkpoint")
    else:
        pass
    if realization_fp8_attention_library is not None and (
        realization_backend != "native" or realization_fp8_attention_calibration is None
    ):
        raise ValueError(
            "Realization FP8 attention library requires native calibrated FP8"
        )
    else:
        pass
    if realization_enable_cuda_graph and (
        realization_backend != "native"
        or realization_sp_mode != "ulysses"
        or realization_stage != "dmd"
    ):
        raise ValueError(
            "Realization CUDA Graph requires native DMD with Ulysses or TP execution"
        )
    else:
        pass
    if realization_optimize_copy_sync and (
        realization_backend != "native" or realization_stage != "dmd"
    ):
        raise ValueError(
            "Realization copy/sync optimization requires native DMD execution"
        )
    else:
        pass
    if realization_cuda_graph_min_history_tokens < 0:
        raise ValueError("Realization graph minimum history must be nonnegative")
    else:
        pass
    if realization_tp_size > 1 and (
        realization_backend != "native" or realization_sp_size != 1
    ):
        raise ValueError("Realization TP requires the native backend and SP1")
    else:
        pass
    if realization_sp_mode == "ring" and realization_backend != "native":
        raise ValueError("Realization Ring requires the native DCP backend")
    else:
        pass
    if intent_backend == "native" and intent_vision_device not in (None, intent_device):
        raise ValueError("Native Intent places vision and text on intent_device")
    else:
        pass
    intent_devices = tuple(
        intent_worker_devices
        if intent_worker_devices is not None
        else range(
            torch.device(intent_device).index,
            torch.device(intent_device).index + intent_tp_size,
        )
    )
    if (
        intent_tp_size not in (1, 2, 4)
        or len(intent_devices) != intent_tp_size
        or len(set(intent_devices)) != intent_tp_size
    ):
        raise ValueError("Intent TP requires one, two or four distinct worker devices")
    elif intent_tp_size > 1 and intent_backend != "native":
        raise ValueError("Intent TP requires the native SGLang backend")
    elif intent_quantization is not None and intent_backend != "native":
        raise ValueError("Intent quantization requires the native SGLang backend")
    else:
        pass
    if intent_kv_window_tokens is not None and intent_backend != "native":
        raise ValueError("Intent sliding KV requires the native backend")
    elif (
        intent_kv_window_tokens is not None
        and not 0 < intent_kv_window_tokens <= max_context_tokens
    ):
        raise ValueError("Intent KV window must be positive and fit context capacity")
    elif realization_kv_window_tokens <= 0:
        raise ValueError("Realization KV window must be positive")
    elif (
        realization_kv_reset_interval_blocks is not None
        and realization_kv_reset_interval_blocks <= 0
    ):
        raise ValueError(
            "Realization KV reset interval must be a positive number of blocks"
        )
    elif (
        realization_kv_reset_interval_blocks is not None
        and realization_checkpoint is None
    ):
        raise ValueError("Automatic KV reset requires a DCP checkpoint")
    elif realization_reference_window_chunks < 0:
        raise ValueError("Realization reference window must be nonnegative")
    elif realization_reference_window_chunks and realization_checkpoint is None:
        raise ValueError("Realization reference window requires a DCP checkpoint")
    else:
        pass
    if overlap_intent_codec:
        planner_devices = set(intent_devices)
        if intent_vision_device is not None:
            planner_devices.add(torch.device(intent_vision_device).index)
        else:
            pass
        codec_devices = set(vae_worker_devices or [torch.device(vae_device).index])
        codec_devices.add(torch.device(vae_device).index)
        if planner_devices & codec_devices:
            raise ValueError("Intent/Codec overlap requires disjoint GPU sets")
        else:
            pass
    else:
        pass
    if realization_kv_workspace_capacity_tokens is not None:
        if realization_backend != "native":
            raise ValueError(
                "Realization workspace reservation requires the native backend"
            )
        elif (
            realization_kv_workspace_capacity_tokens
            < realization_kv_window_tokens + 128
        ):
            raise ValueError(
                "Realization workspace must fit the KV window and 128 padding rows"
            )
        else:
            pass
    else:
        pass
    shutdown = None
    shutdown_intent = None
    release_intent_workspace = False
    if realization_fp8_direct_kernels and realization_modelopt_checkpoint is None:
        raise ValueError("Direct FP8 kernels require a ModelOpt checkpoint")
    else:
        pass
    if realization_modelopt_checkpoint is not None and (
        realization_checkpoint is None
        or realization_backend != "native"
        or realization_fp8_block_linears
    ):
        raise ValueError(
            "ModelOpt Realization requires native DCP execution without online requantization"
        )
    else:
        pass
    if realization_sp_size > 1 and realization_checkpoint is None:
        raise ValueError("Realization SP workers require a DCP checkpoint")
    else:
        pass
    if realization_checkpoint is not None:
        # Load the reference backend only when selected.

        options = [
            "--checkpoint",
            realization_checkpoint,
            "--checkpoint-family",
            "auto",
            "--checkpoint-cache",
            realization_checkpoint,
            "--realization",
            model_path,
            "--intent",
            intent_model_path,
            "--intent-adapter",
            intent_adapter_path,
            "--decode",
            "duo_5_17",
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(steps),
            "--flow-shift",
            str(flow_shift),
            "--max-seq-len",
            "120000",
            "--attention",
            "sdpa",
            "--intent-attention",
            "sdpa",
            "--dtype",
            "bfloat16",
            "--context-parallel-degree",
            "1",
            "--realization-inference-mode",
            "causal",
            "--video-decode-latent",
            "3",
            "--mode",
            "closed_loop",
            "--max-blocks",
            str(max_blocks),
            "--intent-max-pixels",
            "16384",
            "--max-intent-tokens",
            str(max_context_tokens),
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        if not intent_adapter_path:
            options.append("--origin-intent")
        else:
            pass
        reference_args = validate(parser(realization_stage).parse_args(options))
        if intent_backend == "native":

            planner = SRTFrameIntentPlanner(
                reference_args,
                device=torch.device(intent_device),
                devices=intent_devices,
                quantization=intent_quantization,
                kv_window_tokens=(
                    max_context_tokens
                    if intent_kv_window_tokens is None
                    else intent_kv_window_tokens
                ),
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                enable_cuda_graph=enable_cuda_graph,
            )
            shutdown_intent = planner.shutdown
        else:
            reference_args.intent_load_device = intent_device
            processor, intent, identity = load_reference_intent(
                reference_args, dtype=torch.bfloat16
            )
            planner = FrameIntentPlanner(
                processor,
                intent,
                max_context_tokens=max_context_tokens,
                max_new_tokens=max_new_tokens,
                vision_device=intent_vision_device,
                speak_token_id=identity.get("special_ids", {}).get("<speak>"),
            )
            if kernel_backend == "sglang":
                use_intent_kernels(intent)
            else:
                pass
        if realization_sp_size > 1 or realization_backend == "native":
            realization_world_size = realization_sp_size * realization_tp_size

            devices = tuple(
                (
                    [torch.device(realization_device).index]
                    if realization_world_size == 1
                    else range(realization_world_size)
                )
                if realization_worker_devices is None
                else realization_worker_devices
            )
            if (
                len(devices) != realization_world_size
                or len(set(devices)) != realization_world_size
            ):
                raise ValueError(
                    "Realization parallel execution needs one distinct GPU per rank"
                )
            elif any(
                index < 0 or index >= torch.cuda.device_count() for index in devices
            ):
                raise ValueError("Realization worker GPU is not visible")
            else:
                pass
            realization = ParallelDCPRealizationSessionHooks(
                reference_args,
                devices=devices,
                device=torch.device(realization_device),
                use_sglang_kernels=kernel_backend == "sglang",
                native_backbone=realization_backend == "native",
                sp_mode=realization_sp_mode,
                tp_size=realization_tp_size,
                kv_window_tokens=realization_kv_window_tokens,
                reference_window_chunks=realization_reference_window_chunks,
                native_options=NativeDCPOptions(
                    modelopt_checkpoint=realization_modelopt_checkpoint,
                    fp8_direct_kernels=realization_fp8_direct_kernels,
                    merge_fixed_lora=realization_merge_fixed_lora,
                    fp8_block_linears=realization_fp8_block_linears,
                    fp8_attention_calibration=realization_fp8_attention_calibration,
                    fp8_attention_library=realization_fp8_attention_library,
                    batch_history_copies=realization_optimize_copy_sync,
                    reuse_adapter_input_cast=realization_optimize_copy_sync,
                    defer_finite_checks=realization_optimize_copy_sync,
                    direct_attention_output=realization_optimize_copy_sync,
                    enable_cuda_graph=realization_enable_cuda_graph,
                    cuda_graph_min_history_tokens=realization_cuda_graph_min_history_tokens,
                    cuda_graph_max_entries=realization_cuda_graph_max_entries,
                    kv_workspace_capacity_tokens=(
                        realization_kv_workspace_capacity_tokens
                        if realization_kv_workspace_capacity_tokens is not None
                        else (
                            realization_kv_window_tokens + 128
                            if realization_enable_cuda_graph
                            or realization_optimize_copy_sync
                            else 0
                        )
                    ),
                ),
            )
            shutdown = realization.shutdown
            release_intent_workspace = (
                intent_backend == "reference" and planner.device.index in devices
            )
        else:
            with torch.cuda.device(realization_device):
                realization = DCPRealizationSessionHooks(
                    reference_args,
                    device=torch.device(realization_device),
                    use_sglang_kernels=kernel_backend == "sglang",
                    kv_window_tokens=realization_kv_window_tokens,
                    reference_window_chunks=realization_reference_window_chunks,
                )
    else:
        if not (intent_adapter_path and raven_lora_path and trainables_path):
            raise ValueError(
                "Raven mode requires Intent adapter, Raven LoRA and trainables paths"
            )
        else:
            pass
        processor, intent = load_intent(
            intent_model_path, intent_adapter_path, device=intent_device
        )
        processor.video_processor.size = {"shortest_edge": 1024, "longest_edge": 16384}
        planner = LiveIntentPlanner(
            processor,
            intent,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            vision_device=intent_vision_device,
        )
        if kernel_backend == "sglang":
            use_intent_kernels(intent)
        else:
            pass
        with torch.cuda.device(realization_device):
            # FileStore expects a path to a file, not a directory.  A unique
            # per-process path also prevents stale single-process rendezvous data.
            rendezvous = Path(tempfile.mktemp(prefix="realization-realtime-")).as_uri()
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=1, distributed_init_method=rendezvous
            )
            torch.cuda.set_device(realization_device)
            model = load_causal_realization(
                transformer_directory=Path(model_path) / "FL2VA/transformer",
                raven_lora_path=Path(raven_lora_path),
                trainables_path=Path(trainables_path),
                clean_timestep=0.999,
                device=torch.device(realization_device),
                use_flash_attention=True,
                flash_num_splits=1,
                precompute_adaln=True,
                enable_cuda_graph=enable_cuda_graph,
            )

        realization = RealizationSessionHooks(
            model, capacity_tokens=16384, plan_capacity_tokens=128
        )
    set_global_server_args(ServerArgs.from_dict({"model_path": model_path}))
    global_force_attn_backend(AttentionBackendEnum.TORCH_SDPA)
    with torch.cuda.device(vae_device):
        if vae_worker_devices is None:
            codecs = DuoCodecs(
                model_path,
                device=vae_device,
                video_decode_backend=vae_decode_backend,
                video_use_graph=vae_use_graph,
                parallel_audio_decode=True,
                video_attention_backend=AttentionBackendEnum[
                    vae_attention_backend.upper()
                ],
            )
        else:
            codecs = ParallelDuoCodecs(
                model_path,
                device=vae_device,
                video_decode_backend=vae_decode_backend,
                video_use_graph=vae_use_graph,
                devices=tuple(vae_worker_devices),
                video_attention_backend=AttentionBackendEnum[
                    vae_attention_backend.upper()
                ],
            )
    shutdown_realization = shutdown

    def shutdown_codecs() -> None:
        try:
            codecs.shutdown()
        finally:
            if shutdown_realization is not None:
                shutdown_realization()
            else:
                pass
            if shutdown_intent is not None:
                shutdown_intent()
            else:
                pass

    shutdown = shutdown_codecs
    asr: LiveWhisper | LiveQwenASR | None
    if asr_base_url is not None:
        asr_client = LiveQwenASR(asr_base_url, asr_model_name)
        asr = asr_client
        shutdown_local_models = shutdown

        def shutdown_with_asr() -> None:
            try:
                asr_client.close()
            finally:
                shutdown_local_models()

        shutdown = shutdown_with_asr
    elif asr_model_path is not None:
        asr = LiveWhisper(asr_model_path, device=asr_device)
    else:
        asr = None
    input_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="duo-input-asr"
    )
    hooks = LiveDuoSessionHooks(
        planner,
        realization,
        codecs,
        max_blocks=max_blocks,
        asr=asr,
        release_intent_workspace=release_intent_workspace,
        input_executor=input_executor,
    )
    decode_executor = (
        ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="realization-realtime-decode"
        )
        if overlap_intent_codec
        else None
    )
    shutdown_models = shutdown

    def shutdown_with_inputs() -> None:
        try:
            input_executor.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                if decode_executor is not None:
                    decode_executor.shutdown(wait=True)
                else:
                    pass
            finally:
                shutdown_models()

    shutdown = shutdown_with_inputs
    return DuoLiveResources(
        hooks=hooks,
        max_blocks=max_blocks,
        shutdown=shutdown,
        decode_modes=(
            ("duo_5_17",)
            if realization_checkpoint is not None
            else ("duo_1_4", "duo_5_17")
        ),
        supports_realization_kv_reset=realization_checkpoint is not None,
        realization_kv_reset_interval_blocks=realization_kv_reset_interval_blocks,
        reference_size=(height, width),
        decode_executor=decode_executor,
    )


def build_duo_realtime_deployment(resources: DuoLiveResources) -> RealtimeDeployment:
    capabilities = Capabilities(
        interaction="native",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=32000,
        output_modalities=("audio", "video"),
        max_granted_modalities=2,
        input_modalities=("audio", "image"),
        image_frames_per_unit=(4, 17),
        native_unit_ms=167,
        tail_policy="pad",
        supports_truncate=False,
        supports_realization_kv_reset=resources.manual_reset_enabled,
        default_max_slice_nums=2 if resources.decode_modes == ("duo_5_17",) else 1,
        input_samples_per_slice=(
            DUO_DECODE_SPECS["duo_1_4"].input_audio_samples,
            DUO_DECODE_SPECS["duo_5_17"].input_audio_samples,
        ),
    )
    return RealtimeDeployment(
        capabilities=capabilities,
        adapter_factory=lambda: DuoRealtimeAdapter(resources),
        limits=RuntimeLimits(max_input_bytes=120 * 16000 * 2),
        session_limits=SessionLimits(max_pending_chunks=4, max_chunk_bytes=8 << 20),
        max_connections=1,
    )
