# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
import torch.nn.functional as F
from cosyvoice.flow.DiT import dit as cosyvoice_dit
from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from cosyvoice.flow.flow_matching import ConditionalCFM
from cosyvoice.utils.mask import add_optional_chunk_mask as cosyvoice_chunk_mask
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from sglang_omni.models.fun_cosyvoice3.config import reject_conflicting_dit_accelerators
from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
    execute_flow_estimator,
    is_flow_estimator_trt,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    TOKEN_MEL_RATIO,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

# Note (xinran): This is an admission budget, not a maximum supported request
# length. The scheduler admits a request that exceeds it as a singleton Flow
# batch and defers following requests to the next batch.

DEFAULT_FLOW_BATCH_ADMISSION_FRAMES = 8000

AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)

CHUNK_MASK_COMPILE_DISABLED = False

FLOW_CUDA_GRAPH_FRAME_BUCKET = 16
# Note (chenyang):
# Mel-frame step size for buffered flow CUDA Graph keys. Capture shapes
# must use a T that is a multiple of this step size. For example, 489
# frames would be padded to 496 frames, replayed, and then cropped back
# to 489 frames.


@dataclass(frozen=True)
class FlowBatchInput:
    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


@dataclass(frozen=True)
class PackedFlowBatch:
    token: torch.Tensor
    token_mask: torch.Tensor
    combined_token_lengths: tuple[int, ...]
    prompt_token_lengths: tuple[int, ...]
    target_token_lengths: tuple[int, ...]
    prompt_mel_lengths: tuple[int, ...]
    total_mel_lengths: tuple[int, ...]
    combined_token_lengths_tensor: torch.Tensor
    total_mel_lengths_tensor: torch.Tensor
    prompt_mel_lengths_tensor: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


logger = logging.getLogger(__name__)


def pack_flow_inputs(
    flow: CausalMaskedDiffWithDiT, inputs: Sequence[FlowBatchInput]
) -> PackedFlowBatch:
    if not inputs:
        raise ValueError("Flow batch must contain at least one input")
    for index, item in enumerate(inputs):
        if item.token.ndim != 2 or item.token.shape[0] != 1 or item.token.shape[1] <= 0:
            raise ValueError(f"input {index} token must have shape [1, target_tokens]")
        if item.prompt_token.ndim != 2 or item.prompt_token.shape[0] != 1:
            raise ValueError(
                f"input {index} prompt_token must have shape [1, prompt_tokens]"
            )
        if item.prompt_feat.ndim != 3 or item.prompt_feat.shape[0] != 1:
            raise ValueError(
                f"input {index} prompt_feat must have shape [1, prompt_frames, channels]"
            )
        if item.prompt_feat.shape[2] != flow.output_size:
            raise ValueError(
                f"input {index} prompt feature width must equal Flow output_size"
            )
        expected_frames = item.prompt_token.shape[1] * flow.token_mel_ratio
        if item.prompt_feat.shape[1] != expected_frames:
            raise ValueError(
                f"input {index} prompt feature length must equal prompt token length "
                f"times token_mel_ratio ({item.prompt_feat.shape[1]} != {expected_frames})"
            )
        if item.embedding.ndim != 2 or item.embedding.shape[0] != 1:
            raise ValueError(
                f"input {index} embedding must have shape [1, speaker_dim]"
            )
        expected_embedding_size = flow.spk_embed_affine_layer.in_features
        if item.embedding.shape[1] != expected_embedding_size:
            raise ValueError(
                f"input {index} embedding width must be {expected_embedding_size}"
            )

    parameter = next(flow.parameters())
    device, dtype = parameter.device, parameter.dtype
    prompt_lengths = tuple(int(item.prompt_token.shape[1]) for item in inputs)
    target_lengths = tuple(int(item.token.shape[1]) for item in inputs)
    combined_lengths = tuple(
        p + t for p, t in zip(prompt_lengths, target_lengths, strict=True)
    )
    prompt_mel_lengths = tuple(int(item.prompt_feat.shape[1]) for item in inputs)
    total_mel_lengths = tuple(
        length * flow.token_mel_ratio for length in combined_lengths
    )
    combined_token_lengths_tensor = torch.tensor(
        combined_lengths, dtype=torch.int64, device=device
    )
    total_mel_lengths_tensor = torch.tensor(
        total_mel_lengths, dtype=torch.int64, device=device
    )
    prompt_mel_lengths_tensor = torch.tensor(
        prompt_mel_lengths, dtype=torch.int64, device=device
    )

    max_tokens = max(combined_lengths)
    token = torch.zeros(len(inputs), max_tokens, dtype=torch.int32, device=device)
    for index, item in enumerate(inputs):
        prompt_length = prompt_lengths[index]
        token[index, :prompt_length] = item.prompt_token[0].to(
            device=device, dtype=torch.int32
        )
        token[index, prompt_length : combined_lengths[index]] = item.token[0].to(
            device=device, dtype=torch.int32
        )
    token_mask = (
        torch.arange(max_tokens, device=device).unsqueeze(0)
        < combined_token_lengths_tensor.unsqueeze(1)
    ).unsqueeze(-1)

    max_prompt_frames = max(prompt_mel_lengths)
    prompt_feat = torch.zeros(
        len(inputs), max_prompt_frames, flow.output_size, device=device, dtype=dtype
    )
    for index, item in enumerate(inputs):
        prompt_feat[index, : prompt_mel_lengths[index]] = item.prompt_feat[0].to(
            device=device, dtype=dtype
        )
    embedding = torch.cat(
        [item.embedding.to(device=device, dtype=dtype) for item in inputs], dim=0
    )
    return PackedFlowBatch(
        token=token,
        token_mask=token_mask,
        combined_token_lengths=combined_lengths,
        prompt_token_lengths=prompt_lengths,
        target_token_lengths=target_lengths,
        prompt_mel_lengths=prompt_mel_lengths,
        total_mel_lengths=total_mel_lengths,
        combined_token_lengths_tensor=combined_token_lengths_tensor,
        total_mel_lengths_tensor=total_mel_lengths_tensor,
        prompt_mel_lengths_tensor=prompt_mel_lengths_tensor,
        prompt_feat=prompt_feat,
        embedding=embedding,
    )


def solve_flow_euler(
    decoder: ConditionalCFM,
    noisy_mel: torch.Tensor,
    time_span: torch.Tensor,
    token_condition: torch.Tensor,
    mel_mask: torch.Tensor,
    speaker_embedding: torch.Tensor,
    prompt_mel: torch.Tensor,
    *,
    streaming: bool = False,
) -> torch.Tensor:
    batch_size, channels, mel_frame = noisy_mel.shape
    dtype = speaker_embedding.dtype
    noisy_mel_cfg = torch.zeros(
        2 * batch_size, channels, mel_frame, device=noisy_mel.device, dtype=dtype
    )
    mel_mask_cfg = torch.zeros(
        2 * batch_size, 1, mel_frame, device=noisy_mel.device, dtype=dtype
    )
    token_condition_cfg = torch.zeros_like(noisy_mel_cfg)
    flow_time = torch.zeros(1, device=noisy_mel.device, dtype=dtype)
    speaker_embedding_cfg = torch.zeros(
        2 * batch_size,
        speaker_embedding.shape[1],
        device=noisy_mel.device,
        dtype=dtype,
    )
    prompt_mel_cfg = torch.zeros_like(noisy_mel_cfg)
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        noisy_mel_cfg[:batch_size] = noisy_mel
        noisy_mel_cfg[batch_size:] = noisy_mel
        mel_mask_cfg[:batch_size] = mel_mask
        mel_mask_cfg[batch_size:] = mel_mask
        token_condition_cfg[:batch_size] = token_condition
        flow_time[:] = t
        speaker_embedding_cfg[:batch_size] = speaker_embedding
        prompt_mel_cfg[:batch_size] = prompt_mel
        estimator = decoder.estimator
        if isinstance(estimator, torch.nn.Module):
            vector_field = decoder.forward_estimator(
                noisy_mel_cfg,
                mel_mask_cfg,
                token_condition_cfg,
                flow_time,
                speaker_embedding_cfg,
                prompt_mel_cfg,
                streaming=streaming,
            )
        else:
            # Packed Flow is CFG=2N; CosyVoice TRT hardcodes (2, 80, T).
            vector_field = execute_flow_estimator(
                estimator,
                noisy_mel_cfg,
                mel_mask_cfg,
                token_condition_cfg,
                flow_time,
                speaker_embedding_cfg,
                prompt_mel_cfg,
            )
        conditional, unconditional = (
            vector_field[:batch_size],
            vector_field[batch_size:],
        )
        noisy_mel = noisy_mel + dt * (
            (1.0 + decoder.inference_cfg_rate) * conditional
            - decoder.inference_cfg_rate * unconditional
        )
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
    return noisy_mel.float()


def verify_flow_cuda_graph_capture_shapes(
    capture_shapes: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not capture_shapes:
        raise ValueError("flow_cuda_graph_capture_shapes must not be empty")
    for batch_size, mel_frame in capture_shapes:
        if batch_size <= 0 or mel_frame <= 0:
            raise ValueError(
                "flow_cuda_graph_capture_shapes entries must have positive "
                f"batch and mel_frame values; got {(batch_size, mel_frame)!r}"
            )
        if mel_frame % FLOW_CUDA_GRAPH_FRAME_BUCKET != 0:
            raise ValueError(
                "flow_cuda_graph_capture_shapes mel_frame values must be multiples "
                f"of {FLOW_CUDA_GRAPH_FRAME_BUCKET}; got {(batch_size, mel_frame)!r}"
            )
    return capture_shapes


@dataclass
class CapturedFlowCudaGraph:
    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_output: torch.Tensor


class FlowCudaGraphRunner:
    def __init__(
        self,
        flow: FunCosyVoice3Flow,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        self.flow = flow
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.graphs: dict[tuple[int, int], CapturedFlowCudaGraph] = {}
        self.pool: tuple[int, int] | None = None

    def capture_inputs(
        self, batch_size: int, mel_frame: int
    ) -> tuple[torch.Tensor, ...]:
        # Note (chenyang): CUDA Graph capture and replay must share the same
        # tensor storage. This function bulids the static Euler inputs for
        # one (B, T) capture shape; run() later copies the real batch into them.
        parameter = next(self.flow.parameters())
        model_device, parameter_dtype = parameter.device, parameter.dtype
        speaker_dtype = self.autocast_dtype or parameter_dtype
        decoder = self.flow.decoder
        noisy_mel = (
            decoder.rand_noise[:, :, :mel_frame]
            .to(device=model_device, dtype=parameter_dtype)
            .expand(batch_size, -1, -1)
            .clone()
        )
        time_span = torch.linspace(0, 1, 11, device=model_device, dtype=parameter_dtype)
        if decoder.t_scheduler == "cosine":
            time_span = 1 - torch.cos(time_span * 0.5 * torch.pi)
        token_condition = torch.zeros_like(noisy_mel)
        mel_mask = torch.ones(
            batch_size, 1, mel_frame, device=model_device, dtype=parameter_dtype
        )
        speaker_dim = int(self.flow.spk_embed_affine_layer.out_features)
        speaker_embedding = torch.zeros(
            batch_size, speaker_dim, device=model_device, dtype=speaker_dtype
        )
        prompt_mel = torch.zeros_like(noisy_mel)
        return (
            noisy_mel,
            time_span,
            token_condition,
            mel_mask,
            speaker_embedding,
            prompt_mel,
        )

    @torch.inference_mode()
    def capture(self, capture_shapes: tuple[tuple[int, int], ...]) -> None:
        # Note (chenyang): Capture on a side stream so other
        # kernels on default-stream are not recorded.
        graphs: dict[tuple[int, int], CapturedFlowCudaGraph] = {}
        current_stream = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            self.pool = torch.cuda.graph_pool_handle()
            for batch_size, mel_frame in capture_shapes:
                static_inputs = self.capture_inputs(batch_size, mel_frame)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_dtype is not None,
                ):
                    solve_flow_euler(self.flow.decoder, *static_inputs)
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.cuda.graph(
                        graph=graph,
                        pool=self.pool,
                        stream=stream,
                        capture_error_mode="thread_local",
                    ),
                    torch.autocast(
                        device_type=self.device.type,
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_dtype is not None,
                    ),
                ):
                    static_output = solve_flow_euler(self.flow.decoder, *static_inputs)
                graphs[(batch_size, mel_frame)] = CaptureedFlowCudaGraph(
                    graph=graph,
                    static_inputs=static_inputs,
                    static_outputs=static_output,
                )
        current_stream.wait_stream(stream)
        torch.cuda.empty_cache()
        self.graphs = graphs
        return

    @staticmethod
    def right_pad_mel_frames(
        value: torch.Tensor, actual_mel_frame: int, bucket_mel_frame: int
    ) -> torch.Tensor:
        padding = bucket_mel_frame - actual_mel_frame
        if padding == 0:
            return value
        else:
            return F.pad(value, (0, padding), mode="constant", value=0)

    @torch.inference_mode()
    def run(
        self,
        noisy_mel: torch.Tensor,
        time_span: torch.Tensor,
        token_condition: torch.Tensor,
        mel_mask: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_mel: torch.Tensor,
    ) -> torch.Tensor | None:
        frame_inputs = (noisy_mel, token_condition, mel_mask, prompt_mel)
        if noisy_mel.ndim != 3:
            return None
        elif any(
            value.ndim == 0 or value.shape[-1] != int(noisy_mel.shape[2])
            for value in frame_inputs
        ):
            return None
        else:
            batch_size, actual_mel_frame = (
                int(noisy_mel.shape[0]),
                int(noisy_mel.shape[2]),
            )
            bucket_mel_frame = (
                (actual_mel_frame + FLOW_CUDA_GRAPH_FRAME_BUCKET - 1)
                // FLOW_CUDA_GRAPH_FRAME_BUCKET
                * FLOW_CUDA_GRAPH_FRAME_BUCKET
            )
            captured = self.graphs.get((batch_size, bucket_mel_frame))
            if captured is None:
                inputs = None
            else:
                inputs = (
                    self.right_pad_mel_frames(
                        noisy_mel, actual_mel_frame, bucket_mel_frame
                    ),
                    time_span,
                    self.right_pad_mel_frames(
                        token_condition, actual_mel_frame, bucket_mel_frame
                    ),
                    self.right_pad_mel_frames(
                        mel_mask, actual_mel_frame, bucket_mel_frame
                    ),
                    speaker_embedding,
                    self.right_pad_mel_frames(
                        prompt_mel, actual_mel_frame, bucket_mel_frame
                    ),
                )
            if captured is None or inputs is None:
                return None
            elif not all(
                static.shape == value.shape
                and static.dtype == value.dtype
                and static.device == value.device
                for static, value in zip(captured.static_inputs, inputs, strict=True)
            ):
                return None
            else:
                with (
                    torch.cuda.device(self.device),
                    torch.autocast(
                        device_type=self.device.type,
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_dtype is not None,
                    ),
                ):
                    for static, value in zip(
                        captured.static_inputs, inputs, strict=True
                    ):
                        static.copy_(value)
                    captured.graph.replay()
                    return captured.static_output[..., :actual_mel_frame].clone()


@torch.inference_mode()
def generate_flow(
    flow: FunCosyVoice3Flow,
    packed: PackedFlowBatch,
    *,
    streaming: bool = False,
    finalize: bool = True,
) -> torch.Tensor:
    speaker_embedding = flow.spk_embed_affine_layer(
        F.normalize(packed.embedding, dim=1)
    )
    token_embedding = flow.input_embedding(torch.clamp(packed.token, min=0))
    token_embedding = token_embedding * packed.token_mask.to(token_embedding.dtype)
    if finalize:
        lookahead = 0
    else:
        lookahead = flow.pre_lookahead_len
    if finalize or lookahead <= 0:
        token_hidden = flow.pre_lookahead_layer(token_embedding)
    else:
        layer = flow.pre_lookahead_layer
        lengths = packed.combined_token_lengths
        if len(set(int(length) for length in lengths)) <= 1:
            token_hidden = layer(
                token_embedding[:, :-lookahead],
                context=token_embedding[:, -lookahead:],
            )
        else:
            # note (guozhihao-224): packing left-aligns and pads to max
            # combined length. Slice [:, -lookahead:] would read pad on
            # shorter rows.
            body_lens = [max(int(length) - lookahead, 0) for length in lengths]
            max_body = max(body_lens)
            pieces: list[torch.Tensor] = []
            for index, length in enumerate(lengths):
                body_len = body_lens[index]
                hidden = layer(
                    token_embedding[index : index + 1, :body_len],
                    context=token_embedding[
                        index : index + 1, int(length) - lookahead : int(length)
                    ],
                )
                if hidden.shape[1] < max_body:
                    pieces.append(
                        F.pad(hidden, (0, 0, 0, max_body - int(hidden.shape[1])))
                    )
                else:
                    pieces.append(hidden)
            token_hidden = torch.cat(pieces, dim=0)

    token_condition = (
        token_hidden.repeat_interleave(flow.token_mel_ratio, dim=1)
        .transpose(1, 2)
        .contiguous()
    )
    batch_size, channels, max_mel_frame = token_condition.shape
    decoder = flow.decoder
    if channels != flow.output_size:
        raise ValueError("Flow pre-lookahead output width does not match output_size")
    elif max_mel_frame > decoder.rand_noise.shape[2]:
        raise ValueError(
            f"decoder.rand_noise supports {decoder.rand_noise.shape[2]} frames, "
            f"but batch requires {max_mel_frame}"
        )
    else:
        if lookahead > 0:

            total_mel_lengths_tensor = torch.tensor(
                [
                    max(length - lookahead, 0) * flow.token_mel_ratio
                    for length in packed.combined_token_lengths
                ],
                dtype=torch.int64,
                # Note (chenyang): int64 matches torch.arange's default and
                # packed.total_mel_lengths_tensor so the comparison below
                # does not mix integer dtypes.
                device=token_condition.device,
            )
        else:
            total_mel_lengths_tensor = packed.total_mel_lengths_tensor
        mel_mask = (
            (
                torch.arange(max_mel_frame, device=token_condition.device).unsqueeze(0)
                < total_mel_lengths_tensor.unsqueeze(1)
            )
            .unsqueeze(1)
            .to(token_condition.dtype)
        )
        prompt_mel = torch.zeros_like(token_condition)
        for index, prompt_mel_frame in enumerate(packed.prompt_mel_lengths):
            prompt_mel[index, :, :prompt_mel_frame] = packed.prompt_feat[
                index, :prompt_mel_frame
            ].transpose(0, 1)
        noisy_mel = (
            decoder.rand_noise[:, :, :max_mel_frame]
            .to(device=token_condition.device, dtype=token_condition.dtype)
            .expand(batch_size, -1, -1)
            .clone()
        )
        unit_span = torch.linspace(
            0, 1, 11, device=token_condition.device, dtype=token_condition.dtype
        )

        if decoder.t_scheduler == "cosine":
            time_span = 1 - torch.cos(unit_span * 0.5 * torch.pi)
        else:
            time_span = unit_span

        if flow.cuda_graph_runner is None:
            return solve_flow_euler(
                decoder,
                noisy_mel,
                time_span,
                token_condition,
                mel_mask,
                speaker_embedding,
                prompt_mel,
                streaming=streaming,
            )
        else:
            generated = flow.cuda_graph_runner.run(
                noisy_mel,
                time_span,
                token_condition,
                mel_mask,
                speaker_embedding,
                prompt_mel,
            )
            if generated is not None:
                return generated
            else:
                return solve_flow_euler(
                    decoder,
                    noisy_mel,
                    time_span,
                    token_condition,
                    mel_mask,
                    speaker_embedding,
                    prompt_mel,
                    streaming=streaming,
                )


def split_generated_mels(
    flow: CausalMaskedDiffWithDiT,
    packed: PackedFlowBatch,
    generated: torch.Tensor,
    *,
    token_lengths: tuple[int, ...],
    target_token_lengths: tuple[int, ...],
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    ratio = int(flow.token_mel_ratio)
    for index, prompt_frames in enumerate(packed.prompt_mel_lengths):
        total_frames = token_lengths[index] * ratio
        mel = generated[index : index + 1, :, prompt_frames:total_frames]
        expected_frames = target_token_lengths[index] * ratio
        if mel.shape != (1, flow.output_size, expected_frames):
            raise RuntimeError(
                f"Flow output {index} has unexpected shape {tuple(mel.shape)}"
            )
        outputs.append(mel)
    return outputs


class FunCosyVoice3Flow:
    """CosyVoice3 Flow with batch inference enabled as its default API."""

    def __init__(self, flow: CausalMaskedDiffWithDiT) -> None:
        self.flow: CausalMaskedDiffWithDiT = flow
        self.cuda_graph_runner: FlowCudaGraphRunner | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.flow, name)

    def parameters(self):
        return self.flow.parameters()

    def to(self, *args: Any, **kwargs: Any) -> "FunCosyVoice3Flow":
        self.flow.to(*args, **kwargs)
        return self

    def eval(self) -> "FunCosyVoice3Flow":
        self.flow.eval()
        return self

    def attach_cuda_graph_runner(self, runner: FlowCudaGraphRunner) -> None:
        self.cuda_graph_runner = runner

    @torch.inference_mode()
    def inference(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        packed = pack_flow_inputs(self.flow, inputs)
        generated = generate_flow(self, packed)
        return split_generated_mels(
            self.flow,
            packed,
            generated,
            token_lengths=packed.combined_token_lengths,
            target_token_lengths=packed.target_token_lengths,
        )

    @torch.inference_mode()
    def inference_causal(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        # note (guozhihao-224): causal hops (first and follow-up). Same
        # packing as buffered inference, but strip lookahead per row so
        # mixed prompt lengths can share one DiT call. streaming=True
        # keeps the chunk mask aligned with CosyVoice3Model hops.
        packed = pack_flow_inputs(self.flow, inputs)
        generated = generate_flow(self, packed, streaming=True, finalize=False)
        lookahead = self.flow.pre_lookahead_len
        target_token_lengths = tuple(
            max(length - lookahead, 0) for length in packed.target_token_lengths
        )
        combined_token_lengths = tuple(
            max(length - lookahead, 0) for length in packed.combined_token_lengths
        )
        return split_generated_mels(
            self.flow,
            packed,
            generated,
            token_lengths=combined_token_lengths,
            target_token_lengths=target_token_lengths,
        )


def attach_flow_estimator_trt(
    flow: FunCosyVoice3Flow,
    checkpoint_dir: str,
    device: str,
) -> None:
    from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
        build_flow_estimator_trt,
        resolve_flow_estimator_onnx,
    )

    if str(device).split(":", 1)[0].lower() != "cuda":
        raise RuntimeError(
            "enable_flow_estimator_trt requires a CUDA vocoder device, "
            f"got {device!r}"
        )
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        raise RuntimeError(
            "enable_flow_estimator_trt requires NVIDIA CUDA, "
            f"got platform {current_platform.device_type!r}"
        )

    onnx_path = resolve_flow_estimator_onnx(checkpoint_dir)
    # Keep the PyTorch DiT as profile-miss fallback; wrap as nn.Module so
    # CosyVoice's forward_estimator does not take the raw execute_async_v3
    # path (hard-coded CFG batch=2, no profile check) under hop-batch.
    fallback = flow.decoder.estimator
    wrapper = build_flow_estimator_trt(
        onnx_path, device, fallback=fallback, wrap_module=True
    )
    # note (guozhihao-224): CosyVoice registers estimator as an nn.Module child;
    # delete first so assigning the TRT wrapper does not raise TypeError.
    del flow.decoder.estimator
    flow.decoder.estimator = wrapper
    logger.info(
        "Fun-CosyVoice3 Flow DiT estimator is TensorRT Module (%s, max_cfg_batch=%d)",
        onnx_path,
        wrapper.max_batch,
    )


def load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
    *,
    enable_flow_estimator_trt: bool = False,
) -> tuple[FunCosyVoice3Flow, torch.nn.Module]:
    # note (db-ol): the first modelscope import sets every root StreamHandler
    # to ERROR once torch.distributed is initialized, which silences the stage
    # process that hosts both the engine and this vocoder. Undo that change.
    saved = [(handler, handler.level) for handler in logging.getLogger().handlers]
    try:
        importlib.import_module("modelscope")
    except ImportError:
        pass
    for handler, level in saved:
        if handler.level != level:
            handler.setLevel(level)
            logger.info(
                "Restored root log handler level to %s after the modelscope "
                "import changed it",
                logging.getLevelName(level),
            )
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    # note (Dayuxiaoshui): folding weight_norm is the only load-time step
    # batched decode needs.
    folded = 0
    for submodule in list(hift.modules()):
        while is_parametrized(submodule):
            name = next(iter(submodule.parametrizations.keys()))
            remove_parametrizations(submodule, name, leave_parametrized=True)
            folded += 1
    logger.info(
        "Prepared Fun-CosyVoice3 HiFT for inference (folded %d weight_norm "
        "parametrizations)",
        folded,
    )
    del cv.model.llm
    wrapped = FunCosyVoice3Flow(flow)
    if enable_flow_estimator_trt:
        attach_flow_estimator_trt(wrapped, checkpoint_dir, device)
    return wrapped, hift


def compile_dit_backbone(
    flow: FunCosyVoice3Flow,
    *,
    warmup_mel_frames: int = 128,
    warmup_steps: int = 3,
    autocast_dtype: torch.dtype | None = None,
) -> bool:

    estimator = flow.decoder.estimator
    if not isinstance(estimator, torch.nn.Module):
        logger.warning(
            "Fun-CosyVoice3 DiT estimator is not a PyTorch module (%s); "
            "skipping torch.compile",
            type(estimator).__name__,
        )
        return False
    if warmup_mel_frames < 2:
        raise ValueError(f"warmup_mel_frames must be >= 2, got {warmup_mel_frames}")

    original_forward = estimator.forward
    torch._inductor.config.fx_graph_cache = True
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    # note (guozhihao-224): inductor NaN-compares subsequent_chunk_mask in
    # DiT.forward; keep the mask eager.
    global CHUNK_MASK_COMPILE_DISABLED
    if not CHUNK_MASK_COMPILE_DISABLED:
        try:
            import cosyvoice.flow.DiT.dit as dit_mod
        except ImportError:
            dit_mod = None
        if dit_mod is not None:
            dit_mod.add_optional_chunk_mask = torch.compiler.disable(
                dit_mod.add_optional_chunk_mask
            )
            CHUNK_MASK_COMPILE_DISABLED = True
    try:
        estimator.forward = torch.compile(original_forward, dynamic=True)
        param = next(estimator.parameters())
        device, dtype = param.device, param.dtype
        mel_frame = int(warmup_mel_frames)
        with torch.inference_mode():
            for streaming in (False, True):
                for _ in range(warmup_steps):
                    # CFG batch 2; mel dim 80 matches pinned checkpoint proj_out.
                    noisy_mel = torch.randn(
                        2, 80, mel_frame, device=device, dtype=dtype
                    )
                    mel_mask = torch.ones(2, 1, mel_frame, device=device, dtype=dtype)
                    token_condition = torch.randn(
                        2, 80, mel_frame, device=device, dtype=dtype
                    )
                    flow_time = torch.zeros(1, device=device, dtype=dtype)
                    speaker_embedding = torch.randn(2, 80, device=device, dtype=dtype)
                    prompt_mel = torch.randn(
                        2, 80, mel_frame, device=device, dtype=dtype
                    )
                    with torch.autocast(
                        device_type=current_platform.device_type,
                        dtype=autocast_dtype,
                        enabled=autocast_dtype is not None,
                    ):
                        estimator(
                            noisy_mel,
                            mel_mask,
                            token_condition,
                            flow_time,
                            speaker_embedding,
                            prompt_mel,
                            streaming=streaming,
                        )
    except Exception as exc:
        estimator.forward = original_forward
        logger.warning(
            "torch.compile for the Fun-CosyVoice3 DiT backbone failed "
            "(%s: %s); the flow decoder will run eager",
            type(exc).__name__,
            exc,
        )
        return False
    logger.info(
        "Compiled Fun-CosyVoice3 DiT backbone (dynamic=True, autocast_dtype=%s, "
        "warmup_mel_frames=%d, warmup_steps=%d, streaming=False/True)",
        autocast_dtype,
        warmup_mel_frames,
        warmup_steps,
    )
    return True


def create_preprocessing_executor(
    model_path: str,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    del model_path
    # note(chenye): Reference conditioning supports concurrent calls;
    # model prompt finalization is serialized.
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
        max_concurrency=max_concurrency,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    onnx_intra_op_threads: int = 16,
    token_hop_len: int = TOKEN_HOP_LEN,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.engine_builder import (
        FunCosyVoice3EngineBuilder,
    )

    return FunCosyVoice3EngineBuilder(
        token_hop_len=token_hop_len,
        onnx_intra_op_threads=onnx_intra_op_threads,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


@dataclass(frozen=True)
class PreparedFlowRequest:
    index: int
    sample_rate: int
    flow_input: FlowBatchInput
    total_mel_frames: int


def adaptive_flow_requests_grouping(
    requests: Sequence[PreparedFlowRequest],
    *,
    flow_merge_max_gap_frames: int,
    flow_merge_pad_budget_percent: float,
) -> list[list[PreparedFlowRequest]]:
    """Adaptive flow grouping to merge requests with similar padding waste.

    Note (chenyang):
    detailed discussion in https://github.com/sgl-project/sglang-omni/pull/1899
    """
    ordered_requests = tuple(
        sorted(requests, key=lambda request: (request.total_mel_frames, request.index))
    )
    if not ordered_requests:
        return []

    non_patching_workload = sum(
        request.total_mel_frames for request in ordered_requests
    )
    request_count = len(ordered_requests)

    @lru_cache(maxsize=None)
    def optimal_suffix_partition(
        suffix_start: int,
        remaining_group_count: int,
        current_max_group_gap_frames: int,
    ) -> tuple[int, int, tuple[int, ...]] | None:
        if remaining_group_count == 0:
            if suffix_start == request_count:
                return (0, current_max_group_gap_frames, ())
            return None
        if request_count - suffix_start < remaining_group_count:
            return None

        best_plan: tuple[int, int, tuple[int, ...]] | None = None
        shortest_frames = ordered_requests[suffix_start].total_mel_frames
        group_end_limit = request_count - remaining_group_count + 1
        # Note (chenyang): group_end_limit is max possible end index
        # for the current group, since each remaining group must have
        # at least one request.
        for group_end in range(suffix_start + 1, group_end_limit + 1):
            longest_frames = ordered_requests[group_end - 1].total_mel_frames
            group_gap_frames = longest_frames - shortest_frames
            if group_gap_frames > flow_merge_max_gap_frames:
                break
            suffix_plan = optimal_suffix_partition(
                suffix_start=group_end,
                remaining_group_count=remaining_group_count - 1,
                current_max_group_gap_frames=max(
                    current_max_group_gap_frames, group_gap_frames
                ),
            )
            if suffix_plan is None:
                continue
            candidate = (
                (group_end - suffix_start) * longest_frames + suffix_plan[0],
                suffix_plan[1],
                (group_end,) + suffix_plan[2],
            )
            if best_plan is None or candidate < best_plan:
                best_plan = candidate
        return best_plan

    for group_count in range(1, request_count + 1):
        plan = optimal_suffix_partition(
            suffix_start=0,
            remaining_group_count=group_count,
            current_max_group_gap_frames=0,
        )
        if (
            plan is not None
            and (plan[0] / non_patching_workload - 1) * 100
            <= flow_merge_pad_budget_percent + 1e-9
        ):
            start = 0
            groups: list[list[PreparedFlowRequest]] = []
            for end in plan[2]:
                groups.append(list(ordered_requests[start:end]))
                start = end
            return groups

    raise AssertionError("valid Flow requests must have a feasible partition")


class CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: FunCosyVoice3Flow | CausalMaskedDiffWithDiT,
        hift: torch.nn.Module,
        autocast_dtype: torch.dtype | None = None,
        hift_dtype: str = "float32",
        hift_max_padding_waste: float = 1.5,
        flow_merge_max_gap_frames: int = 384,
        flow_merge_pad_budget_percent: float = 25.0,
    ) -> None:
        if hift_max_padding_waste < 1.0:
            raise ValueError("hift_max_padding_waste must be at least 1.0")
        if hift_dtype not in AUTOCAST_DTYPES:
            raise ValueError(
                f"Unsupported Fun-CosyVoice3 HiFT dtype {hift_dtype!r}; "
                f"expected one of {sorted(AUTOCAST_DTYPES)}"
            )
        estimator = flow.decoder.estimator
        if not isinstance(estimator, torch.nn.Module) and not is_flow_estimator_trt(
            estimator
        ):
            raise RuntimeError(
                "Fun-CosyVoice3 Flow estimator must be a PyTorch module or a "
                "TensorRT wrapper exposing acquire_estimator / execute"
            )
        self.flow = (
            flow if isinstance(flow, FunCosyVoice3Flow) else FunCosyVoice3Flow(flow)
        )
        self.hift = hift
        self.autocast_dtype = autocast_dtype
        self.flow_merge_max_gap_frames = flow_merge_max_gap_frames
        self.flow_merge_pad_budget_percent = flow_merge_pad_budget_percent
        self.hift_autocast_dtype = AUTOCAST_DTYPES[hift_dtype]
        self.hift_max_padding_waste = hift_max_padding_waste
        self.hift_samples_per_mel_frame: int | None = None

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_pipeline_state(payload, FunCosyVoice3State)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        # note (guozhihao-224): AR stores one token per step, serialized as
        # [T, 1]; Flow takes a single unbatched sequence.
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        prepared: list[PreparedFlowRequest] = []
        for index, (state, codes) in enumerate(items):
            flow_input = self.make_flow_input(state, codes)
            prepared.append(
                PreparedFlowRequest(
                    index=index,
                    sample_rate=state.sample_rate,
                    flow_input=flow_input,
                    total_mel_frames=(
                        flow_input.prompt_token.shape[1] + flow_input.token.shape[1]
                    )
                    * self.flow.token_mel_ratio,
                )
            )

        results: list[tuple[Any, int] | None] = [None] * len(items)
        flow_groups = adaptive_flow_requests_grouping(
            prepared,
            flow_merge_max_gap_frames=self.flow_merge_max_gap_frames,
            flow_merge_pad_budget_percent=self.flow_merge_pad_budget_percent,
        )

        for flow_group in flow_groups:
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
            ):
                mel_list = self.flow.inference(
                    [request.flow_input for request in flow_group]
                )
            ordered = sorted(
                zip(flow_group, mel_list, strict=True),
                key=lambda pair: int(pair[1].shape[-1]),
            )
            group: list[tuple[Any, torch.Tensor]] = []
            total = 0
            longest = 0
            max_waste = self.hift_max_padding_waste
            hift_groups: list[list[tuple[Any, torch.Tensor]]] = []
            for pair in ordered:
                length = int(pair[1].shape[-1])
                candidate_longest = max(longest, length)
                candidate_total = total + length
                if (
                    group
                    and candidate_longest * (len(group) + 1)
                    > max_waste * candidate_total
                ):
                    hift_groups.append(group)
                    group, total, longest = [], 0, 0
                    candidate_longest = length
                    candidate_total = length
                group.append(pair)
                total, longest = candidate_total, candidate_longest
            if group:
                hift_groups.append(group)
            for group in hift_groups:
                wavs = self.mel2wav_batch([mel for _, mel in group])
                for (request, _), wav in zip(group, wavs, strict=True):
                    results[request.index] = (wav, request.sample_rate)

        if any(result is None for result in results):
            raise RuntimeError("Fun-CosyVoice3 vocoder did not decode every request")
        return [cast(tuple[Any, int], result) for result in results]

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.decode_payloads([payload])
        if len(results) != 1:
            raise RuntimeError(
                f"Fun-CosyVoice3 vocoder returned {len(results)} results for 1 input"
            )
        return results[0]

    def token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        wav, _, _ = self.token2wav_chunk(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            token_offset=0,
            streaming=False,
            finalize=True,
            hift_mel=None,
            speech_offset=0,
        )
        return wav

    def token2wav_chunk(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        token_offset: int,
        streaming: bool,
        finalize: bool,
        hift_mel: torch.Tensor | None,
        speech_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # note (guozhihao-224): causal hops use streaming=True, finalize=False;
        # leftover uses streaming=False, finalize=True, matching CosyVoice3Model.
        # FunCosyVoice3Flow.inference is the buffered batch adapter; hops must
        # call CosyVoice's token/len/streaming signature on the wrapped module.
        if token.shape[1] == 0:
            raise RuntimeError(
                "Fun-CosyVoice3 generation produced no usable speech tokens"
            )
        native_flow = self.flow.flow
        device = next(native_flow.parameters()).device
        offset = max(int(token_offset), 0)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            tts_mel, _ = native_flow.inference(
                token=token.to(device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(device),
                prompt_token=prompt_token.to(device),
                prompt_token_len=torch.tensor(
                    [prompt_token.shape[1]], dtype=torch.int32
                ).to(device),
                prompt_feat=prompt_feat.to(device),
                prompt_feat_len=torch.tensor(
                    [prompt_feat.shape[1]], dtype=torch.int32
                ).to(device),
                embedding=embedding.to(device),
                streaming=streaming,
                finalize=finalize,
            )
        tts_mel = tts_mel[:, :, offset * TOKEN_MEL_RATIO :]
        return self.hift_delta(
            tts_mel, hift_mel=hift_mel, speech_offset=speech_offset, finalize=finalize
        )

    def first_hop_batch(self, items: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        """Causal Flow for equal-shape hops. HiFT stays per request.

        # note (guozhihao-224): first hops and follow-up hops share this
        # path; the scheduler slices new frames at token_offset.
        """
        if not items:
            raise ValueError("first-hop Flow batch must contain at least one input")
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            return self.flow.inference_causal(items)

    def hift_delta(
        self,
        tts_mel: torch.Tensor,
        *,
        hift_mel: torch.Tensor | None,
        speech_offset: int,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if hift_mel is not None:
            tts_mel = torch.cat([hift_mel.to(device=tts_mel.device), tts_mel], dim=2)
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        held = max(int(speech_offset), 0)
        delta = tts_speech[:, held:].detach().cpu()
        return delta, tts_mel.detach(), int(tts_speech.shape[1])

    def make_flow_input(
        self,
        state: FunCosyVoice3State,
        codes: torch.Tensor,
    ) -> FlowBatchInput:
        prompt_token = (
            torch.as_tensor(state.flow_prompt_speech_token, dtype=torch.int32).reshape(
                1, -1
            )
            if state.flow_prompt_speech_token is not None
            else torch.zeros(1, 0, dtype=torch.int32)
        )
        prompt_feat = (
            torch.as_tensor(state.flow_prompt_speech_feat).reshape(1, -1, 80)
            if state.flow_prompt_speech_feat is not None
            else torch.zeros(1, 0, 80)
        )
        embedding = (
            torch.as_tensor(state.flow_embedding).reshape(1, -1)
            if state.flow_embedding is not None
            else torch.zeros(1, 192)
        )
        return FlowBatchInput(
            token=codes.reshape(1, -1).to(torch.int32),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def flow_scheduler_cost(self, payload: StagePayload) -> int:
        state, codes = self.prepare_item(payload)
        flow_input = self.make_flow_input(state, codes)
        return (
            flow_input.prompt_token.shape[1] + flow_input.token.shape[1]
        ) * self.flow.token_mel_ratio

    def mel2wav_batch(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        if not mels:
            return []
        hift_autocast = torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.hift_autocast_dtype,
            enabled=self.hift_autocast_dtype is not None,
        )
        if len(mels) == 1:
            with hift_autocast:
                tts_speech, _ = self.hift.inference(speech_feat=mels[0], finalize=True)
            return [tts_speech.detach().cpu()]
        lengths = [int(mel.shape[2]) for mel in mels]
        longest = max(lengths)
        if min(lengths) == longest:
            padded = torch.cat(mels, dim=0)
        else:
            padded = torch.cat(
                [
                    F.pad(mel, (0, longest - length))
                    for mel, length in zip(mels, lengths)
                ],
                dim=0,
            )
        with hift_autocast:
            wav, _ = self.hift.inference(speech_feat=padded, finalize=True)
        wav = wav.detach()
        if self.hift_samples_per_mel_frame is None:
            stride = int(self.hift.istft_params["hop_len"])
            for rate in self.hift.upsample_rates:
                stride *= int(rate)
            self.hift_samples_per_mel_frame = stride
        samples_per_frame = self.hift_samples_per_mel_frame
        return [
            wav[index : index + 1, : length * samples_per_frame].cpu()
            for index, length in enumerate(lengths)
        ]

    def store_result(
        self,
        payload: StagePayload,
        state: FunCosyVoice3State,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Fun-CosyVoice3 vocoder did not return audio")
        audio_payload = audio_waveform_payload(wav, source_hint="Fun-CosyVoice3")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload = store_pipeline_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 16,
    max_batch_wait_ms: int = 30,
    flow_batch_admission_frames: int = DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
    flow_merge_max_gap_frames: int = 384,
    flow_merge_pad_budget_percent: float = 25.0,
    enable_dit_torch_compile: bool = False,
    enable_flow_cuda_graph: bool = True,
    flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] | None = None,
    enable_flow_estimator_trt: bool = False,
    hift_dtype: str = "float32",
    hift_max_padding_waste: float = 1.5,
    token_hop_len: int = TOKEN_HOP_LEN,
    token_max_hop_len: int = TOKEN_MAX_HOP_LEN,
    disable_hop_growth: bool = False,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
        FunCosyVoice3StreamingVocoderScheduler,
    )

    if flow_batch_admission_frames <= 0:
        raise ValueError("flow_batch_admission_frames must be greater than zero")

    reject_conflicting_dit_accelerators(
        enable_dit_torch_compile=enable_dit_torch_compile,
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )
    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = resolve_checkpoint(model_path)
    if dtype not in AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported Fun-CosyVoice3 vocoder dtype {dtype!r}; "
            f"expected one of {sorted(AUTOCAST_DTYPES)}"
        )
    autocast_dtype = AUTOCAST_DTYPES[dtype]
    flow, hift = load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )

    device_obj = torch.device(device)
    if enable_flow_cuda_graph and (
        device_obj.type != "cuda" or not torch.cuda.is_available()
    ):
        enable_flow_cuda_graph = False

    if enable_flow_cuda_graph:

        def _chunk_mask(
            xs: torch.Tensor,
            masks: torch.Tensor,
            use_dynamic_chunk: bool,
            use_dynamic_left_chunk: bool,
            decoding_chunk_size: int,
            static_chunk_size: int,
            num_decoding_left_chunks: int,
            enable_full_context: bool = True,
        ) -> torch.Tensor:
            if use_dynamic_chunk or static_chunk_size > 0:
                return cosyvoice_chunk_mask(
                    xs,
                    masks,
                    use_dynamic_chunk,
                    use_dynamic_left_chunk,
                    decoding_chunk_size,
                    static_chunk_size,
                    num_decoding_left_chunks,
                    enable_full_context,
                )
            empty_rows = masks.sum(dim=-1, keepdim=True) == 0
            masks.masked_fill_(empty_rows, True)
            return masks

        cosyvoice_dit.add_optional_chunk_mask = _chunk_mask

    if enable_dit_torch_compile:
        compile_dit_backbone(flow, autocast_dtype=autocast_dtype)

    if enable_flow_cuda_graph:
        capture_shapes = verify_flow_cuda_graph_capture_shapes(
            flow_cuda_graph_capture_shapes,
        )
        runner = FlowCudaGraphRunner(
            flow,
            device=device_obj,
            autocast_dtype=autocast_dtype,
        )
        runner.capture(capture_shapes)
        flow.attach_cuda_graph_runner(runner)

    vocoder = CosyVoice3Vocoder(
        flow,
        hift,
        autocast_dtype=autocast_dtype,
        flow_merge_max_gap_frames=flow_merge_max_gap_frames,
        flow_merge_pad_budget_percent=flow_merge_pad_budget_percent,
        hift_dtype=hift_dtype,
        hift_max_padding_waste=hift_max_padding_waste,
    )

    return FunCosyVoice3StreamingVocoderScheduler(
        vocoder,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder.flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
        token_hop_len=token_hop_len,
        token_max_hop_len=token_max_hop_len,
        disable_hop_growth=disable_hop_growth,
    )
