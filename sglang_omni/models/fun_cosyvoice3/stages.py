# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import bisect
import contextlib
import gc
import importlib
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from numpy.typing import ArrayLike
from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
)
from sglang.srt.utils.common import async_d2h, is_pin_memory_available
from torch._inductor.runtime.compile_tasks import _set_triton_libdevice_path
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from x_transformers.x_transformers import apply_rotary_pos_emb

if TYPE_CHECKING:
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM, ConditionalCFM
    from cosyvoice.hifigan.generator import CausalHiFTGenerator, SourceModuleHnNSF
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
        FunCosyVoice3StreamingVocoderScheduler,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

from sglang_omni.models.fun_cosyvoice3.causal_conv import (
    GROUP_CONV_KERNEL_CHANNELS,
    HAS_TRITON,
    FusedConvPositionEmbedding,
)
from sglang_omni.models.fun_cosyvoice3.config import reject_conflicting_dit_accelerators
from sglang_omni.models.fun_cosyvoice3.engine_builder import (
    FunCosyVoice3EngineBuilder,
    engine_headroom_bytes,
)
from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
    execute_flow_estimator,
    is_flow_estimator_trt,
)
from sglang_omni.models.fun_cosyvoice3.hift_channels_last import ChannelsLastHiftDecode
from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    CONV_CONTEXT_FRAMES,
    DIT_INDUCTOR_OPTIONS,
    FA3_DTYPES,
    PackedDiT,
    euler_step_times,
    gather_rows,
    pack_rows,
    scatter_rows,
    solve_flow_euler_packed,
    supports_ragged_attention,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    compile_forward_prefix,
    forward_prefix,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import (
    SolveGraphCapture,
    row_tiers,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
)
from sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph import (
    WholeHistoryCudaGraphRunner,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphPool, ReplayableGraph
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as load_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device, supports_device_streams

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

CAUSAL_CONV_CACHE_PATCHED = False

FLOW_CUDA_GRAPH_FRAME_BUCKET = 16
# note(ratish): the Flow's Euler steps per solve, CosyVoice's n_timesteps.
FLOW_EULER_STEPS = 10
# Note (chenyang):
# Mel-frame step size for buffered flow CUDA Graph keys. Capture shapes
# must use a T that is a multiple of this step size. For example, 489
# frames would be padded to 496 frames, replayed, and then cropped back
# to 489 frames.
# note(ratish): the HiFT decode's left receptive field in mel frames, its causal
# paddings over each stage's samples per frame (28.5, 24 from the stage 0 resblocks).
HIFT_DECODE_CONTEXT_FRAMES = 29


class HiFTDecode(Protocol):
    def __call__(
        self, x: torch.Tensor, s: torch.Tensor = ..., finalize: bool = ...
    ) -> torch.Tensor: ...


class MpsHiFTAdapter:
    """Keep HiFT's float64 F0 branch on CPU while decoding on MPS."""

    upsample_rates: list[int]
    istft_params: dict[str, int]
    conv_pre_look_right: int
    f0_upsamp: torch.nn.Upsample
    m_source: SourceModuleHnNSF
    decode: HiFTDecode

    def __init__(self, hift: "CausalHiFTGenerator", device: str) -> None:
        self.hift = hift
        self.device = torch.device(device)
        self.f0_predictor = hift.f0_predictor
        # Note (yexiaodong): MPS rejects float64 transfers; keep the F0 branch
        # on CPU while the remaining vocoder runs on MPS.
        self.f0_predictor.to(device="cpu")
        self.f0_predictor.to(dtype=torch.float64)

    def __getattr__(self, name: str) -> object:
        return getattr(self.hift, name)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.hift.parameters()

    @torch.inference_mode()
    def inference(
        self, speech_feat: torch.Tensor, finalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cpu_features = speech_feat.detach().to(device="cpu")
        f0 = self.f0_predictor(
            cpu_features.to(dtype=torch.float64),
            finalize=finalize,
        ).to(device=self.device, dtype=speech_feat.dtype)
        source = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = self.hift.m_source(source)
        source = source.transpose(1, 2)
        if finalize:
            generated = self.hift.decode(x=speech_feat, s=source, finalize=True)
        else:
            causal_padding = self.f0_predictor.condnet[0].causal_padding
            generated = self.hift.decode(
                x=speech_feat[:, :, :-causal_padding],
                s=source,
                finalize=False,
            )
        return generated, source


@dataclass(frozen=True)
class FlowBatchInput:
    token: torch.Tensor
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


@dataclass(kw_only=True, frozen=True)
class HiftStepRow:
    """One stream's mel history in a streaming HiFT step and the samples it has emitted."""

    history: torch.Tensor
    emitted_samples: int
    is_final: bool


@dataclass(kw_only=True, frozen=True)
class HiftStepSamples:
    """A streaming HiFT step's new samples per row and each row's emitted samples after
    it. With copy_done the new samples view a pinned host copy, complete once it is."""

    new_samples: list[torch.Tensor]
    emitted_samples: list[int]
    copy_done: torch.cuda.Event | None

    def wait(self) -> list[torch.Tensor]:
        """Each row's new samples on the host, once the copy has landed."""
        if self.copy_done is not None:
            self.copy_done.synchronize()
        else:
            pass
        return self.new_samples


@dataclass(kw_only=True, frozen=True)
class HiftDecodeWindow:
    """The mel frames a row's decode covers, from start_frame, emitting from
    emitted_frame up to end_frame."""

    row: HiftStepRow
    start_frame: int
    emitted_frame: int
    end_frame: int


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
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


logger = logging.getLogger(__name__)


def pack_flow_inputs(
    flow: CausalMaskedDiffWithDiT, inputs: Sequence[FlowBatchInput]
) -> PackedFlowBatch:
    if not inputs:
        raise ValueError("Flow batch must contain at least one input")
    else:
        pass
    for index, item in enumerate(inputs):
        if item.token.ndim != 2 or item.token.shape[0] != 1 or item.token.shape[1] <= 0:
            raise ValueError(f"input {index} token must have shape [1, target_tokens]")
        else:
            pass
        if item.prompt_token.ndim != 2 or item.prompt_token.shape[0] != 1:
            raise ValueError(
                f"input {index} prompt_token must have shape [1, prompt_tokens]"
            )
        else:
            pass
        if item.prompt_feat.ndim != 3 or item.prompt_feat.shape[0] != 1:
            raise ValueError(
                f"input {index} prompt_feat must have shape [1, prompt_frames, channels]"
            )
        else:
            pass
        if item.prompt_feat.shape[2] != flow.output_size:
            raise ValueError(
                f"input {index} prompt feature width must equal Flow output_size"
            )
        else:
            pass
        expected_frames = item.prompt_token.shape[1] * flow.token_mel_ratio
        if item.prompt_feat.shape[1] != expected_frames:
            raise ValueError(
                f"input {index} prompt feature length must equal prompt token length "
                f"times token_mel_ratio ({item.prompt_feat.shape[1]} != {expected_frames})"
            )
        else:
            pass
        if item.embedding.ndim != 2 or item.embedding.shape[0] != 1:
            raise ValueError(
                f"input {index} embedding must have shape [1, speaker_dim]"
            )
        else:
            pass
        expected_embedding_size = flow.spk_embed_affine_layer.in_features
        if item.embedding.shape[1] != expected_embedding_size:
            raise ValueError(
                f"input {index} embedding width must be {expected_embedding_size}"
            )
        else:
            pass

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
        combined_lengths,
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(device),
    ).to(device, non_blocking=True)

    max_tokens = max(combined_lengths)
    token = torch.zeros(len(inputs), max_tokens, dtype=torch.int32, device=device)
    for index, item in enumerate(inputs):
        prompt_length = prompt_lengths[index]
        token[index, :prompt_length] = item.prompt_token[0].to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        token[index, prompt_length : combined_lengths[index]] = item.token[0].to(
            device=device, dtype=torch.int32, non_blocking=True
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
            device=device, dtype=dtype, non_blocking=True
        )
    embedding = torch.cat(
        [
            item.embedding.to(device=device, dtype=dtype, non_blocking=True)
            for item in inputs
        ],
        dim=0,
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
    for t, dt in euler_step_times(time_span):
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
    return noisy_mel.float()


def verify_flow_cuda_graph_capture_shapes(
    capture_shapes: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if not capture_shapes:
        raise ValueError("flow_cuda_graph_capture_shapes must not be empty")
    else:
        pass
    for batch_size, mel_frame in capture_shapes:
        if batch_size <= 0 or mel_frame <= 0:
            raise ValueError(
                "flow_cuda_graph_capture_shapes entries must have positive "
                f"batch and mel_frame values; got {(batch_size, mel_frame)!r}"
            )
        else:
            pass
        if mel_frame % FLOW_CUDA_GRAPH_FRAME_BUCKET != 0:
            raise ValueError(
                "flow_cuda_graph_capture_shapes mel_frame values must be multiples "
                f"of {FLOW_CUDA_GRAPH_FRAME_BUCKET}; got {(batch_size, mel_frame)!r}"
            )
        else:
            pass
    return capture_shapes


@dataclass
class CapturedFlowCudaGraph:
    graph: ReplayableGraph
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
        self.device_module: ModuleType = torch.get_device_module(self.device)
        self.graphs: dict[tuple[int, int], CapturedFlowCudaGraph] = {}
        self.pool: DeviceGraphPool | None = None

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
        time_span = torch.linspace(
            0, 1, FLOW_EULER_STEPS + 1, device=model_device, dtype=parameter_dtype
        )
        if decoder.t_scheduler == "cosine":
            time_span = 1 - torch.cos(time_span * 0.5 * torch.pi)
        else:
            pass
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
        # note(ratish): CosyVoice's loader leaves an onnxruntime session in a cycle;
        # a collection inside the capture would free it there and invalidate the graph.
        gc.collect()
        graph_backend = current_platform.get_device_graph_backend(self.device)
        assert graph_backend is not None, f"{self.device} records no Flow graphs"
        graphs: dict[tuple[int, int], CapturedFlowCudaGraph] = {}
        current_stream = self.device_module.current_stream(self.device)
        stream = self.device_module.Stream(self.device)
        stream.wait_stream(current_stream)
        with (
            current_platform.graph_capture_attention(),
            self.device_module.device(self.device),
            self.device_module.stream(stream),
        ):
            self.pool = graph_backend.graph_pool_handle()
            for batch_size, mel_frame in capture_shapes:
                static_inputs = self.capture_inputs(batch_size, mel_frame)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_dtype is not None,
                ):
                    solve_flow_euler(self.flow.decoder, *static_inputs)
                with (
                    graph_backend.capture(
                        pool=self.pool, stream=stream, thread_local_errors=True
                    ) as graph,
                    torch.autocast(
                        device_type=self.device.type,
                        dtype=self.autocast_dtype,
                        enabled=self.autocast_dtype is not None,
                    ),
                ):
                    static_output = solve_flow_euler(self.flow.decoder, *static_inputs)
                graphs[(batch_size, mel_frame)] = CapturedFlowCudaGraph(
                    graph=graph,
                    static_inputs=static_inputs,
                    static_output=static_output,
                )
        current_stream.wait_stream(stream)
        self.device_module.empty_cache()
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
                    self.device_module.device(self.device),
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


@dataclass(frozen=True)
class FlowConditioning:
    token_condition: torch.Tensor
    mel_lengths: tuple[int, ...]
    speaker_embedding: torch.Tensor
    prompt_mel: torch.Tensor
    noisy_mel: torch.Tensor
    time_span: torch.Tensor


def prepare_flow_conditioning(
    flow: FunCosyVoice3Flow,
    packed: PackedFlowBatch,
    *,
    finalize: bool,
) -> FlowConditioning:
    """Encoder, prompt mel, noise and time schedule of one Flow call in the
    padded (rows, channels, frames) layout, with each row's mel length."""
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
    else:
        pass
    if max_mel_frame > decoder.rand_noise.shape[2]:
        raise ValueError(
            f"decoder.rand_noise supports {decoder.rand_noise.shape[2]} frames, "
            f"but batch requires {max_mel_frame}"
        )
    else:
        pass
    if lookahead > 0:
        mel_lengths = tuple(
            max(length - lookahead, 0) * flow.token_mel_ratio
            for length in packed.combined_token_lengths
        )
    else:
        mel_lengths = packed.total_mel_lengths
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
        0,
        1,
        FLOW_EULER_STEPS + 1,
        device=token_condition.device,
        dtype=token_condition.dtype,
    )
    if decoder.t_scheduler == "cosine":
        time_span = 1 - torch.cos(unit_span * 0.5 * torch.pi)
    else:
        time_span = unit_span
    return FlowConditioning(
        token_condition=token_condition,
        mel_lengths=mel_lengths,
        speaker_embedding=speaker_embedding,
        prompt_mel=prompt_mel,
        noisy_mel=noisy_mel,
        time_span=time_span,
    )


@torch.inference_mode()
def generate_flow(
    flow: FunCosyVoice3Flow,
    packed: PackedFlowBatch,
    *,
    streaming: bool = False,
    finalize: bool = True,
) -> torch.Tensor:
    """Padded Flow call, the graphed non-streaming path or the eager solve."""
    conditioning = prepare_flow_conditioning(flow, packed, finalize=finalize)
    token_condition = conditioning.token_condition
    decoder = flow.decoder
    # Note (chenyang): int64 matches torch.arange's default so the comparison
    # below does not mix integer dtypes.
    mel_lengths = torch.tensor(
        conditioning.mel_lengths, dtype=torch.int64, device=token_condition.device
    )
    mel_mask = (
        (
            torch.arange(
                token_condition.shape[2], device=token_condition.device
            ).unsqueeze(0)
            < mel_lengths.unsqueeze(1)
        )
        .unsqueeze(1)
        .to(token_condition.dtype)
    )
    if streaming or not finalize or flow.cuda_graph_runner is None:
        return solve_flow_euler(
            decoder,
            conditioning.noisy_mel,
            conditioning.time_span,
            token_condition,
            mel_mask,
            conditioning.speaker_embedding,
            conditioning.prompt_mel,
            streaming=streaming,
        )
    else:
        pass
    generated = flow.cuda_graph_runner.run(
        conditioning.noisy_mel,
        conditioning.time_span,
        token_condition,
        mel_mask,
        conditioning.speaker_embedding,
        conditioning.prompt_mel,
    )
    if generated is not None:
        return generated
    else:
        pass
    return solve_flow_euler(
        decoder,
        conditioning.noisy_mel,
        conditioning.time_span,
        token_condition,
        mel_mask,
        conditioning.speaker_embedding,
        conditioning.prompt_mel,
        streaming=False,
    )


@torch.inference_mode()
def generate_flow_packed(
    flow: FunCosyVoice3Flow,
    packed: PackedFlowBatch,
    *,
    streaming: bool,
    finalize: bool,
) -> torch.Tensor:
    """Flow call over the rows packed along the sequence, a non-streaming call
    replayed from the whole history graphs when one holds its rows. Returns the
    padded (rows, channels, frames) layout the mel split reads."""
    conditioning = prepare_flow_conditioning(flow, packed, finalize=finalize)
    token_condition = conditioning.token_condition
    device = token_condition.device
    rows = pack_rows(conditioning.mel_lengths, device)
    noise = gather_rows(conditioning.noisy_mel.transpose(1, 2), rows)
    mu = gather_rows(token_condition.transpose(1, 2), rows)
    mel_conditioning = gather_rows(conditioning.prompt_mel.transpose(1, 2), rows)
    if streaming or flow.whole_history_cuda_graph_runner is None:
        generated = None
    else:
        generated = flow.whole_history_cuda_graph_runner.run(
            noise=noise,
            time_span=conditioning.time_span,
            mu=mu,
            speaker_embeddings=conditioning.speaker_embedding,
            mel_conditioning=mel_conditioning,
            lengths=rows.lengths,
        )
    if generated is None:
        estimator = flow.packed_estimator
        twin_rows = pack_rows(rows.lengths * 2, device)
        generated = solve_flow_euler_packed(
            estimator,
            noise,
            conditioning.time_span,
            mu,
            conditioning.speaker_embedding,
            mel_conditioning,
            twin_rows,
            estimator.row_attention(
                twin_rows,
                streaming=streaming,
                dtype=conditioning.speaker_embedding.dtype,
            ),
            estimator.rope_angles(twin_rows.width),
            cfg_rate=flow.decoder.inference_cfg_rate,
        )
    else:
        pass
    return scatter_rows(generated, rows, token_condition.shape[2]).transpose(1, 2)


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
        else:
            pass
        outputs.append(mel)
    return outputs


class FunCosyVoice3Flow:
    """CosyVoice3 Flow with batch inference enabled as its default API."""

    spk_embed_affine_layer: torch.nn.Linear
    input_embedding: torch.nn.Embedding
    pre_lookahead_layer: PreLookaheadLayer
    pre_lookahead_len: int
    token_mel_ratio: int
    input_frame_rate: int
    output_size: int
    decoder: CausalConditionalCFM

    def __init__(
        self,
        flow: CausalMaskedDiffWithDiT,
        packed_estimator: PackedDiT | None = None,
    ) -> None:
        self.flow: CausalMaskedDiffWithDiT = flow
        self.cuda_graph_runner: FlowCudaGraphRunner | None = None
        # note(ratish): the eager DiT over packed rows; None with the TensorRT
        # estimator, whose fixed (2, 80, T) profile keeps the padded layout.
        self.packed_estimator = packed_estimator
        self.is_buffered_packed = False
        self.prefix_pool: PrefixKVPool | None = None
        self.prefix_cuda_graph_runner: PrefixCudaGraphRunner | None = None
        self.whole_history_cuda_graph_runner: WholeHistoryCudaGraphRunner | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.flow, name)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.flow.parameters()

    def to(
        self,
        *args: str
        | int
        | torch.device
        | torch.dtype
        | torch.Tensor
        | torch.memory_format
        | bool
        | None,
        **kwargs: (
            str
            | int
            | torch.device
            | torch.dtype
            | torch.Tensor
            | torch.memory_format
            | bool
            | None
        ),
    ) -> "FunCosyVoice3Flow":
        self.flow.to(*args, **kwargs)
        return self

    def eval(self) -> "FunCosyVoice3Flow":
        self.flow.eval()
        return self

    def attach_cuda_graph_runner(self, runner: FlowCudaGraphRunner) -> None:
        self.cuda_graph_runner = runner

    @torch.inference_mode()
    def inference(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        if self.is_buffered_packed:
            return self.inference_leftover(inputs)
        else:
            pass
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
    def inference_leftover(
        self, inputs: Sequence[FlowBatchInput]
    ) -> list[torch.Tensor]:
        """Non-streaming Flow over each row's whole token history, the rows
        packed along the sequence."""
        if self.packed_estimator is None:
            return self.inference(inputs)
        else:
            pass
        packed = pack_flow_inputs(self.flow, inputs)
        generated = generate_flow_packed(self, packed, streaming=False, finalize=True)
        return split_generated_mels(
            self.flow,
            packed,
            generated,
            token_lengths=packed.combined_token_lengths,
            target_token_lengths=packed.target_token_lengths,
        )

    @torch.inference_mode()
    def inference_causal_prefix(
        self,
        inputs: Sequence[FlowBatchInput],
        caches: Sequence[tuple[PrefixCacheRow, PrefixCacheRow]],
    ) -> list[torch.Tensor]:
        """Causal hops over the frames each row adds past its cached prefix;
        the returned mels are zero over the prefix."""
        assert self.packed_estimator is not None and self.prefix_pool is not None
        packed = pack_flow_inputs(self.flow, inputs)
        conditioning = prepare_flow_conditioning(self, packed, finalize=False)
        token_condition = conditioning.token_condition
        device = token_condition.device
        total_frames = list(conditioning.mel_lengths)
        prefix_frames = [pair[0].committed_frames for pair in caches]
        new_frames = [
            total - start
            for total, start in zip(total_frames, prefix_frames, strict=True)
        ]
        if min(new_frames) <= 0:
            raise RuntimeError("Fun-CosyVoice3 prefix hop adds no frames")
        else:
            pass
        new_frame_index = torch.cat(
            [
                torch.arange(start, total, device=device)
                + row * token_condition.shape[2]
                for row, (start, total) in enumerate(
                    zip(prefix_frames, total_frames, strict=True)
                )
            ]
        )

        def take_new_frames(padded: torch.Tensor) -> torch.Tensor:
            flat = padded.transpose(1, 2).reshape(-1, padded.shape[1])
            return flat[new_frame_index].unsqueeze(0)

        noise = take_new_frames(conditioning.noisy_mel)
        mu = take_new_frames(token_condition)
        mel_conditioning = take_new_frames(conditioning.prompt_mel)
        if self.prefix_cuda_graph_runner is None:
            generated = None
        else:
            generated = self.prefix_cuda_graph_runner.run(
                noise=noise,
                time_span=conditioning.time_span,
                mu=mu,
                speaker_embeddings=conditioning.speaker_embedding,
                mel_conditioning=mel_conditioning,
                new_frames=new_frames,
                caches=list(caches),
            )
        if generated is None:
            generated = solve_flow_euler_prefix(
                self.packed_estimator,
                self.prefix_pool,
                noise,
                conditioning.time_span,
                mu,
                conditioning.speaker_embedding,
                mel_conditioning,
                new_frames,
                list(caches),
                cfg_rate=self.flow.decoder.inference_cfg_rate,
            )
        else:
            pass
        padded = generated.new_zeros(
            len(inputs), token_condition.shape[2], generated.shape[2]
        )
        padded.view(-1, generated.shape[2])[new_frame_index] = generated[0]
        generated = padded.transpose(1, 2)
        lookahead = self.flow.pre_lookahead_len
        return split_generated_mels(
            self.flow,
            packed,
            generated,
            token_lengths=tuple(
                max(length - lookahead, 0) for length in packed.combined_token_lengths
            ),
            target_token_lengths=tuple(
                max(length - lookahead, 0) for length in packed.target_token_lengths
            ),
        )

    @torch.inference_mode()
    def inference_causal(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        # note (guozhihao-224): causal hops (first and follow-up). Same
        # packing as buffered inference, but strip lookahead per row so
        # mixed prompt lengths can share one DiT call. streaming=True
        # keeps the chunk mask aligned with CosyVoice3Model hops.
        packed = pack_flow_inputs(self.flow, inputs)
        if self.packed_estimator is None:
            generated = generate_flow(self, packed, streaming=True, finalize=False)
        else:
            generated = generate_flow_packed(
                self, packed, streaming=True, finalize=False
            )
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
    else:
        pass
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        raise RuntimeError(
            "enable_flow_estimator_trt requires NVIDIA CUDA, "
            f"got platform {current_platform.device_type!r}"
        )
    else:
        pass

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
    flow.packed_estimator = None
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
    autocast_dtype: torch.dtype | None,
    enable_flow_estimator_trt: bool = False,
) -> "tuple[FunCosyVoice3Flow, CausalHiFTGenerator | MpsHiFTAdapter]":
    if torch.device(device).type == "mps":
        return load_cosyvoice3_flow_hift_lightweight(checkpoint_dir, device=device)
    else:
        pass
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
        else:
            pass
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    keep_hift_constants_on_device(hift, device)
    patch_sine_harmonics(hift, device)
    # note (ratish): a plain attribute, so flow.to(device) leaves it on the host and
    # every Flow call uploads its slice, waiting for the work queued on the stream.
    flow.decoder.rand_noise = flow.decoder.rand_noise.to(device)
    patch_causal_conv_cache()
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
    if autocast_dtype is not None and torch.device(device).type == "cuda":
        # note(ratish): autocast caches no weight cast under inference mode, so
        # each Linear and Conv1d would recast its weights on every Euler step.
        for module in flow.decoder.estimator.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.to(autocast_dtype)
            else:
                pass
        input_embed = flow.decoder.estimator.input_embed
        conv = input_embed.conv_pos_embed.conv1[0]
        if HAS_TRITON and conv.in_channels // conv.groups in GROUP_CONV_KERNEL_CHANNELS:
            input_embed.conv_pos_embed = FusedConvPositionEmbedding(
                input_embed.conv_pos_embed
            )
        else:
            pass
    else:
        pass
    wrapped = FunCosyVoice3Flow(
        flow, packed_estimator=PackedDiT(flow.decoder.estimator, device=device)
    )
    if enable_flow_estimator_trt:
        attach_flow_estimator_trt(wrapped, checkpoint_dir, device)
    else:
        pass
    return wrapped, hift


def patch_chunk_mask() -> None:
    """Build the DiT attention mask without the host sync CosyVoice's
    add_optional_chunk_mask pays to check for empty rows; the rows are
    filled on the device instead, which is also what graph capture needs.
    The check is any, not sum: summing a bool mask first copies it to
    int64, eight bytes per element of a batch by frames squared tensor.
    """
    try:
        from cosyvoice.flow.DiT import dit as cosyvoice_dit
        from cosyvoice.utils.mask import add_optional_chunk_mask as cosyvoice_chunk_mask
        from cosyvoice.utils.mask import subsequent_chunk_mask
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    def chunk_mask(
        xs: torch.Tensor,
        masks: torch.Tensor,
        use_dynamic_chunk: bool,
        use_dynamic_left_chunk: bool,
        decoding_chunk_size: int,
        static_chunk_size: int,
        num_decoding_left_chunks: int,
        enable_full_context: bool = True,
    ) -> torch.Tensor:
        if use_dynamic_chunk:
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
        else:
            pass
        if static_chunk_size > 0:
            chunk = subsequent_chunk_mask(
                xs.size(1), static_chunk_size, num_decoding_left_chunks, xs.device
            )
            masks = masks & chunk.unsqueeze(0)
        else:
            pass
        empty_rows = ~masks.any(dim=-1, keepdim=True)
        masks.masked_fill_(empty_rows, True)
        return masks

    cosyvoice_dit.add_optional_chunk_mask = chunk_mask


def patch_causal_conv_cache() -> None:
    """Allocate CausalConv1d's zero cache on the device. CosyVoice builds it
    on the CPU and copies it in, one host sync per conv per HiFT call.
    """
    global CAUSAL_CONV_CACHE_PATCHED
    if CAUSAL_CONV_CACHE_PATCHED:
        return
    else:
        pass
    try:
        from cosyvoice.transformer.convolution import CausalConv1d
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    original_forward = CausalConv1d.forward

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            cache = x.new_zeros(x.shape[0], x.shape[1], self.causal_padding)
        else:
            pass
        return original_forward(self, x, cache)

    CausalConv1d.forward = forward
    CAUSAL_CONV_CACHE_PATCHED = True


def keep_hift_constants_on_device(hift: "CausalHiFTGenerator", device: str) -> None:
    # note(ratish): plain attributes, not buffers, so hift.to(device) leaves
    # them on the CPU and every HiFT call copies them to the device again.
    hift.stft_window = hift.stft_window.to(device)
    sine_gen = hift.m_source.l_sin_gen
    sine_gen.rand_ini = sine_gen.rand_ini.to(device)
    sine_gen.sine_waves = sine_gen.sine_waves.to(device)
    # note(ratish): HiFT.inference casts the f0 predictor to float64 on every
    # call; done once here the per-call cast finds nothing to convert.
    hift.f0_predictor.to(torch.float64)


def patch_sine_harmonics(hift: "CausalHiFTGenerator", device: str) -> None:
    """Give SineGen2 its harmonic multipliers on the device, bit for bit. Its forward
    builds them on the host every call, and their pageable upload waits for all the work
    queued on the stream."""
    sine_generator = hift.m_source.l_sin_gen
    harmonics = torch.arange(
        1, sine_generator.harmonic_num + 2, dtype=torch.float32, device=device
    ).view(1, 1, -1)

    def forward(f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sine_waves = (
            sine_generator._f02sine(  # noqa: leading-underscore  # upstream name
                torch.multiply(f0, harmonics)
            )
            * sine_generator.sine_amp
        )
        uv = sine_generator._f02uv(f0)  # noqa: leading-underscore  # upstream name
        noise_amp = (
            uv * sine_generator.noise_std + (1 - uv) * sine_generator.sine_amp / 3
        )
        noise = noise_amp * sine_generator.sine_waves[:, : sine_waves.shape[1]]
        return sine_waves * uv + noise, uv, noise

    sine_generator.forward = forward


def load_cosyvoice3_flow_hift_lightweight(
    checkpoint_dir: str,
    *,
    device: str,
) -> "tuple[FunCosyVoice3Flow, CausalHiFTGenerator | MpsHiFTAdapter]":
    """Load only Flow and HiFT for CPU/MPS without constructing a second LLM."""
    try:
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    config_path = os.path.join(checkpoint_dir, "cosyvoice3.yaml")
    flow_path = os.path.join(checkpoint_dir, "flow.pt")
    hift_path = os.path.join(checkpoint_dir, "hift.pt")
    if not all(os.path.isfile(path) for path in (config_path, flow_path, hift_path)):
        raise FileNotFoundError(
            "Fun-CosyVoice3 requires cosyvoice3.yaml, flow.pt and hift.pt in "
            f"{checkpoint_dir}"
        )
    else:
        pass

    with open(config_path, encoding="utf-8") as handle:
        configs = load_hyperpyyaml(
            handle,
            overrides={
                "qwen_pretrain_path": os.path.join(checkpoint_dir, "CosyVoice-BlankEN"),
                "llm": None,
                "hifigan": None,
            },
        )
    flow = configs["flow"]
    hift = configs["hift"]
    flow.load_state_dict(torch.load(flow_path, map_location="cpu", weights_only=True))
    hift_state = {
        key.removeprefix("generator."): value
        for key, value in torch.load(
            hift_path, map_location="cpu", weights_only=True
        ).items()
    }
    hift.load_state_dict(hift_state, strict=True)
    flow.to(device).eval()
    hift.to(device).eval()
    if (
        torch.device(device).type == "mps"
        and not current_platform.is_float64_supported()
    ):
        hift = MpsHiFTAdapter(hift, device)
    else:
        pass
    del configs
    return (
        FunCosyVoice3Flow(
            flow, packed_estimator=PackedDiT(flow.decoder.estimator, device=device)
        ),
        hift,
    )


def set_dit_compile_config() -> None:
    """Process-wide Dynamo and Inductor settings every DiT compile reads."""
    inductor_config.fx_graph_cache = True
    dynamo.config.cache_size_limit = 1024
    dynamo.config.accumulated_cache_size_limit = 1024
    # note(ratish): Inductor picks CUDA's libdevice on its first kernel build;
    # cache entries record the file, so a warm boot's first lookups would miss.
    with inductor_config.patch({"eager_numerics.use_pytorch_libdevice": True}):
        _set_triton_libdevice_path()


def compile_dit_backbone(
    flow: FunCosyVoice3Flow,
    *,
    warmup_mel_frames: int = 128,
    warmup_steps: int = 3,
    autocast_dtype: torch.dtype | None = None,
) -> None:

    estimator = flow.decoder.estimator
    if not isinstance(estimator, torch.nn.Module):
        raise RuntimeError(
            "Fun-CosyVoice3 DiT torch.compile requires a PyTorch estimator, "
            f"got {type(estimator).__name__}"
        )
    else:
        pass
    if warmup_mel_frames < 2:
        raise ValueError(f"warmup_mel_frames must be >= 2, got {warmup_mel_frames}")
    else:
        pass

    try:
        from cosyvoice.flow.DiT import modules as cosyvoice_dit_modules
    except ImportError as exc:
        raise RuntimeError(COSYVOICE_INSTALL_HINT) from exc

    original_forward = estimator.forward
    set_dit_compile_config()
    # note(ratish): the AOT cache refuses the autocast-disabled regions around RoPE;
    # these match without them: an outer product and elementwise ops.
    rotary = estimator.rotary_embed
    assert rotary.scale is None, "the DiT's RoPE has no xpos scale"

    def rotary_frequencies(positions: torch.Tensor) -> tuple[torch.Tensor, float]:
        freqs = positions.to(rotary.inv_freq.dtype)[None, :, None] * rotary.inv_freq
        freqs = freqs / rotary.interpolation_factor
        return torch.stack((freqs, freqs), dim=-1).flatten(-2), 1.0

    rotary.forward = rotary_frequencies
    cosyvoice_dit_modules.apply_rotary_pos_emb = apply_rotary_pos_emb.__wrapped__
    try:
        # note(ratish): not dynamic=True, which makes the LayerNorm eps a symbol;
        # Inductor cannot keep it and restarts the compile.
        estimator.forward = torch.compile(
            original_forward, options=dict(DIT_INDUCTOR_OPTIONS)
        )
        # note(chenye): synthetic inputs must use the dtype serving presents to
        # the estimator, including the effective autocast dtype.
        param = next(flow.parameters())
        device = param.device
        parameter_dtype = param.dtype
        warmup_dtype = autocast_dtype or parameter_dtype
        mel_frame = int(warmup_mel_frames)
        with torch.inference_mode():
            for streaming in (False, True):
                for _ in range(warmup_steps):
                    # CFG batch 2; mel dim 80 matches pinned checkpoint proj_out.
                    noisy_mel = torch.randn(
                        2, 80, mel_frame, device=device, dtype=warmup_dtype
                    )
                    mel_mask = torch.ones(
                        2, 1, mel_frame, device=device, dtype=warmup_dtype
                    )
                    token_condition = torch.randn(
                        2, 80, mel_frame, device=device, dtype=warmup_dtype
                    )
                    flow_time = torch.zeros(1, device=device, dtype=warmup_dtype)
                    speaker_embedding = torch.randn(
                        2, 80, device=device, dtype=warmup_dtype
                    )
                    prompt_mel = torch.randn(
                        2, 80, mel_frame, device=device, dtype=warmup_dtype
                    )
                    # note(ratish): hints, not constraints;
                    # batch and length start symbolic, not learned from later calls.
                    for tensor in (noisy_mel, mel_mask, token_condition, prompt_mel):
                        dynamo.maybe_mark_dynamic(tensor, (0, 2))
                    dynamo.maybe_mark_dynamic(speaker_embedding, 0)
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
        raise RuntimeError(
            "Fun-CosyVoice3 native DiT torch.compile startup warmup failed"
        ) from exc
    logger.info(
        "Compiled Fun-CosyVoice3 DiT backbone "
        f"(automatic dynamic, autocast_dtype={autocast_dtype}, "
        f"warmup_mel_frames={warmup_mel_frames}, warmup_steps={warmup_steps}, "
        "streaming=False/True)"
    )


def build_prefix_pool(
    flow: FunCosyVoice3Flow,
    device: torch.device,
    dtype: torch.dtype,
    *,
    size_gb: float | None,
    max_blocks: int | None,
    stage_reserve_bytes: int,
    headroom_bytes: int,
    share_ceiling_bytes: int | None,
) -> PrefixKVPool | None:
    """The pool at the memory this process can still take less the stage's reserve
    and the runtime headroom, at most size_gb or max_blocks when set; None when that
    holds no block."""
    estimator = flow.decoder.estimator
    attention = estimator.transformer_blocks[0].attn
    layer_num = len(estimator.transformer_blocks)
    head_num = int(attention.heads)
    head_dim = int(attention.inner_dim) // head_num
    bytes_per_frame = PrefixKVPool.bytes_per_frame(
        layer_num=layer_num,
        euler_steps=FLOW_EULER_STEPS,
        head_num=head_num,
        head_dim=head_dim,
        dtype=dtype,
    )
    # note (ratish): a cached row holds a block at least and keeps one conv context
    # beside the pool, so each block is charged a context.
    conv_context_bytes = (
        FLOW_EULER_STEPS
        * 2
        * CONV_CONTEXT_FRAMES
        * int(estimator.input_embed.proj.out_features)
        * torch.tensor([], dtype=dtype).element_size()
    )
    block_bytes = BLOCK_FRAMES * bytes_per_frame + conv_context_bytes
    # note (ratish): the runtime headroom stays free on the device and under the
    # process's allocator cap (total_reserve_bytes); a declared vocoder share
    # (gpu_memory_fraction) bounds this stage's own allocations.
    gc.collect()
    torch.accelerator.empty_cache()
    free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
    reserved_bytes = torch.accelerator.memory_reserved(device)
    allocator_cap_bytes = (
        torch.get_device_module(device).get_per_process_memory_fraction(device)
        * total_bytes
    )
    budget_bytes = (
        min(free_bytes, allocator_cap_bytes - reserved_bytes)
        - headroom_bytes
        - stage_reserve_bytes
    )
    if share_ceiling_bytes is not None:
        budget_bytes = min(
            budget_bytes, share_ceiling_bytes - reserved_bytes - stage_reserve_bytes
        )
    else:
        pass
    if size_gb is None:
        pass
    elif size_gb * 2**30 > budget_bytes:
        logger.warning(
            f"Fun-CosyVoice3 flow_prefix_cache_gb={size_gb} is larger than the "
            f"{budget_bytes / 2**30:.2f} GB left after the reserve, which is used instead"
        )
    else:
        budget_bytes = size_gb * 2**30
    # note (ratish): every key and value tensor also holds the padding block.
    capacity_frames = (
        int((budget_bytes - BLOCK_FRAMES * bytes_per_frame) // block_bytes)
        * BLOCK_FRAMES
    )
    if max_blocks is not None:
        capacity_frames = min(capacity_frames, max_blocks * BLOCK_FRAMES)
    else:
        pass
    reserve_gb = (stage_reserve_bytes + headroom_bytes) / 2**30
    while capacity_frames >= BLOCK_FRAMES:
        pool = PrefixKVPool(
            layer_num=layer_num,
            euler_steps=FLOW_EULER_STEPS,
            head_num=head_num,
            head_dim=head_dim,
            capacity_frames=capacity_frames,
            device=device,
            dtype=dtype,
        )
        # note (ratish): the allocator maps each tensor in its own rounded segment, so
        # the pool's mapped bytes are read back rather than derived.
        pool_bytes = torch.accelerator.memory_reserved(device) - reserved_bytes
        overshoot_bytes = (
            pool_bytes
            + capacity_frames // BLOCK_FRAMES * conv_context_bytes
            - budget_bytes
        )
        if overshoot_bytes <= 0:
            logger.info(
                f"Fun-CosyVoice3 Flow prefix cache: {pool.free_frames} frames "
                f"({pool_bytes / 2**30:.1f} GB mapped, {bytes_per_frame} bytes per "
                f"frame, size_gb={size_gb}, reserve {reserve_gb:.2f} GB)"
            )
            return pool
        else:
            del pool
            torch.accelerator.empty_cache()
            capacity_frames -= -(-int(overshoot_bytes) // block_bytes) * BLOCK_FRAMES
    logger.warning(
        f"Fun-CosyVoice3 Flow prefix cache off: {budget_bytes / 2**30:.2f} GB "
        f"left after the {reserve_gb:.2f} GB reserve"
    )
    return None


def create_preprocessing_executor(
    model_path: str,
    max_concurrency: int = 8,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")
    else:
        pass
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
    server_args_overrides: Mapping[str, object] | None = None,
    onnx_intra_op_threads: int = 16,
    token_hop_len: int = TOKEN_HOP_LEN,
) -> OmniScheduler[CosyVoice3SGLangRequestData]:
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
    else:
        pass

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
            else:
                pass
            return None
        else:
            pass
        if request_count - suffix_start < remaining_group_count:
            return None
        else:
            pass

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
            else:
                pass
            suffix_plan = optimal_suffix_partition(
                suffix_start=group_end,
                remaining_group_count=remaining_group_count - 1,
                current_max_group_gap_frames=max(
                    current_max_group_gap_frames, group_gap_frames
                ),
            )
            if suffix_plan is None:
                continue
            else:
                pass
            candidate = (
                (group_end - suffix_start) * longest_frames + suffix_plan[0],
                suffix_plan[1],
                (group_end,) + suffix_plan[2],
            )
            if best_plan is None or candidate < best_plan:
                best_plan = candidate
            else:
                pass
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
        else:
            pass

    raise AssertionError("valid Flow requests must have a feasible partition")


class CosyVoice3Vocoder(BatchVocoderBase[FunCosyVoice3State, torch.Tensor]):
    def __init__(
        self,
        flow: FunCosyVoice3Flow | CausalMaskedDiffWithDiT,
        hift: "CausalHiFTGenerator | MpsHiFTAdapter",
        autocast_dtype: torch.dtype | None = None,
        hift_dtype: str = "float32",
        hift_max_padding_waste: float = 1.5,
        flow_merge_max_gap_frames: int = 384,
        flow_merge_pad_budget_percent: float = 25.0,
    ) -> None:
        if hift_max_padding_waste < 1.0:
            raise ValueError("hift_max_padding_waste must be at least 1.0")
        else:
            pass
        if hift_dtype not in AUTOCAST_DTYPES:
            raise ValueError(
                f"Unsupported Fun-CosyVoice3 HiFT dtype {hift_dtype!r}; "
                f"expected one of {sorted(AUTOCAST_DTYPES)}"
            )
        else:
            pass
        estimator = flow.decoder.estimator
        if not isinstance(estimator, torch.nn.Module) and not is_flow_estimator_trt(
            estimator
        ):
            raise RuntimeError(
                "Fun-CosyVoice3 Flow estimator must be a PyTorch module or a "
                "TensorRT wrapper exposing acquire_estimator / execute"
            )
        else:
            pass
        self.flow = (
            flow if isinstance(flow, FunCosyVoice3Flow) else FunCosyVoice3Flow(flow)
        )
        self.hift = hift
        self.autocast_dtype = autocast_dtype
        self.flow_merge_max_gap_frames = flow_merge_max_gap_frames
        self.flow_merge_pad_budget_percent = flow_merge_pad_budget_percent
        self.hift_autocast_dtype = AUTOCAST_DTYPES[hift_dtype]
        self.hift_max_padding_waste = hift_max_padding_waste
        self.hift_samples_per_mel_frame = int(
            np.prod(hift.upsample_rates) * hift.istft_params["hop_len"]
        )
        # note(ratish): a non-final HiFT call holds back the F0 predictor's and
        # conv_pre's look right, and the trailing frame its ISTFT trims.
        self.hift_hold_frames = (
            hift.f0_predictor.condnet[0].causal_padding + hift.conv_pre_look_right + 1
        )
        self.hift_decode_cuda_graph_runner: HiftDecodeCudaGraphRunner | None = None
        # note(ratish): the AR shares this process and card on its own top priority
        # stream; on a stream of its own the vocoder's kernels and host copies do not
        # queue behind the AR's. It waits once for what the default stream holds at
        # this point.
        device = next(self.flow.parameters()).device
        if device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            self.stream_context: contextlib.AbstractContextManager[None] = (
                torch.cuda.stream(stream)
            )
            # note (ratish): a ROCm device is a cuda device too, and the kernels' inline
            # PTX cannot compile there.
            self.has_hift_kernels = (
                current_platform.is_cuda() and self.hift_autocast_dtype is None
            )
            if self.has_hift_kernels:
                hift.decode = ChannelsLastHiftDecode(hift, device)
            else:
                pass
            # note (ratish): a step completes after the next one is launched, so two
            # events serve every step in turn; a new event's first record waits while
            # another thread of the process is inside the CUDA runtime.
            self.copy_done_events: list[torch.cuda.Event] = [
                torch.cuda.Event() for _ in range(2)
            ]
        else:
            self.stream_context = contextlib.nullcontext()
            self.has_hift_kernels = False
            self.copy_done_events = []
        self.copy_done_slot: int = 0

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_pipeline_state(payload, FunCosyVoice3State)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        else:
            pass
        # note (guozhihao-224): AR stores one token per step, serialized as
        # [T, 1]; Flow takes a single unbatched sequence.
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
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

        results: list[tuple[torch.Tensor, int] | None] = [None] * len(items)
        flow_groups = adaptive_flow_requests_grouping(
            prepared,
            flow_merge_max_gap_frames=self.flow_merge_max_gap_frames,
            flow_merge_pad_budget_percent=self.flow_merge_pad_budget_percent,
        )
        flow_device = next(self.flow.parameters()).device
        with self.stream_context:
            for flow_group in flow_groups:
                with torch.autocast(
                    device_type=flow_device.type,
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
                group: list[tuple[PreparedFlowRequest, torch.Tensor]] = []
                total = 0
                longest = 0
                max_waste = self.hift_max_padding_waste
                hift_groups: list[list[tuple[PreparedFlowRequest, torch.Tensor]]] = []
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
                    else:
                        pass
                    group.append(pair)
                    total, longest = candidate_total, candidate_longest
                if group:
                    hift_groups.append(group)
                else:
                    pass
                for group in hift_groups:
                    wavs = self.mel2wav_batch([mel for _, mel in group])
                    for (request, _), wav in zip(group, wavs, strict=True):
                        results[request.index] = (wav, request.sample_rate)

        if any(result is None for result in results):
            raise RuntimeError("Fun-CosyVoice3 vocoder did not decode every request")
        else:
            pass
        return [cast(tuple[torch.Tensor, int], result) for result in results]

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.decode_payloads([payload])
        if len(results) != 1:
            raise RuntimeError(
                f"Fun-CosyVoice3 vocoder returned {len(results)} results for 1 input"
            )
        else:
            pass
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
        else:
            pass
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
        tts_mel = tts_mel[:, :, offset * self.flow.token_mel_ratio :]
        return self.hift_delta(
            tts_mel, hift_mel=hift_mel, speech_offset=speech_offset, finalize=finalize
        )

    def hop_batch(self, items: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        """Causal Flow for one hop per row, the rows packed along the sequence
        with attention within each row; the scheduler keeps the frames past
        token_offset and runs HiFT over the step.
        """
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            return self.flow.inference_causal(items)

    def hop_batch_prefix(
        self,
        items: Sequence[FlowBatchInput],
        caches: Sequence[tuple[PrefixCacheRow, PrefixCacheRow]],
    ) -> list[torch.Tensor]:
        """hop_batch over the frames past each row's cached prefix."""
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            return self.flow.inference_causal_prefix(items, caches)

    def prefix_cache_rows(
        self, total_frames: int
    ) -> tuple[PrefixCacheRow, PrefixCacheRow] | None:
        """A fresh cached twin pair with room for total_frames, or None without
        room."""
        pool = self.flow.prefix_pool
        if pool is None:
            return None
        else:
            pair = (PrefixCacheRow(), PrefixCacheRow())
            if grow_rows(pool, list(pair), [total_frames, total_frames]):
                return pair
            else:
                return None

    def grow_prefix_cache(
        self, pair: tuple[PrefixCacheRow, PrefixCacheRow], total_frames: int
    ) -> bool:
        pool = self.flow.prefix_pool
        assert pool is not None
        return grow_rows(pool, list(pair), [total_frames, total_frames])

    def release_prefix_cache(
        self, pair: tuple[PrefixCacheRow, PrefixCacheRow] | None
    ) -> None:
        pool = self.flow.prefix_pool
        if pool is not None and pair is not None:
            release_rows(pool, list(pair))
        else:
            pass

    def leftover_batch(self, items: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        """Non-streaming Flow over each row's whole token history for the
        stream's last chunk, the rows packed along the sequence; the scheduler
        keeps the frames past token_offset and runs HiFT over the step.

        # note (guozhihao-224): the last chunk keeps DiT bidirectional;
        # streaming=True did not move SeedTTS EN stream TTFC/QPS and dropped
        # the tail 0.5 s cosine to about 0.29.
        """
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.autocast_dtype,
            enabled=self.autocast_dtype is not None,
        ):
            return self.flow.inference_leftover(items)

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
        else:
            pass
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
        held = max(int(speech_offset), 0)
        delta = tts_speech[:, held:].detach().cpu()
        return delta, tts_mel.detach(), int(tts_speech.shape[1])

    @torch.inference_mode()
    def hift_step(self, rows: Sequence[HiftStepRow]) -> HiftStepSamples:
        """HiFT over the rows of a streaming step: each row's new samples and its
        emitted sample count afterwards."""
        windows: list[HiftDecodeWindow] = []
        for row in rows:
            history_frames = int(row.history.shape[2])
            emitted_frame = row.emitted_samples // self.hift_samples_per_mel_frame
            if row.is_final:
                end_frame = history_frames
            else:
                end_frame = max(history_frames - self.hift_hold_frames, emitted_frame)
            windows.append(
                HiftDecodeWindow(
                    row=row,
                    start_frame=max(emitted_frame - HIFT_DECODE_CONTEXT_FRAMES, 0),
                    emitted_frame=emitted_frame,
                    end_frame=end_frame,
                )
            )
        # Note (Jiaxin Deng): a final row padded to a wider row would end at
        # the padding, not at its own ISTFT edge, and the whole tail is emitted.
        # note(ratish): hift_group aligns finals right for that. A final whose
        # window starts at the history's start needs its left edge as well.
        groups: dict[int | None, list[int]] = {}
        for index, window in enumerate(windows):
            if window.row.is_final and window.start_frame == 0:
                final_width: int | None = int(window.row.history.shape[2])
            else:
                final_width = None
            groups.setdefault(final_width, []).append(index)
        outputs: dict[int, tuple[torch.Tensor, int]] = {}
        for members in groups.values():
            decoded = self.hift_group([windows[index] for index in members])
            outputs.update(zip(members, decoded, strict=True))
        ordered = [outputs[index] for index in range(len(rows))]
        if self.copy_done_events:
            copy_done: torch.cuda.Event | None = self.copy_done_events[
                self.copy_done_slot
            ]
            self.copy_done_slot ^= 1
            copy_done.record()
        else:
            copy_done = None
        return HiftStepSamples(
            new_samples=[samples for samples, _ in ordered],
            emitted_samples=[emitted for _, emitted in ordered],
            copy_done=copy_done,
        )

    def hift_group(
        self, windows: Sequence[HiftDecodeWindow]
    ) -> list[tuple[torch.Tensor, int]]:
        hift = self.hift
        samples_per_frame = self.hift_samples_per_mel_frame
        channels = self.flow.output_size
        history_frame_counts = [int(window.row.history.shape[2]) for window in windows]
        window_frame_counts = [
            history_frames - window.start_frame
            for history_frames, window in zip(
                history_frame_counts, windows, strict=True
            )
        ]
        max_history_frames = max(history_frame_counts)
        max_window_frames = max(window_frame_counts)
        runner = self.hift_decode_cuda_graph_runner
        # note (ratish): a final from frame 0 needs both its own edges, so it keeps its
        # width and the eager decode.
        if runner is None or any(
            window.row.is_final and window.start_frame == 0 for window in windows
        ):
            frame_tier = None
        else:
            frame_tier = runner.frame_tier(max_window_frames)
        if frame_tier is not None:
            max_window_frames = frame_tier
        else:
            pass
        mel_history = windows[0].row.history.new_zeros(
            len(windows), channels, max_history_frames
        )
        for index, (window, history_frames) in enumerate(
            zip(windows, history_frame_counts, strict=True)
        ):
            mel_history[index, :, :history_frames] = window.row.history[0]
        # note(ratish): a final right aligned ends at the decode's edge as in its own
        # call, and the padding before its context is past the decode's receptive field.
        origin_frames = [
            window.start_frame
            - (max_window_frames - window_frames if window.row.is_final else 0)
            for window, window_frames in zip(windows, window_frame_counts, strict=True)
        ]
        window_mel = mel_history.new_zeros(len(windows), channels, max_window_frames)
        window_source = mel_history.new_zeros(
            len(windows), 1, max_window_frames * samples_per_frame
        )
        for index, (window, history_frames, origin) in enumerate(
            zip(windows, history_frame_counts, origin_frames, strict=True)
        ):
            window_mel[
                index, :, window.start_frame - origin : history_frames - origin
            ] = mel_history[index, :, window.start_frame : history_frames]
        # Note (Jiaxin Deng): F0 and the sine source keep the whole history: the
        # source phase is a cumulative sum and its noise is position-indexed.
        # note(ratish): hops run F0 finalized too. Only its look right frames
        # differ, and a hop's emitted frames end before them.
        f0_device = next(hift.f0_predictor.parameters()).device
        f0 = hift.f0_predictor(
            mel_history.to(device=f0_device, dtype=torch.float64), finalize=True
        ).to(mel_history)
        # note(ratish): source chunks aim at this group's decode frames, at least one
        # whole history; the source needs less memory per frame than the decode.
        source_rows = max(1, len(windows) * max_window_frames // max_history_frames)
        for first in range(0, len(windows), source_rows):
            excitation = hift.f0_upsamp(f0[first : first + source_rows, None])
            excitation, _, _ = hift.m_source(excitation.transpose(1, 2))
            excitation = excitation.transpose(1, 2)
            for index in range(first, min(first + source_rows, len(windows))):
                window = windows[index]
                origin = origin_frames[index]
                history_frames = history_frame_counts[index]
                window_source[
                    index,
                    :,
                    (window.start_frame - origin)
                    * samples_per_frame : (history_frames - origin)
                    * samples_per_frame,
                ] = excitation[
                    index - first,
                    :,
                    window.start_frame
                    * samples_per_frame : history_frames
                    * samples_per_frame,
                ]
        if frame_tier is None:
            speech = hift.decode(x=window_mel, s=window_source, finalize=True)
        else:
            speech = runner.run(window_mel, window_source)
        # note (ratish): queued on this stream, the copy reads a replay's output before
        # any later replay of the pool overwrites it.
        if self.copy_done_events:
            speech = async_d2h(speech.detach())
        else:
            speech = speech.detach().cpu()
        return [
            (
                speech[
                    index : index + 1,
                    (window.emitted_frame - origin)
                    * samples_per_frame : (window.end_frame - origin)
                    * samples_per_frame,
                ],
                window.end_frame * samples_per_frame,
            )
            for index, (window, origin) in enumerate(
                zip(windows, origin_frames, strict=True)
            )
        ]

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
        audio_codes = payload.data.get("audio_codes")
        if audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        else:
            prompt_tokens = payload.data.get("flow_prompt_speech_token")
            token_count = 0
            for value in (prompt_tokens, audio_codes):
                if value is None:
                    continue
                else:
                    pass
                numel = 1
                while isinstance(value, list):
                    if not value:
                        numel = 0
                        break
                    else:
                        pass
                    numel *= len(value)
                    value = value[0]
                token_count += numel
            return token_count * self.flow.token_mel_ratio

    def mel2wav_batch(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        if not mels:
            return []
        else:
            pass
        hift_autocast = torch.autocast(
            device_type=current_platform.device_type,
            dtype=self.hift_autocast_dtype,
            enabled=self.hift_autocast_dtype is not None,
        )
        if len(mels) == 1:
            with hift_autocast:
                tts_speech, _ = self.hift.inference(speech_feat=mels[0], finalize=True)
            return [tts_speech.detach().cpu()]
        else:
            pass
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
        samples_per_frame = self.hift_samples_per_mel_frame
        return [
            wav[index : index + 1, : length * samples_per_frame].cpu()
            for index, length in enumerate(lengths)
        ]

    def store_result(
        self,
        payload: StagePayload,
        state: FunCosyVoice3State,
        wav: ArrayLike | torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Fun-CosyVoice3 vocoder did not return audio")
        else:
            pass
        audio_payload = audio_waveform_payload(wav, source_hint="Fun-CosyVoice3")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload.data = state.to_terminal_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
    max_batch_size: int | None = None,
    max_batch_wait_ms: int = 30,
    flow_batch_admission_frames: int = DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
    flow_merge_max_gap_frames: int = 384,
    flow_merge_pad_budget_percent: float = 25.0,
    enable_dit_torch_compile: bool = True,
    enable_flow_cuda_graph: bool = True,
    enable_flow_prefix_cuda_graph: bool,
    enable_flow_whole_history_cuda_graph: bool,
    enable_hift_cuda_graph: bool,
    flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] | None = None,
    enable_flow_estimator_trt: bool = False,
    hift_dtype: str = "float32",
    hift_max_padding_waste: float = 1.5,
    token_hop_len: int = TOKEN_HOP_LEN,
    token_max_hop_len: int = TOKEN_MAX_HOP_LEN,
    disable_hop_growth: bool = False,
    flow_prefix_cache_gb: float | None,
    total_gpu_memory_fraction: float | None = None,
) -> FunCosyVoice3StreamingVocoderScheduler:
    from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
        FunCosyVoice3StreamingVocoderScheduler,
    )

    if flow_batch_admission_frames <= 0:
        raise ValueError("flow_batch_admission_frames must be greater than zero")
    else:
        pass
    if flow_prefix_cache_gb is not None and flow_prefix_cache_gb < 0:
        raise ValueError("flow_prefix_cache_gb must be >= 0")
    else:
        pass

    reject_conflicting_dit_accelerators(
        enable_dit_torch_compile=enable_dit_torch_compile,
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )
    device = str(resolve_concrete_device(device, gpu_id))

    max_batch_size = 16 if max_batch_size is None else int(max_batch_size)
    dtype = dtype or "bfloat16"
    checkpoint_dir = resolve_checkpoint(model_path)
    if dtype not in AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported Fun-CosyVoice3 vocoder dtype {dtype!r}; "
            f"expected one of {sorted(AUTOCAST_DTYPES)}"
        )
    else:
        pass
    autocast_dtype = AUTOCAST_DTYPES[dtype]
    if (
        torch.device(device).type == "mps"
        and not current_platform.is_float64_supported()
    ):
        # Note (yexiaodong): MPS cannot run the CUDA bf16 autocast path, and
        # HiFT's float64 F0 predictor remains on CPU.
        autocast_dtype = None
    else:
        pass
    device_obj = torch.device(device)
    # note (ratish): decided before the load, because the prefix cache reads free memory
    # before it. The TensorRT estimator replaces the packed DiT.
    has_ragged_attention = (
        not enable_flow_estimator_trt
        and supports_ragged_attention(device_obj)
        and autocast_dtype in FA3_DTYPES
    )
    has_prefix_pool = has_ragged_attention and (
        flow_prefix_cache_gb is None or flow_prefix_cache_gb > 0
    )
    if has_prefix_pool:
        # note (ratish): this stage's allocations start here, read once the allocator
        # has returned what earlier stages freed.
        gc.collect()
        torch.accelerator.empty_cache()
        pre_load_free_bytes, total_bytes = torch.accelerator.get_memory_info(device_obj)
        if total_gpu_memory_fraction is None:
            share_ceiling_bytes: int | None = None
        else:
            share_ceiling_bytes = torch.accelerator.memory_reserved(device_obj) + int(
                total_gpu_memory_fraction * total_bytes
            )
        # note (ratish): the engine keeps 1 - mem_fraction_static of its pre-load memory
        # free for the process's runtime. Without the engine the vocoder keeps the same
        # share of its own, for what no step bounds: each stream's prompt and history.
        if engine_headroom_bytes() > 0:
            headroom_bytes = engine_headroom_bytes()
        else:
            headroom_bytes = int(
                pre_load_free_bytes
                * (1 - FunCosyVoice3EngineBuilder.mem_fraction_static)
            )
    else:
        share_ceiling_bytes = None
        headroom_bytes = 0
    flow, hift = load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
        autocast_dtype=autocast_dtype,
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )

    flow.is_buffered_packed = has_ragged_attention
    if enable_flow_cuda_graph and (
        not supports_device_streams(device_obj)
        or current_platform.get_device_graph_backend(device_obj) is None
    ):
        enable_flow_cuda_graph = False
    elif (
        enable_flow_cuda_graph
        and enable_dit_torch_compile
        and current_platform.get_graph_capture_sdpa_backends()
    ):
        logger.info(
            f"Fun-CosyVoice3 Flow graphs stay off on {device_obj}: the compiled DiT "
            "keeps an SDPA kernel this platform cannot record"
        )
        enable_flow_cuda_graph = False
    else:
        pass

    patch_chunk_mask()

    # note (ratish): the packed route leaves the padded DiT only the fallback decode,
    # which runs it eagerly, so its compile and graphs would serve no request.
    if enable_dit_torch_compile and flow.is_buffered_packed:
        set_dit_compile_config()
    elif enable_dit_torch_compile:
        compile_dit_backbone(flow, autocast_dtype=autocast_dtype)
    else:
        pass

    if enable_flow_cuda_graph and not flow.is_buffered_packed:
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
    else:
        pass

    vocoder = CosyVoice3Vocoder(
        flow,
        hift,
        autocast_dtype=autocast_dtype,
        flow_merge_max_gap_frames=flow_merge_max_gap_frames,
        flow_merge_pad_budget_percent=flow_merge_pad_budget_percent,
        hift_dtype=hift_dtype,
        hift_max_padding_waste=hift_max_padding_waste,
    )

    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        vocoder,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder.flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
        token_hop_len=token_hop_len,
        token_max_hop_len=token_max_hop_len,
        disable_hop_growth=disable_hop_growth,
    )
    packed_estimator = vocoder.flow.packed_estimator
    if packed_estimator is not None:
        # note(ratish): a graph replays its captured dtypes; under autocast a hop's frames
        # keep the token embedding's dtype and its speaker embedding does not.
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device_obj.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            conditioning = prepare_flow_conditioning(
                vocoder.flow,
                pack_flow_inputs(
                    vocoder.flow.flow,
                    [scheduler.make_warmup_flow_input(token_hop_len)],
                ),
                finalize=False,
            )
            packed_estimator.build_modulations(
                conditioning.time_span, conditioning.speaker_embedding.dtype
            )
    else:
        pass
    if enable_dit_torch_compile:
        scheduler.warmup_packed_dit_compile()
    else:
        pass
    has_prefix_graphs = enable_flow_prefix_cuda_graph and has_prefix_pool
    # note(ratish): sglang's ladder starts at 4 frames, so a smaller admission budget
    # has no tier and every final keeps the eager solve.
    whole_history_tier_frames = generate_prefill_cuda_graph_batch_sizes(
        flow_batch_admission_frames
    )
    has_whole_history_graphs = (
        enable_flow_whole_history_cuda_graph
        and has_ragged_attention
        and len(whole_history_tier_frames) > 0
    )
    token_mel_ratio = flow.token_mel_ratio
    if disable_hop_growth:
        longest_hop_tokens = token_hop_len
    else:
        longest_hop_tokens = token_max_hop_len
    # note (ratish): the shortest window is a first hop's; the longest a final's, the
    # tokens after the last hop (less than a hop, and the lookahead) with the frames
    # that hop held and the decode's context before them.
    shortest_window_frames = token_hop_len * token_mel_ratio
    longest_window_frames = (
        (longest_hop_tokens + PRE_LOOKAHEAD_LEN - 1) * token_mel_ratio
        + vocoder.hift_hold_frames
        + HIFT_DECODE_CONTEXT_FRAMES
    )
    first_tier = bisect.bisect_left(whole_history_tier_frames, shortest_window_frames)
    last_tier = bisect.bisect_left(whole_history_tier_frames, longest_window_frames)
    hift_tier_frames = whole_history_tier_frames[first_tier : last_tier + 1]
    # note (ratish): torch.istft reads its window envelope on the host, so HiFT's decode
    # captures only with the vocoder's kernels.
    has_hift_graphs = (
        enable_hift_cuda_graph
        and vocoder.has_hift_kernels
        and len(hift_tier_frames) > 0
    )
    if has_prefix_graphs or has_whole_history_graphs or has_hift_graphs:
        graph_backend = current_platform.get_device_graph_backend(device_obj)
        assert graph_backend is not None
        solve_graphs = SolveGraphCapture(
            graph_backend, device=device_obj, autocast_dtype=autocast_dtype
        )
    else:
        pass
    if has_whole_history_graphs:
        assert flow.packed_estimator is not None
        whole_history_cuda_graph_runner = WholeHistoryCudaGraphRunner(
            flow.packed_estimator,
            graphs=solve_graphs,
            frame_dtype=conditioning.noisy_mel.dtype,
            speaker_dtype=conditioning.speaker_embedding.dtype,
            cfg_rate=flow.decoder.inference_cfg_rate,
            euler_steps=FLOW_EULER_STEPS,
            mel_channels=flow.output_size,
            speaker_channels=flow.spk_embed_affine_layer.out_features,
            max_rows=max_batch_size,
            tier_frames=whole_history_tier_frames,
        )
        whole_history_cuda_graph_runner.capture()
        flow.whole_history_cuda_graph_runner = whole_history_cuda_graph_runner
    else:
        pass
    if has_hift_graphs:
        hift_decode_cuda_graph_runner = HiftDecodeCudaGraphRunner(
            hift,
            graphs=solve_graphs,
            mel_channels=flow.output_size,
            samples_per_frame=vocoder.hift_samples_per_mel_frame,
            row_tiers=row_tiers(max_batch_size),
            frame_tiers=hift_tier_frames,
        )
        hift_decode_cuda_graph_runner.capture()
        vocoder.hift_decode_cuda_graph_runner = hift_decode_cuda_graph_runner
    else:
        pass

    def make_prefix_runner(
        pool: PrefixKVPool, graphs: SolveGraphCapture
    ) -> PrefixCudaGraphRunner:
        assert flow.packed_estimator is not None
        return PrefixCudaGraphRunner(
            flow.packed_estimator,
            pool,
            graphs=graphs,
            frame_dtype=conditioning.noisy_mel.dtype,
            speaker_dtype=conditioning.speaker_embedding.dtype,
            cfg_rate=flow.decoder.inference_cfg_rate,
            mel_channels=flow.output_size,
            speaker_channels=flow.spk_embed_affine_layer.out_features,
            max_rows=max_batch_size,
            hop_frames=longest_hop_tokens * token_mel_ratio,
            min_hop_frames=token_hop_len * token_mel_ratio,
            max_frames=flow.decoder.rand_noise.shape[2],
        )

    if has_prefix_pool:
        assert flow.packed_estimator is not None
        is_prefix_compiled = flow.packed_estimator.is_compiled
        # note (ratish): one compiled callable for the throwaway pool and the real
        # one, so the real pool reuses what compiled on the throwaway one.
        if is_prefix_compiled:
            prefix_forward = compile_forward_prefix()
        else:
            prefix_forward = forward_prefix
        # note (ratish): the pool takes what is free, so it leaves the largest step
        # the scheduler admits: max_batch_size rows of at most the AR's context each.
        step_bytes = scheduler.measure_peak_step_bytes(
            device_obj,
            row_count=max_batch_size,
            generated_tokens_per_row=FunCosyVoice3EngineBuilder.context_length,
        )
        if has_prefix_graphs:
            # note (ratish): the prefix graphs capture after the pool they read, so
            # their memory is measured first on a throwaway pool and graph pool, the
            # pool holding the prefix warmup's hops under the real pool's reserve.
            flow.prefix_pool = build_prefix_pool(
                flow,
                device_obj,
                autocast_dtype,
                size_gb=None,
                max_blocks=scheduler.prefix_warmup_block_count(),
                stage_reserve_bytes=step_bytes,
                headroom_bytes=headroom_bytes,
                share_ceiling_bytes=share_ceiling_bytes,
            )
            # note (ratish): the real pool is sized from no more, so it holds no block
            # either and serving keeps the whole-history hops.
            has_prefix_graphs = flow.prefix_pool is not None
        else:
            pass
        if has_prefix_graphs:
            assert flow.prefix_pool is not None
            flow.prefix_pool.forward = prefix_forward
            if is_prefix_compiled:
                scheduler.warmup_prefix_compile()
            else:
                pass
            torch.accelerator.empty_cache()
            free_bytes = torch.accelerator.get_memory_info(device_obj)[0]
            dry_prefix_runner = make_prefix_runner(
                flow.prefix_pool,
                SolveGraphCapture(
                    graph_backend, device=device_obj, autocast_dtype=autocast_dtype
                ),
            )
            dry_prefix_runner.capture()
            torch.accelerator.empty_cache()
            prefix_graph_bytes = (
                free_bytes - torch.accelerator.get_memory_info(device_obj)[0]
            )
            del dry_prefix_runner
            flow.prefix_pool = None
        else:
            prefix_graph_bytes = 0
        # note (ratish): the step runs on the vocoder's stream while the engine runs on
        # its own, and the allocator never shares a block across streams, so the
        # runtime headroom and the step add up.
        logger.info(
            f"Fun-CosyVoice3 Flow prefix cache reserve: prefix graphs "
            f"{prefix_graph_bytes / 2**30:.2f} GB, runtime headroom "
            f"{headroom_bytes / 2**30:.2f} GB, step {step_bytes / 2**30:.2f} GB"
        )
        flow.prefix_pool = build_prefix_pool(
            flow,
            device_obj,
            autocast_dtype,
            size_gb=flow_prefix_cache_gb,
            max_blocks=None,
            stage_reserve_bytes=prefix_graph_bytes + step_bytes,
            headroom_bytes=headroom_bytes,
            share_ceiling_bytes=share_ceiling_bytes,
        )
        if flow.prefix_pool is not None:
            flow.prefix_pool.forward = prefix_forward
            if is_prefix_compiled:
                scheduler.warmup_prefix_compile()
            else:
                pass
        else:
            pass
    else:
        pass
    if has_prefix_graphs and flow.prefix_pool is not None:
        prefix_cuda_graph_runner = make_prefix_runner(flow.prefix_pool, solve_graphs)
        prefix_cuda_graph_runner.capture()
        flow.prefix_cuda_graph_runner = prefix_cuda_graph_runner
    else:
        pass
    scheduler.warmup_now()
    return scheduler
