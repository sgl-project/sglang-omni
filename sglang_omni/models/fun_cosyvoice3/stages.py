# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
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
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    TOKEN_MEL_RATIO,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

# Note (xinran): This is an admission budget, not a maximum supported request
# length. The scheduler admits a request that exceeds it as a singleton Flow
# batch and defers following requests to the next batch.

_DEFAULT_FLOW_BATCH_ADMISSION_FRAMES = 8000

_AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {
    "float32": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)

_CHUNK_MASK_COMPILE_DISABLED = False


class _MpsHiFTAdapter:
    """Keep HiFT's float64 F0 branch on CPU while decoding on MPS.

    The upstream causal HiFT implementation deliberately evaluates its F0
    predictor in float64.  PyTorch MPS does not implement float64 tensors, but
    the remaining source-filter/ISTFT path is usable on MPS.  Keeping this
    boundary explicit avoids changing the checkpoint's numerical behavior.
    """

    def __init__(self, hift: Any, device: str) -> None:
        self._hift = hift
        self._device = torch.device(device)
        self._f0_predictor = hift.f0_predictor
        # MPS rejects a device transfer that also requests float64. Move the
        # module to CPU first, then preserve upstream's double-precision F0
        # calculation there.
        self._f0_predictor.to(device="cpu")
        self._f0_predictor.to(dtype=torch.float64)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hift, name)

    def parameters(self):
        return self._hift.parameters()

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, finalize: bool = True):
        cpu_features = speech_feat.detach().to(device="cpu")
        f0 = self._f0_predictor(
            cpu_features.to(dtype=torch.float64),
            finalize=finalize,
        ).to(device=self._device, dtype=speech_feat.dtype)
        source = self._hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = self._hift.m_source(source)
        source = source.transpose(1, 2)
        if finalize:
            generated = self._hift.decode(
                x=speech_feat,
                s=source,
                finalize=True,
            )
        else:
            causal_padding = self._f0_predictor.condnet[0].causal_padding
            generated = self._hift.decode(
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


@dataclass(frozen=True)
class _PackedFlowBatch:
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


def _flow_device_and_dtype(flow: Any) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(flow.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError("Flow must expose at least one parameter") from exc
    return parameter.device, parameter.dtype


def _validate_flow_input(flow: Any, item: FlowBatchInput, index: int) -> None:
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
        raise ValueError(f"input {index} embedding must have shape [1, speaker_dim]")
    expected_embedding_size = getattr(flow.spk_embed_affine_layer, "in_features", None)
    if (
        expected_embedding_size is not None
        and item.embedding.shape[1] != expected_embedding_size
    ):
        raise ValueError(
            f"input {index} embedding width must be {expected_embedding_size}"
        )


def _pack_flow_inputs(flow: Any, inputs: Sequence[FlowBatchInput]) -> _PackedFlowBatch:
    if not inputs:
        raise ValueError("Flow batch must contain at least one input")
    for index, item in enumerate(inputs):
        _validate_flow_input(flow, item, index)

    device, dtype = _flow_device_and_dtype(flow)
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
    return _PackedFlowBatch(
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


def _flow_lookahead(flow: Any) -> int:
    layer = getattr(flow, "pre_lookahead_layer", None)
    layer_len = getattr(layer, "pre_lookahead_len", None)
    if layer_len is not None:
        return max(int(layer_len), 0)
    return max(int(getattr(flow, "pre_lookahead_len", PRE_LOOKAHEAD_LEN)), 0)


def _apply_pre_lookahead(
    flow: Any,
    token_embedding: torch.Tensor,
    *,
    finalize: bool,
    lookahead: int,
    combined_token_lengths: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if finalize or lookahead <= 0:
        return flow.pre_lookahead_layer(token_embedding)
    layer = flow.pre_lookahead_layer
    lengths = combined_token_lengths
    if lengths is None or len(set(int(length) for length in lengths)) <= 1:
        body = token_embedding[:, :-lookahead]
        context = token_embedding[:, -lookahead:]
        try:
            return layer(body, context=context)
        except TypeError:
            # note (guozhihao-224): unit fakes use Identity, which has no context kwarg.
            return layer(body)
    # note (guozhihao-224): packing left-aligns and pads to max combined
    # length. Slice [:, -lookahead:] would read pad on shorter rows.
    body_lens = [max(int(length) - lookahead, 0) for length in lengths]
    max_body = max(body_lens)
    pieces: list[torch.Tensor] = []
    for index, length in enumerate(lengths):
        body_len = body_lens[index]
        body = token_embedding[index : index + 1, :body_len]
        context = token_embedding[
            index : index + 1, int(length) - lookahead : int(length)
        ]
        try:
            hidden = layer(body, context=context)
        except TypeError:
            hidden = layer(body)
        if hidden.shape[1] < max_body:
            hidden = F.pad(hidden, (0, 0, 0, max_body - int(hidden.shape[1])))
        pieces.append(hidden)
    return torch.cat(pieces, dim=0)


def _solve_flow_euler(
    decoder: Any,
    x: torch.Tensor,
    t_span: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    *,
    streaming: bool = False,
) -> torch.Tensor:
    batch_size, channels, frames = x.shape
    dtype = spks.dtype
    x_in = torch.zeros(2 * batch_size, channels, frames, device=x.device, dtype=dtype)
    mask_in = torch.zeros(2 * batch_size, 1, frames, device=x.device, dtype=dtype)
    mu_in = torch.zeros_like(x_in)
    t_in = torch.zeros(1, device=x.device, dtype=dtype)
    spks_in = torch.zeros(2 * batch_size, spks.shape[1], device=x.device, dtype=dtype)
    cond_in = torch.zeros_like(x_in)
    t, dt = t_span[0], t_span[1] - t_span[0]
    for step in range(1, len(t_span)):
        x_in[:batch_size] = x
        x_in[batch_size:] = x
        mask_in[:batch_size] = mask
        mask_in[batch_size:] = mask
        mu_in[:batch_size] = mu
        t_in[:] = t
        spks_in[:batch_size] = spks
        cond_in[:batch_size] = cond
        derivative = _forward_flow_estimator(
            decoder,
            x_in,
            mask_in,
            mu_in,
            t_in,
            spks_in,
            cond_in,
            streaming=streaming,
        )
        conditional, unconditional = derivative[:batch_size], derivative[batch_size:]
        x = x + dt * (
            (1.0 + decoder.inference_cfg_rate) * conditional
            - decoder.inference_cfg_rate * unconditional
        )
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x.float()


class _FlowSolveTimer:
    """Elapsed time of one Euler solve, read back without waiting on the device.

    On an accelerator two stream events bracket the solve and the elapsed time
    is only available once the end event has completed. On CPU the ops run
    synchronously, so perf_counter around the call is already exact.
    """

    def __init__(self, device: torch.device) -> None:
        self._on_device = device.type != "cpu"
        if self._on_device:
            self._stream = torch.accelerator.current_stream(device)
            self._start = torch.Event(device=device, enable_timing=True)
            self._end = torch.Event(device=device, enable_timing=True)
        else:
            self._start = self._end = 0.0

    def start(self) -> None:
        if self._on_device:
            self._start.record(self._stream)
        else:
            self._start = time.perf_counter()

    def stop(self) -> None:
        if self._on_device:
            self._end.record(self._stream)
        else:
            self._end = time.perf_counter()

    def elapsed_ms(self) -> float | None:
        """Return the solve time, or None while the end event is still pending."""
        if not self._on_device:
            return (self._end - self._start) * 1000.0
        if not self._end.query():
            return None
        return self._start.elapsed_time(self._end)


@torch.inference_mode()
def _generate_flow(
    flow: Any,
    packed: _PackedFlowBatch,
    *,
    streaming: bool = False,
    finalize: bool = True,
    timer: _FlowSolveTimer | None = None,
) -> torch.Tensor:
    embedding = flow.spk_embed_affine_layer(F.normalize(packed.embedding, dim=1))
    token_embedding = flow.input_embedding(torch.clamp(packed.token, min=0))
    token_embedding = token_embedding * packed.token_mask.to(token_embedding.dtype)
    lookahead = 0 if finalize else _flow_lookahead(flow)
    h = _apply_pre_lookahead(
        flow,
        token_embedding,
        finalize=finalize,
        lookahead=lookahead,
        combined_token_lengths=packed.combined_token_lengths,
    )
    mu = h.repeat_interleave(flow.token_mel_ratio, dim=1).transpose(1, 2).contiguous()
    batch_size, channels, max_mel = mu.shape
    if channels != flow.output_size:
        raise ValueError("Flow pre-lookahead output width does not match output_size")
    if lookahead > 0:
        total_mel_lengths_tensor = torch.tensor(
            [
                max(length - lookahead, 0) * flow.token_mel_ratio
                for length in packed.combined_token_lengths
            ],
            dtype=torch.int64,
            device=mu.device,
        )
    else:
        total_mel_lengths_tensor = packed.total_mel_lengths_tensor
    mask = (
        (
            torch.arange(max_mel, device=mu.device).unsqueeze(0)
            < total_mel_lengths_tensor.unsqueeze(1)
        )
        .unsqueeze(1)
        .to(mu.dtype)
    )
    cond = torch.zeros_like(mu)
    for index, prompt_frames in enumerate(packed.prompt_mel_lengths):
        cond[index, :, :prompt_frames] = packed.prompt_feat[
            index, :prompt_frames
        ].transpose(0, 1)
    decoder = flow.decoder
    if max_mel > decoder.rand_noise.shape[2]:
        raise ValueError(
            f"decoder.rand_noise supports {decoder.rand_noise.shape[2]} frames, "
            f"but batch requires {max_mel}"
        )
    z = (
        decoder.rand_noise[:, :, :max_mel]
        .to(device=mu.device, dtype=mu.dtype)
        .expand(batch_size, -1, -1)
        .clone()
    )
    t_span = torch.linspace(0, 1, 11, device=mu.device, dtype=mu.dtype)
    if decoder.t_scheduler == "cosine":
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    if timer is not None:
        timer.start()
    mel = _solve_flow_euler(
        decoder, z, t_span, mu, mask, embedding, cond, streaming=streaming
    )
    if timer is not None:
        timer.stop()
    return mel


def _split_generated_mels(
    flow: Any,
    packed: _PackedFlowBatch,
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


def _forward_flow_estimator(
    decoder: Any,
    x: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    *,
    streaming: bool = False,
) -> torch.Tensor:
    # note (guozhihao-224): CosyVoice forward_estimator hardcodes TRT shapes to
    # (2, 80, T). Packed Flow is CFG=2N, so TRT uses execute_flow_estimator.
    # TRT ONNX freezes attention, so streaming is only meaningful for PyTorch.
    estimator = decoder.estimator
    if isinstance(estimator, torch.nn.Module):
        return decoder.forward_estimator(
            x, mask, mu, t, spks, cond, streaming=streaming
        )
    return execute_flow_estimator(estimator, x, mask, mu, t, spks, cond)


class FunCosyVoice3Flow:
    """CosyVoice3 Flow with batch inference enabled as its default API."""

    def __init__(self, flow: Any) -> None:
        self._flow = flow
        self._last_solve: tuple[int, _FlowSolveTimer] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._flow, name)

    def parameters(self):
        return self._flow.parameters()

    def to(self, *args: Any, **kwargs: Any) -> "FunCosyVoice3Flow":
        self._flow.to(*args, **kwargs)
        return self

    def eval(self) -> "FunCosyVoice3Flow":
        self._flow.eval()
        return self

    def log_last_solve(self) -> None:
        # note (db-ol): the vocoder calls this after the bucket's audio reached
        # the host. A still pending end event is skipped, never waited for.
        if self._last_solve is None:
            return
        items, timer = self._last_solve
        self._last_solve = None
        elapsed_ms = timer.elapsed_ms()
        if elapsed_ms is None:
            return
        logger.debug(
            "Fun-CosyVoice3 flow solve: batch_items=%d solve_elapsed_ms=%.1f",
            items,
            elapsed_ms,
        )

    @torch.inference_mode()
    def inference(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        packed = _pack_flow_inputs(self._flow, inputs)
        # note (db-ol): the solve is timed only while the debug record can be
        # seen, so the default INFO path runs it untouched.
        timer = None
        if logger.isEnabledFor(logging.DEBUG):
            timer = _FlowSolveTimer(packed.token.device)
            self._last_solve = (len(inputs), timer)
        generated = _generate_flow(self._flow, packed, timer=timer)
        return _split_generated_mels(
            self._flow,
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
        packed = _pack_flow_inputs(self._flow, inputs)
        generated = _generate_flow(self._flow, packed, streaming=True, finalize=False)
        lookahead = _flow_lookahead(self._flow)
        target_token_lengths = tuple(
            max(length - lookahead, 0) for length in packed.target_token_lengths
        )
        combined_token_lengths = tuple(
            max(length - lookahead, 0) for length in packed.combined_token_lengths
        )
        return _split_generated_mels(
            self._flow,
            packed,
            generated,
            token_lengths=combined_token_lengths,
            target_token_lengths=target_token_lengths,
        )


def load_state(payload: StagePayload) -> FunCosyVoice3State:
    return _load_pipeline_state(payload, FunCosyVoice3State)


def store_state(payload: StagePayload, state: FunCosyVoice3State) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _attach_flow_estimator_trt(
    flow: Any,
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


def _fold_weight_norm(module: torch.nn.Module) -> int:
    folded = 0
    for submodule in list(module.modules()):
        while is_parametrized(submodule):
            name = next(iter(submodule.parametrizations.keys()))
            remove_parametrizations(submodule, name, leave_parametrized=True)
            folded += 1
    return folded


def _hift_samples_per_mel_frame(hift: Any) -> int:
    rates = getattr(hift, "upsample_rates", None)
    hop_len = (
        getattr(hift, "istft_params", {}).get("hop_len")
        if hasattr(hift, "istft_params")
        else None
    )
    if not rates or not hop_len:
        raise RuntimeError(
            "Fun-CosyVoice3 HiFT generator is missing upsample_rates / "
            "istft_params; refusing to guess the mel->wave stride"
        )
    stride = int(hop_len)
    for rate in rates:
        stride *= int(rate)
    return stride


def _prepare_hift_for_inference(hift: Any) -> None:
    # note (Dayuxiaoshui): folding weight_norm is the only load-time step
    # batched decode needs. The pinned CausalHiFTGenerator already squeezes
    # the source to [B, T] before its STFT and casts f0_predictor to float64
    # inside inference(), and right-zero-padded mels reproduce per-request
    # output bit-for-bit except in the final mel frame of padded requests.
    folded = _fold_weight_norm(hift)
    logger.info(
        "Prepared Fun-CosyVoice3 HiFT for inference (folded %d weight_norm "
        "parametrizations)",
        folded,
    )


def _import_modelscope_preserving_root_handlers() -> None:
    # note (db-ol): the first modelscope import sets every root StreamHandler
    # to ERROR once torch.distributed is initialized, which silences the stage
    # process that hosts both the engine and this vocoder. Undo that change.
    saved = [(handler, handler.level) for handler in logging.getLogger().handlers]
    try:
        importlib.import_module("modelscope")
    except ImportError:
        # note (db-ol): cosyvoice imports modelscope itself and raises a
        # clearer error below.
        pass
    for handler, level in saved:
        if handler.level != level:
            handler.setLevel(level)
            logger.info(
                "Restored root log handler level to %s after the modelscope "
                "import changed it",
                logging.getLevelName(level),
            )


def _load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
    *,
    enable_flow_estimator_trt: bool = False,
) -> tuple[Any, Any]:
    if torch.device(device).type == "mps":
        return _load_cosyvoice3_flow_hift_lightweight(
            checkpoint_dir,
            device=device,
        )
    _import_modelscope_preserving_root_handlers()
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(_COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    _prepare_hift_for_inference(hift)
    del cv.model.llm
    wrapped = FunCosyVoice3Flow(flow)
    if enable_flow_estimator_trt:
        _attach_flow_estimator_trt(wrapped, checkpoint_dir, device)
    return wrapped, hift


def _load_cosyvoice3_flow_hift_lightweight(
    checkpoint_dir: str,
    *,
    device: str,
) -> tuple[Any, Any]:
    """Load only Flow and HiFT for CPU/MPS without constructing a second LLM."""
    try:
        from hyperpyyaml import load_hyperpyyaml
    except ImportError as exc:
        raise RuntimeError(_COSYVOICE_INSTALL_HINT) from exc

    config_path = os.path.join(checkpoint_dir, "cosyvoice3.yaml")
    flow_path = os.path.join(checkpoint_dir, "flow.pt")
    hift_path = os.path.join(checkpoint_dir, "hift.pt")
    if not all(os.path.isfile(path) for path in (config_path, flow_path, hift_path)):
        raise FileNotFoundError(
            "Fun-CosyVoice3 requires cosyvoice3.yaml, flow.pt and hift.pt in "
            f"{checkpoint_dir}"
        )

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
            hift_path,
            map_location="cpu",
            weights_only=True,
        ).items()
    }
    hift.load_state_dict(hift_state, strict=True)
    flow.to(device).eval()
    hift.to(device).eval()
    if (
        torch.device(device).type == "mps"
        and not current_platform.is_float64_supported()
    ):
        hift = _MpsHiFTAdapter(hift, device)
    del configs
    return FunCosyVoice3Flow(flow), hift


def _resolve_cosyvoice3_mlx_artifact(
    model_path: str,
    *,
    revision: str | None,
) -> str:
    """Resolve and inspect the exact MLX snapshot before loading its weights."""
    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    model_dir = resolve_model_directory(model_path, revision=revision)
    ensure_remote_code_allowed(model_dir, trust_remote_code=False)
    return str(model_dir)


def _load_cosyvoice3_mlx_vocoder(
    model_path: str,
    *,
    revision: str | None,
    expected_dtype: str | None,
) -> Any:
    model_dir = _resolve_cosyvoice3_mlx_artifact(model_path, revision=revision)
    from sglang_omni.models.fun_cosyvoice3.mlx.vocoder import FunCosyVoice3MlxVocoder

    return FunCosyVoice3MlxVocoder.from_pretrained(
        model_dir, revision=None, expected_dtype=expected_dtype
    )


def _get_mlx_core() -> Any:
    import mlx.core as mx

    return mx


def _configure_dit_torch_compile() -> None:
    """Enable the Inductor/Dynamo flags the DiT graph wants, without pulling in
    the full sglang.srt stack (the vocoder is a plain pipeline process)."""
    torch._inductor.config.fx_graph_cache = True
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 1024


def _disable_compile_on_chunk_mask() -> None:
    # note (guozhihao-224): inductor NaN-compares subsequent_chunk_mask in
    # DiT.forward; keep the mask eager.
    global _CHUNK_MASK_COMPILE_DISABLED
    if _CHUNK_MASK_COMPILE_DISABLED:
        return
    try:
        import cosyvoice.flow.DiT.dit as dit_mod
        import cosyvoice.utils.mask as mask_mod
    except ImportError:
        return
    disabled = torch.compiler.disable(mask_mod.add_optional_chunk_mask)
    mask_mod.add_optional_chunk_mask = disabled
    dit_mod.add_optional_chunk_mask = disabled
    _CHUNK_MASK_COMPILE_DISABLED = True


def _run_dit_estimator(
    estimator: Any,
    mel_frames: int,
    *,
    compute_dtype: torch.dtype | None = None,
    streaming: bool = False,
) -> None:

    param = next(estimator.parameters())
    device, dtype = param.device, param.dtype
    t = int(mel_frames)
    # note (guozhihao-224): CFG batch 2; mel dim 80 matches pinned checkpoint proj_out.
    x = torch.randn(2, 80, t, device=device, dtype=dtype)
    mask = torch.ones(2, 1, t, device=device, dtype=dtype)
    mu = torch.randn(2, 80, t, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=dtype)
    spks = torch.randn(2, 80, device=device, dtype=dtype)
    cond = torch.randn(2, 80, t, device=device, dtype=dtype)
    with torch.autocast(
        device_type=current_platform.device_type,
        dtype=compute_dtype,
        enabled=compute_dtype is not None,
    ):
        estimator(x, mask, mu, timestep, spks, cond, streaming=streaming)


def _compile_dit_backbone(
    flow: Any,
    *,
    warmup_mel_frames: int = 128,
    warmup_steps: int = 3,
    compute_dtype: torch.dtype | None = None,
) -> bool:

    estimator = getattr(getattr(flow, "decoder", None), "estimator", None)
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
    _configure_dit_torch_compile()
    _disable_compile_on_chunk_mask()
    try:
        estimator.forward = torch.compile(original_forward, dynamic=True)
        with torch.inference_mode():
            for streaming in (False, True):
                for _ in range(warmup_steps):
                    _run_dit_estimator(
                        estimator,
                        warmup_mel_frames,
                        compute_dtype=compute_dtype,
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
        "Compiled Fun-CosyVoice3 DiT backbone (dynamic=True, compute_dtype=%s, "
        "warmup_mel_frames=%d, warmup_steps=%d, streaming=False/True)",
        compute_dtype,
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
    mlx_model_path: str | None = None,
    mlx_model_revision: str | None = None,
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
        mlx_model_path=mlx_model_path,
        mlx_model_revision=mlx_model_revision,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


@dataclass(frozen=True)
class _PreparedFlowRequest:
    index: int
    sample_rate: int
    flow_input: FlowBatchInput


def _group_by_padding_waste(
    items: Sequence[tuple[Any, torch.Tensor]],
    *,
    max_waste: float,
) -> Iterator[list[tuple[Any, torch.Tensor]]]:
    ordered = sorted(items, key=lambda pair: int(pair[1].shape[-1]))
    group: list[tuple[Any, torch.Tensor]] = []
    total = 0
    longest = 0
    for pair in ordered:
        length = int(pair[1].shape[-1])
        candidate_longest = max(longest, length)
        candidate_total = total + length
        if group and candidate_longest * (len(group) + 1) > max_waste * candidate_total:
            yield group
            group, total, longest = [], 0, 0
            candidate_longest = length
            candidate_total = length
        group.append(pair)
        total, longest = candidate_total, candidate_longest
    if group:
        yield group


def _prepare_vocoder_item(
    payload: StagePayload,
) -> tuple[FunCosyVoice3State, torch.Tensor]:
    state = load_state(payload)
    if state.audio_codes is None:
        raise RuntimeError(
            "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
        )
    return state, torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)


def _make_flow_input(state: FunCosyVoice3State, codes: torch.Tensor) -> FlowBatchInput:
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


def _store_vocoder_result(
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
    payload = store_state(payload, state)
    payload.data.update(audio_payload)
    payload.data["sample_rate"] = state.sample_rate
    payload.data["modality"] = "audio"
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        compute_dtype: torch.dtype | None = None,
        flow_batch_bucket_frames: int = 50,
        hift_compute_dtype: str = "float32",
        hift_max_padding_waste: float = 1.5,
    ) -> None:
        if flow_batch_bucket_frames <= 0:
            raise ValueError("flow_batch_bucket_frames must be greater than zero")
        if hift_max_padding_waste < 1.0:
            raise ValueError("hift_max_padding_waste must be at least 1.0")
        if hift_compute_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"Unsupported Fun-CosyVoice3 HiFT dtype {hift_compute_dtype!r}; "
                f"expected one of {sorted(_AUTOCAST_DTYPES)}"
            )
        estimator = flow.decoder.estimator
        if not isinstance(estimator, torch.nn.Module) and not is_flow_estimator_trt(
            estimator
        ):
            raise RuntimeError(
                "Fun-CosyVoice3 Flow estimator must be a PyTorch module or a "
                "TensorRT wrapper exposing acquire_estimator / execute"
            )
        self._flow = (
            flow if isinstance(flow, FunCosyVoice3Flow) else FunCosyVoice3Flow(flow)
        )
        self._hift = hift
        self._compute_dtype = compute_dtype
        self._flow_batch_bucket_frames = flow_batch_bucket_frames
        self._hift_compute_dtype = _AUTOCAST_DTYPES[hift_compute_dtype]
        self._hift_max_padding_waste = hift_max_padding_waste
        self._hift_samples_per_mel_frame: int | None = None

    def _mel_stride(self) -> int:
        if self._hift_samples_per_mel_frame is None:
            self._hift_samples_per_mel_frame = _hift_samples_per_mel_frame(self._hift)
        return self._hift_samples_per_mel_frame

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
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
        prepared = [
            _PreparedFlowRequest(
                index=index,
                sample_rate=state.sample_rate,
                flow_input=self._make_flow_input(state, codes),
            )
            for index, (state, codes) in enumerate(items)
        ]
        results: list[tuple[Any, int] | None] = [None] * len(prepared)
        buckets: dict[int, list[_PreparedFlowRequest]] = defaultdict(list)
        for request in prepared:
            buckets[self._flow_bucket_key(request.flow_input)].append(request)

        flow_device, _ = _flow_device_and_dtype(self._flow)
        for bucket in buckets.values():
            with torch.autocast(
                device_type=flow_device.type,
                dtype=self._compute_dtype,
                enabled=self._compute_dtype is not None,
            ):
                mel_list = self._flow.inference(
                    [request.flow_input for request in bucket]
                )

                pairs = list(zip(bucket, mel_list, strict=True))
                for group in _group_by_padding_waste(
                    pairs, max_waste=self._hift_max_padding_waste
                ):
                    wavs = self._mel2wav_batch([mel for _, mel in group])
                    for (request, _), wav in zip(group, wavs, strict=True):
                        results[request.index] = (wav, request.sample_rate)
            self._flow.log_last_solve()

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
        native_flow = getattr(self._flow, "_flow", self._flow)
        device = next(native_flow.parameters()).device
        offset = max(int(token_offset), 0)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=self._compute_dtype,
            enabled=self._compute_dtype is not None,
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
        return self._hift_delta(
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
            dtype=self._compute_dtype,
            enabled=self._compute_dtype is not None,
        ):
            return self._flow.inference_causal(items)

    def _hift_delta(
        self,
        tts_mel: torch.Tensor,
        *,
        hift_mel: torch.Tensor | None,
        speech_offset: int,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if hift_mel is not None:
            tts_mel = torch.cat([hift_mel.to(device=tts_mel.device), tts_mel], dim=2)
        tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=finalize)
        held = max(int(speech_offset), 0)
        delta = tts_speech[:, held:].detach().cpu()
        return delta, tts_mel.detach(), int(tts_speech.shape[1])

    def _make_flow_input(
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

    def _flow_bucket_key(self, item: FlowBatchInput) -> int:
        total_mel = self._flow_total_mel_frames(item)
        return (
            total_mel + self._flow_batch_bucket_frames - 1
        ) // self._flow_batch_bucket_frames

    def _flow_total_mel_frames(self, item: FlowBatchInput) -> int:
        total_tokens = item.prompt_token.shape[1] + item.token.shape[1]
        return total_tokens * self._flow.token_mel_ratio

    def _flow_scheduler_cost(self, payload: StagePayload) -> int:
        state, codes = self.prepare_item(payload)
        total_mel = self._flow_total_mel_frames(self._make_flow_input(state, codes))
        return (
            (total_mel + self._flow_batch_bucket_frames - 1)
            // self._flow_batch_bucket_frames
            * self._flow_batch_bucket_frames
        )

    def _mel2wav(self, tts_mel: torch.Tensor) -> torch.Tensor:
        with self._hift_autocast():
            tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech.detach().cpu()

    def _hift_autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=current_platform.device_type,
            dtype=self._hift_compute_dtype,
            enabled=self._hift_compute_dtype is not None,
        )

    def _mel2wav_batch(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        if not mels:
            return []
        if len(mels) == 1:
            return [self._mel2wav(mels[0])]
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
        with self._hift_autocast():
            wav, _ = self._hift.inference(speech_feat=padded, finalize=True)
        wav = wav.detach()
        samples_per_frame = self._mel_stride()
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

        payload = store_state(payload, state)
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


class _CosyVoice3MlxVocoderAdapter(BatchVocoderBase):
    """Bridge pipeline state into the native batch-one MLX Flow/HiFT API."""

    def __init__(self, vocoder: Any) -> None:
        self._vocoder = vocoder
        self._mx = _get_mlx_core()
        self._stream = self._mx.new_thread_local_stream(self._mx.gpu)
        self.sample_rate = int(vocoder.sample_rate)
        self.token_mel_ratio = int(vocoder.token_mel_ratio)

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        return _prepare_vocoder_item(payload)

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        if len(items) != 1:
            raise RuntimeError(
                "Fun-CosyVoice3 native MLX vocoder requires exactly one request per decode batch"
            )
        state, codes = items[0]
        flow_input = _make_flow_input(state, codes)
        mx = self._mx
        with mx.stream(self._stream):
            wav = self._vocoder.decode_mx(
                token=mx.array(flow_input.token.detach().cpu().numpy(), dtype=mx.int32),
                prompt_token=mx.array(
                    flow_input.prompt_token.detach().cpu().numpy(), dtype=mx.int32
                ),
                prompt_feat=mx.array(
                    flow_input.prompt_feat.detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy(),
                    dtype=mx.float32,
                ),
                embedding=mx.array(
                    flow_input.embedding.detach().to(dtype=torch.float32).cpu().numpy(),
                    dtype=mx.float32,
                ),
            )
            mx.eval(wav)
            wav = np.ascontiguousarray(np.asarray(wav, dtype=np.float32))
        return [(wav, self.sample_rate)]

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.decode_payloads([payload])
        return results[0]

    def decode_tokens(
        self,
        *,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Decode accumulated stream tokens through the native MLX graph."""
        mx = self._mx
        with mx.stream(self._stream):
            wav = self._vocoder.decode_mx(
                token=mx.array(token.detach().cpu().numpy(), dtype=mx.int32),
                prompt_token=mx.array(
                    prompt_token.detach().cpu().numpy(), dtype=mx.int32
                ),
                prompt_feat=mx.array(
                    prompt_feat.detach().to(dtype=torch.float32).cpu().numpy(),
                    dtype=mx.float32,
                ),
                embedding=mx.array(
                    embedding.detach().to(dtype=torch.float32).cpu().numpy(),
                    dtype=mx.float32,
                ),
            )
            mx.eval(wav)
        return torch.from_numpy(np.ascontiguousarray(np.asarray(wav, dtype=np.float32)))

    def store_result(
        self,
        payload: StagePayload,
        state: FunCosyVoice3State,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        return _store_vocoder_result(payload, state, wav, sample_rate)


@dataclass
class _FunCosyVoice3MlxStreamState:
    tokens: list[int] = field(default_factory=list)
    prompt_token: torch.Tensor | None = None
    prompt_feat: torch.Tensor | None = None
    embedding: torch.Tensor | None = None


class _FunCosyVoice3MlxStreamingVocoderScheduler(
    StreamingVocoderBase[_FunCosyVoice3MlxStreamState, None]
):
    """Stream-aware MLX scheduler with whole-utterance final decode.

    The converted MLX Flow/HiFT artifact is currently a non-causal decoder.
    This scheduler preserves Omni's stream_chunk/stream_done contract and
    emits one final waveform instead of silently dropping chunks in
    ``SimpleScheduler``. Incremental MLX Flow/HiFT decoding can replace the
    accumulated-token decode later without changing the stage contract.
    """

    def __init__(
        self, vocoder: _CosyVoice3MlxVocoderAdapter, *, max_batch_wait_ms: int
    ) -> None:
        self._vocoder = vocoder
        super().__init__(
            vocoder.decode_payload,
            batch_compute_fn=vocoder.decode_payloads,
            sample_rate=vocoder.sample_rate,
            stream_source_hint="Fun-CosyVoice3",
            max_batch_size=1,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _FunCosyVoice3MlxStreamState:
        del request_id
        return _FunCosyVoice3MlxStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _FunCosyVoice3MlxStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin == "payload":
            pipeline_state = FunCosyVoice3State.from_dict(source.data)
            prompt = (
                pipeline_state.flow_prompt_speech_token,
                pipeline_state.flow_prompt_speech_feat,
                pipeline_state.flow_embedding,
            )
        else:
            prompt = (
                source.get("flow_prompt_speech_token"),
                source.get("flow_prompt_speech_feat"),
                source.get("flow_embedding"),
            )
        if all(value is not None for value in prompt):
            prompt_tensors = tuple(
                torch.as_tensor(value).detach().cpu() for value in prompt
            )
            if state.prompt_token is not None and (
                not torch.equal(state.prompt_token, prompt_tensors[0])
                or not torch.equal(state.prompt_feat, prompt_tensors[1])
                or not torch.equal(state.embedding, prompt_tensors[2])
            ):
                raise ValueError(
                    "Fun-CosyVoice3 MLX stream prompt tensors changed mid-request"
                )
            state.prompt_token, state.prompt_feat, state.embedding = prompt_tensors

    def validate_chunk(
        self,
        request_id: str,
        state: _FunCosyVoice3MlxStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        codes = codes.to(dtype=torch.long)
        if codes.ndim == 2 and codes.shape[-1] == 1:
            codes = codes.reshape(-1)
        if codes.ndim != 1:
            raise ValueError(
                f"Fun-CosyVoice3 MLX stream chunk must be 1-D, got {codes.shape}"
            )
        return codes.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _FunCosyVoice3MlxStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.tokens.extend(int(token) for token in codes.tolist())

    def should_decode(
        self, state: _FunCosyVoice3MlxStreamState, *, is_final: bool
    ) -> bool:
        del state
        return is_final

    def decode_delta(
        self,
        request_id: str,
        state: _FunCosyVoice3MlxStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        if not is_final or not state.tokens:
            return None
        if (
            state.prompt_token is None
            or state.prompt_feat is None
            or state.embedding is None
        ):
            raise RuntimeError(
                "Fun-CosyVoice3 MLX stream is missing prompt conditioning"
            )
        return self._vocoder.decode_tokens(
            token=torch.tensor(state.tokens, dtype=torch.int32).reshape(1, -1),
            prompt_token=state.prompt_token,
            prompt_feat=state.prompt_feat,
            embedding=state.embedding,
        )

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _FunCosyVoice3MlxStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        result = {"modality": "audio", "sample_rate": self._sample_rate}
        usage = build_usage(pipeline_state)
        if usage is not None:
            result["usage"] = usage
        return result

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Fun-CosyVoice3",
        )

    def release_stream_resources(
        self, request_id: str, state: _FunCosyVoice3MlxStreamState
    ) -> None:
        del request_id
        state.tokens.clear()
        state.prompt_token = None
        state.prompt_feat = None
        state.embedding = None


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
    max_batch_size: int | None = None,
    max_batch_wait_ms: int = 30,
    flow_batch_bucket_frames: int = 50,
    flow_batch_admission_frames: int = _DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
    enable_dit_torch_compile: bool = False,
    enable_flow_estimator_trt: bool = False,
    hift_dtype: str = "float32",
    hift_max_padding_waste: float = 1.5,
    token_hop_len: int = TOKEN_HOP_LEN,
    token_max_hop_len: int = TOKEN_MAX_HOP_LEN,
    disable_hop_growth: bool = False,
    mlx_model_path: str | None = None,
    mlx_model_revision: str | None = None,
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

    from sglang.srt.hardware_backend.mlx.runtime import use_mlx

    if use_mlx():
        if not current_platform.is_mps():
            raise RuntimeError("Fun-CosyVoice3 native MLX vocoder requires Apple Metal")
        if mlx_model_path is None:
            raise ValueError(
                "Fun-CosyVoice3 native MLX vocoder requires mlx_model_path"
            )
        if max_batch_size not in (None, 1):
            raise ValueError(
                "Fun-CosyVoice3 native MLX vocoder requires max_batch_size=1"
            )
        if enable_dit_torch_compile:
            raise ValueError(
                "enable_dit_torch_compile is unavailable on the native MLX vocoder"
            )
        vocoder = _CosyVoice3MlxVocoderAdapter(
            _load_cosyvoice3_mlx_vocoder(
                mlx_model_path, revision=mlx_model_revision, expected_dtype=dtype
            )
        )
        return _FunCosyVoice3MlxStreamingVocoderScheduler(
            vocoder,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    max_batch_size = 16 if max_batch_size is None else int(max_batch_size)
    dtype = dtype or "bfloat16"
    if enable_flow_estimator_trt and enable_dit_torch_compile:
        raise ValueError(
            "enable_flow_estimator_trt and enable_dit_torch_compile both "
            "target flow.decoder.estimator; enable only one"
        )
    if enable_dit_torch_compile is None:
        enable_dit_torch_compile = not enable_flow_estimator_trt
    checkpoint_dir = resolve_checkpoint(model_path)
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported Fun-CosyVoice3 vocoder dtype {dtype!r}; "
            f"expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    compute_dtype = _AUTOCAST_DTYPES[dtype]
    if (
        torch.device(device).type == "mps"
        and not current_platform.is_float64_supported()
    ):
        # Keep the declarative CUDA default (bf16) unchanged while avoiding
        # an autocast scope around the MPS Flow/HiFT path. The MPS adapter also
        # keeps HiFT's required float64 F0 predictor on CPU.
        compute_dtype = None
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
        enable_flow_estimator_trt=enable_flow_estimator_trt,
    )
    if enable_dit_torch_compile:
        _compile_dit_backbone(flow, compute_dtype=compute_dtype)

    vocoder = _CosyVoice3Vocoder(
        flow,
        hift,
        compute_dtype=compute_dtype,
        flow_batch_bucket_frames=flow_batch_bucket_frames,
        hift_compute_dtype=hift_dtype,
        hift_max_padding_waste=hift_max_padding_waste,
    )

    return FunCosyVoice3StreamingVocoderScheduler(
        vocoder,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder._flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
        token_hop_len=token_hop_len,
        token_max_hop_len=token_max_hop_len,
        disable_hop_growth=disable_hop_growth,
    )
