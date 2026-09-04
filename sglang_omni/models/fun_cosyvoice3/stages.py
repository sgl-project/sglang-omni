# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence, cast

import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec

# Note (xinran): This is an admission budget, not a maximum supported request
# length. The scheduler admits a request that exceeds it as a singleton Flow
# batch and defers following requests to the next batch.
_DEFAULT_FLOW_BATCH_ADMISSION_FRAMES = 2000

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


def _solve_flow_euler(
    decoder: Any,
    x: torch.Tensor,
    t_span: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    batch_size, channels, frames = x.shape
    dtype = spks.dtype
    x_in = torch.zeros(2 * batch_size, channels, frames, device=x.device, dtype=dtype)
    mask_in = torch.zeros(2 * batch_size, 1, frames, device=x.device, dtype=dtype)
    mu_in = torch.zeros_like(x_in)
    t_in = torch.zeros(2 * batch_size, device=x.device, dtype=dtype)
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
        derivative = decoder.forward_estimator(
            x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False
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


@torch.inference_mode()
def _generate_flow(flow: Any, packed: _PackedFlowBatch) -> torch.Tensor:
    embedding = flow.spk_embed_affine_layer(F.normalize(packed.embedding, dim=1))
    token_embedding = flow.input_embedding(torch.clamp(packed.token, min=0))
    h = flow.pre_lookahead_layer(
        token_embedding * packed.token_mask.to(token_embedding.dtype)
    )
    mu = h.repeat_interleave(flow.token_mel_ratio, dim=1).transpose(1, 2).contiguous()
    batch_size, channels, max_mel = mu.shape
    if channels != flow.output_size:
        raise ValueError("Flow pre-lookahead output width does not match output_size")
    mask = (
        (
            torch.arange(max_mel, device=mu.device).unsqueeze(0)
            < packed.total_mel_lengths_tensor.unsqueeze(1)
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
    return _solve_flow_euler(decoder, z, t_span, mu, mask, embedding, cond)


class FunCosyVoice3Flow:
    """CosyVoice3 Flow with batch inference enabled as its default API."""

    def __init__(self, flow: Any) -> None:
        self._flow = flow

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

    @torch.inference_mode()
    def inference(self, inputs: Sequence[FlowBatchInput]) -> list[torch.Tensor]:
        packed = _pack_flow_inputs(self._flow, inputs)
        generated = _generate_flow(self._flow, packed)
        outputs: list[torch.Tensor] = []
        for index, prompt_frames in enumerate(packed.prompt_mel_lengths):
            mel = generated[
                index : index + 1,
                :,
                prompt_frames : packed.total_mel_lengths[index],
            ]
            expected_frames = (
                packed.target_token_lengths[index] * self._flow.token_mel_ratio
            )
            if mel.shape != (1, self._flow.output_size, expected_frames):
                raise RuntimeError(
                    f"Flow output {index} has unexpected shape {tuple(mel.shape)}"
                )
            outputs.append(mel)
        return outputs


def load_state(payload: StagePayload) -> FunCosyVoice3State:
    return _load_pipeline_state(payload, FunCosyVoice3State)


def store_state(payload: StagePayload, state: FunCosyVoice3State) -> StagePayload:
    return _store_pipeline_state(payload, state)


def _load_cosyvoice3_flow_hift(
    checkpoint_dir: str,
    device: str,
    fp16: bool = False,
) -> tuple[Any, Any]:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except ImportError as exc:
        raise RuntimeError(_COSYVOICE_INSTALL_HINT) from exc

    cv = CosyVoice3(checkpoint_dir, fp16=fp16)
    flow = cv.model.flow
    hift = cv.model.hift
    flow.to(device).eval()
    hift.to(device).eval()
    del cv.model.llm
    return FunCosyVoice3Flow(flow), hift


def _configure_dit_torch_compile() -> None:
    """Enable the Inductor/Dynamo flags the DiT graph wants, without pulling in
    the full ``sglang.srt`` stack (the vocoder is a plain pipeline process)."""
    torch._inductor.config.fx_graph_cache = True
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 1024


def _run_dit_estimator(
    estimator: Any,
    mel_frames: int,
    *,
    compute_dtype: torch.dtype | None = None,
) -> None:

    param = next(estimator.parameters())
    device, dtype = param.device, param.dtype
    t = int(mel_frames)
    # CFG batch 2, mel dim 80 (proj_out.out_features for the pinned checkpoint).
    x = torch.randn(2, 80, t, device=device, dtype=dtype)
    mask = torch.ones(2, 1, t, device=device, dtype=dtype)
    mu = torch.randn(2, 80, t, device=device, dtype=dtype)
    timestep = torch.zeros(2, device=device, dtype=dtype)
    spks = torch.randn(2, 80, device=device, dtype=dtype)
    cond = torch.randn(2, 80, t, device=device, dtype=dtype)
    with torch.autocast(
        device_type=current_platform.device_type,
        dtype=compute_dtype,
        enabled=compute_dtype is not None,
    ):
        estimator(x, mask, mu, timestep, spks, cond, streaming=False)


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
    try:
        estimator.forward = torch.compile(original_forward, dynamic=True)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                _run_dit_estimator(
                    estimator,
                    warmup_mel_frames,
                    compute_dtype=compute_dtype,
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
        "warmup_mel_frames=%d, warmup_steps=%d)",
        compute_dtype,
        warmup_mel_frames,
        warmup_steps,
    )
    return True


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    # note(chenye): Reference conditioning supports concurrent calls;
    # model prompt finalization is serialized.
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
        max_concurrency=4,
        abort_callback=cleanup_prepared_cosyvoice3_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    from sglang_omni.models.fun_cosyvoice3.engine_builder import (
        FunCosyVoice3EngineBuilder,
    )

    return FunCosyVoice3EngineBuilder().build(
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


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        compute_dtype: torch.dtype | None = None,
        flow_batch_bucket_frames: int = 50,
    ) -> None:
        if flow_batch_bucket_frames <= 0:
            raise ValueError("flow_batch_bucket_frames must be greater than zero")
        if not isinstance(flow.decoder.estimator, torch.nn.Module):
            raise RuntimeError(
                "Fun-CosyVoice3 requires the PyTorch Flow estimator from the pinned "
                "CosyVoice commit; TensorRT Flow is not supported"
            )
        self._flow = (
            flow if isinstance(flow, FunCosyVoice3Flow) else FunCosyVoice3Flow(flow)
        )
        self._hift = hift
        self._compute_dtype = compute_dtype
        self._flow_batch_bucket_frames = flow_batch_bucket_frames

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )

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

        for bucket in buckets.values():
            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=self._compute_dtype,
                enabled=self._compute_dtype is not None,
            ):
                mel_list = self._flow.inference(
                    [request.flow_input for request in bucket]
                )
                for request, mel in zip(bucket, mel_list, strict=True):
                    results[request.index] = (
                        self._mel2wav(mel),
                        request.sample_rate,
                    )

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
        tts_speech, _ = self._hift.inference(speech_feat=tts_mel, finalize=True)
        return tts_speech.detach().cpu()

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


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    flow_batch_bucket_frames: int = 50,
    flow_batch_admission_frames: int = _DEFAULT_FLOW_BATCH_ADMISSION_FRAMES,
    enable_dit_torch_compile: bool = False,
) -> SimpleScheduler:
    if flow_batch_admission_frames <= 0:
        raise ValueError("flow_batch_admission_frames must be greater than zero")
    device = resolve_device_spec(device, gpu_id)
    checkpoint_dir = resolve_checkpoint(model_path)
    if dtype not in _AUTOCAST_DTYPES:
        raise ValueError(
            f"Unsupported Fun-CosyVoice3 vocoder dtype {dtype!r}; "
            f"expected one of {sorted(_AUTOCAST_DTYPES)}"
        )
    compute_dtype = _AUTOCAST_DTYPES[dtype]
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
    )
    if enable_dit_torch_compile:
        _compile_dit_backbone(flow, compute_dtype=compute_dtype)

    vocoder = _CosyVoice3Vocoder(
        flow,
        hift,
        compute_dtype=compute_dtype,
        flow_batch_bucket_frames=flow_batch_bucket_frames,
    )

    return SimpleScheduler(
        vocoder.decode_payload,
        batch_compute_fn=vocoder.decode_payloads,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        request_cost_fn=vocoder._flow_scheduler_cost,
        max_batch_cost=flow_batch_admission_frames,
    )
