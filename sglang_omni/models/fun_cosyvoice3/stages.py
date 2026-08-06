# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fun-CosyVoice3 pipeline."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    cleanup_prepared_cosyvoice3_request,
    preprocess_cosyvoice3_payload,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_COSYVOICE_INSTALL_HINT = (
    "Fun-CosyVoice3 support requires the `cosyvoice` package. "
    "Clone the official repository and set PYTHONPATH, or install it "
    "in the serving environment before launching Fun-CosyVoice3."
)


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
    return flow, hift


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    return SimpleScheduler(
        preprocess_cosyvoice3_payload,
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


class _CosyVoice3Vocoder(BatchVocoderBase):
    def __init__(
        self,
        flow: Any,
        hift: Any,
        fp16: bool = False,
    ) -> None:
        self._flow = flow
        self._hift = hift
        self._fp16 = fp16

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FunCosyVoice3State, torch.Tensor]:
        state = load_state(payload)
        if state.audio_codes is None:
            raise RuntimeError(
                "Fun-CosyVoice3 vocoder requires audio_codes from tts_engine"
            )
        # The AR runner stores one-element tensors per step, which serialize as
        # ``[num_tokens, 1]``. Flow consumes one unbatched token sequence here.
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long).reshape(-1)
        return state, codes

    async def decode_batch(
        self, items: list[tuple[FunCosyVoice3State, torch.Tensor]]
    ) -> list[tuple[Any, int]]:
        results = []
        for state, codes in items:
            prompt_token = (
                torch.as_tensor(state.flow_prompt_speech_token, dtype=torch.int32)
                if state.flow_prompt_speech_token is not None
                else torch.zeros(1, 0, dtype=torch.int32)
            )
            prompt_feat = (
                torch.as_tensor(state.flow_prompt_speech_feat)
                if state.flow_prompt_speech_feat is not None
                else torch.zeros(1, 0, 80)
            )
            embedding = (
                torch.as_tensor(state.flow_embedding)
                if state.flow_embedding is not None
                else torch.zeros(1, 192)
            )
            wav = self._token2wav(
                token=codes.unsqueeze(0),
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                speed=state.speed,
            )
            results.append((wav, state.sample_rate))
        return results

    def _token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        speed: float = 1.0,
    ) -> torch.Tensor:
        device = next(self._flow.parameters()).device
        # note: the HiFT vocoder requires a minimum mel length for its
        # convolutional upsampling. Pad short token sequences with a
        # zero token so the vocoder has enough frames to process.
        min_tokens = 10
        if token.shape[1] < min_tokens:
            pad_len = min_tokens - token.shape[1]
            pad = torch.zeros(1, pad_len, dtype=token.dtype, device=token.device)
            token = torch.cat([token, pad], dim=1)

        with torch.autocast(device_type="cuda", enabled=self._fp16):
            tts_mel, _ = self._flow.inference(
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
                streaming=False,
                finalize=True,
            )
        if speed != 1.0:
            tts_mel = torch.nn.functional.interpolate(
                tts_mel,
                size=int(tts_mel.shape[2] / speed),
                mode="linear",
            )
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
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    checkpoint_dir = model_path
    flow, hift = _load_cosyvoice3_flow_hift(
        checkpoint_dir,
        device=device,
        fp16=(dtype == "float16"),
    )

    return _CosyVoice3Vocoder(flow, hift, fp16=(dtype == "float16")).build_scheduler(
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
