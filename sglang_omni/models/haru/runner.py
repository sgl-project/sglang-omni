# SPDX-License-Identifier: Apache-2.0
"""Advance Haru one conversation frame over scheduler-owned hybrid state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from haru.audio import decode_audio, prepare_reference
from haru.models.loaders import load_tokenizer
from haru.models.vibevoice_codec import open_stream
from haru.prompt import build_prompt
from haru.sampling import DryConfig, DryState, StreamDetokenizer, sample_token
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.haru.config import HaruSampling
from sglang_omni.models.haru.sglang_model import HaruSGLangModel
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.serve.realtime.types import MS_PER_SECOND

PCM16_SCALE = 32768.0


class CodecStream(Protocol):
    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


@dataclass(kw_only=True)
class HaruSessionState:
    sampling: HaruSampling
    instructions: str
    reference_wav: bytes | None
    codec: CodecStream
    generator: torch.Generator
    detokenizer: StreamDetokenizer
    dry_state: DryState | None
    positions: int = 0
    frame_index: int = 0
    previous_token: torch.Tensor | None = None
    previous_latent: torch.Tensor | None = None


@dataclass(kw_only=True)
class HaruRequestData(SGLangARRequestData):
    state: HaruSessionState
    input_embeddings: torch.Tensor
    retained_positions: int
    frame: TimedChunk | None = None


class HaruModelRunner(ModelRunner):
    model: HaruSGLangModel

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.LAST

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        embeddings = []
        for index, request in enumerate(requests):
            request_data = request.data
            assert isinstance(request_data, HaruRequestData)
            retained_positions = int(forward_batch.extend_prefix_lens_cpu[index])
            new_positions = int(forward_batch.extend_seq_lens_cpu[index])
            # Note (wbn): Codec feedback and the recurrent cache must advance together.
            if (
                retained_positions != request_data.retained_positions
                or new_positions != request_data.input_embeddings.shape[0]
            ):
                raise RuntimeError(
                    "Haru input frames are out of sync with the LM cache"
                )
            else:
                pass
            embeddings.append(request_data.input_embeddings)
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=torch.cat(embeddings))
        )

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        logits_output = result.logits_output
        assert logits_output is not None
        logits_rows = logits_output.next_token_logits
        hidden_rows = logits_output.hidden_states
        assert logits_rows is not None and hidden_rows is not None
        model = self.model
        config = model.config
        frame_duration_ms = config.frame_samples * MS_PER_SECOND / config.sample_rate
        stream_ids = [config.pad_token_id, config.eop_token_id]
        banned_token_ids = sorted(set(stream_ids) | ({config.eos_token_id} - {None}))
        tokens = []
        for request, logits, hidden in zip(
            requests, logits_rows, hidden_rows, strict=True
        ):
            request_data = request.data
            assert isinstance(request_data, HaruRequestData)
            state = request_data.state
            sampling = state.sampling
            logits = logits.float()
            if state.dry_state is not None:
                state.dry_state.apply(logits)
            else:
                pass
            logits[stream_ids] = model.logits_fp32(hidden, stream_ids)
            logits[config.eop_token_id] += sampling.eop_bias
            if state.previous_token is None and sampling.speak_first:
                logits[banned_token_ids] = -torch.inf
            else:
                pass
            token = sample_token(
                logits.unsqueeze(0),
                sampling.temperature,
                sampling.top_k,
                sampling.top_p,
                state.generator,
            )
            noise = torch.randn(
                (1, config.latent_dim),
                device=hidden.device,
                dtype=model.dtype,
                generator=state.generator,
            )
            latent = model.unscale_latents(
                model.head.sample(hidden.unsqueeze(0), noise, sampling.guidance_scale)
            )
            audio = state.codec.decode(latent.unsqueeze(0))[0].float()
            pcm = (
                (audio.clamp(-1.0, 1.0) * (PCM16_SCALE - 1))
                .round()
                .to(torch.int16)
                .cpu()
                .numpy()
                .astype("<i2", copy=False)
                .tobytes()
            )
            state.previous_token = token
            state.previous_latent = latent
            token_id = int(token)
            if state.dry_state is not None:
                state.dry_state.update(token_id)
            else:
                pass
            request_data.frame = TimedChunk(
                modality="audio",
                t_start_ms=state.frame_index * frame_duration_ms,
                duration_ms=frame_duration_ms,
                seq=state.frame_index,
                payload={"text": state.detokenizer.push(token_id), "pcm": pcm},
            )
            state.frame_index += 1
            tokens.append(token)
        result.next_token_ids = torch.cat(tokens)


class HaruSessionAdapter(ARSessionAdapter):
    def __init__(
        self,
        runner: HaruModelRunner,
        *,
        model_path: str,
        context_length: int,
        codec_backend: Literal["torch", "triton", "auto"],
    ) -> None:
        self.model = runner.model
        self.tokenizer = load_tokenizer(model_path)
        self.context_length = context_length
        self.codec_backend = codec_backend
        self.states: dict[SessionIdentity, HaruSessionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        config = self.model.config
        sampling = HaruSampling.model_validate(
            {
                name: request.params[name]
                for name in HaruSampling.model_fields
                if name in request.params
            }
        )
        instructions = request.params.get("instructions", "")
        reference_wav = request.params.get("reference_audio")
        if not isinstance(instructions, str):
            raise ValueError("Haru instructions must be a string")
        elif reference_wav is not None and not isinstance(reference_wav, bytes):
            raise ValueError("Haru reference audio must be WAV bytes")
        else:
            pass
        generator = torch.Generator(device=self.model.device)
        if sampling.seed is None:
            generator.seed()
        else:
            generator.manual_seed(sampling.seed)
        silent_ids = {config.pad_token_id, config.eop_token_id}
        banned_token_ids = silent_ids | ({config.eos_token_id} - {None})
        self.states[session_identity] = HaruSessionState(
            sampling=sampling,
            instructions=instructions,
            reference_wav=reference_wav,
            codec=open_stream(self.model.codec, backend=self.codec_backend),
            generator=generator,
            detokenizer=StreamDetokenizer(self.tokenizer, silent_ids),
            dry_state=DryState(DryConfig(), banned_token_ids) if sampling.dry else None,
        )

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        return payload

    @torch.inference_mode()
    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> HaruRequestData:
        model = self.model
        config = model.config
        state = self.states[session_identity]
        if state.positions == 0:
            if chunk.payload:
                raise ValueError("Haru needs to prefill before the first caller frame")
            else:
                pass
            reference = (
                torch.from_numpy(
                    prepare_reference(
                        decode_audio(state.reference_wav, config.sample_rate),
                        sample_rate=config.sample_rate,
                    )
                )
                if state.reference_wav is not None
                else None
            )
            prompt = build_prompt(
                config,
                model.codec,
                self.tokenizer,
                state.instructions,
                reference,
                state.sampling.speak_first,
                state.generator,
            )
            state.reference_wav = None
            input_ids = prompt.tokens.tolist()
            input_embeddings = model.embed(
                prompt.tokens.unsqueeze(0),
                prompt.model_latents.unsqueeze(0),
                prompt.user_latents.unsqueeze(0),
                prompt.model_mask.unsqueeze(0),
                prompt.user_mask.unsqueeze(0),
            )[0]
        else:
            assert (
                state.previous_token is not None and state.previous_latent is not None
            )
            pcm = chunk.payload
            frame_bytes = config.frame_samples * 2
            if not isinstance(pcm, bytes) or len(pcm) != frame_bytes:
                raise ValueError(
                    f"Haru needs an input frame of {frame_bytes} PCM16 bytes"
                )
            else:
                pass
            waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float()
            waveform = waveform.unsqueeze(0) / PCM16_SCALE
            user_latent = model.codec.sample(
                state.codec.encode(waveform), state.generator
            )
            input_ids = []
            input_embeddings = model.embed(
                state.previous_token.unsqueeze(0),
                state.previous_latent.unsqueeze(0),
                user_latent.float(),
            )[0]
        retained_positions = state.positions
        positions = retained_positions + input_embeddings.shape[0]
        if positions + 1 > self.context_length - 1:
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: Haru reached its context length; start a new session"
            )
        else:
            pass
        state.positions = positions
        sampling_params = SamplingParams(
            max_new_tokens=1, temperature=0.0, ignore_eos=True
        )
        sampling_params.normalize(tokenizer=None)
        native_request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=int(config.lm["vocab_size"]),
        )
        return HaruRequestData(
            req=native_request,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=1,
            temperature=0.0,
            state=state,
            input_embeddings=input_embeddings,
            retained_positions=retained_positions,
        )

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        return request_data.stage_payload

    def flush(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> Iterable[TimedChunk]:
        assert isinstance(request_data, HaruRequestData)
        assert request_data.frame is not None
        return (request_data.frame,)
