# SPDX-License-Identifier: Apache-2.0
"""Torch MPS runner for Chatterbox-Turbo T3."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load_file

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.chatterbox.request_builders import START_SPEECH_TOKEN
from sglang_omni.models.chatterbox.stages import CHATTERBOX_INSTALL_HINT


def install_torch_mps_t3_model(
    checkpoint_dir: str, device: str, placeholder: Any | None = None
) -> Any:
    """Load chatterbox-tts's T3 on MPS, replacing the SGLang placeholder model."""
    if placeholder is not None:
        # Drop the placeholder weights before loading T3 so the two models
        # never peak together on MPS.
        placeholder.to("meta")
        gc.collect()
        torch.mps.empty_cache()
    try:
        from chatterbox.models.t3 import T3
        from chatterbox.models.t3.modules.t3_config import T3Config
    except ImportError as exc:
        raise RuntimeError(CHATTERBOX_INSTALL_HINT) from exc

    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)
    state = load_file(os.path.join(checkpoint_dir, "t3_turbo_v1.safetensors"))
    if "model" in state:
        state = state["model"]
    t3.load_state_dict(state)
    del t3.tfmr.wte
    t3.to(device).eval()

    conds_path = os.path.join(checkpoint_dir, "conds.pt")
    if os.path.exists(conds_path):
        from chatterbox.tts_turbo import Conditionals

        conds = Conditionals.load(conds_path, map_location="cpu")
        conds.to(device)
        t3._builtin_t3_cond = conds.t3
    return t3


@dataclass
class _DecodeState:
    past_key_values: Any = None
    past_token_ids: list[int] = field(default_factory=list)
    logits_processors: Any = None
    temperature: float = 1.0
    generator: Any = None


class ChatterboxT3TorchMpsModelRunner(ModelRunner):
    """Single-request T3 prefill and cached Hugging Face Torch decoding."""

    model_name = "Chatterbox-Turbo"

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._states: dict[str, _DecodeState] = {}

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    def _one_request(self, requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                f"{self.model_name} Torch MPS requires max_running_requests=1"
            )
        return requests[0]

    def _next_token_result(self, next_token_ids: torch.Tensor) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def _build_logits_processors(self, data: Any) -> Any:
        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        processors = LogitsProcessorList()
        if data.temperature > 0 and data.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(data.temperature))
        if data.top_k > 0:
            processors.append(TopKLogitsWarper(data.top_k))
        if data.top_p < 1.0:
            processors.append(TopPLogitsWarper(data.top_p))
        if data.repetition_penalty != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(data.repetition_penalty)
            )
        return processors

    def _sample(self, logits: torch.Tensor, state: _DecodeState) -> torch.Tensor:
        input_ids = torch.tensor(
            [state.past_token_ids], dtype=torch.long, device=logits.device
        )
        processed = state.logits_processors(input_ids, logits)
        if state.temperature <= 0:
            return processed.argmax(-1)
        probs = torch.softmax(processed, dim=-1)
        return torch.multinomial(
            probs, num_samples=1, generator=state.generator
        ).squeeze(-1)

    @torch.inference_mode()
    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch
        from chatterbox.models.t3.modules.cond_enc import T3Cond

        scheduler_request = self._one_request(requests)
        data = scheduler_request.data
        t3 = self.model

        text_tokens = torch.tensor(
            [data.text_tokens], dtype=torch.long, device=self.device
        )
        cond_tokens = torch.tensor(
            [data.cond_prompt_speech_tokens], dtype=torch.long, device=self.device
        )
        speaker = data.speaker_embedding
        if speaker is None:
            t3_cond = t3._builtin_t3_cond
        else:
            speaker = speaker.to(device=self.device)
            t3_cond = T3Cond(
                speaker_emb=speaker,
                cond_prompt_speech_tokens=cond_tokens,
                emotion_adv=torch.zeros(1, 1, 1, device=self.device),
            )

        start_tokens = torch.full_like(text_tokens[:, :1], START_SPEECH_TOKEN)
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=start_tokens,
            cfg_weight=0.0,
        )

        output = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        hidden = output.last_hidden_state[:, -1:]
        logits = t3.speech_head(hidden)[:, -1, :]

        generator = None
        if data.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(data.seed))
        state = _DecodeState(
            past_key_values=output.past_key_values,
            past_token_ids=[START_SPEECH_TOKEN],
            logits_processors=self._build_logits_processors(data),
            temperature=data.temperature,
            generator=generator,
        )
        next_token = self._sample(logits, state)
        state.past_token_ids.append(int(next_token.item()))
        self._states[scheduler_request.request_id] = state
        return self._next_token_result(next_token)

    @torch.inference_mode()
    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch
        scheduler_request = self._one_request(requests)
        request_id = scheduler_request.request_id
        try:
            state = self._states[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"{self.model_name} Torch MPS decode has no cache for {request_id}"
            ) from exc

        current_token = torch.tensor(
            [[state.past_token_ids[-1]]], dtype=torch.long, device=self.device
        )
        t3 = self.model
        current_embed = t3.speech_emb(current_token)
        output = t3.tfmr(
            inputs_embeds=current_embed,
            past_key_values=state.past_key_values,
            use_cache=True,
        )
        state.past_key_values = output.past_key_values
        hidden = output.last_hidden_state[:, -1:]
        logits = t3.speech_head(hidden)[:, -1, :]

        next_token = self._sample(logits, state)
        state.past_token_ids.append(int(next_token.item()))
        return self._next_token_result(next_token)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del req_data
        self._states.pop(request_id, None)

    def abort_request(self, request_id: str) -> None:
        self._states.pop(request_id, None)
