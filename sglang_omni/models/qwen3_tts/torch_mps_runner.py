# SPDX-License-Identifier: Apache-2.0
"""Canonical Torch compatibility runner for Qwen3-TTS on MPS."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


def _materialize_rotary_buffers(talker: torch.nn.Module) -> None:
    """Recreate non-persistent RoPE buffers omitted from checkpoint weights."""
    for module in talker.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or not inv_freq.is_meta:
            continue
        rope_init_fn = getattr(module, "rope_init_fn", None)
        config = getattr(module, "config", None)
        if rope_init_fn is None or config is None:
            raise RuntimeError(
                "Qwen3-TTS Torch MPS cannot initialize a meta rotary buffer"
            )
        initialized, attention_scaling = rope_init_fn(config, torch.device("cpu"))
        module.inv_freq = initialized
        module.original_inv_freq = initialized.clone()
        module.attention_scaling = attention_scaling

    meta_buffers = [name for name, value in talker.named_buffers() if value.is_meta]
    if meta_buffers:
        raise RuntimeError(
            "Qwen3-TTS Torch MPS talker has uninitialized buffers: "
            f"{meta_buffers[:10]}"
        )


def _load_talker_weights(talker: torch.nn.Module, checkpoint: Path) -> None:
    expected = set(talker.state_dict())
    state_dict: dict[str, torch.Tensor] = {}
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"Qwen3-TTS checkpoint has no safetensors in {checkpoint}")

    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as weights:
            for checkpoint_name in weights.keys():
                if not checkpoint_name.startswith("talker."):
                    continue
                model_name = checkpoint_name.removeprefix("talker.")
                if model_name in expected:
                    state_dict[model_name] = weights.get_tensor(checkpoint_name)

    missing = expected - set(state_dict)
    if missing:
        raise ValueError(
            "Qwen3-TTS Torch MPS talker checkpoint is incomplete: "
            f"{sorted(missing)[:10]}"
        )
    talker.load_state_dict(state_dict, strict=True, assign=True)
    _materialize_rotary_buffers(talker)


def install_torch_mps_talker(model: Any, model_path: str) -> None:
    """Replace SGLang execution modules with the upstream Torch talker."""
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    checkpoint = _resolve_model_path(model_path)
    old_parameter = next(model.parameters())
    device, dtype = old_parameter.device, old_parameter.dtype
    config = model.config
    # qwen-tts 0.1.1 omits this inherited field. The talker embedding is in
    # codec space, so the text-domain TTS pad id would be out of range.
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = int(config.codec_pad_id)

    for name in ("model", "text_projection", "codec_head", "code_predictor"):
        delattr(model, name)
    # SGLang caches every parameter for its custom weight loader. Keeping the
    # mapping would retain the complete replaced talker even after its modules
    # are detached.
    model._cached_params_dict = {}
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    with torch.device("meta"):
        talker = Qwen3TTSTalkerForConditionalGeneration(config)
    _load_talker_weights(talker, checkpoint)
    talker = talker.eval().to(device=device, dtype=dtype)

    # The prompt-builder mixin continues to use these public components. They
    # alias the modules owned by the canonical talker, so there is one copy of
    # every parameter shared by preprocessing and generation.
    model.torch_mps_talker = talker
    model.model = talker.model
    model.text_projection = talker.text_projection
    model.codec_head = talker.codec_head
    model.code_predictor = talker.code_predictor
    model.model._feedback_buffer = torch.empty(0, device=device, dtype=dtype)
    logger.info("Installed upstream Qwen3-TTS Torch talker on %s (%s)", device, dtype)


@dataclass
class _TorchMpsRequestState:
    past_key_values: Any
    past_hidden: torch.Tensor
    generation_step: int
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    attention_mask: torch.Tensor
    next_logits: torch.Tensor
    semantic_generator: torch.Generator


class Qwen3TTSTorchMpsModelRunner(ModelRunner):
    """Run one request through the upstream eager Torch talker and its PKV."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._request_states: dict[str, _TorchMpsRequestState] = {}

    @property
    def talker(self) -> Any:
        return self.model.torch_mps_talker

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    @staticmethod
    def _one_request(requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                "Qwen3-TTS Torch MPS currently requires max_running_requests=1"
            )
        return requests[0]

    @staticmethod
    def _batch_result(next_token_ids: torch.Tensor) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    @staticmethod
    def _trailing_text(data: Any, pad_embed: torch.Tensor) -> torch.Tensor:
        rows = list(data.pending_text_queue)
        if not rows:
            return pad_embed.new_empty((1, 0, pad_embed.shape[-1]))
        return torch.stack(rows, dim=0).unsqueeze(0)

    @staticmethod
    def _seed_subtalker(data: Any, device: torch.device) -> None:
        seed = int(data.subtalker_sampling_seed)
        torch.manual_seed(seed)
        if device.type == "mps":
            torch.mps.manual_seed(seed)

    def _sample_semantic(
        self,
        logits: torch.Tensor,
        data: Any,
        generator: torch.Generator,
    ) -> torch.Tensor:
        sampling = data.req.sampling_params
        scores = logits.detach().float().reshape(-1).cpu().clone()
        vocab_size = min(int(self.model.config.vocab_size), int(scores.numel()))
        scores = scores[:vocab_size]

        suppress_start = max(0, vocab_size - 1024)
        eos_id = int(self.model.config.codec_eos_token_id)
        if suppress_start < vocab_size:
            eos_score = scores[eos_id].clone() if 0 <= eos_id < vocab_size else None
            scores[suppress_start:vocab_size] = float("-inf")
            if eos_score is not None:
                scores[eos_id] = eos_score

        penalty = float(sampling.repetition_penalty)
        if penalty != 1.0:
            for token_id in set(int(token) for token in data.req.output_ids):
                if not 0 <= token_id < vocab_size:
                    continue
                scores[token_id] = (
                    scores[token_id] * penalty
                    if scores[token_id] < 0
                    else scores[token_id] / penalty
                )

        temperature = float(sampling.temperature)
        if temperature <= 0:
            token = int(scores.argmax())
        else:
            scores.div_(temperature)
            top_k = int(sampling.top_k)
            if 0 < top_k < vocab_size:
                threshold = torch.topk(scores, top_k).values[-1]
                scores[scores < threshold] = float("-inf")
            top_p = float(sampling.top_p)
            if top_p < 1.0:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                scores[sorted_indices[remove]] = float("-inf")
            probabilities = torch.softmax(scores, dim=-1)
            token = int(
                torch.multinomial(
                    probabilities,
                    1,
                    generator=generator,
                ).item()
            )
        return torch.tensor([token], dtype=torch.long, device=self.device)

    def _advance_frame(
        self,
        scheduler_request: Any,
        state: _TorchMpsRequestState,
        semantic_id: torch.Tensor,
    ) -> None:
        data = scheduler_request.data
        if int(semantic_id.item()) == int(self.model.config.codec_eos_token_id):
            return

        attention_mask = torch.cat(
            [state.attention_mask, state.attention_mask.new_ones((1, 1))], dim=1
        )
        cache_position = torch.tensor(
            [state.attention_mask.shape[1]], dtype=torch.long, device=self.device
        )
        output = self.talker(
            input_ids=semantic_id.reshape(1, 1),
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            past_hidden=state.past_hidden,
            trailing_text_hidden=state.trailing_text_hidden,
            tts_pad_embed=state.tts_pad_embed,
            generation_step=state.generation_step,
            subtalker_dosample=bool(data.subtalker_dosample),
            subtalker_top_p=float(data.subtalker_top_p),
            subtalker_top_k=int(data.subtalker_top_k),
            subtalker_temperature=float(data.subtalker_temperature),
            use_cache=True,
            cache_position=cache_position,
            return_dict=True,
        )
        codec_ids = output.hidden_states[1]
        if codec_ids is None or codec_ids.shape[0] != 1:
            raise RuntimeError("Qwen3-TTS Torch MPS talker returned no codec frame")
        data.output_codes.append(codec_ids[0].detach().clone())
        state.past_key_values = output.past_key_values
        state.past_hidden = output.past_hidden
        state.generation_step = int(output.generation_step)
        state.attention_mask = attention_mask
        state.next_logits = output.logits[:, -1, :]

    @torch.inference_mode()
    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch, schedule_batch
        scheduler_request = self._one_request(requests)
        data = scheduler_request.data
        prompt = data.prompt_input_embeds
        if prompt is None:
            raise RuntimeError("Qwen3-TTS Torch MPS prefill requires prompt embeddings")
        prompt = prompt.to(device=self.device, dtype=self.talker.dtype).unsqueeze(0)
        attention_mask = torch.ones(
            (1, prompt.shape[1]), dtype=torch.long, device=self.device
        )
        pad_embed = data.tts_pad_embed.to(
            device=self.device, dtype=prompt.dtype
        ).reshape(1, 1, -1)
        trailing = self._trailing_text(data, pad_embed).to(
            device=self.device, dtype=prompt.dtype
        )
        output = self.talker(
            inputs_embeds=prompt,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing,
            tts_pad_embed=pad_embed,
            use_cache=True,
            return_dict=True,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(data.semantic_sampling_seed))
        state = _TorchMpsRequestState(
            past_key_values=output.past_key_values,
            past_hidden=output.past_hidden,
            generation_step=int(output.generation_step),
            trailing_text_hidden=trailing,
            tts_pad_embed=pad_embed,
            attention_mask=attention_mask,
            next_logits=output.logits[:, -1, :],
            semantic_generator=generator,
        )
        self._request_states[scheduler_request.request_id] = state
        self._seed_subtalker(data, self.device)
        semantic_id = self._sample_semantic(
            state.next_logits, data, state.semantic_generator
        )
        self._advance_frame(scheduler_request, state, semantic_id)
        return self._batch_result(semantic_id)

    @torch.inference_mode()
    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch, schedule_batch
        scheduler_request = self._one_request(requests)
        try:
            state = self._request_states[scheduler_request.request_id]
        except KeyError as exc:
            raise RuntimeError(
                "Qwen3-TTS Torch MPS decode has no request state for "
                f"{scheduler_request.request_id}"
            ) from exc
        semantic_id = self._sample_semantic(
            state.next_logits,
            scheduler_request.data,
            state.semantic_generator,
        )
        self._advance_frame(scheduler_request, state, semantic_id)
        return self._batch_result(semantic_id)

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del req_data
        self._request_states.pop(request_id, None)

    def abort_request(self, request_id: str) -> None:
        self._request_states.pop(request_id, None)


__all__ = ["Qwen3TTSTorchMpsModelRunner", "install_torch_mps_talker"]
