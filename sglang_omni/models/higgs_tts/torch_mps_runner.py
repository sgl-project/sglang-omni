# SPDX-License-Identifier: Apache-2.0
"""Single-request Torch MPS language execution for Higgs TTS."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import Qwen3ForCausalLM

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner


def load_torch_language_model(checkpoint: str, *, device, dtype):
    """Load the unfused HF Qwen3 weights, checking every language parameter."""
    config = HiggsMultimodalQwen3Config.from_pretrained(checkpoint).get_text_config()
    # The rotary-buffer reconstruction below implements default RoPE only.
    # Never silently replace a checkpoint's scaled frequencies with that formula.
    if config.rope_parameters.get("rope_type", "default") != "default":
        raise ValueError("Higgs Torch MPS currently requires default RoPE")
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        language_model = Qwen3ForCausalLM(config)
    prefixes = {
        "tied.embedding.text_embedding.": "model.embed_tokens.",
        "tied.head.text_head.": "lm_head.",
        "body.layers.": "model.layers.",
        "body.norm.": "model.norm.",
    }
    state = {}
    for path in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                for prefix, target in prefixes.items():
                    if key.startswith(prefix):
                        name = target + key[len(prefix) :]
                        if name in state:
                            raise ValueError(f"Duplicate Higgs language weight: {name}")
                        state[name] = weights.get_tensor(key)
                        break
    if config.tie_word_embeddings and "model.embed_tokens.weight" in state:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
    language_model.load_state_dict(state, strict=True, assign=True)
    language_model.tie_weights()
    # Meta construction also leaves the nonpersistent rotary buffer on meta.
    rotary = language_model.model.rotary_emb
    theta = float(config.rope_parameters["rope_theta"])
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    )
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    language_model = language_model.eval().to(device=device, dtype=dtype)
    # Keep frequencies in fp32; HF rounds the resulting cos/sin to the model
    # dtype. Rounding frequencies themselves accumulates position-dependent error.
    rotary.inv_freq = inv_freq.to(device=device)
    rotary.original_inv_freq = rotary.inv_freq.clone()
    return language_model


def install_torch_mps_language_model(model: Any, checkpoint: str) -> None:
    parameter = next(model.backbone.parameters())
    device, dtype = parameter.device, parameter.dtype
    del parameter
    del model.backbone
    gc.collect()
    torch.mps.empty_cache()
    model.backbone = load_torch_language_model(checkpoint, device=device, dtype=dtype)


class HiggsTorchMpsModelRunner(HiggsTTSModelRunner):
    """Keep Higgs sampling/streaming hooks; replace only language forwards."""

    def __init__(self, tp_worker, output_processor):
        super().__init__(tp_worker, output_processor)
        self._past_key_values: dict[str, Any] = {}

    def lookahead_eligible(self, batch):
        return False

    @staticmethod
    def _request_id(requests):
        if len(requests) != 1:
            raise ValueError("Higgs Torch MPS requires exactly one active request")
        return requests[0].request_id

    def reset_request(self, request_id):
        self._past_key_values.pop(request_id, None)
        self.model.reset_request(request_id)
        self._cg_launch_key = None

    def on_request_finished(self, request_id, req_data):
        try:
            super().on_request_finished(request_id, req_data)
        finally:
            self._past_key_values.pop(request_id, None)

    def _forward(self, request_id, embeddings, *, prefill):
        if prefill:
            self._past_key_values.pop(request_id, None)
            cache = None
        else:
            if request_id not in self._past_key_values:
                raise RuntimeError(f"Higgs MPS decode has no KV cache for {request_id}")
            cache = self._past_key_values[request_id]
        try:
            output = self.model.backbone.model(
                inputs_embeds=embeddings.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
            )
        except Exception:
            self.reset_request(request_id)
            raise
        self._past_key_values[request_id] = output.past_key_values
        return output.last_hidden_state[:, -1, :]

    @staticmethod
    def _result(logits, hidden):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=logits, hidden_states=hidden
            ),
            next_token_ids=None,
            can_run_cuda_graph=False,
        )

    @torch.inference_mode()
    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        rid = self._request_id(requests)
        req = requests[0].data.req
        if req.extend_range.start != 0 or req.inflight_middle_chunks > 0:
            raise ValueError(
                "Higgs MPS requires whole-prompt prefill without prefix reuse"
            )
        inputs = get_omni_prefill_inputs(forward_batch)
        hidden = self._forward(rid, inputs.input_embeds, prefill=True)
        params = self.model._gen_params_for_batch(forward_batch.sampling_info, 1)
        logits = self.model.decode_codebooks_batch(hidden, [rid], params)
        return self._result(logits, hidden)

    @torch.inference_mode()
    def custom_decode_forward(self, forward_batch, schedule_batch, requests):
        rid = self._request_id(requests)
        embeddings = self.model._decode_step_embeds_cg(forward_batch.input_ids, 1)
        hidden = self._forward(rid, embeddings, prefill=False)
        logits = self.model.decode_codebooks_batch_cg(hidden)
        return self._result(logits, hidden)
