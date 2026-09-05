# SPDX-License-Identifier: Apache-2.0
"""Breeze prompt segments and T5Gemma2 reference/text encoding.

Token layout follows breezeblue-ai/breeze-tts (Apache-2.0), revision
43e2ea1595297c4059477e2e4a300653761c759b. Each text segment is encoded separately;
reference audio and the generated audio must never share the text encoder mask.
"""

from dataclasses import asdict

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

from sglang_omni.proto import StagePayload

from .checkpoint import load_component, read_config
from .request import BreezeRequest, parse_request

CONTEXT_LENGTH = 1024


def text_segments(request: BreezeRequest, *, negative: bool = False) -> list[str]:
    target = "[S0]"
    if request.instructions and not negative:
        target += f"<ins_bos>{request.instructions}<ins_eos>"
    target += request.text
    if request.ref_audio is not None:
        return [f"[S0]{request.ref_text}", target]
    return [target]


class BreezeFrontend(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        text_config = T5Gemma2TextConfig(**config["text_encoder_config"])
        text_config._attn_implementation = "sdpa"
        self.text_encoder = T5Gemma2TextEncoder(text_config)
        self.text_encoder_proj = nn.Linear(
            text_config.hidden_size, config["hidden_size"], bias=False
        )
        self.audio_embeddings = nn.Embedding(
            config["num_codebooks"] * config["vocab_size"], config["audio_embed_size"]
        )
        self.num_codebooks = config["num_codebooks"]
        self.audio_vocab_size = config["vocab_size"]
        self.audio_eos = config["codebook_eos_token_id"]

    @classmethod
    def from_checkpoint(cls, checkpoint: str, device: str):
        config = read_config(checkpoint)
        with init_empty_weights(include_buffers=False):
            frontend = cls(config)
        load_component(frontend.text_encoder, checkpoint, "text_encoder.")
        load_component(frontend.text_encoder_proj, checkpoint, "text_encoder_proj.")
        # This is tied to backbone_model.embed_tokens in the original model;
        # safetensors stores only the depth-decoder copy.
        load_component(
            frontend.audio_embeddings, checkpoint, "depth_decoder.model.embed_tokens."
        )
        return frontend.to(device=device, dtype=torch.bfloat16).eval()

    def encode_text(self, tokenizer, text: str) -> torch.Tensor:
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= CONTEXT_LENGTH:
            raise ValueError("Breeze-TTS-2 text segment exceeds the 1024-token context")
        ids = ids.to(self.audio_embeddings.weight.device)
        hidden = self.text_encoder(input_ids=ids).last_hidden_state
        return self.text_encoder_proj(hidden)[0]

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = (
            torch.arange(self.num_codebooks, device=codes.device)
            * self.audio_vocab_size
        )
        return self.audio_embeddings(codes.long() + offsets).sum(-2)

    @torch.no_grad()
    def prepare(
        self, payload: StagePayload, tokenizer, audio_tokenizer
    ) -> StagePayload:
        request = parse_request(payload)
        device = self.audio_embeddings.weight.device
        reference = None
        if request.ref_audio is not None:
            from sglang_omni.utils.audio import load_audio

            rate = audio_tokenizer.get_input_sample_rate()
            waveform = load_audio(request.ref_audio, target_sample_rate=rate)
            if waveform.size == 0:
                raise ValueError("Breeze-TTS-2 reference audio is empty")
            encoded = audio_tokenizer.encode(waveform, sr=rate)
            codes = encoded.audio_codes[0].to(device=device, dtype=torch.long)
            if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
                raise ValueError(
                    "Breeze-TTS-2 reference codec must return [frames, 16]"
                )
            eos = codes.new_full((1, self.num_codebooks), self.audio_eos)
            reference = self.embed_audio(torch.cat((codes, eos), dim=0))

        def branch(negative: bool):
            segments = [
                self.encode_text(tokenizer, text)
                for text in text_segments(request, negative=negative)
            ]
            if reference is not None:
                segments.insert(1, reference)
            return torch.cat(segments, dim=0).detach()

        cond = branch(False)
        uncond = branch(True) if request.sampling.cfg_scale != 1.0 else cond
        prompt_len = max(len(cond), len(uncond))
        # Do not silently truncate references or text. Match the reference
        # runtime's bounded generation window without dropping the final frame.
        remaining = CONTEXT_LENGTH - prompt_len
        if remaining < 1:
            raise ValueError(
                "Breeze-TTS-2 prompt leaves no generation room in the 1024-token context"
            )
        sampling = asdict(request.sampling)
        sampling["max_new_tokens"] = min(sampling["max_new_tokens"], remaining)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "prompt_embeds": cond,
                "negative_embeds": uncond,
                "sampling": sampling,
            },
        )


def load_text_tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint, fix_mistral_regex=False)
