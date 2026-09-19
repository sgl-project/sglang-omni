# SPDX-License-Identifier: Apache-2.0
"""Whole-request Torch/MPS AR scheduler for MiniMax Music 3."""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, Qwen3ForCausalLM

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

from .chunking import chunk_windows
from .constants import AR_CFG_SCALE, AR_CFG_TOP_K
from .payload_types import MiniMaxMusic3State
from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS, validate_tokenizer_ids
from .rvq_decoder import RVQDepthDecoder

logger = logging.getLogger(__name__)

_SEMANTIC_VOCAB_SIZE = 16_384
_MODEL_PATTERNS = (
    "config.json",
    "LICENSE",
    "language_model/*",
    "rvq_depth_decoder/*",
    "tokenizer/*",
    "flowmatching_vae.pth",
    "dav.pth",
)


def resolve_torch_mps_directory(
    model_path: str,
    revision: str | None = None,
) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_path,
            revision=revision,
            allow_patterns=list(_MODEL_PATTERNS),
        )
    )


def _load_rvq_depth_decoder(
    model_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[RVQDepthDecoder, nn.Embedding]:
    component_dir = model_dir / "rvq_depth_decoder"
    config = json.loads((component_dir / "config.json").read_text())
    state = load_file(str(component_dir / "diffusion_pytorch_model.safetensors"))

    with torch.device("meta"):
        decoder = RVQDepthDecoder(
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_attention_heads"]),
            intermediate_size=int(config["intermediate_size"]),
            audio_vocab_size=int(config["audio_vocab_size"]),
            num_codebooks=int(config["num_codebooks"]),
            max_seq_len=int(config.get("max_position_embeddings", 16)),
        )
        audio_embeddings = nn.Embedding(
            int(config["audio_vocab_size"]) * (int(config["num_codebooks"]) - 1),
            int(config["hidden_size"]),
        )

    mapped: dict[str, torch.Tensor] = {}
    for name in ("projection.weight", "pos_embedding.weight", "norm.weight"):
        mapped[name] = state[name]
    for index in range(int(config["num_codebooks"]) - 1):
        name = f"audio_heads.{index}.weight"
        mapped[name] = state[name]
    for index in range(int(config["num_layers"])):
        source = f"layers.{index}"
        target = source
        mapped[f"{target}.input_layernorm.weight"] = state[
            f"{source}.input_layernorm.weight"
        ]
        mapped[f"{target}.post_attention_layernorm.weight"] = state[
            f"{source}.post_attention_layernorm.weight"
        ]
        mapped[f"{target}.self_attn.in_proj_weight"] = torch.cat(
            [
                state[f"{source}.attn.to_{projection}.weight"]
                for projection in ("q", "k", "v")
            ],
            dim=0,
        )
        mapped[f"{target}.self_attn.out_proj.weight"] = state[
            f"{source}.attn.to_out.weight"
        ]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            mapped[f"{target}.{projection}.weight"] = state[
                f"{source}.{projection}.weight"
            ]

    decoder.load_state_dict(mapped, strict=True, assign=True)
    audio_embeddings.load_state_dict(
        {"weight": state["audio_embeddings.weight"]},
        strict=True,
        assign=True,
    )
    del state, mapped
    decoder = decoder.to(device=device, dtype=dtype).eval()
    audio_embeddings = audio_embeddings.to(device=device, dtype=dtype).eval()
    return decoder, audio_embeddings


class MiniMaxMusic3TorchMpsARModel:
    def __init__(self, model_dir: Path) -> None:
        self.device = torch.device("mps")
        self.dtype = torch.bfloat16
        self.language_model = Qwen3ForCausalLM.from_pretrained(
            model_dir / "language_model",
            dtype=self.dtype,
            device_map={"": str(self.device)},
            low_cpu_mem_usage=True,
        ).eval()
        self.rvq_depth_decoder, self.audio_embeddings = _load_rvq_depth_decoder(
            model_dir,
            device=self.device,
            dtype=self.dtype,
        )
        self.audio_vocab_size = self.rvq_depth_decoder.audio_vocab_size
        self.num_codebooks = self.rvq_depth_decoder.num_codebooks
        self.audio_embedding_offsets = (
            torch.arange(self.num_codebooks - 1, device=self.device)
            * self.audio_vocab_size
        ).unsqueeze(0)
        self.frame_embedding_scale = self.num_codebooks**-0.5
        self.c0_logit_ids = torch.cat(
            (
                torch.tensor([SPECIAL_TOKEN_IDS["<|audio_end|>"]], device=self.device),
                torch.arange(
                    AUDIO_CODE_OFFSET,
                    AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE,
                    device=self.device,
                ),
            )
        )

    def _sample_top_k(
        self,
        logits: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        threshold = torch.topk(values, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        probabilities = torch.softmax(
            values.masked_fill(values < threshold, -torch.inf), dim=-1
        )
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)

    def _embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        c0 = self.language_model.model.embed_tokens(codes[:, :1] + AUDIO_CODE_OFFSET)
        residual = self.audio_embeddings(
            codes[:, 1:] + self.audio_embedding_offsets
        ).sum(dim=1, keepdim=True)
        return (c0 + residual.to(c0.dtype)) * self.frame_embedding_scale

    def _depth_codes(
        self,
        last_hidden: torch.Tensor,
        c0: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.rvq_depth_decoder
        paired_c0 = c0.repeat(2)
        c0_embedding = self.language_model.model.embed_tokens(
            paired_c0 + AUDIO_CODE_OFFSET
        )
        sequence = [
            decoder.projection(last_hidden).unsqueeze(1),
            decoder.projection(c0_embedding).unsqueeze(1),
        ]
        codes = [paired_c0]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            hidden = decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = decoder.audio_heads[index - 1](hidden).float()
            guided = logits[1:2] + (logits[:1] - logits[1:2]) * AR_CFG_SCALE
            sampled = self._sample_top_k(guided, generator)
            paired = sampled.repeat(2)
            codes.append(paired)
            if index < self.num_codebooks - 1:
                embedding = self.audio_embeddings(
                    paired + (index - 1) * self.audio_vocab_size
                )
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    @torch.inference_mode()
    def generate(
        self,
        text_ids: torch.Tensor,
        *,
        max_frames: int,
        seed: int,
        should_abort: Callable[[], bool],
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        embeddings = self.language_model.model.embed_tokens(
            text_ids.to(device=self.device)
        )
        output = self.language_model.model(
            inputs_embeds=embeddings,
            use_cache=True,
            return_dict=True,
        )
        last_hidden = output.last_hidden_state[:, -1]
        cache = output.past_key_values
        frames = []
        for frame_index in range(max_frames + 1):
            if should_abort():
                raise InterruptedError("MiniMax Music 3 Torch MPS generation aborted")
            logits = self.language_model.lm_head(last_hidden).float()
            narrowed = logits.index_select(1, self.c0_logit_ids)
            guided = narrowed[1:2] + (narrowed[:1] - narrowed[1:2]) * AR_CFG_SCALE
            conditional_threshold = torch.topk(
                narrowed[:1], AR_CFG_TOP_K, dim=-1
            ).values[..., -1, None]
            guided = guided.masked_fill(
                narrowed[:1] < conditional_threshold, -torch.inf
            )
            sampled = self._sample_top_k(guided, generator)
            if int(sampled.item()) == 0:
                break
            codes, depth_hidden = self._depth_codes(
                last_hidden,
                sampled - 1,
                generator,
            )
            if frame_index > 0:
                frames.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frames) >= max_frames:
                    break
            feedback = self._embed_audio_frame(codes)
            output = self.language_model.model(
                inputs_embeds=feedback,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            last_hidden = output.last_hidden_state[:, -1]
            cache = output.past_key_values
        if not frames:
            raise ValueError("MiniMax Music 3 generated zero audio frames")
        return torch.stack(frames, dim=1)


def _build_text_pair(
    prompt: str,
    tokenizer: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > 5_000:
        raise ValueError(
            f"MiniMax Music 3 prompt has {input_ids.shape[1]} tokens; "
            "the maximum is 5000"
        )
    conditional = input_ids.to(device=device)
    unconditional = conditional
    if unconditional.shape[1] > 3:
        middle = torch.full(
            (1, unconditional.shape[1] - 3),
            SPECIAL_TOKEN_IDS["<|audio_cfg|>"],
            dtype=torch.long,
            device=device,
        )
        unconditional = torch.cat(
            [unconditional[:, :1], middle, unconditional[:, -2:]], dim=1
        )
    return torch.cat([conditional, unconditional], dim=0)


class MiniMaxMusic3TorchMpsARScheduler(SimpleScheduler):
    """Generate Torch/MPS frame hiddens and stream CPU chunks."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
    ) -> None:
        model_dir = resolve_torch_mps_directory(model_path, revision)
        self.model = MiniMaxMusic3TorchMpsARModel(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer",
            trust_remote_code=False,
        )
        validate_tokenizer_ids(self.tokenizer)
        self._abort_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        gc.collect()
        torch.mps.empty_cache()
        super().__init__(self._generate, max_concurrency=1)

    def _abort_event(self, request_id: str) -> threading.Event:
        with self._events_lock:
            return self._abort_events.setdefault(request_id, threading.Event())

    def _generate(self, payload: StagePayload) -> StagePayload:
        state = MiniMaxMusic3State.from_dict(payload.data)
        if state.prompt is None:
            raise ValueError("MiniMax Music 3 preprocessing did not build a prompt")
        abort_event = self._abort_event(payload.request_id)
        started = time.perf_counter()
        try:
            text_ids = _build_text_pair(
                state.prompt,
                self.tokenizer,
                device=self.model.device,
            )
            hidden = self.model.generate(
                text_ids,
                max_frames=state.max_audio_frames,
                seed=state.seed,
                should_abort=abort_event.is_set,
            )
            generated_frames = int(hidden.shape[1])
            for window in chunk_windows(generated_frames):
                if abort_event.is_set():
                    raise InterruptedError(
                        "MiniMax Music 3 Torch MPS generation aborted"
                    )
                transport = hidden[:, window.start : window.end].to(
                    device="cpu",
                    dtype=torch.float16,
                )
                self.outbox.put(
                    OutgoingMessage(
                        request_id=payload.request_id,
                        type="stream",
                        data=transport,
                        metadata={
                            "stream": True,
                            "modality": "ttm_hidden",
                            "chunk_idx": window.index,
                            "start_frame": window.start,
                            "end_frame": window.end,
                            "is_final": window.is_last,
                            "seed": state.seed,
                        },
                    )
                )
        finally:
            with self._events_lock:
                self._abort_events.pop(payload.request_id, None)

        state.generated_frames = generated_frames
        state.finish_reason = (
            "length" if generated_frames >= state.max_audio_frames else "stop"
        )
        state.prompt = None
        state.caption = ""
        state.lyrics = ""
        logger.info(
            "MiniMax Music 3 Torch MPS AR done request=%s frames=%d elapsed=%.1fs",
            payload.request_id,
            generated_frames,
            time.perf_counter() - started,
        )
        return store_state(payload, state)

    def abort(self, request_id: str) -> None:
        with self._events_lock:
            event = self._abort_events.get(request_id)
            if event is not None:
                event.set()
        super().abort(request_id)


__all__ = [
    "MiniMaxMusic3TorchMpsARModel",
    "MiniMaxMusic3TorchMpsARScheduler",
    "resolve_torch_mps_directory",
]
