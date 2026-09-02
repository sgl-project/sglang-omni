# SPDX-License-Identifier: Apache-2.0
"""Talker (TTS) component for MiniCPM-o.

Wraps the checkpoint's remote-code ``MiniCPMTTS`` (``tts.`` prefix): a small
llama backbone that autoregressively emits s3tokenizer codec tokens. The
condition construction mirrors the remote code's non-streaming path
(``_generate_speech_non_streaming``): per thinker token,
``emb_text(token) + l2_normalize(projector_semantic(hidden))``, followed by
the ``text_eos`` and ``audio_bos`` embeddings; speaker embeddings are empty.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.minicpm_o.components.talker_decode import TalkerDecodeLoop
from sglang_omni.models.weight_loader import (
    load_module,
    resolve_dtype,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

MINICPMO_TALKER_FAST_ENV = "SGLANG_OMNI_MINICPMO_TALKER_FAST"


def _fast_decode_enabled() -> bool:
    value = os.environ.get(MINICPMO_TALKER_FAST_ENV, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


class MiniCPMOTalker(nn.Module):
    """Remote-code MiniCPMTTS wrapper: thinker tokens + hidden → codec tokens."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        model_dir = str(resolve_model_path(model_path))
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self._device = torch.device(device)
        self._dtype = torch_dtype

        tts_cls = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMTTS", model_dir
        )
        self._sampling_params_cls = get_class_from_dynamic_module(
            "utils.TTSSamplingParams", model_dir
        )

        # AutoConfig may resolve to a generic config class that leaves
        # tts_config as a raw dict; coerce it through the remote config class.
        tts_config = config.tts_config
        if isinstance(tts_config, dict):
            tts_config_cls = get_class_from_dynamic_module(
                "configuration_minicpmo.MiniCPMTTSConfig", model_dir
            )
            tts_config = tts_config_cls(**tts_config)
        # MiniCPMTTS.__init__ reads generation defaults that transformers v4
        # provided on every PretrainedConfig and v5 removed; backfill the v4
        # values (generation is driven by TTSSamplingParams, not these).
        for attr, default in (
            ("top_p", 1.0),
            ("top_k", 50),
            ("repetition_penalty", 1.0),
        ):
            if not hasattr(tts_config, attr):
                setattr(tts_config, attr, default)
        # Mirrors the remote code's init_tts_module attention pin.
        tts_config.attn_implementation = "eager"
        tts = tts_cls(tts_config, audio_tokenizer=None)
        self.tts = load_module(
            tts, model_dir, prefix=("tts.",), dtype=torch_dtype, device=device
        )
        self.tts.eval()

        cfg = self.tts.config
        self.codec_eos_id = int(cfg.num_audio_tokens) - 1

        self._decode_loop: TalkerDecodeLoop | None = None
        if _fast_decode_enabled():
            gen_logits_fn = get_class_from_dynamic_module(
                "modeling_minicpmo.gen_logits", model_dir
            )
            self._decode_loop = TalkerDecodeLoop(self.tts, gen_logits_fn=gen_logits_fn)

    def _build_condition(
        self, tts_token_ids: torch.Tensor, tts_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Per-token condition + text_eos + audio_bos, shape (1, T+2, hidden)."""
        tokens = tts_token_ids.to(self._device, dtype=torch.long)
        hidden = tts_hidden.to(self._device, dtype=self._dtype)

        llm_embeds = self.tts.emb_text(tokens)
        hidden_embeds = self.tts.projector_semantic(hidden)
        if self.tts.config.normalize_projected_hidden:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        tts_embeds = llm_embeds + hidden_embeds

        boundary = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id, self.tts.audio_bos_token_id],
                device=self._device,
                dtype=torch.long,
            )
        )
        return torch.cat([tts_embeds, boundary], dim=0).unsqueeze(0)

    @torch.inference_mode()
    def forward(
        self,
        *,
        tts_token_ids: torch.Tensor,
        tts_hidden: torch.Tensor,
        min_new_token: int = 50,
        max_new_token: int = 2048,
        sampling_overrides: dict[str, float] | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Generate codec tokens for one utterance.

        Args:
            tts_token_ids: ``(T,)`` thinker token ids inside the
                ``<|tts_bos|>``/``<|tts_eos|>`` span.
            tts_hidden: ``(T, llm_dim)`` last-layer thinker hidden states at
                the same positions.

        Returns:
            ``codec_tokens``: ``(N,)`` s3tokenizer codes, EOS stripped.
        """
        if tts_token_ids.numel() == 0:
            return {"codec_tokens": torch.empty(0, dtype=torch.long)}
        if tts_token_ids.shape[0] != tts_hidden.shape[0]:
            raise ValueError(
                f"tts_token_ids ({tts_token_ids.shape[0]}) and tts_hidden "
                f"({tts_hidden.shape[0]}) must be position-aligned"
            )

        inputs_embeds = self._build_condition(tts_token_ids, tts_hidden)
        sampling_params = self._sampling_params_cls(**(sampling_overrides or {}))
        eos_token = torch.tensor(
            [self.codec_eos_id], dtype=torch.long, device=self._device
        )
        if self._decode_loop is not None:
            new_ids = self._decode_loop.generate(
                inputs_embeds,
                eos_token,
                min_new_token=min_new_token,
                max_new_token=max_new_token,
                sampling_params=sampling_params,
            )
        else:
            outputs = self.tts.generate(
                inputs_embeds=inputs_embeds,
                eos_token=eos_token,
                min_new_token=min_new_token,
                max_new_token=max_new_token,
                show_tqdm=False,
                sampling_params=sampling_params,
            )
            new_ids = outputs.new_ids
        # new_ids: (1, N, num_vq=1); both paths exclude the EOS step.
        codec = new_ids.squeeze(0).squeeze(-1).to("cpu", dtype=torch.long)
        return {"codec_tokens": codec}
