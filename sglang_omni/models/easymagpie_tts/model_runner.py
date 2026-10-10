# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie request lifecycle on top of the shared SGLang model runner.

Each AR step conditions on three delayed streams: the text token at the
request offset, the previous phoneme row once the phoneme delay has passed,
and the previous acoustic frame once the speech delay has passed.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.easymagpie_tts.request_builders import (
    EasyMagpieSGLangRequestData,
)


class EasyMagpieTTSModelRunner(ModelRunner):
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def build_prompt_embeds(
        self, data: EasyMagpieSGLangRequestData, device: torch.device
    ) -> torch.Tensor:
        """Speaker rows, context text, then the text lead-in with phoneme BOS.

        The lead-in rows are every step whose conditioning is known before
        any prediction exists, so they are folded into prefill.
        """
        state = data.state
        heads = self.model.heads
        config = self.model.tts_config
        dtype = self.model_dtype()
        rows = []
        if state.speaker_embedding is not None:
            rows.append(state.speaker_embedding.to(device=device, dtype=dtype))
        else:
            pass
        if state.context_token_ids:
            context = torch.tensor(state.context_token_ids, device=device)
            rows.append(heads.text_embedding(context).to(dtype))
        else:
            pass
        lead_in = torch.zeros(
            (state.text_prefill_num, config.embedding_dim), device=device, dtype=dtype
        )
        lead_in_ids = state.text_token_ids[: state.text_prefill_num]
        if lead_in_ids:
            ids = torch.tensor(lead_in_ids, device=device)
            lead_in[: len(lead_in_ids)] = heads.text_embedding(ids).to(dtype)
        else:
            pass
        bos = torch.full(
            (1, config.phoneme_stacking_factor),
            config.phoneme_bos_id,
            device=device,
            dtype=torch.long,
        )
        lead_in[state.phoneme_delay] += heads.embed_phonemes(bos)[0].to(dtype)
        rows.append(lead_in)
        return torch.cat(rows, dim=0)

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        device = forward_batch.input_ids.device
        pieces = []
        for request in requests:
            data = request.data
            prompt = self.build_prompt_embeds(data, device)
            start = len(data.req.prefix_indices)
            length = int(data.req.extend_range.length)
            piece = prompt[start : start + length]
            if piece.shape[0] != length:
                raise RuntimeError(
                    f"EasyMagpie prefill has {piece.shape[0]} rows, "
                    f"the scheduler expects {length}"
                )
            else:
                pass
            pieces.append(piece)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(pieces, dim=0),
                input_embeds_are_projected=True,
            ),
        )
        return None

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch, schedule_batch
        phonemes = self.model.last_phoneme_tokens[: len(requests)].clone()
        is_eos = (phonemes == self.model.tts_config.phoneme_eos_id).any(dim=1)
        for request, row, row_is_eos in zip(requests, phonemes, is_eos.tolist()):
            request.data.last_phoneme_tokens = row
            request.data.last_phoneme_is_eos = row_is_eos

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        config = self.model.tts_config
        device = forward_batch.input_ids.device
        channels = config.num_stacked_codebooks
        phoneme_bos = torch.full(
            (config.phoneme_stacking_factor,), config.phoneme_bos_id, device=device
        )
        phoneme_zero = torch.zeros_like(phoneme_bos)
        audio_bos = torch.full((channels,), config.audio_bos_id, device=device)
        audio_zero = torch.zeros_like(audio_bos)
        text_tokens, text_valid = [], []
        phoneme_rows, phoneme_valid = [], []
        audio_rows, audio_valid = [], []
        temperatures, top_ks, seeds, positions = [], [], [], []
        for request in requests:
            data = request.data
            state = data.state
            step = data.decode_offset
            if step < len(state.text_token_ids):
                text_tokens.append(state.text_token_ids[step])
                text_valid.append(True)
            else:
                text_tokens.append(0)
                text_valid.append(False)

            if not data.phoneme_ended and step >= state.phoneme_delay:
                if step == state.phoneme_delay or data.last_phoneme_tokens is None:
                    phoneme_rows.append(phoneme_bos)
                else:
                    phoneme_rows.append(data.last_phoneme_tokens.to(device))
                phoneme_valid.append(True)
                # A predicted phoneme EOS is fed once, then the channel closes.
                data.phoneme_ended = data.last_phoneme_is_eos
            else:
                phoneme_rows.append(phoneme_zero)
                phoneme_valid.append(False)

            if step >= state.speech_delay:
                if step == state.speech_delay or data.last_audio_codes is None:
                    audio_rows.append(audio_bos)
                else:
                    audio_rows.append(data.last_audio_codes.to(device))
                audio_valid.append(True)
            else:
                audio_rows.append(audio_zero)
                audio_valid.append(False)

            temperatures.append(state.temperature)
            top_ks.append(state.top_k)
            seeds.append(data.sampling_seed)
            # Prefill consumes sampling position zero; decode positions stay
            # request-local regardless of which batch a request lands in.
            positions.append((step + 1) * channels)
            data.decode_offset = step + 1

        audio_valid_tensor = torch.tensor(audio_valid, device=device)
        conditioning = self.model.compose_conditioning(
            text_tokens=torch.tensor(text_tokens, device=device),
            text_valid=torch.tensor(text_valid, device=device),
            phoneme_tokens=torch.stack(phoneme_rows),
            phoneme_valid=torch.tensor(phoneme_valid, device=device),
            previous_audio_codes=torch.stack(audio_rows),
            audio_valid=audio_valid_tensor,
        )
        self.model.decode_buffers.stage(
            conditioning=conditioning.to(self.model_dtype()),
            audio_valid=audio_valid_tensor,
            temperatures=torch.tensor(temperatures, dtype=torch.float32),
            top_ks=torch.tensor(top_ks),
            seeds=torch.tensor(seeds),
            positions=torch.tensor(positions),
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch, schedule_batch
        rows = len(requests)
        config = self.model.tts_config
        buffers = self.model.decode_buffers
        codes = buffers.codes[:rows].clone()
        phonemes = buffers.phonemes[:rows].clone()
        phoneme_eos = (phonemes == config.phoneme_eos_id).any(dim=1).tolist()
        audio_eos = buffers.eos[:rows].tolist()
        for row, request in enumerate(requests):
            data = request.data
            data.last_audio_codes = codes[row]
            data.last_phoneme_tokens = phonemes[row]
            data.last_phoneme_is_eos = phoneme_eos[row]
            emits_audio = data.decode_offset > data.state.speech_delay
            if emits_audio and not audio_eos[row]:
                data.output_codes.append(codes[row].cpu())
            else:
                pass


__all__ = ["EasyMagpieTTSModelRunner"]
