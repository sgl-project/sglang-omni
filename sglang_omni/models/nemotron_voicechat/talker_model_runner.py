from __future__ import annotations

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

NUM_ITER = 8


def char_vocab_from_tokenizer(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    token_vocabulary = tokenizer.get_vocab()
    characters = sorted(
        (token for token in token_vocabulary if len(token) == 1),
        key=lambda token: token_vocabulary[token],
    )
    return {character: index for index, character in enumerate(characters)}


class NemotronVoiceChatTalkerModelRunner(ModelRunner):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        speech = self.model.config.nemotron_speech
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            speech["tokenizer_name"],
            bos_token=speech.get("bos_token"),
            eos_token=speech.get("eos_token"),
            pad_token=speech.get("pad_token"),
        )
        self.char_vocab: dict[str, int] = char_vocab_from_tokenizer(self.tokenizer)
        self.char_padding_idx: int = len(self.char_vocab)
        self.initialize_character_lookup()
        # From the checkpoint's names above: the tokenizer's own eos_token_id
        # is <SPECIAL_12>, the text channel's PAD, which means still speaking.
        self.text_pad_id: int = int(self.tokenizer.pad_token_id)
        self.text_eos_id: int = int(self.tokenizer.eos_token_id)
        self.exponent: float = float(speech["tts_config"]["exponent"])
        self.top_p: float = float(speech["inference_top_p_or_k"])
        self.noise_scale: float = float(speech["inference_noise_scale"])
        self.force_silence: bool = bool(speech["inference_force_speech_silence_on_eos"])
        self.speech_pad_id: int = int(speech["codec_config"]["codebook_size"])
        self.warmup_rows: torch.Tensor | None = None

    def fusion_device(self) -> torch.device:
        return self.model.fusion_buffer.device

    def initialize_character_lookup(self) -> None:
        self.token_id_count: int = (
            max(self.tokenizer.get_vocab().values(), default=-1) + 1
        )
        character_sequences = [
            [
                self.char_vocab[character]
                for character in (self.tokenizer.convert_ids_to_tokens(token_id) or "")
                if character in self.char_vocab
            ]
            or [self.char_padding_idx]
            for token_id in range(self.token_id_count)
        ]
        self.character_lengths_cpu: list[int] = [
            len(sequence) for sequence in character_sequences
        ]
        character_ids_cpu = torch.full(
            (len(character_sequences), max(self.character_lengths_cpu, default=1)),
            self.char_padding_idx,
            dtype=torch.long,
            device="cpu",
        )
        for row, sequence in enumerate(character_sequences):
            character_ids_cpu[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device="cpu"
            )
        device = self.fusion_device()
        self.token_ids_by_token: torch.Tensor = torch.arange(
            self.token_id_count, dtype=torch.long, device=device
        )
        self.character_ids_by_token: torch.Tensor = character_ids_cpu.to(device)
        self.character_lengths_by_token: torch.Tensor = torch.tensor(
            self.character_lengths_cpu, dtype=torch.long, device=device
        )

    def char_batch(
        self, token_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(token_ids) == 1 and 0 <= token_ids[0] < self.token_id_count:
            token_id = token_ids[0]
            character_width = self.character_lengths_cpu[token_id]
            return (
                self.token_ids_by_token[token_id : token_id + 1],
                self.character_ids_by_token[token_id : token_id + 1, :character_width],
                self.character_lengths_by_token[token_id : token_id + 1],
            )
        elif all(0 <= token_id < self.token_id_count for token_id in token_ids):
            character_width = max(
                self.character_lengths_cpu[token_id] for token_id in token_ids
            )
            batch_token_ids = torch.tensor(token_ids, device=self.fusion_device())
            return (
                batch_token_ids,
                self.character_ids_by_token[:, :character_width].index_select(
                    0, batch_token_ids
                ),
                self.character_lengths_by_token.index_select(0, batch_token_ids),
            )
        else:
            device = self.fusion_device()
            character_sequences = [
                [
                    self.char_vocab[character]
                    for character in (
                        self.tokenizer.convert_ids_to_tokens(token_id) or ""
                    )
                    if character in self.char_vocab
                ]
                or [self.char_padding_idx]
                for token_id in token_ids
            ]
            character_width = max(len(sequence) for sequence in character_sequences)
            character_ids = torch.full(
                (len(character_sequences), character_width),
                self.char_padding_idx,
                dtype=torch.long,
                device=device,
            )
            for row, sequence in enumerate(character_sequences):
                character_ids[row, : len(sequence)] = torch.tensor(
                    sequence, device=device
                )
            character_lengths = torch.tensor(
                [len(sequence) for sequence in character_sequences], device=device
            )
            return (
                torch.tensor(token_ids, device=device),
                character_ids,
                character_lengths,
            )

    def warmup(self):
        if self.warmup_rows is None:
            model = self.model
            talker = model.talker
            frames = model.audio_prompt_latent.shape[0]
            # The prompt's codes are consumed shifted by one, so the frame the
            # model starts speaking from carries the silence behind it.
            audio = torch.cat(
                [
                    model.audio_prompt_latent[:-1],
                    talker.embed_codes(model.codec_silence_tokens.unsqueeze(0))
                    + talker.bos_emb,
                ]
            )
            ids, chars, lengths = self.char_batch(
                [self.text_pad_id] * (frames - 1) + [self.text_eos_id]
            )
            mask = torch.zeros(frames, dtype=torch.bool, device=self.fusion_device())
            mask[frames - 2 :] = True
            text = talker.embed_subword(ids, chars, lengths, mask)
            self.warmup_rows = talker.gated_fusion_audio_text(audio, text)
        else:
            pass
        return self.warmup_rows

    def pad_codes(self) -> torch.Tensor:
        return torch.full(
            (1, self.model.talker.num_quantizers),
            self.speech_pad_id,
            dtype=torch.long,
            device=self.fusion_device(),
        )

    def step_row(self, prev_codes: torch.Tensor, token: int) -> torch.Tensor:
        """The one row a frame forwards: last frame's codes, this frame's text."""
        talker = self.model.talker
        if self.force_silence and token == self.text_eos_id:
            prev_codes = self.model.codec_silence_tokens.unsqueeze(0)
        else:
            pass
        audio_1D = talker.embed_codes(prev_codes)
        ids, chars, lengths = self.char_batch([token])
        text_1D = talker.embed_subword(ids, chars, lengths)
        return talker.gated_fusion_audio_text(audio_1D, text_1D)

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        del schedule_batch
        rows = [self.warmup() for _ in requests]
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                # Back to the backbone's dtype: the rows were built in float32.
                input_embeds=torch.cat(rows, dim=0).to(self.model.fusion_buffer.dtype),
                input_embeds_are_projected=True,
            ),
        )

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        del result, forward_batch, schedule_batch
        for request in requests:
            inputs = request.data.talker_model_inputs
            inputs["codes_rows"] = []
            inputs["prev_codes"] = self.pad_codes()

    @staticmethod
    def is_terminating(req) -> bool:
        return req.to_finish is not None or req.finished()

    def is_decode_batch_ready(self, schedule_batch) -> bool:
        # An aborted request still needs one forward: sglang turns to_finish
        # into finished_reason during the step, and only then releases the
        # slot and its KV. Holding the batch back until a text token arrives
        # would strand a request whose thinker has already stopped.
        return all(
            len(req.omni_data.pending_text_queue) > 0
            or self.is_terminating(
                req
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            for req in schedule_batch.reqs
        )

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ) -> None:
        del forward_batch, is_lookahead
        model = self.model
        rows = []
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            queue = data.pending_text_queue
            if queue:
                token = queue.popleft()
            elif self.is_terminating(req):
                # Carries the forward that retires the request; its codes go
                # nowhere.
                token = self.text_pad_id
            else:
                raise RuntimeError(
                    f"talker request {request.request_id} reached decode with no "
                    "text token and no finish reason"
                )
            rows.append(
                self.step_row(
                    data.talker_model_inputs["prev_codes"],
                    token,
                )
            )
        batch = len(rows)
        model.fusion_buffer[:batch] = torch.cat(rows, dim=0)
        model.fusion_mask[:batch] = True

    def generate_codes(self, index: int) -> torch.Tensor:
        model = self.model
        return model.talker.generate_codes(
            model.hidden_out[index : index + 1].float(),
            model.mog_head,
            num_iter=NUM_ITER,
            exponent=self.exponent,
            top_p=self.top_p,
            noise_scale=self.noise_scale,
        )

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        del result, forward_batch, schedule_batch
        for index, request in enumerate(requests):
            inputs = request.data.talker_model_inputs
            codes = self.generate_codes(index)
            inputs["prev_codes"] = codes
            inputs["codes_rows"].append(codes[0])
            inputs["stream_chunk"] = codes.cpu()
