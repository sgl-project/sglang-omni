# SPDX-License-Identifier: Apache-2.0

"""Tokenizer utilities for Fun-CosyVoice3."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class CosyVoice3Tokenizer:

    def __init__(self, token_path: str, skip_special_tokens: bool = True):
        from transformers import AutoTokenizer

        special_tokens = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": _COSYVOICE3_SPECIAL_TOKENS,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer([text], return_tensors="pt")
        return tokens["input_ids"][0].cpu().tolist()

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        return self.tokenizer.batch_decode(
            [tokens], skip_special_tokens=self.skip_special_tokens
        )[0]

    def __call__(self, text: str) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor([ids], dtype=torch.int32)


class SpeechTokenizerV3:

    def __init__(self, model_path: str, device: str = "cpu"):
        import onnxruntime

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=option, providers=providers
        )
        self.device = device

    def extract_speech_token(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> torch.Tensor:
        import whisper

        if sample_rate != 16000:
            raise ValueError(
                f"Speech tokenizer expects 16kHz audio, got {sample_rate}Hz"
            )

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        if audio.shape[1] / sample_rate > 30:
            raise ValueError(
                "Audio longer than 30s is not supported for speech token extraction"
            )

        feat = whisper.log_mel_spectrogram(torch.from_numpy(audio), n_mels=128)
        feat_len = feat.shape[2]
        result = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                self.session.get_inputs()[1].name: np.array([feat_len], dtype=np.int32),
            },
        )[0]
        return torch.tensor([result.flatten().tolist()], dtype=torch.int32)


class SpeakerEncoder:

    def __init__(self, model_path: str, device: str = "cpu"):
        import onnxruntime

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self.device = device

    def extract_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> torch.Tensor:
        import torchaudio.compliance.kaldi as kaldi

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        speech = torch.from_numpy(audio)
        feat = kaldi.fbank(
            speech, num_mel_bins=80, dither=0, sample_frequency=sample_rate
        )
        feat = feat - feat.mean(dim=0, keepdim=True)

        embedding = self.session.run(
            None,
            {self.session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0]
        return torch.tensor([embedding.flatten().tolist()])


def _run_cosyvoice3_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """Run the official CosyVoice3 prompt-mel configuration."""
    from matcha.utils.audio import mel_spectrogram

    return mel_spectrogram(
        waveform,
        n_fft=1920,
        num_mels=80,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )


def extract_prompt_speech_feat(
    audio: np.ndarray,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """Extract Flow conditioning mel features in ``[batch, frames, bins]`` layout.

    CosyVoice3 computes prompt mel features directly from 24 kHz audio using
    the parameters in ``cosyvoice3.yaml``.  The Flow implementation expects
    the same features transposed from Matcha's ``[batch, bins, frames]``
    output, so this helper owns both the configuration and layout conversion.
    """
    if sample_rate != 24000:
        raise ValueError(
            f"CosyVoice3 prompt mel extraction expects 24000Hz audio, got {sample_rate}Hz"
        )

    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim == 1:
        audio_array = audio_array.reshape(1, -1)
    if audio_array.ndim != 2:
        raise ValueError(
            "CosyVoice3 prompt audio must have shape [samples] or [channels, samples]"
        )

    waveform = torch.from_numpy(audio_array)
    mel = _run_cosyvoice3_mel_spectrogram(waveform)
    if mel.ndim != 3 or mel.shape[0] != waveform.shape[0] or mel.shape[1] != 80:
        raise RuntimeError(
            "CosyVoice3 mel extractor returned an unexpected shape: "
            f"{tuple(mel.shape)}"
        )
    return mel.transpose(1, 2).contiguous()


def build_llm_prompt_embeddings(
    *,
    text_token: torch.Tensor,
    text_embed: torch.Tensor,
    prompt_speech_token: torch.Tensor,
    speech_embed: Callable[[torch.Tensor], torch.Tensor],
    embedding: torch.Tensor,
    sos_id: int,
    task_id: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # note: CosyVoice3 inference does NOT inject the speaker embedding into
    # the LLM prompt — speaker identity is carried through prompt_speech_token
    # (the reference audio's speech tokens). The CAMPPlus embedding is reserved
    # for flow/vocoder conditioning (flow_embedding).
    del embedding

    sos_emb = speech_embed(torch.tensor([[sos_id]], device=device)).reshape(1, 1, -1)

    task_emb = speech_embed(torch.tensor([[task_id]], device=device)).reshape(1, 1, -1)

    if prompt_speech_token.shape[1] > 0:
        prompt_speech_emb = speech_embed(prompt_speech_token.to(device=device))
    else:
        prompt_speech_emb = torch.zeros(1, 0, hidden_size, device=device, dtype=dtype)

    lm_input = torch.cat([sos_emb, text_embed, task_emb, prompt_speech_emb], dim=1)
    return lm_input


_COSYVOICE3_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
    "<|endofsystem|>",
    # ARPABET phonemes
    "[AA]",
    "[AA0]",
    "[AA1]",
    "[AA2]",
    "[AE]",
    "[AE0]",
    "[AE1]",
    "[AE2]",
    "[AH]",
    "[AH0]",
    "[AH1]",
    "[AH2]",
    "[AO]",
    "[AO0]",
    "[AO1]",
    "[AO2]",
    "[AW]",
    "[AW0]",
    "[AW1]",
    "[AW2]",
    "[AY]",
    "[AY0]",
    "[AY1]",
    "[AY2]",
    "[B]",
    "[CH]",
    "[D]",
    "[DH]",
    "[EH]",
    "[EH0]",
    "[EH1]",
    "[EH2]",
    "[ER]",
    "[ER0]",
    "[ER1]",
    "[ER2]",
    "[EY]",
    "[EY0]",
    "[EY1]",
    "[EY2]",
    "[F]",
    "[G]",
    "[HH]",
    "[IH]",
    "[IH0]",
    "[IH1]",
    "[IH2]",
    "[IY]",
    "[IY0]",
    "[IY1]",
    "[IY2]",
    "[JH]",
    "[K]",
    "[L]",
    "[M]",
    "[N]",
    "[NG]",
    "[OW]",
    "[OW0]",
    "[OW1]",
    "[OW2]",
    "[OY]",
    "[OY0]",
    "[OY1]",
    "[OY2]",
    "[P]",
    "[R]",
    "[S]",
    "[SH]",
    "[T]",
    "[TH]",
    "[UH]",
    "[UH0]",
    "[UH1]",
    "[UH2]",
    "[UW]",
    "[UW0]",
    "[UW1]",
    "[UW2]",
    "[V]",
    "[W]",
    "[Y]",
    "[Z]",
    "[ZH]",
    # Pinyin phonemes
    "[a]",
    "[ai]",
    "[an]",
    "[ang]",
    "[ao]",
    "[b]",
    "[c]",
    "[ch]",
    "[d]",
    "[e]",
    "[ei]",
    "[en]",
    "[eng]",
    "[f]",
    "[g]",
    "[h]",
    "[i]",
    "[ian]",
    "[in]",
    "[ing]",
    "[iu]",
    "[ià]",
    "[iàn]",
    "[iàng]",
    "[iào]",
    "[iá]",
    "[ián]",
    "[iáng]",
    "[iáo]",
    "[iè]",
    "[ié]",
    "[iòng]",
    "[ióng]",
    "[iù]",
    "[iú]",
    "[iā]",
    "[iān]",
    "[iāng]",
    "[iāo]",
    "[iē]",
    "[iě]",
    "[iōng]",
    "[iū]",
    "[iǎ]",
    "[iǎn]",
    "[iǎng]",
    "[iǎo]",
    "[iǒng]",
    "[iǔ]",
    "[j]",
    "[k]",
    "[l]",
    "[m]",
    "[n]",
    "[o]",
    "[ong]",
    "[ou]",
    "[p]",
    "[q]",
    "[r]",
    "[s]",
    "[sh]",
    "[t]",
    "[u]",
    "[uang]",
    "[ue]",
    "[un]",
    "[uo]",
    "[uà]",
    "[uài]",
    "[uàn]",
    "[uàng]",
    "[uá]",
    "[uái]",
    "[uán]",
    "[uáng]",
    "[uè]",
    "[ué]",
    "[uì]",
    "[uí]",
    "[uò]",
    "[uó]",
    "[uā]",
    "[uāi]",
    "[uān]",
    "[uāng]",
    "[uē]",
    "[uě]",
    "[uī]",
    "[uō]",
    "[uǎ]",
    "[uǎi]",
    "[uǎn]",
    "[uǎng]",
    "[uǐ]",
    "[uǒ]",
    "[vè]",
    "[w]",
    "[x]",
    "[y]",
    "[z]",
    "[zh]",
    "[à]",
    "[ài]",
    "[àn]",
    "[àng]",
    "[ào]",
    "[á]",
    "[ái]",
    "[án]",
    "[áng]",
    "[áo]",
    "[è]",
    "[èi]",
    "[èn]",
    "[èng]",
    "[èr]",
    "[é]",
    "[éi]",
    "[én]",
    "[éng]",
    "[ér]",
    "[ì]",
    "[ìn]",
    "[ìng]",
    "[í]",
    "[ín]",
    "[íng]",
    "[ò]",
    "[òng]",
    "[òu]",
    "[ó]",
    "[óng]",
    "[óu]",
    "[ù]",
    "[ùn]",
    "[ú]",
    "[ún]",
    "[ā]",
    "[āi]",
    "[ān]",
    "[āng]",
    "[āo]",
    "[ē]",
    "[ēi]",
    "[ēn]",
    "[ēng]",
    "[ě]",
    "[ěi]",
    "[ěn]",
    "[ěng]",
    "[ěr]",
    "[ī]",
    "[īn]",
    "[īng]",
    "[ō]",
    "[ōng]",
    "[ōu]",
    "[ū]",
    "[ūn]",
    "[ǎ]",
    "[ǎi]",
    "[ǎn]",
    "[ǎng]",
    "[ǎo]",
    "[ǐ]",
    "[ǐn]",
    "[ǐng]",
    "[ǒ]",
    "[ǒng]",
    "[ǒu]",
    "[ǔ]",
    "[ǔn]",
    "[ǘ]",
    "[ǚ]",
    "[ǜ]",
]
