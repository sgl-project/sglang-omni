# SPDX-License-Identifier: Apache-2.0
"""X-ASR uploaded-audio transcription stage."""

import re
from pathlib import Path

import kaldi_native_fbank as fbank
import numpy as np
import sentencepiece as sentencepiece

from sglang_omni.models.xasr.native_model import NativeXASR
from sglang_omni.preprocessing.transcription import (
    police_request_audio,
    resolve_audio_source,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.checkpoint import resolve_checkpoint

SAMPLE_RATE_HZ = 16000
TAIL_SILENCE_SAMPLES = 4800
MIN_PADDED_SAMPLES = 10080


class XASRWorker:
    def __init__(self, model_path: str, device: str):
        model_dir = Path(resolve_checkpoint(model_path))
        self.model = NativeXASR(model_dir, device=device)
        self.tokens = {}
        for line in (model_dir / "tokens.txt").read_text().splitlines():
            piece, token_id = line.rsplit(" ", 1)
            self.tokens[int(token_id)] = piece
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(model_dir / "bpe.model")
        )

    def transcribe_one(self, payload: StagePayload) -> StagePayload:
        result = self.transcribe_batch([payload])[0]
        if isinstance(result, Exception):
            raise result
        else:
            pass
        return result

    def transcribe_batch(self, payloads: list[StagePayload]):
        results = [None] * len(payloads)
        states = {}
        features = {}
        try:
            for index, payload in enumerate(payloads):
                try:
                    waveform = load_audio(
                        police_request_audio(resolve_audio_source(payload)),
                        source_name="xasr",
                        target_sample_rate=SAMPLE_RATE_HZ,
                    )
                    if (
                        waveform.ndim != 1
                        or not waveform.size
                        or not np.isfinite(waveform).all()
                    ):
                        raise ValueError(
                            "X-ASR audio must be finite, non-empty, and mono"
                        )
                    elif len(waveform) > 30 * SAMPLE_RATE_HZ:
                        raise ValueError("X-ASR accepts audio up to 30 seconds")
                    else:
                        pass
                    options = fbank.FbankOptions()
                    options.frame_opts.dither = 0
                    options.frame_opts.snip_edges = False
                    options.frame_opts.window_type = "povey"
                    options.frame_opts.samp_freq = SAMPLE_RATE_HZ
                    options.mel_opts.num_bins = 80
                    options.mel_opts.high_freq = -400
                    extractor = fbank.OnlineFbank(options)
                    silence_samples = max(
                        TAIL_SILENCE_SAMPLES, MIN_PADDED_SAMPLES - len(waveform)
                    )
                    samples = np.pad(waveform, (0, silence_samples))
                    extractor.accept_waveform(SAMPLE_RATE_HZ, samples.tolist())
                    frames = np.stack(
                        [
                            extractor.get_frame(i)
                            for i in range(extractor.num_frames_ready)
                        ]
                    )
                    state = self.model.new_stream()
                    features[index] = [
                        frames[i : i + 61] for i in range(0, len(frames) - 61 + 1, 48)
                    ]
                    states[index] = state
                except Exception as exc:
                    results[index] = exc
            for step in range(
                max((len(chunks) for chunks in features.values()), default=0)
            ):
                active = [index for index in states if step < len(features[index])]
                self.model.step_batch(
                    [states[i] for i in active], [features[i][step] for i in active]
                )
            for index, state in states.items():
                token_ids = state.tokens[2:]
                text = self.tokenizer.decode_pieces(
                    [self.tokens[token] for token in token_ids]
                ).strip()
                text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
                payload = payloads[index]
                results[index] = StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data={"text": text, "token_ids": token_ids, "modality": "text"},
                )
        finally:
            for state in states.values():
                self.model.close(state)
        return results


def create_xasr_executor(
    model_path: str, max_batch_size: int, max_batch_wait_ms: int, gpu_id: int = 0
):
    worker = XASRWorker(model_path, device=f"cuda:{gpu_id}")
    return SimpleScheduler(
        worker.transcribe_one,
        batch_compute_fn=worker.transcribe_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
