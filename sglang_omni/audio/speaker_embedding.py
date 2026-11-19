# SPDX-License-Identifier: Apache-2.0
"""Speaker embeddings of 16 kHz speech from a 3D-Speaker CAM++ ONNX model."""

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download

SPEAKER_EMBEDDING_SAMPLE_RATE_HZ = 16000
SPEAKER_EMBEDDING_FILENAME = (
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
SPEAKER_EMBEDDING_SHA256 = (
    "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
)
MEL_BINS = 80


@dataclass(frozen=True)
class SpeakerEmbeddingModel:
    session: onnxruntime.InferenceSession

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """The unit-length embedding of mono float32 samples in [-1, 1]."""
        mel_features = kaldi.fbank(
            torch.from_numpy(samples).unsqueeze(0),
            num_mel_bins=MEL_BINS,
            dither=0,
            sample_frequency=SPEAKER_EMBEDDING_SAMPLE_RATE_HZ,
        )
        mel_features = mel_features - mel_features.mean(dim=0, keepdim=True)
        embedding = self.session.run(
            None,
            {self.session.get_inputs()[0].name: mel_features.unsqueeze(0).numpy()},
        )[0].reshape(-1)
        return embedding / np.linalg.norm(embedding)


@functools.cache
def load_speaker_embedding_model(model_path: str) -> SpeakerEmbeddingModel:
    """The model at a local file or directory, or in a Hugging Face repository; one per process."""
    path = Path(model_path).expanduser()
    if not path.exists() and model_path.count("/") == 1:
        path = Path(hf_hub_download(model_path, SPEAKER_EMBEDDING_FILENAME))
    elif path.is_dir():
        path = path / SPEAKER_EMBEDDING_FILENAME
    else:
        pass
    if not path.is_file():
        raise FileNotFoundError(f"speaker embedding model not found at {path}")
    else:
        pass
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SPEAKER_EMBEDDING_SHA256:
        raise ValueError(
            f"speaker embedding checksum mismatch for {path}: expected {SPEAKER_EMBEDDING_SHA256}, got {digest}"
        )
    else:
        pass
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return SpeakerEmbeddingModel(
        onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    )
