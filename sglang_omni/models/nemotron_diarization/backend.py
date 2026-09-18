# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint loading and inference; NeMo is only a test reference."""

from __future__ import annotations

import io
import math
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

from sglang_omni.client.types import DiarizationResult, DiarizationSegment

CHECKPOINT_FILENAME = "Nemotron-3-Diarization-preview.nemo"
SAMPLE_RATE = 16000
_PROFILES = {
    # Cache, FIFO, chunk, right context and update period, in 80 ms frames.
    "offline": (264, 40, 340, 40, 300),
    "low_latency": (264, 264, 9, 4, 222),
}

# Native inference implements this published architecture only. In particular,
# shape-compatible changes to normalization, attention or cache scoring must not
# silently load with different semantics. Training-only settings are irrelevant.
_REQUIRED_CONFIG = {
    "preprocessor": {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "normalize": "NA",
        "window_size": 0.025,
        "window_stride": 0.01,
        "sample_rate": 16000,
        "window": "hann",
        "features": 128,
        "n_fft": 512,
        "frame_splicing": 1,
    },
    "encoder": {
        "_target_": "nemo.collections.asr.modules.TransformerEncoder",
        "feat_in": 128,
        "feat_out": -1,
        "n_layers": 31,
        "d_model": 512,
        "n_heads": 8,
        "subsampling": "feature_stacking",
        "subsampling_factor": 8,
        "ff_expansion": 4.0,
        "self_attention_model": "rope",
        "xscaling": False,
        "qkv_bias": False,
        "qk_norm": False,
        "pre_block_norm": True,
        "attn_mode": "full",
    },
    "sortformer_modules": {
        "_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
        "num_spks": 8,
        "fc_d_model": 512,
        "tf_d_model": 192,
        "chunk_left_context": 0,
        "spkcache_sil_frames_per_spk": 1,
        "pred_score_threshold": 0.25,
        "max_index": 99999,
        "scores_boost_latest": 0.05,
        "strong_boost_rate": 0.75,
        "weak_boost_rate": 1.5,
        "min_pos_scores_rate": 0.5,
        "use_learnable_sil_emb": True,
    },
}
_DEFAULT_CONFIG = {
    "preprocessor": {
        "preemph": 0.97,
        "log": True,
        "log_zero_guard_type": "add",
        "log_zero_guard_value": 2**-24,
        "mag_power": 2.0,
        "exact_pad": False,
        "pad_value": 0.0,
        "use_grads": False,
    },
    "encoder": {"rope_base": 10000.0, "rotary_fraction": 1.0, "causal_tail_len": 0},
    "sortformer_modules": {"use_activity_head": False},
}


def resolve_nemo_checkpoint(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file():
        if path.suffix != ".nemo":
            raise ValueError("Nemotron diarization requires a .nemo checkpoint")
        return path
    if path.is_dir():
        checkpoint = path / CHECKPOINT_FILENAME
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        return checkpoint
    if (
        path.is_absolute()
        or model_path.startswith((".", "~"))
        or path.suffix == ".nemo"
    ):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    from huggingface_hub import hf_hub_download

    repo_id, _, revision = model_path.partition("@")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=CHECKPOINT_FILENAME,
            revision=revision or None,
        )
    )


def validate_checkpoint(path: Path) -> None:
    # Inspect configuration without instantiating any of its NeMo targets.
    with tarfile.open(path) as archive:
        configs = [
            member
            for member in archive.getmembers()
            if member.name in {"model_config.yaml", "./model_config.yaml"}
        ]
        if len(configs) != 1 or not configs[0].isfile() or configs[0].size > 1024**2:
            raise ValueError("Expected one model_config.yaml in the .nemo archive")
        with archive.extractfile(configs[0]) as handle:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Invalid Nemotron diarization configuration")
    if not (
        all(isinstance(config.get(section), dict) for section in _REQUIRED_CONFIG)
        and config.get("target")
        == "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel"
        and config.get("sample_rate") == SAMPLE_RATE
        and config.get("high_resolution") is True
        and config.get("output_subsampling_factor") == 1
        and config.get("streaming_mode") is True
        and config.get("max_num_of_spks") == 8
        and config.get("transformer_encoder") is None
        and not config.get("activity_weight", 0)
    ):
        raise ValueError(
            "Checkpoint is not the supported Nemotron 3 diarization layout"
        )
    for section, expected in _REQUIRED_CONFIG.items():
        if any(config[section].get(key) != value for key, value in expected.items()):
            raise ValueError(f"Checkpoint is not the supported {section} configuration")
    for section, expected in _DEFAULT_CONFIG.items():
        if any(
            config[section].get(key, value) != value for key, value in expected.items()
        ):
            raise ValueError(f"Checkpoint is not the supported {section} configuration")


def parse_segments(lines: list[str], duration: float) -> DiarizationResult:
    """Preserve overlapping speakers and clip frame-rounded ends to the waveform."""
    segments = []
    for line in lines:
        try:
            start_text, end_text, speaker = line.split()
            start, end = float(start_text), float(end_text)
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(f"Invalid diarization segment: {line!r}") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or re.fullmatch(r"speaker_[0-7]", speaker) is None
            or end > duration + 0.01
        ):
            raise RuntimeError(f"Invalid diarization segment: {line!r}")
        end = min(end, duration)
        if start < end:
            segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return DiarizationResult(duration=duration, segments=segments)


class NemotronDiarizer:
    def __init__(
        self, model_path: str, *, device: torch.device, profile: str = "offline"
    ):
        if profile not in _PROFILES:
            raise ValueError(
                f"Unknown diarization profile {profile!r}; use {list(_PROFILES)}"
            )
        if torch.device(device).type != "cuda":
            raise ValueError("Nemotron 3 diarization requires an NVIDIA CUDA device")
        from sglang_omni.models.nemotron_diarization.model import (
            NemotronDiarizationModel,
        )

        checkpoint = resolve_nemo_checkpoint(model_path)
        validate_checkpoint(checkpoint)
        self.device = torch.device(device)
        self.model = NemotronDiarizationModel(profile=_PROFILES[profile])
        self.model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True)
        self.model.to(device=self.device).eval()

    @torch.inference_mode()
    def probabilities(self, waveform: np.ndarray) -> torch.Tensor:
        signal = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        return self.model(signal.unsqueeze(0))

    @torch.inference_mode()
    def diarize(self, waveform: np.ndarray) -> DiarizationResult:
        predictions = self.probabilities(waveform)[0].cpu().numpy()
        return probabilities_to_segments(
            predictions, duration=len(waveform) / SAMPLE_RATE
        )


def load_checkpoint_weights(path: Path) -> dict[str, torch.Tensor]:
    """Read tensors without extracting paths or executing pickle globals."""
    with tarfile.open(path) as archive:
        members = [
            m
            for m in archive.getmembers()
            if m.name in {"model_weights.ckpt", "./model_weights.ckpt"}
        ]
        if len(members) != 1 or not members[0].isfile() or members[0].size > 1024**3:
            raise ValueError("Expected one model_weights.ckpt in the .nemo archive")
        with archive.extractfile(members[0]) as handle:
            weights = torch.load(
                io.BytesIO(handle.read()), map_location="cpu", weights_only=True
            )
    if not isinstance(weights, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()
    ):
        raise ValueError("Expected a tensor state dictionary in model_weights.ckpt")
    return weights


def probabilities_to_segments(
    predictions: np.ndarray, *, duration: float
) -> DiarizationResult:
    """NeMo's default 0.5 hysteresis at 10 ms, preserving equal-threshold state."""
    if (
        predictions.ndim != 2
        or predictions.shape[1] != 8
        or not np.isfinite(predictions).all()
    ):
        raise RuntimeError("Invalid diarization probabilities")
    lines = []
    for speaker in range(8):
        values = predictions[:, speaker]
        positions = np.arange(1, len(values) + 1)
        events = np.where(values != 0.5, positions, 0)
        last_event = np.maximum.accumulate(events)
        active = np.concatenate(([False], values > 0.5))[last_event]
        changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        lines.extend(
            f"{start / 100:.3f} {end / 100:.3f} speaker_{speaker}"
            for start, end in zip(starts, ends)
        )
    return parse_segments(lines, duration)
