# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint loading and speaker interval decoding."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

from sglang_omni.client.types import DiarizationResult, DiarizationSegment
from sglang_omni.models.nemotron_diarization.model import NemotronDiarizationModel

CHECKPOINT_FILENAME = "Nemotron-3-Diarization.nemo"
SAMPLE_RATE = 16000
DiarizationProfile = Literal["offline", "low_latency"]
PROFILES: dict[DiarizationProfile, tuple[int, int, int, int, int]] = {
    # note (Ryan Stewart): Cache, FIFO, chunk, right context, and update period in 80 ms frames.
    "offline": (264, 40, 340, 40, 300),
    "low_latency": (264, 264, 9, 4, 222),
}

# note (Ryan Stewart): Shape-compatible variants must not silently load with different semantics.
REQUIRED_CONFIG = {
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
DEFAULT_CONFIG = {
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
}


def resolve_nemo_checkpoint(model_path: str) -> Path:
    """Return a local archive, downloading only the archive from a hub repository."""
    path = Path(model_path).expanduser()
    if path.is_dir():
        path = path / CHECKPOINT_FILENAME
    elif (
        path.suffix == ".nemo"
        or path.is_absolute()
        or model_path.startswith((".", "~"))
    ):
        pass
    else:
        repo_id, _, revision = model_path.partition("@")
        path = Path(
            hf_hub_download(
                repo_id=repo_id, filename=CHECKPOINT_FILENAME, revision=revision or None
            )
        )
    if path.suffix != ".nemo" or not path.is_file():
        raise FileNotFoundError(f"Missing .nemo checkpoint: {path}")
    else:
        pass
    return path


def validate_checkpoint(path: Path) -> None:
    """Inspect the configuration without instantiating any of its NeMo targets."""
    with tarfile.open(path) as archive:
        configs = [
            member
            for member in archive.getmembers()
            if member.name in {"model_config.yaml", "./model_config.yaml"}
        ]
        if len(configs) != 1 or not configs[0].isfile() or configs[0].size > 1024**2:
            raise ValueError("Expected one model_config.yaml in the .nemo archive")
        else:
            pass
        with archive.extractfile(configs[0]) as handle:
            config = yaml.safe_load(handle)
    if not (
        isinstance(config, dict)
        and all(isinstance(config.get(section), dict) for section in REQUIRED_CONFIG)
        and config.get("target")
        == "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel"
        and config.get("sample_rate") == SAMPLE_RATE
        and config.get("high_resolution") is True
        and config.get("output_subsampling_factor") == 1
        and config.get("streaming_mode") is True
        and config.get("max_num_of_spks") == 8
        and config.get("transformer_encoder") is None
    ):
        raise ValueError(
            "Checkpoint is not the supported Nemotron 3 diarization layout"
        )
    else:
        pass
    for section in REQUIRED_CONFIG:
        values = config[section]
        if any(
            values.get(field_name) != value
            for field_name, value in REQUIRED_CONFIG[section].items()
        ) or any(
            values.get(field_name, value) != value
            for field_name, value in DEFAULT_CONFIG.get(section, {}).items()
        ):
            raise ValueError(f"Checkpoint is not the supported {section} configuration")
        else:
            pass


def load_checkpoint_weights(path: Path) -> dict[str, torch.Tensor]:
    """Read tensors without extracting paths or executing pickle globals."""
    with open(path, "rb") as raw, tarfile.open(fileobj=raw) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name in {"model_weights.ckpt", "./model_weights.ckpt"}
        ]
        if len(members) != 1 or not members[0].isfile() or members[0].size > 1024**3:
            raise ValueError("Expected one model_weights.ckpt in the .nemo archive")
        else:
            pass
        with archive.extractfile(members[0]) as handle:
            # note (Richard Wang): Buffer compressed streams to avoid repeated decompression.
            source = handle if archive.fileobj is raw else io.BytesIO(handle.read())
            weights = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in weights.items()
    ):
        raise ValueError("Expected a tensor state dictionary in model_weights.ckpt")
    else:
        pass
    return weights


def load_diarization_model(
    model_path: str, *, device: torch.device, profile: DiarizationProfile
) -> NemotronDiarizationModel:
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown diarization profile {profile!r}, use one of {list(PROFILES)}"
        )
    elif device.type != "cuda":
        raise ValueError("Nemotron 3 diarization requires an NVIDIA CUDA device")
    else:
        pass
    checkpoint = resolve_nemo_checkpoint(model_path)
    validate_checkpoint(checkpoint)
    with torch.device("meta"):
        model = NemotronDiarizationModel(profile=PROFILES[profile])
    parameters = model.state_dict()
    # note (Richard Wang): Load BF16 or FP32 archives as FP32, minus the unused activity head.
    weights = {
        name: (
            tensor.to(parameters[name].dtype).contiguous()
            if name in parameters
            else tensor
        )
        for name, tensor in load_checkpoint_weights(checkpoint).items()
        if ".activity_head." not in name
    }
    model.load_state_dict(weights, strict=True, assign=True)
    return model.to(device).eval()


def probabilities_to_segments(
    predictions: np.ndarray,
    *,
    duration_s: float,
    start_frame: int = 0,
    active: np.ndarray | None = None,
) -> DiarizationResult:
    """NeMo's default 0.5 hysteresis at 10 ms, preserving equal-threshold state."""
    if (
        predictions.ndim != 2
        or predictions.shape[1] != 8
        or not np.isfinite(predictions).all()
    ):
        raise RuntimeError("Invalid diarization probabilities")
    else:
        pass
    active = np.zeros(8, dtype=bool) if active is None else active
    positions = np.arange(1, len(predictions) + 1)
    segments = []
    for speaker in range(8):
        values = predictions[:, speaker]
        events = np.where(values != 0.5, positions, 0)
        last_event = np.maximum.accumulate(events)
        activity = np.concatenate(([active[speaker]], values > 0.5))[last_event]
        active[speaker] = activity[-1] if len(activity) else active[speaker]
        changes = np.diff(np.pad(activity.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for start_frame_offset, end_frame_offset in zip(starts, ends):
            start_s = (start_frame + int(start_frame_offset)) / 100
            end_s = (start_frame + int(end_frame_offset)) / 100
            if end_s > duration_s + 0.01:
                raise RuntimeError("Diarization segment extends past the audio")
            elif start_s < min(end_s, duration_s):
                segments.append(
                    DiarizationSegment(
                        start=start_s,
                        end=min(end_s, duration_s),
                        speaker=f"speaker_{speaker}",
                    )
                )
            else:
                pass
    segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker))
    return DiarizationResult(duration=duration_s, segments=segments)
