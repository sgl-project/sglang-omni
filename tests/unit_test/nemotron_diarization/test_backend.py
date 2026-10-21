# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection, archive safety, masks and frame-rounded intervals."""

import importlib
import io
import tarfile
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch
import yaml
from torch._inductor import config as inductor_config

from sglang_omni.models.nemotron_diarization import model
from sglang_omni.models.nemotron_diarization.backend import (
    REQUIRED_CONFIG,
    load_checkpoint_weights,
    probabilities_to_segments,
    resolve_nemo_checkpoint,
    validate_checkpoint,
)
from sglang_omni.models.nemotron_diarization.model import TransformerEncoder


def write_archive(
    path: Path, name: str, payload: bytes, *, copies: int = 1, symlink: bool = False
) -> Path:
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(name)
        if symlink:
            member.type, member.linkname = tarfile.SYMTYPE, "/outside/archive"
        else:
            member.size = len(payload)
        for _ in range(copies):
            archive.addfile(member, io.BytesIO(payload))
    return path


def test_segments_preserve_overlap_and_clip_only_the_partial_last_frame() -> None:
    predictions = np.zeros((251, 8))
    predictions[:200, 0] = 1
    predictions[100:, 7] = 1
    result = probabilities_to_segments(predictions, duration_s=2.504)
    assert msgspec.to_builtins(result) == {
        "duration": 2.504,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.504, "speaker": "speaker_7"},
        ],
    }
    with pytest.raises(RuntimeError, match="segment extends past the audio"):
        probabilities_to_segments(np.ones((300, 8)), duration_s=2.0)


def test_threshold_ties_retain_previous_activity_and_preserve_overlap() -> None:
    predictions = np.zeros((7, 8), dtype=np.float32)
    predictions[:, 0] = [0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    predictions[2:4, 7] = 0.9
    result = probabilities_to_segments(predictions, duration_s=0.07)
    assert [
        (segment.start, segment.end, segment.speaker) for segment in result.segments
    ] == [
        (0.01, 0.03, "speaker_0"),
        (0.02, 0.04, "speaker_7"),
        (0.05, 0.07, "speaker_0"),
    ]


@pytest.mark.parametrize(
    "overrides, accepted",
    [
        ({}, True),
        ({"high_resolution": False}, False),
        ({"preprocessor": {"normalize": "per_feature"}}, False),
        ({"preprocessor": {"preemph": 0.0}}, False),
        ({"transformer_encoder": {}}, False),
    ],
)
def test_checkpoint_validator_accepts_only_the_released_layout(
    tmp_path: Path,
    overrides: dict[str, bool | dict[str, str | float]],
    accepted: bool,
) -> None:
    config = {
        "target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
        "sample_rate": 16000,
        "high_resolution": True,
        "output_subsampling_factor": 1,
        "streaming_mode": True,
        "max_num_of_spks": 8,
        # note (Richard Wang): The released checkpoint trains an auxiliary activity head.
        "activity_weight": 0.5,
        **{section: dict(values) for section, values in REQUIRED_CONFIG.items()},
    }
    for field_name, value in overrides.items():
        config[field_name] = (
            {**config[field_name], **value} if field_name in REQUIRED_CONFIG else value
        )
    path = write_archive(
        tmp_path / "test.nemo", "model_config.yaml", yaml.safe_dump(config).encode()
    )
    if accepted:
        validate_checkpoint(path)
    else:
        with pytest.raises(ValueError, match="not the supported"):
            validate_checkpoint(path)


@pytest.mark.parametrize("name", ["unrelated.nemo", "missing.nemo"])
def test_local_paths_never_select_another_archive_or_the_hub(
    tmp_path: Path, name: str
) -> None:
    (tmp_path / "unrelated.nemo").touch()
    target = tmp_path if name == "unrelated.nemo" else tmp_path / name
    with pytest.raises(FileNotFoundError, match="Missing .nemo checkpoint"):
        resolve_nemo_checkpoint(str(target))


@pytest.mark.parametrize("variant", ["duplicate", "symlink", "non_tensor"])
def test_weights_must_be_one_regular_tensor_state_dict(
    tmp_path: Path, variant: str
) -> None:
    serialized = io.BytesIO()
    state = {"weight": torch.zeros(1)}
    torch.save(
        {**state, "config": "text"} if variant == "non_tensor" else state, serialized
    )
    path = write_archive(
        tmp_path / "bad.nemo",
        "model_weights.ckpt",
        serialized.getvalue(),
        copies=2 if variant == "duplicate" else 1,
        symlink=variant == "symlink",
    )
    with pytest.raises(ValueError, match="model_weights|tensor state dictionary"):
        load_checkpoint_weights(path)


def test_reused_attention_masks_preserve_each_valid_length() -> None:
    encoder = TransformerEncoder(max_frames=9)
    positions = torch.arange(9)
    for valid_length in [9, 3, 8, 3, 9]:
        mask = encoder.attention_mask(9, valid_length)
        actual = mask.mask_mod(0, 0, 0, positions)
        torch.testing.assert_close(actual, positions < valid_length)


@pytest.mark.parametrize("deterministic", [False, True])
def test_model_compiler_policy_leaves_other_models_unchanged(
    deterministic: bool,
) -> None:
    with inductor_config.patch(
        {"deterministic": deterministic, "triton.autotune_pointwise": True}
    ):
        importlib.reload(model)
        assert inductor_config.deterministic is deterministic
        assert inductor_config.triton.autotune_pointwise is True
        config = model.compiled_flex_attention.get_compiler_config()
        assert config["deterministic"] is True
        assert config["triton.autotune_pointwise"] is False
