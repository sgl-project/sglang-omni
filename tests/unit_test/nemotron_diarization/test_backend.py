# SPDX-License-Identifier: Apache-2.0
"""Guard checkpoint-family selection and overlapping, frame-rounded intervals."""

import io
import tarfile

import msgspec
import numpy as np
import pytest
import torch
import yaml

from sglang_omni.models.nemotron_diarization.backend import (
    load_checkpoint_weights,
    parse_segments,
    probabilities_to_segments,
    resolve_nemo_checkpoint,
    validate_checkpoint,
)


def test_segments_preserve_overlap_and_clip_only_the_partial_last_frame():
    result = parse_segments(
        ["1.000 2.510 speaker_7", "0.000 2.000 speaker_0"], duration=2.504
    )
    assert msgspec.to_builtins(result) == {
        "duration": 2.504,
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "speaker_0"},
            {"start": 1.0, "end": 2.504, "speaker": "speaker_7"},
        ],
    }


@pytest.mark.parametrize(
    "line",
    [
        "nan 1 speaker_0",
        "0 inf speaker_0",
        "-1 1 speaker_0",
        "1 0 speaker_0",
        "0 1 speaker_8",
        "0 20 speaker_0",
        "0 1",
        "0 0 speaker_0",
    ],
)
def test_invalid_model_segments_are_not_silently_repaired(line):
    with pytest.raises(RuntimeError, match="Invalid diarization segment"):
        parse_segments([line], duration=2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"high_resolution": False},
        {"sample_rate": 8000},
        {"target": "other.Model"},
        {"streaming_mode": False},
        {"max_num_of_spks": 4},
        {"preprocessor": {"window_stride": 0.08, "sample_rate": 16000}},
        {"preprocessor": {"normalize": "per_feature"}},
        {"preprocessor": {"preemph": 0.0}},
        {"encoder": {"attn_mode": "causal"}},
        {"encoder": {"rope_base": 500000.0}},
        {"sortformer_modules": {"scores_boost_latest": 0.0}},
        {"transformer_encoder": {}},
    ],
)
def test_checkpoint_validator_rejects_other_nemo_layouts(tmp_path, overrides):
    config = {
        "target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
        "sample_rate": 16000,
        "high_resolution": True,
        "output_subsampling_factor": 1,
        "streaming_mode": True,
        "max_num_of_spks": 8,
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
            "d_model": 512,
            "n_heads": 8,
            "subsampling": "feature_stacking",
            "ff_expansion": 4.0,
            "xscaling": False,
            "qkv_bias": False,
            "qk_norm": False,
            "pre_block_norm": True,
            "attn_mode": "full",
            "self_attention_model": "rope",
            "n_layers": 31,
            "subsampling_factor": 8,
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
    path = tmp_path / "test.nemo"

    def write_archive(values):
        data = yaml.safe_dump(values).encode()
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo("model_config.yaml")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    write_archive(config)
    validate_checkpoint(path)
    updated = dict(config)
    for key, value in overrides.items():
        updated[key] = (
            {**config[key], **value} if isinstance(config.get(key), dict) else value
        )
    write_archive(updated)
    with pytest.raises(ValueError, match="not the supported"):
        validate_checkpoint(path)


def test_local_checkpoint_directory_does_not_select_an_arbitrary_archive(tmp_path):
    (tmp_path / "unrelated.nemo").touch()
    with pytest.raises(FileNotFoundError, match="Nemotron-3-Diarization-preview.nemo"):
        resolve_nemo_checkpoint(str(tmp_path))


def test_missing_local_checkpoint_does_not_become_a_hub_repository(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing checkpoint"):
        resolve_nemo_checkpoint(str(tmp_path / "missing.nemo"))


def test_threshold_ties_retain_previous_activity_and_preserve_overlap():
    predictions = np.zeros((7, 8), dtype=np.float32)
    predictions[:, 0] = [0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5]
    predictions[2:4, 7] = 0.9
    result = probabilities_to_segments(predictions, duration=0.07)
    assert [(s.start, s.end, s.speaker) for s in result.segments] == [
        (0.01, 0.03, "speaker_0"),
        (0.02, 0.04, "speaker_7"),
        (0.05, 0.07, "speaker_0"),
    ]


@pytest.mark.parametrize("frames", [0, 10])
def test_silent_or_subframe_recording_has_no_segments(frames):
    assert probabilities_to_segments(np.zeros((frames, 8)), duration=0.1).segments == []


@pytest.mark.parametrize("duplicate,symlink", [(True, False), (False, True)])
def test_weights_member_must_be_unique_and_regular(tmp_path, duplicate, symlink):
    path = tmp_path / "bad.nemo"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("model_weights.ckpt")
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside/archive"
        archive.addfile(info)
        if duplicate:
            archive.addfile(info)
    with pytest.raises(ValueError, match="Expected one model_weights"):
        load_checkpoint_weights(path)


def test_checkpoint_rejects_non_tensor_state_entries(tmp_path):
    serialized = io.BytesIO()
    torch.save({"weight": torch.zeros(1), "config": "not a tensor"}, serialized)
    path = tmp_path / "bad.nemo"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("model_weights.ckpt")
        info.size = len(serialized.getvalue())
        archive.addfile(info, io.BytesIO(serialized.getvalue()))
    with pytest.raises(ValueError, match="tensor state dictionary"):
        load_checkpoint_weights(path)
