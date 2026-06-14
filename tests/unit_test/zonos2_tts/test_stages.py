# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.config.placement import StagePlacementPlanner
from sglang_omni.config.topology import build_process_topology_plan
from sglang_omni.models.zonos2_tts.stages import (
    _resolve_codec_device,
    _slice_batched_dac_waveforms,
    prepare_dac_codes_for_decode,
)
from sglang_omni.models.zonos2_tts.config import (
    Zonos2TTSMultiGPUPipelineConfig,
    Zonos2TTSPipelineConfig,
)


def test_slice_batched_dac_waveforms_removes_padding_tail() -> None:
    wavs = torch.arange(2 * 100, dtype=torch.float32).view(2, 100)

    trimmed = _slice_batched_dac_waveforms(wavs, [10, 5])

    assert trimmed[0].shape == (100,)
    assert trimmed[1].shape == (50,)
    torch.testing.assert_close(trimmed[0], wavs[0])
    torch.testing.assert_close(trimmed[1], wavs[1, :50])


def test_slice_batched_dac_waveforms_accepts_dac_channel_dim() -> None:
    wavs = torch.arange(2 * 1 * 80, dtype=torch.float32).view(2, 1, 80)

    trimmed = _slice_batched_dac_waveforms(wavs, [4, 2])

    assert [tuple(wav.shape) for wav in trimmed] == [(80,), (40,)]
    torch.testing.assert_close(trimmed[1], wavs[1, 0, :40])


def test_prepare_dac_codes_drops_delay_flush_without_eos() -> None:
    codes = torch.arange(12 * 9, dtype=torch.long).view(12, 9)

    prepared = prepare_dac_codes_for_decode(
        codes,
        n_codebooks=9,
        audio_pad_id=1025,
        codebook_size=1024,
        eos_frame=None,
    )

    assert prepared is not None
    assert prepared.shape == (4, 9)


def test_prepare_dac_codes_respects_eos_frame_after_deshearing() -> None:
    codes = torch.arange(12 * 9, dtype=torch.long).view(12, 9)

    prepared = prepare_dac_codes_for_decode(
        codes,
        n_codebooks=9,
        audio_pad_id=1025,
        codebook_size=1024,
        eos_frame=2,
    )

    assert prepared is not None
    assert prepared.shape == (2, 9)


def test_resolve_codec_device_prefers_launcher_gpu_id() -> None:
    assert _resolve_codec_device("cuda:1", 0) == "cuda:0"
    assert _resolve_codec_device("cuda:0", 1) == "cuda:1"


def test_default_pipeline_uses_async_talker_and_colocated_vocoder() -> None:
    config = Zonos2TTSPipelineConfig(model_path="Zyphra/ZONOS2")
    stage_names = [stage.name for stage in config.stages]
    stages = {stage.name: stage for stage in config.stages}
    topology = build_process_topology_plan(
        config, StagePlacementPlanner(config).build()
    )

    assert stage_names == ["preprocessing", "speaker_encode", "tts_engine", "vocoder"]
    assert stages["preprocessing"].next == "speaker_encode"
    assert stages["speaker_encode"].next == "tts_engine"
    assert stages["speaker_encode"].process == "pipeline"
    assert stages["speaker_encode"].factory_args["device"] == "cuda:0"
    assert stages["speaker_encode"].gpu == 0
    assert stages["tts_engine"].factory_args["enable_async_decode"] is True
    assert stages["tts_engine"].gpu == 0
    assert stages["vocoder"].process == "pipeline"
    assert stages["vocoder"].factory_args["device"] == "cuda:0"
    assert stages["vocoder"].factory_args["gpu_id"] is None
    assert stages["vocoder"].gpu == 0
    assert [(group.name, group.gpu_id) for group in topology.groups] == [
        ("pipeline", 0)
    ]


def test_multi_gpu_pipeline_uses_second_gpu_for_speaker_and_vocoder() -> None:
    config = Zonos2TTSMultiGPUPipelineConfig(model_path="Zyphra/ZONOS2")
    stages = {stage.name: stage for stage in config.stages}
    topology = build_process_topology_plan(
        config, StagePlacementPlanner(config).build()
    )

    assert stages["speaker_encode"].process == "vocoder"
    assert stages["speaker_encode"].factory_args["device"] == "cuda:1"
    assert stages["speaker_encode"].gpu == 1
    assert stages["tts_engine"].process == "pipeline"
    assert stages["tts_engine"].gpu == 0
    assert stages["vocoder"].process == "vocoder"
    assert stages["vocoder"].factory_args["device"] == "cuda:1"
    assert stages["vocoder"].factory_args["gpu_id"] is None
    assert stages["vocoder"].gpu == 1
    assert [(group.name, group.gpu_id) for group in topology.groups] == [
        ("pipeline", 0),
        ("vocoder", 1),
    ]
