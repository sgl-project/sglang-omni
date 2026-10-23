# SPDX-License-Identifier: Apache-2.0
"""Drama 3 stage factories on CPU: codec path resolution and what each factory builds."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest
import torch
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_drama3 import stages
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_CODEC_PATH,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    Drama3PipelineConfig,
    Drama3VocoderFactoryArgs,
)
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
from sglang_omni.models.fishaudio_drama3.vocoder import Drama3VocoderScheduler
from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import WindowDecoder
from tests.unit_test.fishaudio_drama3.test_prompt import (
    DEFAULT_SYSTEM_TURN,
    END,
    USER,
    VOICE_TURN,
    build_toy_tokenizer,
    expected_rows,
)
from tests.unit_test.fishaudio_drama3.test_speech_request import (
    generation_input_of,
    lowered,
    refused,
    toy_config,
)
from tests.unit_test.fishaudio_drama3.test_vocoder import FrameNumberCodec

COMMIT = "1de9996b6be38b745688de084d87a5633f714e4e"


class HubDownloads:
    """Stands in for the isolated Hub downloader and records each call."""

    def __init__(self, local_file: Path) -> None:
        self.local_file = local_file
        self.calls: list[tuple[str, str, str]] = []

    def __call__(
        self,
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        assert timeout == 120.0
        self.calls.append(tuple(command[-3:]))
        return subprocess.CompletedProcess(command, 0, stdout=f"{self.local_file}\n")


@pytest.fixture
def hub_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HubDownloads:
    downloads = HubDownloads(tmp_path / "snapshot" / "codec.pth")
    monkeypatch.setattr(stages.subprocess, "run", downloads)
    return downloads


def test_a_local_codec_file_or_directory_resolves_without_the_hub(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    assert stages.resolve_codec_directory(
        str(codec_file), download_timeout_seconds=120.0
    ) == str(tmp_path)
    assert stages.resolve_codec_directory(
        str(tmp_path), download_timeout_seconds=120.0
    ) == str(tmp_path)
    assert hub_downloads.calls == []


def test_a_hub_codec_downloads_only_codec_pth_at_its_commit(
    hub_downloads: HubDownloads,
) -> None:
    assert stages.resolve_codec_directory(
        DEFAULT_CODEC_PATH, download_timeout_seconds=120.0
    ) == str(hub_downloads.local_file.parent)
    assert hub_downloads.calls == [("fishaudio/s2-pro", "codec.pth", COMMIT)]


@pytest.mark.parametrize(
    "download_error",
    [
        OSError("Cannot start downloader"),
        subprocess.CalledProcessError(1, ["python"], stderr="Hub is unreachable"),
    ],
)
def test_hub_failure_names_the_codec_and_local_override(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    download_error: OSError | subprocess.CalledProcessError,
) -> None:
    def fail_download(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        raise download_error

    monkeypatch.setattr(stages.subprocess, "run", fail_download)
    with caplog.at_level(logging.INFO, logger=stages.__name__):
        with pytest.raises(RuntimeError) as failure:
            stages.resolve_codec_directory(
                DEFAULT_CODEC_PATH, download_timeout_seconds=120.0
            )

    assert DEFAULT_CODEC_PATH in str(failure.value)
    assert "--vocoder.factory.codec_path" in str(failure.value)
    assert "HF_HUB_OFFLINE=1" in str(failure.value)
    assert failure.value.__cause__ is download_error
    assert DEFAULT_CODEC_PATH in caplog.text


def test_an_offline_hub_cache_hit_resolves_in_the_downloader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot_directory = tmp_path / "models--fishaudio--s2-pro" / "snapshots" / COMMIT
    snapshot_directory.mkdir(parents=True)
    (snapshot_directory / "codec.pth").write_bytes(b"")

    assert stages.resolve_codec_directory(
        DEFAULT_CODEC_PATH, download_timeout_seconds=5.0
    ) == str(snapshot_directory)


def test_an_offline_hub_cache_miss_reports_the_codec_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    with pytest.raises(RuntimeError) as failure:
        stages.resolve_codec_directory(DEFAULT_CODEC_PATH, download_timeout_seconds=5.0)

    assert DEFAULT_CODEC_PATH in str(failure.value)
    assert "--vocoder.factory.codec_path" in str(failure.value)
    assert "HF_HUB_OFFLINE=1" in str(failure.value)
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    assert "LocalEntryNotFoundError" in str(failure.value)


def test_a_stalled_hub_download_is_terminated_before_stage_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = subprocess.run
    download_processes: list[subprocess.Popen[str]] = []
    popen = subprocess.Popen

    def record_process(
        command: list[str], *, stdout: int, stderr: int, text: bool
    ) -> subprocess.Popen[str]:
        process = popen(command, stdout=stdout, stderr=stderr, text=text)
        download_processes.append(process)
        return process

    def stalled_download(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )

    monkeypatch.setattr(subprocess, "Popen", record_process)
    monkeypatch.setattr(stages.subprocess, "run", stalled_download)
    started = time.monotonic()
    with pytest.raises(TimeoutError) as failure:
        stages.resolve_codec_directory(DEFAULT_CODEC_PATH, download_timeout_seconds=0.2)

    assert time.monotonic() - started < 5.0
    assert "0.2s" in str(failure.value)
    assert DEFAULT_CODEC_PATH in str(failure.value)
    assert "--vocoder.factory.codec_path" in str(failure.value)
    assert "--vocoder.factory.codec_download_timeout_seconds" in str(failure.value)
    assert isinstance(failure.value.__cause__, subprocess.TimeoutExpired)
    assert len(download_processes) == 1
    assert download_processes[0].poll() is not None


def test_both_stages_use_the_bundled_codec_without_the_hub(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    (tmp_path / "codec.pth").write_bytes(b"")
    config = Drama3PipelineConfig(model_path=str(tmp_path))

    for stage_name in ("preprocessing", "vocoder"):
        codec_path = config.stage_factory_kwargs(stage_name)["codec_path"]
        assert isinstance(codec_path, str)
        assert stages.resolve_codec_directory(
            codec_path, download_timeout_seconds=120.0
        ) == str(tmp_path)
    assert hub_downloads.calls == []


def test_missing_explicit_codec_does_not_fall_back_to_bundled_codec(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    (tmp_path / "codec.pth").write_bytes(b"")
    config = Drama3PipelineConfig(model_path=str(tmp_path))
    config.stage_named("vocoder").factory.codec_path = str(
        tmp_path / "missing/codec.pth"
    )

    for stage_name in ("preprocessing", "vocoder"):
        codec_path = config.stage_factory_kwargs(stage_name)["codec_path"]
        assert isinstance(codec_path, str)
        with pytest.raises(ValueError, match="--vocoder.factory.codec_path"):
            stages.resolve_codec_directory(codec_path, download_timeout_seconds=120.0)
    assert hub_downloads.calls == []


@pytest.mark.parametrize(
    "codec_path",
    [
        pytest.param("fishaudio/s2-pro", id="no-revision"),
        pytest.param("fishaudio/s2-pro@main", id="branch"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT[:7]}", id="short-commit"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT.upper()}", id="upper-case-commit"),
        pytest.param("missing/codec.pth", id="missing-file"),
    ],
)
def test_a_codec_path_that_could_drift_or_does_not_exist_is_refused(
    hub_downloads: HubDownloads, codec_path: str
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(codec_path, download_timeout_seconds=120.0)
    assert hub_downloads.calls == []


def test_a_codec_file_of_another_name_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    renamed_codec_file = tmp_path / "renamed-codec.pth"
    renamed_codec_file.write_bytes(b"")

    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(
            str(renamed_codec_file), download_timeout_seconds=120.0
        )


def test_a_directory_without_codec_pth_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(str(tmp_path), download_timeout_seconds=120.0)


class CodecLoads:
    """Stands in for load_codec and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, checkpoint_dir: str, device: str) -> FrameNumberCodec:
        self.calls.append((checkpoint_dir, device))
        return FrameNumberCodec()


@pytest.fixture
def codec_loads(monkeypatch: pytest.MonkeyPatch) -> CodecLoads:
    loads = CodecLoads()
    monkeypatch.setattr(stages, "load_codec", loads)
    return loads


def write_toy_checkpoint(tmp_path: Path) -> Tokenizer:
    tokenizer = build_toy_tokenizer()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    toy_config(tokenizer).save_pretrained(str(checkpoint_dir))
    (tmp_path / "codec.pth").write_bytes(b"")
    return tokenizer


def test_preprocessing_reads_the_checkpoint_and_loads_the_codec_on_cpu(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    tokenizer = write_toy_checkpoint(tmp_path)
    rows = expected_rows(
        tokenizer, DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN]
    )

    scheduler = stages.create_preprocessing_executor(
        str(tmp_path / "checkpoint"),
        device="cpu",
        max_concurrency=2,
        max_prompt_rows=rows.shape[0],
        codec_path=str(tmp_path / "codec.pth"),
        codec_download_timeout_seconds=120.0,
        reference_encode_dtype="float32",
        max_reference_audio_seconds=30.0,
    )

    preprocessor = scheduler.fn
    assert isinstance(preprocessor, Drama3SpeechPreprocessor)
    assert scheduler.max_concurrency == 2
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    hook = preprocessor.reference_encoder.hook
    assert isinstance(hook, FishReferenceEncodeHook)
    assert (hook.device, hook.autocast_dtype, hook.max_reference_audio_seconds) == (
        torch.device("cpu"),
        None,
        30.0,
    )
    assert torch.equal(
        generation_input_of(preprocessor(lowered({"input": "hello"}))).prompt_rows,
        rows,
    )
    assert "maximum allowed length" in refused(
        preprocessor, lowered({"input": "hello there"})
    )


def test_a_bfloat16_reference_encode_on_the_cpu_is_refused(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    write_toy_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="reference_encode_dtype to float32"):
        stages.create_preprocessing_executor(
            str(tmp_path / "checkpoint"),
            device="cpu",
            max_concurrency=1,
            max_prompt_rows=64,
            codec_path=str(tmp_path / "codec.pth"),
            codec_download_timeout_seconds=120.0,
            reference_encode_dtype="bfloat16",
            max_reference_audio_seconds=60.0,
        )

    assert codec_loads.calls == []


class EncodeRecordingCodec:
    sample_rate: int = 44100

    def __init__(self) -> None:
        self.encoded_shapes: list[tuple[int, ...]] = []

    def encode(
        self, audios: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoded_shapes.append(tuple(audios.shape))
        return torch.zeros((1, 10, 1), dtype=torch.long), audio_lengths


@pytest.mark.parametrize(
    ("device", "reference_encode_dtype", "expected_encoded_shapes"),
    [
        pytest.param("cpu", "float32", [], id="cpu"),
        pytest.param(
            "cuda",
            "bfloat16",
            [(1, EncodeRecordingCodec.sample_rate)],
            id="cuda",
            marks=pytest.mark.accelerator,
        ),
    ],
)
def test_preprocessing_warms_the_encoder_up_once_on_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: str,
    reference_encode_dtype: Literal["float32", "bfloat16"],
    expected_encoded_shapes: list[tuple[int, ...]],
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    write_toy_checkpoint(tmp_path)
    codec = EncodeRecordingCodec()
    monkeypatch.setattr(stages, "load_codec", lambda checkpoint_dir, device: codec)

    stages.create_preprocessing_executor(
        str(tmp_path / "checkpoint"),
        device=device,
        max_concurrency=1,
        max_prompt_rows=64,
        codec_path=str(tmp_path / "codec.pth"),
        codec_download_timeout_seconds=120.0,
        reference_encode_dtype=reference_encode_dtype,
        max_reference_audio_seconds=60.0,
    )

    assert codec.encoded_shapes == expected_encoded_shapes


def test_the_engine_factory_hands_its_knobs_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> str:
        assert (
            model_path,
            device,
            gpu_id,
            server_args_overrides,
            total_gpu_memory_fraction,
        ) == ("checkpoint", None, 1, {"mem_fraction_static": 0.5}, None)
        assert (
            builder.context_length,
            builder.max_running_requests,
            builder.max_prompt_rows,
            builder.max_new_tokens_cap,
            builder.enable_async_decode,
            builder.async_decode_min_batch_size,
        ) == (4096, 4, 1024, 512, True, 3)
        return "engine"

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    engine = stages.create_tts_engine_executor(
        "checkpoint",
        gpu_id=1,
        context_length=4096,
        max_running_requests=4,
        max_new_tokens=512,
        max_prompt_rows=1024,
        server_args_overrides={"mem_fraction_static": 0.5},
        enable_async_decode=True,
        async_decode_min_batch_size=3,
    )

    assert engine == "engine"


@pytest.mark.parametrize(
    (
        "process_total_gpu_memory_fraction",
        "server_args_overrides",
        "kv_cache_budget",
        "expected_overrides",
    ),
    [
        pytest.param(None, None, 0.7, {"mem_fraction_static": 0.7}, id="own-share"),
        pytest.param(0.8, None, 0.8, {"mem_fraction_static": 0.7}, id="process-share"),
        pytest.param(
            None,
            {"mem_fraction_static": 0.7},
            0.7,
            {"mem_fraction_static": 0.7},
            id="equal-static-fraction",
        ),
    ],
)
def test_the_engine_sizes_its_kv_cache_from_its_declared_share(
    monkeypatch: pytest.MonkeyPatch,
    process_total_gpu_memory_fraction: float | None,
    server_args_overrides: Mapping[str, object] | None,
    kv_cache_budget: float,
    expected_overrides: Mapping[str, object],
) -> None:
    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> tuple[float | None, Mapping[str, object] | None]:
        return total_gpu_memory_fraction, server_args_overrides

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    engine = stages.create_tts_engine_executor(
        "checkpoint",
        gpu_id=0,
        total_gpu_memory_fraction=0.7,
        process_total_gpu_memory_fraction=process_total_gpu_memory_fraction,
        server_args_overrides=server_args_overrides,
    )

    assert engine == (kv_cache_budget, expected_overrides)


def test_a_static_fraction_other_than_the_stage_share_is_refused_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: list[str] = []

    def build(
        builder: Drama3EngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
        total_gpu_memory_fraction: float | None,
    ) -> str:
        builds.append(model_path)
        return "engine"

    monkeypatch.setattr(Drama3EngineBuilder, "build", build)

    with pytest.raises(ValueError, match="--tts_engine.gpu_memory_fraction"):
        stages.create_tts_engine_executor(
            "checkpoint",
            gpu_id=0,
            total_gpu_memory_fraction=0.7,
            server_args_overrides={"mem_fraction_static": 0.5},
        )

    assert builds == []


def cpu_vocoder(factory_args: Drama3VocoderFactoryArgs) -> Drama3VocoderScheduler:
    return stages.create_vocoder_executor(
        device="cpu",
        codec_path=factory_args.codec_path,
        codec_download_timeout_seconds=factory_args.codec_download_timeout_seconds,
        stream_stride=factory_args.stream_stride,
        stream_followup_stride_ramp=factory_args.stream_followup_stride_ramp,
        stream_followup_stride=factory_args.stream_followup_stride,
        stream_overlap_tokens=factory_args.stream_overlap_tokens,
        stream_crossfade_samples=factory_args.stream_crossfade_samples,
        max_batch_size=factory_args.max_batch_size,
        max_batch_wait_ms=factory_args.max_batch_wait_ms,
        window_cuda_graphs=factory_args.window_cuda_graphs,
        decode_dtype=factory_args.decode_dtype,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS_CAP,
    )


def test_the_vocoder_factory_loads_the_codec_on_its_device_with_the_windows(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    vocoder = cpu_vocoder(
        Drama3VocoderFactoryArgs(codec_path=str(codec_file)).model_copy(
            update={"window_cuda_graphs": False, "decode_dtype": "float32"}
        )
    )

    assert isinstance(vocoder, Drama3VocoderScheduler)
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    assert isinstance(vocoder.codec, FrameNumberCodec)
    assert vocoder.codec.is_weight_norm_folded
    assert vocoder.decode_window == vocoder.codec.from_indices
    assert (
        vocoder.stream_stride,
        vocoder.stream_followup_stride_ramp,
        vocoder.stream_followup_stride,
        vocoder.stream_overlap_tokens,
        vocoder.stream_crossfade_samples,
        vocoder.max_batch_size,
    ) == (10, (10, 12, 32), 64, 32, 0, 8)
    assert vocoder.max_padded_frames_per_decode is None


def test_window_cuda_graphs_on_a_cpu_device_are_refused(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    with pytest.raises(RuntimeError, match="window_cuda_graphs to false"):
        cpu_vocoder(
            Drama3VocoderFactoryArgs(codec_path=str(codec_file)).model_copy(
                update={"decode_dtype": "float32"}
            )
        )


def bytes_per_frame_decoder(bytes_per_frame: int) -> WindowDecoder:
    def decode(indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (indices.shape[0], 1, indices.shape[-1] * bytes_per_frame // 4),
            dtype=torch.float32,
            device=indices.device,
        )

    return decode


@pytest.mark.accelerator
def test_the_padded_frame_limit_is_the_share_left_after_held_memory() -> None:
    device = torch.device("cuda")
    bytes_per_frame = 4096
    share_bytes = int(0.01 * torch.cuda.get_device_properties(device).total_memory)
    torch.cuda.empty_cache()
    held_before = torch.cuda.memory_reserved(device)

    max_padded_frames_per_decode = stages.measure_max_padded_frames_per_decode(
        bytes_per_frame_decoder(bytes_per_frame),
        device=device,
        num_codebooks=4,
        total_gpu_memory_fraction=0.01,
        max_new_tokens=16,
    )

    held_after = torch.cuda.memory_reserved(device)
    assert (
        (share_bytes - held_after) // bytes_per_frame
        <= max_padded_frames_per_decode
        <= (share_bytes - held_before) // bytes_per_frame
    )


@pytest.mark.accelerator
def test_a_share_that_cannot_hold_one_longest_request_is_refused() -> None:
    with pytest.raises(ValueError, match="--vocoder.gpu_memory_fraction"):
        stages.measure_max_padded_frames_per_decode(
            bytes_per_frame_decoder(4096),
            device=torch.device("cuda"),
            num_codebooks=4,
            total_gpu_memory_fraction=0.01,
            max_new_tokens=10**9,
        )
