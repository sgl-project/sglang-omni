# SPDX-License-Identifier: Apache-2.0
"""Native TTS evaluation: audio preparation and port reuse."""

import base64
import io
import json
import socket
import wave
from array import array
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks.eval import benchmark_tts_seedtts as tts
from sglang_omni_mlx.native.ci import evaluate_tts as native_tts


@pytest.mark.parametrize(
    "sample_rate, channel_count", [(16000, 1), (44100, 1), (48000, 2)]
)
def test_native_reference_audio_preparation(
    tmp_path: Path, sample_rate: int, channel_count: int
) -> None:
    reference_path = tmp_path / "reference.wav"
    channel_samples = [8192] if channel_count == 1 else [8192, 24576]
    with wave.open(str(reference_path), "wb") as waveform:
        waveform.setparams((channel_count, 2, sample_rate, 0, "NONE", "not compressed"))
        waveform.writeframes(
            array("h", channel_samples * (sample_rate // 10)).tobytes()
        )
    original_bytes = reference_path.read_bytes()
    metadata_path = tmp_path / "meta.lst"
    metadata_path.write_text(
        "first|参考一|reference.wav|目标一\n"
        "second|参考二|reference.wav|目标二\n"
        "third|参考三|reference.wav|目标三\n"
    )
    config = tts.TtsSeedttsBenchmarkConfig(
        model="test/ming",
        meta=str(metadata_path),
        voice_clone=True,
        sample_offset=1,
        max_samples=1,
    )
    samples = native_tts.prepare_reference_samples(config)
    assert len(samples) == 1
    assert (samples[0].sample_id, samples[0].ref_text, samples[0].target_text) == (
        "second",
        "参考二",
        "目标二",
    )
    prefix, encoded = samples[0].ref_audio.split(",", 1)
    assert prefix == "data:audio/wav;base64"
    with wave.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as waveform:
        assert waveform.getparams()[:4] == (1, 2, 16000, 1600)
        pcm16 = array("h", waveform.readframes(waveform.getnframes()))
    expected_amplitude = sum(channel_samples) / channel_count
    assert pcm16[100:-100] == pytest.approx([expected_amplitude] * 1400, rel=0.001)
    assert reference_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "sample_rate, channel_count", [(16000, 1), (44100, 1), (48000, 2)]
)
def test_native_asr_audio_preparation(
    tmp_path: Path, sample_rate: int, channel_count: int
) -> None:
    audio_path = tmp_path / "generated.wav"
    channel_samples = [8192] if channel_count == 1 else [8192, 24576]
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setparams((channel_count, 2, sample_rate, 0, "NONE", "not compressed"))
        wav_file.writeframes(
            array("h", channel_samples * (sample_rate // 10)).tobytes()
        )
    original_bytes = audio_path.read_bytes()
    generated = [
        {
            "sample_id": "generated",
            "wav_path": str(audio_path),
            "is_success": True,
            "target_text": "hello world",
            "audio_duration_s": 0.1,
            "latency_s": 0.5,
        },
        {"sample_id": "failed", "is_success": False, "error": "generation failed"},
    ]
    metadata_path = tmp_path / "generated.json"
    metadata_path.write_text(json.dumps(generated))
    original_metadata = metadata_path.read_bytes()
    (tmp_path / "speed_results.json").write_text('{"metrics": {"rtf_mean": 5}}')
    config = tts.TtsSeedttsBenchmarkConfig(
        model="test/ming", meta="test/seedtts", output_dir=str(tmp_path)
    )
    prepared = native_tts.prepare_asr_audio(config)
    scoring_directory = Path(prepared.output_dir)
    scoring_entries = json.loads((scoring_directory / "generated.json").read_text())
    converted = Path(scoring_entries[0]["wav_path"])
    with wave.open(str(converted), "rb") as wav_file:
        assert wav_file.getparams()[:4] == (1, 2, 16000, 1600)
        samples = array("h", wav_file.readframes(1600))
    expected_amplitude = sum(channel_samples) / channel_count
    assert samples[100:-100] == pytest.approx([expected_amplitude] * 1400, rel=0.001)
    assert scoring_entries[0] == {**generated[0], "wav_path": str(converted)}
    assert scoring_entries[1] == generated[1]
    assert (scoring_directory / "speed_results.json").read_bytes() == (
        tmp_path / "speed_results.json"
    ).read_bytes()
    assert config.output_dir == str(tmp_path)
    assert audio_path.read_bytes() == original_bytes
    assert metadata_path.read_bytes() == original_metadata
    if sample_rate == 16000 and channel_count == 1:
        assert converted.read_bytes() == original_bytes
    else:
        pass


@pytest.mark.parametrize("server_running", [False, True])
def test_native_port_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server_running: bool
) -> None:
    process = MagicMock()
    process.poll.return_value = None
    launch = MagicMock(return_value=process)
    monkeypatch.setattr(native_tts.subprocess, "Popen", launch)
    monkeypatch.setattr(native_tts, "wait_healthy", MagicMock())
    monkeypatch.setattr(native_tts, "stop_server", MagicMock())
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        if not server_running:
            with socket.create_connection(address) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                    assert client.recv(1) == b""
                    client.shutdown(socket.SHUT_WR)
                    assert connection.recv(1) == b""
            listener.close()
        else:
            pass
        server = native_tts.managed_native_server(
            binary=tmp_path / "qwen3_asr_server",
            model_path=str(tmp_path),
            revision=None,
            port=address[1],
            host=address[0],
            log_file=tmp_path / "asr.log",
            timeout_seconds=10,
        )
        if server_running:
            with pytest.raises(OSError, match="Address already in use"), server:
                pytest.fail("A live listener must prevent startup")
            launch.assert_not_called()
        else:
            with server:
                pass
            launch.assert_called_once()
