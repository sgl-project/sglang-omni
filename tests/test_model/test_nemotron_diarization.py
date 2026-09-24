# SPDX-License-Identifier: Apache-2.0
"""Opt-in native/NeMo probability and HTTP parity.

No checkpoint or evaluation output is downloaded or included in the repository.
Additional permitted conversational WAV fixtures can be supplied through
NEMOTRON_DIARIZATION_AUDIO_DIR. These checks establish integration parity,
not model accuracy; DER requires reference speaker annotations.
"""

import io
import os
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import msgspec
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.g711 import wrap_g711_as_wav

pytestmark = pytest.mark.accelerator


def _wav(waveform, sample_rate=16000):
    stream = io.BytesIO()
    sf.write(stream, waveform, sample_rate, format="WAV", subtype="FLOAT")
    return stream.getvalue()


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("NEMOTRON_DIARIZATION_CHECKPOINT")
    if not path:
        pytest.skip("Set NEMOTRON_DIARIZATION_CHECKPOINT to a local gated checkpoint")
    if not Path(path).exists():
        pytest.fail(f"Checkpoint does not exist: {path}")
    if not torch.cuda.is_available():
        pytest.skip("Nemotron diarization requires CUDA")
    return path


@pytest.fixture(scope="module")
def deterministic_reference():
    from nemo.collections.asr.modules import transformer_encoder_utils
    from torch.nn.attention.flex_attention import flex_attention

    # The exact oracle specifies its own compiler policy independently.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            transformer_encoder_utils,
            "_flex_attention_compiled",
            torch.compile(
                flex_attention, dynamic=True, options={"deterministic": True}
            ),
        )
        yield


@pytest.fixture(scope="module", params=["offline", "low_latency"])
def reference_models(checkpoint, request, deterministic_reference):
    from nemo.collections.asr.models import SortformerEncLabelModel

    from sglang_omni.models.nemotron_diarization.backend import (
        NemotronDiarizer,
        resolve_nemo_checkpoint,
    )

    profile = request.param
    # Construct the reference independently from the wrapper's profile table.
    direct = SortformerEncLabelModel.restore_from(
        str(resolve_nemo_checkpoint(checkpoint)), map_location="cuda:0", strict=True
    ).eval()
    settings = (
        dict(
            spkcache_len=264,
            fifo_len=40,
            chunk_len=340,
            chunk_right_context=40,
            spkcache_update_period=300,
        )
        if profile == "offline"
        else dict(
            spkcache_len=264,
            fifo_len=264,
            chunk_len=9,
            chunk_right_context=4,
            spkcache_update_period=222,
        )
    )
    for key, value in settings.items():
        setattr(direct.sortformer_modules, key, value)
    direct._check_streaming_parameters()
    native = NemotronDiarizer(checkpoint, device="cuda:0", profile=profile)
    yield direct, native, profile
    del direct
    del native
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[1, 2])
def deployment(checkpoint, reference_models, request, tmp_path_factory):
    from sglang_omni.utils import find_available_port
    from tests.utils import start_server_from_cmd, stop_server

    direct, native, profile = reference_models
    concurrency = request.param
    directory = tmp_path_factory.mktemp(f"nemotron_{profile}_{concurrency}")
    config = directory / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "config_cls": "NemotronDiarizationPipelineConfig",
                "model_path": checkpoint,
            }
        )
    )
    # Enforce the production dependency boundary in the server and its workers,
    # even though this test process installs NeMo as the parity oracle.
    (directory / "sitecustomize.py").write_text(
        "import sys\n"
        "class RejectNeMo:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'nemo' or fullname.startswith('nemo.'):\n"
        "            raise ImportError('NeMo is unavailable in the native server')\n"
        "sys.meta_path.insert(0, RejectNeMo())\n"
    )
    port = find_available_port()
    proc = start_server_from_cmd(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(config),
            "--diarization.factory.profile",
            profile,
            "--diarization.factory.max_concurrency",
            str(concurrency),
            "--port",
            str(port),
        ],
        directory / "server.log",
        port,
        timeout=300,
        env={
            "PYTHONPATH": os.pathsep.join(
                [
                    str(directory),
                    str(Path(__file__).parents[2]),
                    os.environ.get("PYTHONPATH", ""),
                ]
            )
        },
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=180, trust_env=False
        ) as client:
            yield direct, native, client
    finally:
        stop_server(proc)


@pytest.fixture(scope="module")
def recordings():
    source = Path(__file__).parents[1] / "data" / "query_to_draw_8k.ulaw"
    audio = wrap_g711_as_wav(source.read_bytes(), "mulaw")
    waveform = load_audio(audio, target_sample_rate=16000)
    # 67 seconds traverses multiple offline chunks and speaker-cache updates.
    long_audio = np.tile(waveform, 14)[: 67 * 16000 + 37]
    cases = {
        "speech": audio,
        "silence": _wav(np.zeros(32000, dtype=np.float32)),
        "partial_tail": _wav(waveform[:16003]),
        "stereo_resampled": _wav(np.column_stack([waveform[::2], waveform[::2]]), 8000),
        "cache_updates": _wav(long_audio),
    }
    extra = os.environ.get("NEMOTRON_DIARIZATION_AUDIO_DIR")
    if extra:
        cases.update(
            {str(path): path.read_bytes() for path in sorted(Path(extra).glob("*.wav"))}
        )
    return cases


def _post(client, audio):
    response = client.post(
        "/v1/audio/diarizations", files={"file": ("audio.wav", audio)}
    )
    assert response.status_code == 200, response.text
    return response


def test_http_matches_direct_nemo_and_requests_are_isolated(deployment, recordings):
    direct, native, client = deployment
    # The first requests also exercise compilation from concurrent worker threads.
    initial_names = ["cache_updates", "speech"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        initial_responses = list(
            pool.map(lambda name: _post(client, recordings[name]), initial_names)
        )
    expected = {}
    for name, audio in recordings.items():
        waveform = load_audio(audio, target_sample_rate=16000)
        with torch.inference_mode():
            lines, probabilities = direct.diarize(
                audio=[waveform],
                sample_rate=16000,
                batch_size=1,
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
        assert torch.isfinite(probabilities[0]).all()
        assert probabilities[0].shape[-1] == 8
        assert abs(probabilities[0].shape[1] - len(waveform) / 160) <= 1
        # Measured on the pinned FP32 runtime: features, attention, speaker-head
        # output and cache decisions match exactly, including long recordings.
        actual = native.probabilities(waveform).cpu()
        torch.testing.assert_close(actual, probabilities[0].cpu(), rtol=0, atol=0)
        duration = len(waveform) / 16000
        segments = [
            {
                "start": float(start),
                "end": min(float(end), duration),
                "speaker": speaker,
            }
            for start, end, speaker in (line.split() for line in lines[0])
            if float(start) < duration
        ]
        segments.sort(
            key=lambda segment: (segment["start"], segment["end"], segment["speaker"])
        )
        expected[name] = {"duration": duration, "segments": segments}
        assert _post(client, audio).json() == expected[name]
    assert expected["silence"]["segments"] == []
    for name, response in zip(initial_names, initial_responses):
        assert response.json() == expected[name]
    # Mixed lengths exercise separate caches, multiple chunks and queued work.
    names = ["cache_updates", "silence", "partial_tail", "speech", "cache_updates"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda name: _post(client, recordings[name]), names))
    assert len({response.headers["x-request-id"] for response in responses}) == len(
        names
    )
    for name, response in zip(names, responses):
        assert response.json() == expected[name]


def test_concurrent_stage_reuses_bounded_workers_and_cuda_streams(
    reference_models, recordings, monkeypatch
):
    from sglang_omni.client import Client, GenerateRequest
    from sglang_omni.models.nemotron_diarization import stages
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.messages import IncomingMessage

    _, native, _ = reference_models
    barrier = threading.Barrier(2, timeout=30)
    worker_streams = {}
    original = native.diarize
    expected = {
        name: msgspec.to_builtins(
            original(load_audio(recordings[name], target_sample_rate=16000))
        )
        for name in ("speech", "silence")
    }

    original_preprocess = native.model.preprocessor.forward

    def simultaneous(waveform):
        stream = torch.cuda.current_stream(native.device).cuda_stream
        worker_streams.setdefault(threading.get_ident(), set()).add(stream)
        barrier.wait()
        return original_preprocess(waveform)

    monkeypatch.setattr(native.model.preprocessor, "forward", simultaneous)
    monkeypatch.setattr(stages, "NemotronDiarizer", lambda *args, **kwargs: native)
    scheduler = stages.create_diarization_executor(
        "unused", device="cuda", gpu_id=0, max_concurrency=2
    )
    # More queued work than slots catches accidentally raising the worker limit.
    names = ["speech", "silence", "speech", "silence"]
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for batch in range(6):
            pending = {}
            for index, name in enumerate(names):
                request_id = f"{batch}-{index}"
                pending[request_id] = expected[name]
                request = Client.build_omni_request(
                    GenerateRequest(
                        prompt={"audio_bytes": recordings[name]},
                        stream=False,
                        metadata={"task": "diarization"},
                    )
                )
                scheduler.inbox.put(
                    IncomingMessage(
                        request_id,
                        "new_request",
                        StagePayload(request_id=request_id, request=request, data={}),
                    )
                )
            for _ in names:
                output = scheduler.outbox.get(timeout=60)
                assert output.type == "result", output.data
                assert output.data.request_id == output.request_id
                assert output.data.data["diarization"] == pending.pop(output.request_id)
            assert not pending
    finally:
        barrier.abort()
        scheduler.stop()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(worker_streams) == 2
    assert all(len(owned) == 1 for owned in worker_streams.values())
    streams = set().union(*worker_streams.values())
    assert len(streams) == 2
    assert torch.cuda.default_stream(native.device).cuda_stream not in streams


@pytest.mark.parametrize(
    "audio", [b"", b"not audio", _wav(np.array([], dtype=np.float32))]
)
def test_bad_upload_returns_400_and_worker_recovers(deployment, audio):
    _, _, client = deployment
    response = client.post("/v1/audio/diarizations", files={"file": ("bad.wav", audio)})
    assert response.status_code == 400, response.text
    assert (
        _post(client, _wav(np.zeros(16000, dtype=np.float32))).json()["segments"] == []
    )


@pytest.mark.parametrize("variant", ["incomplete", "dtype_and_layout"])
def test_native_restore_preserves_weight_contract(checkpoint, tmp_path, variant):
    from sglang_omni.models.nemotron_diarization.backend import (
        NemotronDiarizer,
        load_checkpoint_weights,
        resolve_nemo_checkpoint,
    )

    # Keep the real architecture and check rejection and copy-load semantics.
    with tarfile.open(resolve_nemo_checkpoint(checkpoint)) as archive:
        config_member = next(
            m
            for m in archive.getmembers()
            if m.name in {"model_config.yaml", "./model_config.yaml"}
        )
        config = archive.extractfile(config_member).read()
    weights = io.BytesIO()
    state = {"unexpected.weight": torch.zeros(1)}
    key = "encoder.layers.0.attn.out_proj.weight"
    if variant == "dtype_and_layout":
        state = load_checkpoint_weights(resolve_nemo_checkpoint(checkpoint))
        state[key] = state[key].double().t().contiguous().t()
    torch.save(state, weights)
    broken = tmp_path / f"{variant}.nemo"
    with tarfile.open(broken, "w") as archive:
        for name, data in [
            ("model_config.yaml", config),
            ("model_weights.ckpt", weights.getvalue()),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    if variant == "incomplete":
        with pytest.raises(RuntimeError, match="Missing key.*state_dict"):
            NemotronDiarizer(str(broken), device="cuda:0")
    else:
        model = NemotronDiarizer(str(broken), device="cuda:0").model
        restored = model.state_dict()[key]
        assert restored.dtype == torch.float32 and restored.is_contiguous()
        torch.testing.assert_close(restored.cpu(), state[key].float(), rtol=0, atol=0)


@pytest.mark.parametrize("reference_models", ["low_latency"], indirect=True)
def test_live_probabilities_match_nemo_across_message_boundaries(
    reference_models, recordings
):
    from sglang_omni.models.nemotron_diarization.streaming import LiveDiarization

    direct, native, _ = reference_models
    names = ["cache_updates", "speech", "silence", "partial_tail"]
    names.extend(name for name in recordings if name.endswith(".wav"))
    for name in names:
        waveform = load_audio(recordings[name], target_sample_rate=16000)
        _, reference = direct.diarize(
            audio=[waveform],
            sample_rate=16000,
            batch_size=1,
            include_tensor_outputs=True,
            num_workers=0,
            verbose=False,
        )
        # Uneven network messages must not become acoustic chunk boundaries.
        stream = LiveDiarization(native.model, device=native.device)
        chunks = []
        for offset in range(0, len(waveform), 1777):
            chunks.append(stream.probabilities(waveform[offset : offset + 1777]).cpu())
            assert stream.audio.size < 104 * 160 + 256 + 320
            assert stream.state.cache.shape[1] <= 264
            assert stream.state.fifo.shape[1] <= 264
        if len(waveform) > 17000:
            assert sum(chunk.shape[1] for chunk in chunks) > 0
        chunks.append(stream.probabilities(np.empty(0, np.float32), final=True).cpu())
        actual = torch.cat(chunks, dim=1)
        expected = reference[0].cpu()
        # Full and incremental mel products have different reduction shapes.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        assert torch.equal(actual > 0.5, expected > 0.5)
        repacketized = LiveDiarization(native.model, device=native.device)
        other = [
            repacketized.probabilities(waveform[offset : offset + 16000]).cpu()
            for offset in range(0, len(waveform), 16000)
        ]
        other.append(
            repacketized.probabilities(np.empty(0, np.float32), final=True).cpu()
        )
        torch.testing.assert_close(actual, torch.cat(other, dim=1), rtol=0, atol=0)


def _merge_live_segments(updates):
    merged = []
    for speaker in range(8):
        intervals = [
            s
            for update in updates
            for s in update["segments"]
            if s["speaker"] == f"speaker_{speaker}"
        ]
        for segment in intervals:
            if (
                merged
                and merged[-1]["speaker"] == segment["speaker"]
                and merged[-1]["end"] == segment["start"]
            ):
                merged[-1]["end"] = segment["end"]
            else:
                merged.append(dict(segment))
    return sorted(merged, key=lambda s: (s["start"], s["end"], s["speaker"]))


@pytest.mark.parametrize("reference_models", ["low_latency"], indirect=True)
def test_live_websocket_interleaves_recordings_and_recovers(deployment, recordings):
    import json

    from websockets.sync.client import connect

    from tests.utils import disable_proxy

    _, native, client = deployment
    url = (
        str(client.base_url).replace("http://", "ws://").rstrip("/")
        + "/v1/audio/diarizations/stream"
    )
    inputs = []
    for name in ("speech", "partial_tail", "silence"):
        waveform = load_audio(recordings[name], target_sample_rate=16000)
        pcm = np.clip(np.rint(waveform * 32768), -32768, 32767).astype("<i2")
        expected = msgspec.to_builtins(native.diarize(pcm.astype(np.float32) / 32768))
        inputs.append((pcm.tobytes(), expected))

    barrier = threading.Barrier(len(inputs), timeout=60)

    def run(case):
        pcm, expected = case
        updates = []
        with connect(url, open_timeout=30) as ws:
            ready = json.loads(ws.recv(timeout=60))
            assert ready["type"] == "session.ready"
            barrier.wait()
            for offset in range(0, len(pcm), 3200):
                ws.send(pcm[offset : offset + 3200])
                while True:
                    event = json.loads(ws.recv(timeout=60))
                    if event["type"] == "audio.ack":
                        break
                    assert event["type"] == "diarization.update", event
                    assert event["start"] == (updates[-1]["end"] if updates else 0)
                    updates.append(event)
            if len(pcm) >= 2 * (104 * 160 + 256):
                assert updates  # Results arrive before audio.end.
            else:
                assert not updates  # Short recordings need the final lookahead flush.
            ws.send(json.dumps({"type": "audio.end"}))
            while True:
                event = json.loads(ws.recv(timeout=60))
                if event["type"] == "diarization.done":
                    assert event["duration"] == expected["duration"]
                    break
                assert event["type"] == "diarization.update", event
                assert event["start"] == (updates[-1]["end"] if updates else 0)
                updates.append(event)
        assert _merge_live_segments(updates) == expected["segments"]
        return ready["session_id"]

    with disable_proxy(), ThreadPoolExecutor(max_workers=len(inputs)) as pool:
        ids = list(pool.map(run, inputs))
    assert len(set(ids)) == len(inputs)
    # More than the session limit over time: both protocol errors and ordinary
    # disconnects must release capacity, rather than accumulating stale sessions.
    for index in range(10):
        with disable_proxy(), connect(url, open_timeout=30) as ws:
            assert json.loads(ws.recv(timeout=60))["type"] == "session.ready"
            ws.send(b"x" if index % 2 else inputs[0][0][:3200])
            event = json.loads(ws.recv(timeout=60))
            assert event["type"] == ("error" if index % 2 else "audio.ack")
    assert _post(client, recordings["speech"]).status_code == 200
