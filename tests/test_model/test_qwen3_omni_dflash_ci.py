# SPDX-License-Identifier: Apache-2.0
"""Opt-in native DFlash GPU smoke, including in-flight cancellation.

Run from the repository root with two visible GPUs:
  export SGLANG_OMNI_DFLASH_TEST_CONFIG=/path/to/speech.yaml
  python -m pytest tests/test_model/test_qwen3_omni_dflash_ci.py -v -s -x

The YAML must include stages.thinker and speech-stage placements/memory budgets
valid for the selected TP setting. DFlash defaults to enabled and requires a
local draft path in thinker.engine.speculative_draft_model_path or the
SGLANG_OMNI_TEST_DFLASH_MODEL environment variable.

Optional overrides: SGLANG_OMNI_DFLASH_TEST_OUTPUT, SGLANG_OMNI_DFLASH_TEST_TP
(1 or 2), SGLANG_OMNI_DFLASH_TEST_ENABLED (1 or 0), SGLANG_OMNI_DFLASH_TEST_PORT,
and SGLANG_OMNI_TEST_QWEN3_MODEL.
Run this module separately for each TP/off-on configuration.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml

from sglang_omni.utils import find_available_port
from tests.utils import start_server_from_cmd, stop_server

pytestmark = pytest.mark.benchmark
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUEST_TIMEOUT = 120
PROMPTS = (
    "Please say exactly: Hello, world.",
    "Please read aloud: The morning train arrived beside the quiet river, "
    "and the passengers carried their bags across the old stone bridge.",
)


def _events(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        # note(wenyao): A writer can be appending the last line while we poll.
        data = path.read_bytes()
        for line in data[: data.rfind(b"\n") + 1].splitlines():
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: row["monotonic_ns"])


def _wait_events(server, predicate, *, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = _events(server.trace)
        if predicate(rows):
            return rows
        assert server.proc.poll() is None, "server exited; inspect server.log"
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for trace evidence in {server.trace}")


@pytest.fixture(scope="module")
def dflash_server(tmp_path_factory):
    source_config = os.environ.get("SGLANG_OMNI_DFLASH_TEST_CONFIG")
    if not source_config:
        pytest.skip("set SGLANG_OMNI_DFLASH_TEST_CONFIG to run the GPU smoke")
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("speech smoke requires two visible GPUs")
    config = yaml.safe_load(Path(source_config).read_text())
    thinker = config["stages"]["thinker"]
    tp = int(os.environ.get("SGLANG_OMNI_DFLASH_TEST_TP", thinker.get("tp_size", 1)))
    enabled_value = os.environ.get("SGLANG_OMNI_DFLASH_TEST_ENABLED", "1")
    assert tp in (1, 2) and enabled_value in ("0", "1")
    enabled = enabled_value == "1"
    thinker["tp_size"] = tp
    thinker["gpu"] = [0, 1] if tp == 2 else 0
    factory = thinker.setdefault("factory", {})
    factory.update(talker_stream_token_only=True, capture_speech_hidden_states=False)
    engine = thinker.setdefault("engine", {})
    engine["chunked_prefill_size"] = 512
    if enabled:
        draft = os.environ.get("SGLANG_OMNI_TEST_DFLASH_MODEL") or engine.get(
            "speculative_draft_model_path"
        )
        assert draft, "provide a local DFlash draft in the config or environment"
        engine.update(
            speculative_algorithm="DFLASH",
            speculative_draft_model_path=draft,
            speculative_num_draft_tokens=8,
        )
        factory["enable_async_decode"] = False
    else:
        for key in list(engine):
            if key.startswith("speculative_"):
                del engine[key]
    if os.environ.get("SGLANG_OMNI_TEST_QWEN3_MODEL"):
        config["model_path"] = os.environ["SGLANG_OMNI_TEST_QWEN3_MODEL"]
    output = os.environ.get("SGLANG_OMNI_DFLASH_TEST_OUTPUT")
    folder = (
        Path(output).resolve()
        if output
        else tmp_path_factory.mktemp(f"dflash-tp{tp}-{'on' if enabled else 'off'}")
    )
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "server.log").exists(), "use a fresh smoke output directory"
    trace = folder / "trace"
    trace.mkdir()
    probe = folder / "probe"
    probe.mkdir()
    (probe / "sitecustomize.py").write_text(
        "from tests.test_model.qwen3_omni_dflash_probe import install; install()\n"
    )
    config["endpoints"] = {"base_path": str(folder / "ipc")}
    config_path = folder / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    port_override = os.environ.get("SGLANG_OMNI_DFLASH_TEST_PORT")
    port = int(port_override) if port_override else find_available_port()
    env = {
        "PYTHONPATH": os.pathsep.join((str(probe), str(PROJECT_ROOT))),
        "OMNI_DFLASH_TEST_TRACE_DIR": str(trace),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    (folder / "launch.json").write_text(
        json.dumps({"command": cmd, "environment": env, "tp": tp, "enabled": enabled})
    )
    proc = start_server_from_cmd(
        cmd, folder / "server.log", port, timeout=900, env=env, strip_proxy=True
    )
    server = SimpleNamespace(
        proc=proc,
        url=f"http://127.0.0.1:{port}/v1/chat/completions",
        folder=folder,
        trace=trace,
        tp=tp,
        enabled=enabled,
        model=config["model_path"],
    )
    try:
        yield server
    finally:
        # note(wenyao): Release our probe barrier and stop only our process group.
        (trace / "resume_request").touch()
        stop_server(proc)


def _payload(server, request_id, prompt, *, max_tokens=128):
    return {
        "model": server.model,
        "request_id": request_id,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "Ethan", "format": "pcm"},
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 123456,
        "talker_temperature": 0.9,
        "talker_top_k": 50,
        "talker_top_p": 1.0,
        "talker_max_new_tokens": 2048,
    }


def _stream(server, request_id, prompt):
    payload = _payload(server, request_id, prompt)
    text, pcm, chunks, reasons = [], bytearray(), [], []
    done = False
    with requests.Session() as session:
        session.trust_env = False
        with session.post(
            server.url, json=payload, stream=True, timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                assert chunk["id"] == f"chatcmpl-{request_id}"
                chunks.append(chunk)
                for choice in chunk["choices"]:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        text.append(delta["content"])
                    if delta.get("audio", {}).get("data"):
                        audio = base64.b64decode(delta["audio"]["data"], validate=True)
                        assert audio and len(audio) % 2 == 0
                        pcm.extend(audio)
                    if choice.get("finish_reason"):
                        reasons.append(choice["finish_reason"])
    (server.folder / f"{request_id}.json").write_text(
        json.dumps({"request": payload, "chunks": chunks, "text": "".join(text)})
    )
    with wave.open(str(server.folder / f"{request_id}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    assert done and reasons == ["stop"], reasons
    assert "".join(text).strip() and pcm, "missing streamed text or speech"
    return request_id


def _assert_completed(server, request_id):
    def complete(rows):
        terminals = [
            row
            for row in rows
            if row["event"] == "terminal" and row["request_id"] == request_id
        ]
        return len(terminals) >= server.tp + 1

    rows = _wait_events(server, complete)
    terminals = [
        row
        for row in rows
        if row["event"] == "terminal" and row["request_id"] == request_id
    ]
    thinkers = [row for row in terminals if row["stage"] == "thinker"]
    talkers = [row for row in terminals if row["stage"] == "talker_ar"]
    assert sorted(row["tp_rank"] for row in thinkers) == list(range(server.tp))
    assert len(talkers) == 1
    for row in terminals:
        assert row["ids"], row
        reason = row["finished_reason"]
        eos = set(row["eos_token_ids"])
        tokenizer_eos = row["tokenizer_eos_token_id"]
        if isinstance(tokenizer_eos, int):
            eos.add(tokenizer_eos)
        elif tokenizer_eos is not None:
            eos.update(tokenizer_eos)
        assert not row["aborted"], row
        assert reason["type"] == "stop" and row["ids"][-1] in eos, row
        assert reason["matched"] == row["ids"][-1], row
    assert all(row["ids"] == thinkers[0]["ids"] for row in thinkers)
    for rank in range(server.tp):
        for target in ("decode", "talker_ar"):
            tokens = [
                row["token_id"]
                for row in rows
                if row["event"] == "stream"
                and row["request_id"] == request_id
                and row["stage"] == "thinker"
                and row["tp_rank"] == rank
                and row["target"] == target
            ]
            assert tokens == thinkers[0]["ids"], (rank, target, tokens, thinkers)
    entry = next(row for row in thinkers if row["tp_rank"] == 0)
    if server.enabled:
        assert entry["spec_verify_ct"] > 0, entry
    return entry


def test_native_streaming_and_eos(dflash_server):
    server = dflash_server
    records = []
    for repeat in range(2):
        for concurrency in (1, 2):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(
                        _stream,
                        server,
                        f"smoke-r{repeat}-c{concurrency}-p{index}",
                        prompt,
                    )
                    for index, prompt in enumerate(PROMPTS)
                ]
                records.extend(
                    _assert_completed(server, future.result()) for future in futures
                )
    if server.enabled:
        assert sum(row["spec_num_correct_drafts"] for row in records) > 0, records
    (server.folder / "streaming-result.json").write_text(json.dumps(records, indent=2))


@pytest.mark.parametrize("phase", ["verification", "chunked_prefill"])
def test_cancel_during_model_work_then_reuse(dflash_server, phase):
    server = dflash_server
    if phase == "verification" and not server.enabled:
        pytest.skip("the in-flight verification barrier requires DFlash")
    request_id = f"cancel-{uuid.uuid4().hex}"
    (server.trace / "resume_request").unlink(missing_ok=True)
    if phase == "verification":
        marker, event = "pause_request", "verify_paused"
        prompt = "Count from one to five hundred, writing every number in words."
    else:
        marker, event = "pause_chunked_request", "chunked_paused"
        prompt = (
            "Read this background silently: "
            + "The morning train crossed the bridge beside the river. " * 160
            + "Now say exactly: Hello, world."
        )
    (server.trace / marker).write_text(request_id)
    response = None
    with requests.Session() as session:
        session.trust_env = False
        try:
            response = session.post(
                server.url,
                json=_payload(
                    server,
                    request_id,
                    prompt,
                    max_tokens=1024,
                ),
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            rows = _wait_events(
                server,
                lambda rows: len(
                    {
                        row["tp_rank"]
                        for row in rows
                        if row["event"] == event and row["request_id"] == request_id
                    }
                )
                == server.tp,
            )
            assert not any(
                row["event"] == "terminal" and row["request_id"] == request_id
                for row in rows
            ), "cancellation must happen before native completion"
            paused_pids = {
                row["pid"]
                for row in rows
                if row["event"] == event and row["request_id"] == request_id
            }
            if phase == "chunked_prefill":
                assert all(
                    row["req_pool_idx"] is not None
                    and row["prefix_tokens"] >= 512
                    and not row["batch_aliases"]
                    for row in rows
                    if row["event"] == event and row["request_id"] == request_id
                )
            response.close()
            _wait_events(
                server,
                lambda rows: {
                    row["pid"]
                    for row in rows
                    if row["event"] == "abort"
                    and row["request_id"] == request_id
                    and row["stage"] == "thinker"
                    and row["found"]
                }
                == paused_pids,
                timeout=15,
            )
        finally:
            if response is not None:
                response.close()
            (server.trace / "resume_request").touch()
    _wait_events(
        server,
        lambda rows: len(
            {
                row["pid"]
                for row in rows
                if row["event"] == "release"
                and row["request_id"] == request_id
                and row["pid"] in paused_pids
                and row["req_pool_idx_before"] is not None
                and row["req_pool_idx_after"] is None
            }
        )
        == server.tp,
    )
    _assert_completed(server, _stream(server, f"after-{phase}", PROMPTS[1]))
    rows = _events(server.trace)
    abort_times = {}
    for row in rows:
        if (
            row["event"] == "abort"
            and row["request_id"] == request_id
            and row["pid"] in paused_pids
        ):
            abort_times.setdefault(row["pid"], row["monotonic_ns"])
    assert not any(
        row["event"] == "stream"
        and row["request_id"] == request_id
        and row["pid"] in abort_times
        and row["monotonic_ns"] > abort_times[row["pid"]]
        for row in rows
    ), "Thinker emitted tokens after cancellation"
