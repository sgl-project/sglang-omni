# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR NPU serving and transcript regressions using local audio fixtures."""

from __future__ import annotations

import json
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNI_RUN_NPU_TESTS") != "1",
    reason="Set OMNI_RUN_NPU_TESTS=1 with a reserved NPU and local model",
)


def _normalize(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def _load_cases(path: Path) -> dict:
    cases = json.loads(path.read_text())
    assert set(cases) == {"English", "Chinese"}, "Provide both language fixtures"
    for case in cases.values():
        assert _normalize(case["text"]), "A reference transcript is required"
        # No permissive default: calibrate the ceiling against fixed audio.
        assert isinstance(case["max_cer"], (int, float))
        assert 0 <= case["max_cer"] < 1, "Set a calibrated CER ceiling below 1"
        audio = (path.parent / case["audio"]).resolve()
        assert audio.is_file(), f"Missing fixture: {audio}"
        case["audio"] = str(audio)
    return cases


def _stream_text(lines) -> str:
    deltas = []
    final = None
    ended = False
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        assert not ended, "Unexpected event after [DONE]"
        data = line[5:].strip()
        if data == "[DONE]":
            ended = True
            continue
        event = json.loads(data)
        if event["type"] == "transcript.text.delta":
            assert final is None, "Delta after final transcript"
            deltas.append(event["delta"])
        elif event["type"] == "transcript.text.done":
            assert final is None, "Duplicate final transcript"
            final = event["text"]
        else:
            raise AssertionError(f"Unexpected transcription event: {event}")
    assert ended and final is not None, "Incomplete transcription stream"
    if deltas:
        assert "".join(deltas) == final, "Deltas differ from final transcript"
    return final


@pytest.fixture(scope="module")
def asr_server(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    from benchmarks.benchmarker.utils import managed_omni_server
    from sglang_omni.platforms import current_platform
    from tests.test_model.omni_router_utils import _find_available_port_range

    assert current_platform.is_npu() and torch.npu.is_available()
    assert torch.npu.device_count() == 1, "Reserve one visible NPU"
    model = Path(os.environ["OMNI_NPU_ASR_MODEL"]).resolve()
    config = Path(os.environ["OMNI_NPU_ASR_CONFIG"]).resolve()
    assert (model / "config.json").is_file()
    assert config.is_file()
    cases = _load_cases(Path(os.environ["OMNI_NPU_ASR_CASES"]))
    output = Path(
        os.environ.get("OMNI_NPU_ASR_OUTPUT", str(tmp_path_factory.mktemp("npu-asr")))
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    port = _find_available_port_range(1)
    base_url = f"http://127.0.0.1:{port}"
    with managed_omni_server(
        model_path=str(model),
        server_config=str(config),
        port=port,
        host="127.0.0.1",
        log_file=output / "server.log",
        wait_for_gpu_release=False,
    ):
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(base_url + "/v1/models", timeout=10)
            response.raise_for_status()
            model_id = response.json()["data"][0]["id"]
        yield base_url, model_id, cases, output


def _request(server, language, name, stream=False):
    from jiwer import cer

    base_url, model_id, cases, output = server
    case = cases[language]
    start = time.monotonic()
    with requests.Session() as session, open(case["audio"], "rb") as audio:
        session.trust_env = False
        with session.post(
            base_url + "/v1/audio/transcriptions",
            files={"file": (Path(case["audio"]).name, audio, "audio/wav")},
            data={
                "model": model_id,
                "language": language,
                "response_format": "json",
                "stream": str(stream).lower(),
            },
            stream=stream,
            timeout=(10, 300),
        ) as response:
            assert response.status_code == 200, response.text[:2000]
            if stream:
                assert "text/event-stream" in response.headers["content-type"]
                # Persist events before asserting completeness, including failures.
                lines = list(response.iter_lines())
                (output / f"{name}.sse").write_bytes(b"\n".join(lines))
                text = _stream_text(lines)
            else:
                result = response.json()
                (output / f"{name}-response.json").write_text(json.dumps(result))
                text = result["text"]
    actual = _normalize(text)
    error_rate = cer(_normalize(case["text"]), actual)
    (output / f"{name}.json").write_text(
        json.dumps(
            {"text": text, "cer": error_rate, "elapsed_s": time.monotonic() - start},
            ensure_ascii=False,
            indent=2,
        )
    )
    assert actual, "Empty transcript"
    assert error_rate <= case["max_cer"], f"CER {error_rate}: {text!r}"


@pytest.mark.parametrize("language", ["English", "Chinese"])
def test_transcription(asr_server, language):
    _request(asr_server, language, language.lower())


def test_stream_transcription(asr_server):
    _request(asr_server, "English", "stream", stream=True)


def test_concurrent_transcriptions(asr_server):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_request, asr_server, language, f"concurrent-{language}")
            for language in ("English", "Chinese")
        ]
        for future in futures:
            future.result()


def test_transcription_after_rejection(asr_server):
    base_url, model_id, _, _ = asr_server
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            base_url + "/v1/audio/transcriptions",
            data={"model": model_id},
            timeout=10,
        )
        assert response.status_code in {400, 422}, response.text
    _request(asr_server, "English", "after-rejection")
