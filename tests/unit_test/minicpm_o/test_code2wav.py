import base64
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import (
    code2wav_reference_audio,
    project_talker_to_code2wav,
    project_thinker_to_talker,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.pipeline.helpers import run_scheduler


def _data_uri(audio):
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")


def _payload(*, request_id="test", params=None, metadata=None):
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params=params or {}, metadata=metadata or {}),
        data=MiniCPMOPipelineState(
            engine_outputs={"talker": {"codec_tokens": torch.tensor([1, 2])}}
        ).to_dict(),
    )


@pytest.mark.parametrize("field", ["ref_audio", "prompt_wav"])
@pytest.mark.parametrize("source", ["stage", "audio_config", "tts_params", "params"])
def test_reference_survives_projection(field, source):
    config = {field: _data_uri(b"reference")}
    params = {}
    metadata = {}
    if source == "stage":
        params = {"stage_params": {"code2wav": config}}
    elif source == "params":
        params = config
    else:
        metadata[source] = config
    payload = _payload(params=params, metadata=metadata)
    payload = project_thinker_to_talker(payload)
    payload = project_talker_to_code2wav(payload)
    payload = StagePayload.from_dict(payload.to_dict())
    assert code2wav_reference_audio(payload) == b"reference"


def test_explicit_reference_precedence_and_default():
    payload = _payload(
        params={"stage_params": {"code2wav": {"ref_audio": _data_uri(b"stage")}}},
        metadata={"audio_config": {"ref_audio": _data_uri(b"audio")}},
    )
    assert code2wav_reference_audio(payload) == b"stage"
    payload.request.params = {}
    assert code2wav_reference_audio(payload) == b"audio"
    payload.request.metadata = {}
    payload.request.inputs = {"audios": [_data_uri(b"understanding input")]}
    assert code2wav_reference_audio(payload) is None


def test_reference_accepts_inline_audio_descriptor():
    reference = {"data": base64.b64encode(b"reference").decode("ascii")}
    payload = _payload(metadata={"audio_config": {"ref_audio": reference}})
    assert code2wav_reference_audio(payload) == b"reference"


@pytest.mark.parametrize("field", ["audio", "stage_params"])
def test_chat_api_forwards_reference_to_vocoder(field):
    from sglang_omni.client.client import _build_params
    from sglang_omni.serve.openai_api import (
        ChatCompletionRequest,
        _build_chat_generate_request,
    )

    reference = _data_uri(b"reference")
    fields = (
        {"audio": {"format": "wav", "ref_audio": reference}}
        if field == "audio"
        else {"stage_params": {"code2wav": {"prompt_wav": reference}}}
    )
    request = ChatCompletionRequest(
        model="minicpm-o",
        messages=[{"role": "user", "content": "Hello"}],
        modalities=["text", "audio"],
        **fields,
    )
    generate_request = _build_chat_generate_request(request)
    payload = _payload(
        params=_build_params(generate_request), metadata=generate_request.metadata
    )
    assert code2wav_reference_audio(project_talker_to_code2wav(payload)) == b"reference"


@pytest.mark.parametrize(
    "reference",
    ["", b"", "/tmp/ref.wav", "file:///tmp/ref.wav", "https://example.com/ref.wav", {}],
)
def test_invalid_reference_does_not_silently_use_default(reference):
    payload = _payload(params={"ref_audio": reference})
    with pytest.raises(ValueError, match="inline audio"):
        code2wav_reference_audio(payload)


def test_invalid_base64_reference_is_rejected():
    payload = _payload(params={"ref_audio": "data:audio/wav;base64,%%%"})
    with pytest.raises(ValueError, match="Invalid base64"):
        code2wav_reference_audio(payload)


def _model(prepare_prompt):
    model = MiniCPMOCode2Wav.__new__(MiniCPMOCode2Wav)
    torch.nn.Module.__init__(model)
    model.token2wav = SimpleNamespace(cache=None, _prepare_prompt=prepare_prompt)
    model._prompt_cache_key = None
    model._prompt_wav = None
    model._device_ctx = nullcontext()
    return model


def test_prompt_cache_switches_references_and_reuses_only_current(tmp_path):
    prepared = []

    def prepare_prompt(path):
        content = Path(path).read_bytes()
        prepared.append(content)
        return (content,)

    model = _model(prepare_prompt)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    for path, expected in [
        (first, b"first"),
        (first, b"first"),
        (second, b"second"),
        (first, b"first"),
    ]:
        assert model._get_prompt(str(path)) == (expected,)
    assert prepared == [b"first", b"second", b"first"]
    first.write_bytes(b"other")
    assert model._get_prompt(str(first)) == (b"other",)
    assert prepared[-1] == b"other"


def test_inline_prompt_cache_cleans_tempfiles_and_preserves_cache_on_failure():
    paths = []

    def prepare_prompt(path):
        paths.append(Path(path))
        content = Path(path).read_bytes()
        if content == b"invalid":
            raise ValueError("invalid audio")
        return (content,)

    model = _model(prepare_prompt)
    assert model._get_prompt(b"first") == (b"first",)
    assert model._get_prompt(b"first") == (b"first",)
    assert len(paths) == 1
    original_key = model._prompt_cache_key
    with pytest.raises(ValueError, match="invalid audio"):
        model._get_prompt(b"invalid")
    assert model._prompt_cache_key == original_key
    assert model.token2wav.cache == (b"first",)
    assert model._get_prompt(b"second") == (b"second",)
    assert all(not path.exists() for path in paths)


def test_forward_restores_default_after_custom_reference(tmp_path, monkeypatch):
    model = _model(lambda path: (Path(path).read_bytes(),))
    default = tmp_path / "default.wav"
    default.write_bytes(b"default")
    model._prompt_wav = str(default)
    references = []

    def vocode(tokens, reference):
        references.append(model._get_prompt(reference)[0])
        return np.ones(len(tokens), dtype=np.float32)

    monkeypatch.setattr(model, "_vocode", vocode)
    for reference in (None, b"custom", b"custom", None):
        output = model(codec_tokens=torch.tensor([1, 2]), prompt_wav=reference)
        assert output["sample_rate"] == 24000
    assert references == [b"default", b"custom", b"custom", b"default"]
    empty = model(codec_tokens=torch.tensor([]), prompt_wav=b"unused")
    assert empty["waveform"].size == 0
    assert len(references) == 4


def test_missing_default_reference_has_explicit_error():
    model = _model(lambda path: pytest.fail("should not prepare missing reference"))
    with pytest.raises(ValueError, match="No speaker-reference"):
        model._get_prompt(None)


def test_vocoder_passes_each_requests_reference(monkeypatch):
    from sglang_omni.models.minicpm_o.components import code2wav
    from sglang_omni.models.minicpm_o.stages import create_code2wav_executor

    received = []

    def vocode(**model_inputs):
        received.append(model_inputs)
        return {"waveform": np.zeros(24, dtype=np.float32), "sample_rate": 24000}

    monkeypatch.setattr(code2wav, "MiniCPMOCode2Wav", lambda *args, **kwargs: vocode)
    scheduler = create_code2wav_executor("model", device="cpu")
    payloads = [
        _payload(
            request_id="first",
            metadata={"audio_config": {"ref_audio": _data_uri(b"first")}},
        ),
        _payload(
            request_id="second",
            params={"stage_params": {"code2wav": {"prompt_wav": b"second"}}},
        ),
        _payload(request_id="default"),
    ]
    for payload in payloads:
        scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    outputs = run_scheduler(scheduler, [], output_count=len(payloads))
    assert [item["prompt_wav"] for item in received] == [b"first", b"second", None]
    assert all(item["codec_tokens"].tolist() == [1, 2] for item in received)
    assert [output.request_id for output in outputs] == ["first", "second", "default"]
    assert all(output.type == "result" for output in outputs)
    assert all(output.data.data["sample_rate"] == 24000 for output in outputs)
    assert all("engine_outputs" not in output.data.data for output in outputs)


def test_vocoder_returns_completed_request_before_next_finishes(monkeypatch):
    from sglang_omni.models.minicpm_o.components import code2wav
    from sglang_omni.models.minicpm_o.stages import create_code2wav_executor

    second_started = threading.Event()
    release_second = threading.Event()

    def vocode(**model_inputs):
        if model_inputs["prompt_wav"] == b"second":
            second_started.set()
            assert release_second.wait(timeout=5.0)
        return {"waveform": np.zeros(24, dtype=np.float32), "sample_rate": 24000}

    monkeypatch.setattr(code2wav, "MiniCPMOCode2Wav", lambda *args, **kwargs: vocode)
    scheduler = create_code2wav_executor("model", device="cpu")
    for request_id in ("first", "second"):
        payload = _payload(
            request_id=request_id, params={"ref_audio": request_id.encode()}
        )
        scheduler.inbox.put(IncomingMessage(request_id, "new_request", payload))

    def collect_first():
        try:
            assert second_started.wait(timeout=2.0)
            output = scheduler.outbox.get_nowait()
            assert output.request_id == "first"
            assert output.type == "result"
        finally:
            release_second.set()

    outputs = run_scheduler(scheduler, [], output_count=1, before_collect=collect_first)
    assert outputs[0].request_id == "second"
    assert outputs[0].type == "result"


@pytest.mark.parametrize("failure", ["reference", "vocode"])
@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_vocoder_failure_is_request_local(monkeypatch, failure, failed_index):
    from sglang_omni.models.minicpm_o.components import code2wav
    from sglang_omni.models.minicpm_o.stages import create_code2wav_executor

    def vocode(**model_inputs):
        if model_inputs["prompt_wav"] == b"invalid":
            raise RuntimeError("vocode failed")
        return {"waveform": np.zeros(24, dtype=np.float32), "sample_rate": 24000}

    monkeypatch.setattr(code2wav, "MiniCPMOCode2Wav", lambda *args, **kwargs: vocode)
    scheduler = create_code2wav_executor("model", device="cpu")
    for request_index in range(3):
        params = {}
        if request_index == failed_index:
            params["ref_audio"] = (
                "/tmp/ref.wav" if failure == "reference" else b"invalid"
            )
        payload = _payload(request_id=f"req-{request_index}", params=params)
        scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))

    outputs = run_scheduler(scheduler, [], output_count=3)
    assert [output.request_id for output in outputs] == ["req-0", "req-1", "req-2"]
    for request_index, output in enumerate(outputs):
        if request_index == failed_index:
            assert output.type == "error"
            error_type = ValueError if failure == "reference" else RuntimeError
            assert isinstance(output.data, error_type)
        else:
            assert output.type == "result"
            assert output.data.data["sample_rate"] == 24000
    assert scheduler.outbox.empty()
