# SPDX-License-Identifier: Apache-2.0
"""Image generation thinker bootstrap plumbing tests."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _install_fake_sglang_request_modules(monkeypatch, *, reject_zero: bool = False):
    attempts: list[int] = []

    class FakeReq:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.output_ids = []

    class FakeSamplingParams:
        def __init__(self, *, max_new_tokens, temperature, **kwargs):
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            for key, value in kwargs.items():
                setattr(self, key, value)

        def normalize(self, tokenizer):
            self.tokenizer = tokenizer

        def verify(self, vocab_size):
            attempts.append(self.max_new_tokens)
            if reject_zero and self.max_new_tokens == 0:
                raise ValueError("max_new_tokens must be positive")
            self.vocab_size = vocab_size

    schedule_batch_module = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch_module.Req = FakeReq
    sampling_params_module = ModuleType("sglang.srt.sampling.sampling_params")
    sampling_params_module.SamplingParams = FakeSamplingParams

    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers", ModuleType("sglang.srt.managers")
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.schedule_batch",
        schedule_batch_module,
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.sampling", ModuleType("sglang.srt.sampling")
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.sampling.sampling_params",
        sampling_params_module,
    )

    @dataclass
    class FakeSGLangARRequestData:
        input_ids: Any = None
        attention_mask: Any = None
        model_inputs: dict[str, Any] = field(default_factory=dict)
        capture_model_output_keys: tuple[str, ...] = ()
        max_new_tokens: int | None = None
        temperature: float = 0.0
        output_ids: list[int] = field(default_factory=list)
        req: Any = None
        stage_payload: Any = None

    sglang_backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")
    sglang_backend_module.SGLangARRequestData = FakeSGLangARRequestData
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        sglang_backend_module,
    )
    return attempts


def _build_image_gen_payload(*, params=None, capture_keys=("hidden_states",)):
    torch = pytest.importorskip("torch")

    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.proto import OmniRequest, StagePayload

    state = MingOmniPipelineState(
        prompt={
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
        mm_inputs={"image_gen": {"prefill_only": True}},
        thinker_inputs={
            "capture_model_output_keys": capture_keys,
            "model_inputs": {"image_embeds": torch.ones(1, 2)},
        },
    )
    return StagePayload(
        request_id="image-gen-prefill",
        request=OmniRequest(inputs={}, params=dict(params or {})),
        data=state.to_dict(),
    )


def _request_builder():
    from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters

    tokenizer = SimpleNamespace(
        vocab_size=32000,
        eos_token_id=2,
        unk_token_id=0,
    )
    builder, _ = make_thinker_scheduler_adapters(tokenizer=tokenizer, vocab_size=32000)
    return builder


def test_request_builder_marks_image_gen_prefill_hidden_capture(monkeypatch) -> None:
    _install_fake_sglang_request_modules(monkeypatch)
    payload = _build_image_gen_payload()

    req_data = _request_builder()(payload)

    assert req_data.max_new_tokens == 0
    assert req_data.req.sampling_params.max_new_tokens == 0
    assert req_data.req._omni_prefill_only is True
    assert req_data.req._omni_capture_hidden is True


def test_request_builder_retries_prefill_zero_rejection_with_one_token(
    monkeypatch,
) -> None:
    attempts = _install_fake_sglang_request_modules(monkeypatch, reject_zero=True)
    payload = _build_image_gen_payload()

    req_data = _request_builder()(payload)

    assert attempts == [0, 1]
    assert req_data.max_new_tokens == 1
    assert req_data.req.sampling_params.max_new_tokens == 1
    assert req_data.req._omni_prefill_only is True
    assert req_data.req._omni_capture_hidden is True


def _load_runner_with_fake_sglang(monkeypatch):
    scheduler_module = ModuleType("sglang.srt.managers.scheduler")

    class GenerationBatchResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    scheduler_module.GenerationBatchResult = GenerationBatchResult

    forward_batch_module = ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch_module.CaptureHiddenMode = SimpleNamespace(NULL="null", LAST="last")

    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.managers", ModuleType("sglang.srt.managers")
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler",
        scheduler_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor",
        ModuleType("sglang.srt.model_executor"),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.forward_batch_info",
        forward_batch_module,
    )

    module_name = "sglang_omni.model_runner.ming_thinker_model_runner"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    _purge_cached_module(module_name, module)
    return module


def _purge_cached_module(module_name: str, module: ModuleType) -> None:
    sys.modules.pop(module_name, None)
    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if isinstance(parent, ModuleType) and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


def test_ming_thinker_runner_capture_modes_for_image_gen_prefill(monkeypatch) -> None:
    module = _load_runner_with_fake_sglang(monkeypatch)
    runner = module.MingThinkerModelRunner.__new__(module.MingThinkerModelRunner)

    capture_req = SimpleNamespace(_omni_capture_hidden=True)
    normal_req = SimpleNamespace(_omni_capture_hidden=False)

    assert (
        runner.requested_capture_hidden_mode_prefill(
            SimpleNamespace(reqs=[normal_req, capture_req]), []
        )
        == "null"
    )
    assert (
        runner.requested_capture_hidden_mode_prefill(
            SimpleNamespace(reqs=[normal_req]), []
        )
        == "null"
    )
    assert (
        runner.requested_capture_hidden_mode_decode(
            SimpleNamespace(reqs=[capture_req]), []
        )
        == "null"
    )


def test_forward_with_omni_embeds_attaches_hidden_states_when_requested(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner_with_fake_sglang(monkeypatch)
    runner = module.MingThinkerModelRunner.__new__(module.MingThinkerModelRunner)

    hidden_states = torch.ones(2, 4)
    truncated_hidden_states = torch.zeros(1, 4)
    logits_output = SimpleNamespace(hidden_states=truncated_hidden_states)

    class FakeAttnBackend:
        def init_forward_metadata(self, forward_batch):
            self.forward_batch = forward_batch

    class FakeOuter:
        lm_head = object()

        def model(self, **kwargs):
            self.model_kwargs = kwargs
            return hidden_states

        def logits_processor(self, *args):
            self.logits_args = args
            return logits_output

    fake_outer = FakeOuter()
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=FakeAttnBackend())
    )
    runner._outer_model = fake_outer
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        mrope_positions=None,
    )

    result = runner._forward_with_omni_embeds(
        forward_batch,
        torch.zeros(2, 4),
        capture_hidden=True,
    )

    assert logits_output.hidden_states is hidden_states
    assert result.logits_output is logits_output
    assert result.can_run_cuda_graph is False


def test_result_adapter_hides_fallback_token_for_prefill_only_request() -> None:
    from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters
    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.proto import OmniRequest, StagePayload

    tokenizer = SimpleNamespace(eos_token_id=2)
    _, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer, vocab_size=32000
    )
    payload = StagePayload(
        request_id="image-gen-prefill",
        request=OmniRequest(inputs={}, params={}),
        data=MingOmniPipelineState().to_dict(),
    )
    data = SimpleNamespace(
        stage_payload=payload,
        output_ids=[123],
        finish_reason=None,
        extra_model_outputs={"hidden_states": "kept"},
        req=SimpleNamespace(_omni_prefill_only=True),
    )

    result = result_adapter(data)
    state = MingOmniPipelineState.from_dict(result.data)

    assert state.thinker_out["output_ids"] == []
    assert state.thinker_out["extra_model_outputs"] == {"hidden_states": "kept"}


def _build_image_gen_payload_with_patch_tokens(*, request_id="image-gen-prefill"):
    torch = pytest.importorskip("torch")

    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.proto import OmniRequest, StagePayload

    state = MingOmniPipelineState(
        prompt={
            "input_ids": torch.tensor([11, 157157, 157157, 12], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
        },
        mm_inputs={
            "image_gen": {
                "prefill_only": True,
                "image_patch_token_id": 157157,
                "gen_mask": [0, 1, 1, 0],
            }
        },
        thinker_inputs={
            "capture_model_output_keys": ("hidden_states",),
            "model_inputs": {"image_embeds": torch.ones(2, 3)},
        },
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )


def test_request_builder_substitutes_query_patch_tokens_per_request(
    monkeypatch,
) -> None:
    _install_fake_sglang_request_modules(monkeypatch)

    req_data = _request_builder()(_build_image_gen_payload_with_patch_tokens())

    ids = list(req_data.req.origin_input_ids)
    assert 157157 not in ids
    pad = req_data.model_inputs["pad_values"]["image_gen"]
    assert pad >= 32000
    assert ids == [11, pad, pad, 12]
    tensor_ids = req_data.input_ids.tolist()
    assert tensor_ids == [11, pad, pad, 12]


def test_request_builder_query_pad_is_unique_per_request_id(monkeypatch) -> None:
    _install_fake_sglang_request_modules(monkeypatch)
    builder = _request_builder()

    pad_a = builder(
        _build_image_gen_payload_with_patch_tokens(request_id="req-a")
    ).model_inputs["pad_values"]["image_gen"]
    pad_b = builder(
        _build_image_gen_payload_with_patch_tokens(request_id="req-b")
    ).model_inputs["pad_values"]["image_gen"]

    assert pad_a != pad_b


def test_request_builder_without_patch_token_id_keeps_input_ids(monkeypatch) -> None:
    _install_fake_sglang_request_modules(monkeypatch)

    req_data = _request_builder()(_build_image_gen_payload())

    assert list(req_data.req.origin_input_ids) == [11, 12, 13]
    assert "pad_values" not in req_data.model_inputs


def test_request_builder_gen_pad_only_covers_gen_mask_positions(monkeypatch) -> None:
    _install_fake_sglang_request_modules(monkeypatch)
    torch = pytest.importorskip("torch")

    from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters
    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.proto import OmniRequest, StagePayload

    state = MingOmniPipelineState(
        prompt={
            "input_ids": torch.tensor([11, 157157, 157157, 12], dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
        },
        mm_inputs={
            "image_gen": {
                "prefill_only": True,
                "image_patch_token_id": 157157,
                "gen_mask": [0, 0, 1, 0],
            }
        },
        thinker_inputs={
            "capture_model_output_keys": ("hidden_states",),
            "model_inputs": {"image_gen_embeds": torch.ones(1, 3)},
        },
    )
    payload = StagePayload(
        request_id="req-mask",
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )
    builder, _ = make_thinker_scheduler_adapters(
        tokenizer=SimpleNamespace(
            vocab_size=32000,
            eos_token_id=2,
            unk_token_id=0,
        ),
        vocab_size=32000,
    )

    req_data = builder(payload)
    ids = list(req_data.req.origin_input_ids)
    pad = req_data.model_inputs["pad_values"]["image_gen"]

    assert ids == [11, 157157, pad, 12]


def test_runner_injects_image_and_image_gen_at_disjoint_positions(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner_with_fake_sglang(monkeypatch)
    runner = module.MingThinkerModelRunner.__new__(module.MingThinkerModelRunner)

    hidden = 4
    runner._image_token_id = 157157
    runner._video_token_id = None
    runner._audio_token_id = None

    class FakeEmbed:
        num_embeddings = 100000

        def __call__(self, ids):
            return torch.zeros(ids.shape[0], hidden)

    runner._embed_tokens = FakeEmbed()

    image_pad = 50000
    gen_pad = 60000
    input_ids = torch.tensor([11, image_pad, image_pad, gen_pad, gen_pad, 12])
    image_embeds = torch.ones(2, hidden)
    gen_embeds = torch.full((2, hidden), 7.0)

    req = SimpleNamespace(
        omni_model_inputs={
            "image_embeds": image_embeds,
            "image_gen_embeds": gen_embeds,
            "pad_values": {"image": image_pad, "image_gen": gen_pad},
        },
        is_chunked=0,
        rid="r",
    )
    forward_batch = SimpleNamespace(
        input_ids=input_ids,
        extend_seq_lens_cpu=[input_ids.shape[0]],
    )
    schedule_batch = SimpleNamespace(reqs=[req])

    out = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(out[1], image_embeds[0])
    assert torch.equal(out[2], image_embeds[1])
    assert torch.equal(out[3], gen_embeds[0])
    assert torch.equal(out[4], gen_embeds[1])
    assert torch.equal(out[0], torch.zeros(hidden))
    assert torch.equal(out[5], torch.zeros(hidden))


def test_runner_image_static_fallback_disjoint_from_gen_positions(monkeypatch) -> None:
    """Lock the static-fallback disjointness invariant.

    When pad_values contains "image_gen" but NOT "image", _resolve_match_id
    returns self._image_token_id (157157) for the "image" modality and the
    explicit gen_pad for "image_gen". The two match IDs are always distinct
    (157157 vs gen_pad which is a scheduler-allocated value outside the vocab),
    so image_embeds land on the 157157 positions and image_gen_embeds land on
    the gen_pad positions with no overlap.
    """
    torch = pytest.importorskip("torch")
    module = _load_runner_with_fake_sglang(monkeypatch)
    runner = module.MingThinkerModelRunner.__new__(module.MingThinkerModelRunner)

    hidden = 4
    runner._image_token_id = 157157
    runner._video_token_id = None
    runner._audio_token_id = None

    class FakeEmbed:
        num_embeddings = 200000

        def __call__(self, ids):
            return torch.zeros(ids.shape[0], hidden)

    runner._embed_tokens = FakeEmbed()

    gen_pad = 70000
    input_ids = torch.tensor([11, 157157, 157157, gen_pad, gen_pad, 12])
    image_embeds = torch.ones(2, hidden) * 3.0
    gen_embeds = torch.full((2, hidden), 9.0)

    req = SimpleNamespace(
        omni_model_inputs={
            "image_embeds": image_embeds,
            "image_gen_embeds": gen_embeds,
            "pad_values": {"image_gen": gen_pad},
        },
        is_chunked=0,
        rid="r2",
    )
    forward_batch = SimpleNamespace(
        input_ids=input_ids,
        extend_seq_lens_cpu=[input_ids.shape[0]],
    )
    schedule_batch = SimpleNamespace(reqs=[req])

    out = runner._inject_multimodal_embeds(forward_batch, schedule_batch)

    assert torch.equal(out[1], image_embeds[0])
    assert torch.equal(out[2], image_embeds[1])
    assert torch.equal(out[3], gen_embeds[0])
    assert torch.equal(out[4], gen_embeds[1])
    assert not torch.equal(out[1], gen_embeds[0])
    assert not torch.equal(out[3], image_embeds[0])
    assert torch.equal(out[0], torch.zeros(hidden))
    assert torch.equal(out[5], torch.zeros(hidden))
