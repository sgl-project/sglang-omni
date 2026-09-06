# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.breeze_tts.depth_decoder import BreezeDepthDecoder
from sglang_omni.models.breeze_tts.engine_builder import BreezeEngineBuilder
from sglang_omni.models.breeze_tts.hf_config import BreezeConfig
from sglang_omni.models.breeze_tts.model_runner import BreezeModelRunner
from sglang_omni.models.breeze_tts.request_builders import (
    CFG_SUFFIX,
    apply_result,
    build_request,
    stream_output,
)
from sglang_omni.models.breeze_tts.sampling import SamplingConfig
from sglang_omni.models.breeze_tts.scheduler import BreezeScheduler
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.fixture(autouse=True)
def cpu_runner_device(monkeypatch):
    monkeypatch.setattr(
        "sglang_omni.model_runner.base.current_platform.get_device",
        lambda _: torch.device("cpu"),
    )


def make_model(config):
    depth = BreezeDepthDecoder(config["depth_decoder_config"]).eval()
    torch.nn.init.normal_(depth.codebooks_head.weight, std=0.1)
    return SimpleNamespace(
        config=BreezeConfig(**config),
        depth_decoder=depth,
        lm_head=torch.nn.Linear(16, 12),
        model=SimpleNamespace(norm=torch.nn.LayerNorm(16)),
    )


def make_request(model, rid="request", seed=42):
    payload = StagePayload(
        request_id=rid,
        request=OmniRequest(inputs="Hello", params={"stream": True}),
        data={
            "prompt_embeds": torch.randn(5, 16),
            "negative_embeds": torch.randn(3, 16),
            "sampling": asdict(
                SamplingConfig(temperature=0, max_new_tokens=3, seed=seed)
            ),
        },
    )
    return build_request(payload, model)


def test_branch_ownership_prefill_and_feedback(tiny_config):
    model = make_model(tiny_config)
    data = make_request(model)
    twin = data.cfg_uncond
    assert twin.req.rid == data.req.rid + CFG_SUFFIX
    assert twin.generation is data.generation
    assert data.stage_payload.data == {}  # ownership moved out of relay payload
    assert data.req.origin_input_ids == [0] * 5
    runner = BreezeModelRunner(
        SimpleNamespace(gpu_id=0, model_runner=SimpleNamespace(model=model)), None
    )
    requests = [SimpleNamespace(data=data), SimpleNamespace(data=twin)]
    for sr in requests:
        sr.data.req.extend_range = SimpleNamespace(
            length=len(sr.data.prefill_input_embeds)
        )
        sr.data.req.prefix_indices = []
    fb = SimpleNamespace(
        replace_embeds=None, input_ids=torch.zeros(8, dtype=torch.long)
    )
    runner.before_prefill(fb, None, requests)
    expected = torch.cat([data.prefill_input_embeds, twin.prefill_input_embeds])
    torch.testing.assert_close(get_omni_prefill_inputs(fb).input_embeds, expected)
    logits = torch.zeros(2, 12)
    logits[:, 3] = 10
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            next_token_logits=logits, hidden_states=torch.randn(2, 16)
        ),
        next_token_ids=None,
    )
    runner._advance(result, requests)
    assert result.next_token_ids.tolist() == [3, 3]
    assert len(data.generation.codes) == 1
    runner.before_decode(fb, None, requests)
    torch.testing.assert_close(fb.input_embeds[0], fb.input_embeds[1])
    torch.testing.assert_close(
        fb.input_embeds[0], model.depth_decoder.embed_frames(data.generation.codes[-1])
    )
    # A companion row must not steal the primary row's pending codec frame.
    assert list(stream_output(twin.req.rid, twin, None)) == []
    chunks = list(stream_output("request", data, None))
    assert len(chunks) == 1
    assert chunks[0].data.shape == (1, 4)
    assert list(stream_output("request", data, None)) == []  # no duplicate audio
    final = apply_result(data)
    assert final.data["completion_tokens"] == 1
    assert final.data["ref_code_len"] == 0  # references condition AR, not codec prefix


def test_eos_never_reaches_depth_or_codec(tiny_config, monkeypatch):
    model = make_model(tiny_config)
    data = make_request(model)
    runner = BreezeModelRunner(
        SimpleNamespace(gpu_id=0, model_runner=SimpleNamespace(model=model)), None
    )

    def forbidden(*args, **kwargs):
        pytest.fail("EOS must not run the depth decoder")

    monkeypatch.setattr(model.depth_decoder, "decode_frame", forbidden)
    logits = torch.zeros(2, 12)
    logits[:, 11] = 100
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=logits), next_token_ids=None
    )
    runner._advance(
        result, [SimpleNamespace(data=data), SimpleNamespace(data=data.cfg_uncond)]
    )
    assert result.next_token_ids.tolist() == [11, 11]
    assert list(stream_output("request", data, None)) == []
    with pytest.raises(ValueError, match="no audio"):
        apply_result(data)


def test_requests_do_not_share_rng_feedback_or_codec_history(tiny_config):
    model = make_model(tiny_config)
    first, second = make_request(model), make_request(model, "other")
    assert first.generation is not second.generation
    assert first.generation.generator is not second.generation.generator
    first.generation.history.append(7)
    first.generation.codes.append(torch.ones(4))
    assert second.generation.history == []
    assert second.generation.codes == []
    assert second.generation.feedback is None
    with pytest.raises(RuntimeError, match="adjacent CFG pair"):
        BreezeModelRunner._generation(
            [SimpleNamespace(data=first), SimpleNamespace(data=second)]
        )


def test_nonstreaming_requests_keep_codes_without_emitting_partial_audio(tiny_config):
    data = make_request(make_model(tiny_config))
    data.stage_payload.request.params["stream"] = False
    data.generation.codes.append(torch.tensor([1, 2, 3, 4]))
    data.generation.pending_chunk = data.generation.codes[-1]
    assert list(stream_output("request", data, None)) == []
    assert data.generation.pending_chunk is None
    assert apply_result(data).data["audio_codes"].tolist() == [[1, 2, 3, 4]]


def test_late_prepared_request_cannot_enqueue_orphan_cfg_twin(tiny_config, monkeypatch):
    # The shared scheduler drops an already-aborted primary request. Even
    # though preprocessing completed, its auxiliary CFG row must also vanish.
    scheduler = object.__new__(BreezeScheduler)
    scheduler._request_admission_lock = RLock()
    scheduler.waiting_queue = []
    monkeypatch.setattr(OmniScheduler, "_enqueue_built_request", lambda *a, **k: None)
    data = make_request(make_model(tiny_config))
    scheduler._enqueue_built_request(data.stage_payload, False, data)
    assert scheduler.waiting_queue == []


def test_scheduler_admits_whole_pair_only_and_restores_waiting_queue(monkeypatch):
    scheduler = object.__new__(BreezeScheduler)
    scheduler._request_admission_lock = RLock()
    scheduler.waiting_queue = list(range(6))
    seen = []
    updated_running = SimpleNamespace(reqs=[])

    def prefill(self, running):
        seen.append(list(self.waiting_queue))
        self.waiting_queue.clear()
        return NextBatchPlan(
            batch_to_run=SimpleNamespace(reqs=["cond", "uncond"]),
            running_batch=updated_running,
        )

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", prefill)
    active = SimpleNamespace(reqs=["active"])
    held = scheduler.get_new_batch_prefill(active)
    assert held.batch_to_run is None
    assert held.running_batch is active
    assert scheduler.waiting_queue == list(range(6))
    plan = scheduler.get_new_batch_prefill(SimpleNamespace(reqs=[]))
    assert plan.batch_to_run.reqs == ["cond", "uncond"]
    assert plan.running_batch is updated_running
    assert seen == [[0, 1]]
    assert scheduler.waiting_queue == [2, 3, 4, 5]


def test_idle_scheduler_returns_upstream_plan_before_first_request():
    scheduler = object.__new__(BreezeScheduler)
    scheduler._request_admission_lock = RLock()
    scheduler.waiting_queue = []
    running = SimpleNamespace(reqs=[])
    plan = scheduler.get_new_batch_prefill(running)
    # The real SGLang event loop dereferences this even before the first HTTP
    # request. Returning None crashes the worker immediately after startup.
    assert plan.batch_to_run is None
    assert plan.running_batch is running


def test_scheduler_abort_also_retires_cfg_twin(monkeypatch):
    aborted = []
    monkeypatch.setattr(
        OmniScheduler, "abort", lambda self, rid, **kwargs: aborted.append(rid)
    )
    scheduler = object.__new__(BreezeScheduler)
    scheduler._request_admission_lock = RLock()
    scheduler.abort("request")
    assert aborted == ["request", "request" + CFG_SUFFIX]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 2),
        ("max_running_requests", 4),
        ("disable_radix_cache", False),
        ("disable_cuda_graph", False),
        ("chunked_prefill_size", 16),
        ("dtype", "float16"),
        ("quantization", "fp8"),
        ("max_prefill_tokens", 1024),
    ],
)
def test_unsupported_execution_cannot_bypass_cfg_invariants(field, value):
    builder = BreezeEngineBuilder()
    settings = builder.generation_defaults(dtype="bfloat16")
    settings[field] = value
    with pytest.raises(ValueError, match="Breeze"):
        builder.adjust_overrides(settings)
