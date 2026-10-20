# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Cosmos3 UMM adapter, without model execution."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import xgrammar as xgr
from sglang.srt.constrained.json_schema_validation import validate_xgrammar_json_schema
from xgrammar.testing import _is_grammar_accept_string

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.cosmos3 import umm as cosmos3_umm
from sglang_omni.models.cosmos3.umm import (
    MAX_PROMPT_CHARS,
    MAX_TEXT_CHARS,
    Cosmos3UMMAdapter,
    interpret_reasoner,
    worst_case_bytes,
)
from sglang_omni.native_generation import stage as generation_stage
from sglang_omni.native_generation.stage import (
    DEADLINE_METADATA_KEY,
    INLINE_MEDIA_LIMIT_BYTES,
    NativeGenerationScheduler,
    build_sampling_params,
    data_url,
)
from sglang_omni.pipeline.umm import (
    CONTINUATION_METADATA_KEY,
    UMMController,
    UMMLimits,
    route_umm,
)
from sglang_omni.preprocessing import resource_connector
from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.native_chat_scheduler import build_chat_fields
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.protocol import ChatCompletionRequest

PHASES = {"reason": "reasoner", "generate": "generation"}


def controller(clock=time.monotonic, **bounds: int) -> UMMController:
    limits = {
        "max_dispatches": 17,
        "max_outputs": 17,
        "max_active": 32,
        "max_retained_bytes": 128 * 1024 * 1024,
        "timeout_s": 3600.0,
        **bounds,
    }
    return UMMController(
        Cosmos3UMMAdapter("checkpoint", clock=clock),
        limits=UMMLimits(**limits),
        phases=PHASES,
        output_stage="output",
        clock=clock,
    )


def submit(umm: UMMController, inputs="Draw a cube.", params=None, rid="r", **meta):
    request = OmniRequest(inputs, params or {}, meta)
    return umm.advance(StagePayload(rid, request, request.inputs))


def reply(dispatched: StagePayload, data: object) -> StagePayload:
    return StagePayload(dispatched.request_id, dispatched.request, data)


def reasoner_reply(kind="generate", text="", modality="image", prompt="A blue cube"):
    generation = None if kind == "final" else {"modality": modality, "prompt": prompt}
    return {
        "text": json.dumps({"kind": kind, "text": text, "generation": generation}),
        "finish_reason": "stop",
    }


def media(content=b"image fixture", kind="image"):
    mime = "image/png" if kind == "image" else "video/mp4"
    return {
        "media": [
            {
                "kind": kind,
                "mime_type": mime,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": data_url(mime, content),
            }
        ]
    }


def rejection(umm: UMMController) -> str:
    (message,) = [umm.outbox.get_nowait() for _ in range(umm.outbox.qsize())]
    assert message.type == "error"
    return str(message.data)


@pytest.mark.parametrize(
    "inputs",
    [
        {"messages": [{"role": "user"}, 1]},
        {"messages": [{"role": "tool", "content": "result"}]},
    ],
)
def test_invalid_chat_history_is_a_client_error_before_any_dispatch(inputs):
    umm = controller()
    original = deepcopy(inputs)
    assert submit(umm, inputs) is None
    assert generation_error_status_code(RuntimeError(rejection(umm))) == 400
    assert not umm.contexts and not umm.adapter.histories
    assert inputs == original
    dispatched = submit(umm, "hello", rid="healthy")
    assert route_umm("healthy", dispatched) == "reasoner"


@pytest.mark.parametrize(
    "body, temperature",
    [
        (
            {
                "max_tokens": 64,
                "stage_params": {
                    "reasoner": {
                        "stream": True,
                        "response_format": {"type": "text"},
                        "max_new_tokens": 4096,
                        "top_k": 5,
                    }
                },
            },
            0.0,
        ),
        # An explicit value equal to the stage's placeholder still applies.
        ({"temperature": 1.0}, 1.0),
    ],
)
def test_decisions_keep_their_contract_and_only_explicit_client_sampling(
    body, temperature
):
    chat = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Make a blue cube"}],
        images=[data_url("image/png", bytes(3))],
        **body,
    )
    request = Client.build_omni_request(build_chat_generate_request(chat))
    internal = submit(controller(), request.inputs, request.params, **request.metadata)
    fields = build_chat_fields(
        internal,
        "cosmos3",
        {
            "stream",
            "response_format",
            "temperature",
            "top_k",
            "max_tokens",
            "max_completion_tokens",
            "chat_template_kwargs",
            "n",
            "logprobs",
        },
        "reasoner",
    )
    assert fields["stream"] is False
    assert fields["response_format"]["json_schema"]["strict"]
    assert fields["chat_template_kwargs"]["enable_thinking"] is False
    assert fields["temperature"] == temperature
    assert fields.get("top_k") == body.get("stage_params", {}).get("reasoner", {}).get(
        "top_k"
    )
    # The decision schema alone bounds a decision, and none returns logprobs.
    assert not {"max_tokens", "max_completion_tokens", "logprobs"} & set(fields)
    assert fields["messages"][1]["content"][1]["image_url"]["url"].endswith("AAAA")


def decision_text(kind, text, generation):
    return {"text": json.dumps({"kind": kind, "text": text, "generation": generation})}


@pytest.mark.parametrize(
    "response",
    [
        {"text": "Here is an image prompt: A blue cube"},
        {"text": '{"kind":"final","kind":"final","text":"done","generation":null}'},
        {
            "text": '{"kind":"final","text":"done","generation":null}',
            "finish_reason": "length",
        },
        {"text": None},
        decision_text("final", "x", {}),
        decision_text("generate", "", {"modality": "audio", "prompt": "x"}),
        decision_text("generate", "", {"modality": "image", "prompt": "x", "y": 1}),
        # Model output beyond the server limits the byte reservation plans for.
        decision_text("final", "x" * (MAX_TEXT_CHARS + 1), None),
        decision_text(
            "generate",
            "",
            {"modality": "image", "prompt": "x" * (MAX_PROMPT_CHARS + 1)},
        ),
    ],
)
def test_unstructured_or_malformed_decisions_fail_as_server_errors(response):
    umm = controller()
    dispatched = submit(umm)
    with pytest.raises(ValueError) as rejected:
        umm.advance(reply(dispatched, response))
    assert generation_error_status_code(rejected.value) == 500
    assert not umm.contexts and not umm.adapter.histories


def generate_turn(umm, decision=None, params=None, inputs="task"):
    dispatched = submit(umm, inputs, params)
    return umm.advance(reply(dispatched, decision or reasoner_reply()))


def test_generation_decision_controls_prompt_and_modality_with_native_tuning(tmp_path):
    params = {
        "stream": True,
        "diffusion": {"prompt": "stale", "seed": 0, "num_frames": 33},
        "stage_params": {
            "generation": {
                "prompt": "also stale",
                "guidance_scale": 0.0,
                "num_outputs_per_prompt": 9,
            }
        },
    }
    original = deepcopy(params)
    generation = generate_turn(controller(), params=params)
    assert route_umm("r", generation) == "generation"
    native = build_sampling_params(generation, str(tmp_path))
    assert native["prompt"] == "A blue cube"
    assert native["num_frames"] == 1
    assert native["num_outputs_per_prompt"] == 1
    assert native["seed"] == 0
    assert native["guidance_scale"] == 0.0
    assert params == original


@pytest.mark.parametrize("diffusion, frames", [({}, 33), ({"num_frames": 65}, 65)])
def test_video_decisions_use_native_frame_options(diffusion, frames):
    video = reasoner_reply(modality="video")
    generation = generate_turn(controller(), video, {"diffusion": diffusion})
    assert generation.request.params["diffusion"]["num_frames"] == frames


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/foreign.png"},
        {"kind": "audio"},
        {"kind": "video"},
        {"mime_type": "image/jpeg"},
        {"size_bytes": True},
        {"size_bytes": INLINE_MEDIA_LIMIT_BYTES + 1},
        {"sha256": "0" * 64},
        {"url": "file:///foreign.png"},
        {"url": data_url("image/png", b"") + "????"},
        None,
    ],
)
def test_generated_media_must_be_one_owned_item_of_the_chosen_modality(change):
    umm = controller()
    generation = generate_turn(umm)
    value = media()
    if change is None:
        value["media"].append(deepcopy(value["media"][0]))
    else:
        value["media"][0].update(change)
    with pytest.raises(ValueError):
        umm.advance(reply(generation, value))
    assert not umm.contexts and not umm.adapter.histories


def test_last_generation_turn_allows_only_a_final_decision():
    umm = controller(max_dispatches=3)
    first = submit(umm, "Draw a cube, then describe it.")
    generation = umm.advance(reply(first, reasoner_reply()))
    last = umm.advance(reply(generation, media()))
    schemas = [
        payload.request.params["response_format"]["json_schema"]["schema"]
        for payload in (first, last)
    ]
    kinds = [
        [
            branch["properties"]["kind"]["enum"]
            for branch in schema.get("anyOf", [schema])
        ]
        for schema in schemas
    ]
    assert kinds == [[["final"], ["generate"]], [["final"]]]
    # Native SGLang must accept each schema, its grammar must admit text and
    # prompts with JSON escapes and blanks, and the adapter must accept them.
    for schema, offered in zip(schemas, kinds, strict=True):
        validate_xgrammar_json_schema(schema)
        grammar = xgr.Grammar.from_json_schema(
            json.dumps(schema), any_whitespace=True, strict_mode=True
        )
        for value in ("", " \t\n", "\u3000", 'Red\nGreen "Blue" \\ \x01 é'):
            generation = {"modality": "image", "prompt": value}
            for decision in (
                {"kind": "final", "text": value, "generation": None},
                {"kind": "generate", "text": value, "generation": generation},
            ):
                if [decision["kind"]] in offered:
                    text = json.dumps(decision, ensure_ascii=False)
                    assert _is_grammar_accept_string(grammar, text), text
                    interpret_reasoner({"text": text, "finish_reason": "stop"})
                else:
                    pass
    final = umm.advance(reply(last, reasoner_reply("final", "A cube.")))
    assert route_umm("r", final) == "output"
    parsed = Client.default_result_builder("r", final.data)
    assert [segment.kind for segment in parsed.segments] == ["image", "text"]
    assert parsed.text == "A cube."
    assert not umm.contexts and not umm.adapter.histories


@pytest.mark.parametrize("spare", [0, -1])
def test_admitted_requests_never_exceed_retained_bytes_through_model_output(
    monkeypatch, spare
):
    # Maximal outputs at a small inline limit, so the derived bounds stay small.
    monkeypatch.setattr(generation_stage, "INLINE_MEDIA_LIMIT_BYTES", 1024)
    monkeypatch.setattr(cosmos3_umm, "INLINE_MEDIA_LIMIT_BYTES", 1024)
    monkeypatch.setattr(cosmos3_umm, "WORST_CASE_BYTES", worst_case_bytes())
    probe = controller(max_dispatches=5)
    submit(probe)
    (context,) = probe.contexts.values()
    budget = probe.adapter.usage(context) + spare
    umm = controller(max_dispatches=5, max_retained_bytes=budget)
    current = submit(umm)
    if spare < 0:
        assert current is None and not umm.contexts
        assert generation_error_status_code(RuntimeError(rejection(umm))) == 400
        return
    else:
        pass
    texts = iter(["\x01" * MAX_TEXT_CHARS, "\U0001f600" * MAX_TEXT_CHARS] * 2)
    held = budget
    while route_umm("r", current) != "output":
        # What the request holds and may still add never grows.
        context = umm.contexts["r"]
        assert context.output_bytes + umm.adapter.usage(context) <= held
        held = context.output_bytes + umm.adapter.usage(context)
        if route_umm("r", current) == "reasoner":
            final = "anyOf" not in (
                current.request.params["response_format"]["json_schema"]["schema"]
            )
            prompt = "\x01" * MAX_PROMPT_CHARS
            data = reasoner_reply(
                "final" if final else "generate", next(texts), "video", prompt
            )
        else:
            data = media(bytes(range(256)) * 4, kind="video")
        current = umm.advance(reply(current, data))
    assert [s["kind"] for s in current.data["segments"]] == [
        "text",
        "video",
        "text",
        "video",
        "text",
    ]


@pytest.mark.parametrize("inside", [True, False])
def test_generation_media_paths_follow_the_server_media_policy(
    tmp_path, monkeypatch, inside
):
    allowed = tmp_path / "media"
    allowed.mkdir()
    image = (allowed if inside else tmp_path) / "input.png"
    image.write_bytes(b"image")
    connector = MultiModalResourceConnector(allowed_local_media_path=allowed)
    monkeypatch.setattr(resource_connector, "_global_connector", connector)
    params = {"stage_params": {"generation": {"image_path": str(image)}}}
    umm = controller()
    dispatched = submit(umm, "x", params)
    if inside:
        assert route_umm("r", dispatched) == "reasoner"
    else:
        assert dispatched is None
        error = rejection(umm)
        assert "refused by the server media policy" in error
        assert generation_error_status_code(RuntimeError(error)) == 400


class SavedGenerator:
    local_scheduler_process = None

    def __init__(self, content=b"png fixture", foreign=None, symlink=False):
        self.content, self.foreign, self.symlink = content, foreign, symlink

    def generate(self, *, sampling_params_kwargs):
        target = Path(sampling_params_kwargs["output_path"]) / "result.png"
        if self.symlink:
            target.symlink_to(self.foreign)
        elif self.foreign is None:
            target.write_bytes(self.content)
        else:
            target = self.foreign
        return [
            SimpleNamespace(
                output_file_path=str(target),
                size=(16, 16),
                prompt="cube",
                generation_time=0,
                peak_memory_mb=0,
                metrics={},
            )
        ]

    def shutdown(self):
        pass


def inline_scheduler(tmp_path, generator=None):
    return NativeGenerationScheduler(
        generator or SavedGenerator(), str(tmp_path), inline_media=True
    )


@pytest.mark.parametrize("refused", [False, True, "aborted"])
def test_inline_output_is_owned_hashed_and_released_before_forwarding(
    tmp_path, refused
):
    keep = tmp_path / "keep"
    keep.write_text("keep")
    generator = SavedGenerator()
    scheduler = inline_scheduler(tmp_path, generator)
    if refused:
        # Native returns no output for a generation it refused, could not run
        # or that an abort stopped, and none of them becomes a result.
        def no_output(**_):
            if refused == "aborted":
                scheduler.cancel_native("r")
            else:
                pass
            return None

        generator.generate = no_output
    else:
        pass
    if refused:
        with pytest.raises(RuntimeError, match="without producing an output"):
            scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
        assert list(tmp_path.iterdir()) == [keep]
        return
    else:
        pass
    result = scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == [keep]
    assert not scheduler.native_requests
    assert scheduler.claim_result(result, terminal=False)
    item = result.data["media"][0]
    assert "path" not in item
    assert item["sha256"] == hashlib.sha256(b"png fixture").hexdigest()
    assert base64.b64decode(item["url"].split(",", 1)[1]) == b"png fixture"


def test_inline_generation_serves_the_earliest_deadline_first(tmp_path):
    scheduler = inline_scheduler(tmp_path)
    for name, deadline in (("late", 3.0), ("none", None), ("early", 1.0), ("b", 2.0)):
        metadata = {} if deadline is None else {DEADLINE_METADATA_KEY: deadline}
        payload = StagePayload(name, OmniRequest("cube", {}, metadata), None)
        scheduler.inbox.put(IncomingMessage(name, "new_request", payload))
    order = [scheduler.next_message().request_id for _ in range(4)]
    assert order == ["early", "b", "late", "none"]


@pytest.mark.parametrize("symlink", [False, True])
def test_inline_generation_rejects_foreign_files_without_removing_them(
    tmp_path, symlink
):
    foreign = tmp_path / "foreign.png"
    foreign.write_bytes(b"retain")
    scheduler = inline_scheduler(
        tmp_path, SavedGenerator(foreign=foreign, symlink=symlink)
    )
    with pytest.raises(ValueError, match="outside its owned request directory"):
        scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == [foreign]
    assert foreign.read_bytes() == b"retain"
    assert not scheduler.native_requests


@pytest.mark.parametrize("content", [b"1234", b""])
def test_inline_byte_bounds_fail_and_release_only_owned_directory(
    tmp_path, monkeypatch, content
):
    monkeypatch.setattr(generation_stage, "INLINE_MEDIA_LIMIT_BYTES", 3)
    scheduler = inline_scheduler(tmp_path, SavedGenerator(content))
    with pytest.raises(ValueError, match="byte limit"):
        scheduler.generate(StagePayload("r", OmniRequest("cube"), None))
    assert list(tmp_path.iterdir()) == []
    assert not scheduler.native_requests


def test_umm_variant_wires_the_controller_and_inline_generation(monkeypatch):
    from sglang_omni.config import manager
    from sglang_omni.config.runtime import resolve_stage_typed_kwargs
    from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
    from sglang_omni.models.cosmos3.umm_config import Cosmos3UMMPipelineConfig
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
    from sglang_omni.pipeline.umm import create_umm_controller
    from tests.unit_test.pipeline.helpers import build_compiled_process_topology

    # Both ways of selecting the variant reach the lazily imported class.
    monkeypatch.setattr(
        manager, "resolve_config_cls_for_model_path", lambda _: Cosmos3PipelineConfig
    )
    config = manager.ConfigManager.from_model_path("checkpoint", "umm").config
    assert type(config) is Cosmos3UMMPipelineConfig
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name("Cosmos3UMMPipelineConfig")
        is Cosmos3UMMPipelineConfig
    )
    build_compiled_process_topology(config)
    assert config.stage_factory_kwargs("generation") == {"inline_media": True}
    args = {
        **resolve_stage_typed_kwargs(config.stage_named("orchestrator")),
        **config.stage_factory_kwargs("orchestrator"),
    }
    umm = create_umm_controller("checkpoint", **args)
    assert isinstance(umm.adapter, Cosmos3UMMAdapter)
    assert config.entry_stage_admission_limit() == umm.limits.max_active
    umm.stop()


@pytest.mark.parametrize(
    "service_s, admitted, offered, ended",
    [
        (100.0, True, True, False),
        (200.0, False, True, False),
        (1200.0, True, False, True),
    ],
)
def test_admission_refuses_requests_generation_cannot_finish_in_time(
    service_s, admitted, offered, ended
):
    now = {"t": 0.0}
    umm = controller(lambda: now["t"])
    # Before any generation was measured, requests are admitted.
    first = submit(umm, rid="a")
    queued = submit(umm, rid="c")
    now["t"] = 1.0
    generation = umm.advance(reply(first, reasoner_reply()))
    umm.advance(reply(queued, reasoner_reply()))
    assert generation.request.metadata[DEADLINE_METADATA_KEY] == [3600.0, 0]
    now["t"] = 1.0 + service_s
    resumed = umm.advance(reply(generation, {**media(), "service_s": service_s}))
    # Another generation, a younger one running ahead of it, and two
    # Reasoner turns of the measured second must end by the deadline 3600.
    schema = resumed.request.params["response_format"]["json_schema"]["schema"]
    assert ("anyOf" in schema) is offered
    # The first measurement plans again: c's queued generation, a younger one
    # running ahead of it, and its final answer must still end by 3600, or c
    # ends now with the capacity error instead of failing at its deadline.
    umm.expire()
    assert ("c" in umm.contexts) is not ended
    if ended:
        notices = [umm.outbox.get_nowait() for _ in range(umm.outbox.qsize())]
        assert [(n.request_id, n.type) for n in notices] == [
            ("c", "admitted"),
            ("c", "error"),
        ]
        # The client gets the error the coordinator rebuilds from its text.
        error = QueueFullError.from_message(str(notices[1].data))
        assert generation_error_status_code(error) == 503
        assert "cannot finish it in time" in str(error)
    else:
        pass
    # The new request plans the generations that end by its deadline behind
    # those still planned for a and c, each behind one younger generation and
    # before a Reasoner turn, and is refused when they leave it fewer than an
    # idle server would.
    second = submit(umm, rid="b")
    if admitted:
        assert route_umm("b", second) == "reasoner"
    else:
        assert second is None and "b" not in umm.contexts
        assert generation_error_status_code(RuntimeError(rejection(umm))) == 503
    assert "a" in umm.contexts


def test_queued_generation_is_not_shed_for_another_requests_longer_one():
    now = {"t": 0.0}
    umm = controller(lambda: now["t"])
    fast = submit(umm, rid="f")
    now["t"] = 1.0
    generation = umm.advance(reply(fast, reasoner_reply()))
    now["t"] = 11.0
    umm.advance(reply(generation, {**media(), "service_s": 10.0}))
    first, queued = submit(umm, rid="a"), submit(umm, rid="c")
    now["t"] = 12.0
    generation = umm.advance(reply(first, reasoner_reply()))
    umm.advance(reply(queued, reasoner_reply()))
    now["t"] = 1212.0
    umm.advance(reply(generation, {**media(), "service_s": 1200.0}))
    # c's queued image still ends by its deadline at the 10 s an image took.
    umm.expire()
    assert "c" in umm.contexts


@pytest.mark.parametrize(
    "kind, prompt, status",
    [
        ("image", 'A sign reading "你好" on a cube.\nUse the lighting.', "completed"),
        ("video", 'A sign reading "你好" on a cube.\nUse the lighting.', "completed"),
        # Native generation refuses a blank prompt, so none is dispatched.
        ("image", " \n", "not_executed"),
    ],
)
def test_receipt_binds_model_prompt_and_resubmits_history(kind, prompt, status):
    umm = controller()
    resumed = generate_turn(umm, reasoner_reply(modality=kind, prompt=prompt))
    generated = media(b"completed result", kind=kind)
    original = deepcopy(generated)
    if status == "completed":
        resumed = umm.advance(reply(resumed, generated))
    else:
        pass
    assert route_umm("r", resumed) == "reasoner"
    messages = resumed.request.inputs["messages"]
    receipt = messages[-1]
    text = receipt["content"][0]["text"]
    record = json.loads(text.split("Result receipt: ", 1)[1].split("\n", 1)[0])
    assert record == {"status": status, "modality": kind, "requested_prompt": prompt}
    assert receipt["content"][1:] == (
        [{"type": f"{kind}_url", f"{kind}_url": {"url": generated["media"][0]["url"]}}]
        if status == "completed"
        else []
    )
    assert json.loads(messages[-2]["content"])["generation"]["prompt"] == prompt
    assert generated == original


@pytest.mark.parametrize("seed", [0, 7])
def test_umm_sdk_sampling_reaches_decisions_and_generation(seed):
    sampling = SamplingParams(seed=seed, temperature=0.9)
    request = Client.build_omni_request(
        GenerateRequest(prompt="draw a lake", sampling=sampling)
    )
    umm = controller()
    first = submit(umm, request.inputs, request.params, **request.metadata)
    assert first.request.params["temperature"] == 0.9
    generation = umm.advance(reply(first, reasoner_reply()))
    native = build_sampling_params(generation, "outputs")
    assert native["seed"] == seed


def test_forward_dynamics_offers_only_video_generations():
    options = {"action_mode": "forward_dynamics", "action": [[0.0] * 8] * 4}
    umm = controller()
    dispatched = submit(umm, "Predict motion", {"diffusion": options})
    schema = dispatched.request.params["response_format"]["json_schema"]["schema"]
    generate = schema["anyOf"][1]["properties"]["generation"]["properties"]
    assert generate["modality"]["enum"] == ["video"]
    with pytest.raises(ValueError, match="requires a video decision"):
        umm.advance(reply(dispatched, reasoner_reply(modality="image")))


def test_forward_dynamics_video_keeps_the_known_actions():
    options = {"action_mode": "forward_dynamics", "action": [[0.0] * 8] * 4}
    generation = generate_turn(
        controller(), reasoner_reply(modality="video"), {"diffusion": options}
    )
    assert generation.request.params["diffusion"]["action"] == options["action"]
    assert generation.request.params["diffusion"]["num_frames"] > 1
    assert CONTINUATION_METADATA_KEY in generation.request.metadata


@pytest.mark.parametrize(
    "params, message",
    [
        ({"stage_params": {"reasoner": []}}, "stage_params.reasoner"),
        ({"chat_template_kwargs": "invalid"}, "chat_template_kwargs"),
        ({"return_logprob": True}, "return_logprob"),
        ({"diffusion": []}, "diffusion"),
        ({"stage_params": {"generation": {"num_frames": 1}}}, "more than one frame"),
        ({"diffusion": {"action_mode": " POLICY "}}, "action-only"),
        ({"stage_params": {"generation": {"output_path": "/tmp"}}}, "output_path"),
    ],
)
def test_invalid_options_are_refused_before_any_turn(params, message):
    # Refused even when the Reasoner would have answered without generating.
    umm = controller()
    assert submit(umm, "hello", params) is None
    error = rejection(umm)
    assert message in error
    assert generation_error_status_code(RuntimeError(error)) == 400
    assert not umm.contexts and not umm.adapter.histories


def test_start_check_lists_the_fields_native_cosmos3_declares():
    from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        get_sampling_request_extra_fields,
    )

    declared = get_sampling_request_extra_fields(
        Cosmos3SamplingParams, "image"
    ) | get_sampling_request_extra_fields(Cosmos3SamplingParams, "video")
    assert cosmos3_umm.COSMOS3_REQUEST_FIELDS == declared
