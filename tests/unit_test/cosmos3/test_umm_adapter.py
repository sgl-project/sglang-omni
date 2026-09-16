# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Cosmos3 adapter, without model execution."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.cosmos3.reasoner import build_chat_fields
from sglang_omni.models.cosmos3.stages import (
    NativeGenerationScheduler,
    build_sampling_params,
)
from sglang_omni.models.cosmos3.umm import INLINE_MEDIA_LIMIT_BYTES, Cosmos3UMMAdapter
from sglang_omni.pipeline.umm import UMMDecision
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.continuation import ContinuationToken


def decision(kind="generate", text="", modality="image", prompt="A blue cube"):
    generation = None if kind == "final" else {"modality": modality, "prompt": prompt}
    return Cosmos3UMMAdapter().interpret_reasoner(
        {
            "text": json.dumps({"kind": kind, "text": text, "generation": generation}),
            "finish_reason": "stop",
        }
    )


def media(content=b"image fixture", kind="image"):
    mime = "image/png" if kind == "image" else "video/mp4"
    return {
        "media": [
            {
                "kind": kind,
                "mime_type": mime,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": f"data:{mime};base64," + base64.b64encode(content).decode(),
            }
        ]
    }


@pytest.mark.parametrize("messages", [[1], [None], ["bad"], [{"role": "user"}, 1]])
@pytest.mark.parametrize("media", [{}, {"images": ["image.png"]}])
def test_invalid_chat_elements_reject_before_umm_session(messages, media):
    request = OmniRequest({"messages": messages, **media})
    original = deepcopy(request)
    with pytest.raises(ValueError):
        Cosmos3UMMAdapter().start(request)
    assert request == original


def test_native_chat_contract_and_user_options_survive_internal_decision():
    request = OmniRequest(
        {
            "messages": [{"role": "user", "content": "Make a blue cube"}],
            "images": ["data:image/png;base64,AAAA"],
        },
        {
            "stream": True,
            "stage_params": {
                "reasoner": {
                    "stream": True,
                    "response_format": {"type": "text"},
                    "temperature": 0.25,
                    "max_new_tokens": 4096,
                }
            },
        },
    )
    original = deepcopy(request)
    adapter = Cosmos3UMMAdapter()
    history = adapter.start(request)
    internal = adapter.reasoner_request(history, request)
    fields = build_chat_fields(
        StagePayload("r", internal, None),
        "cosmos3",
        {
            "stream",
            "response_format",
            "temperature",
            "max_tokens",
            "chat_template_kwargs",
            "n",
        },
    )
    assert fields["stream"] is False
    assert fields["response_format"]["json_schema"]["strict"]
    assert fields["chat_template_kwargs"]["enable_thinking"] is False
    assert fields["temperature"] == 0.25
    assert fields["max_tokens"] == 4096
    assert fields["messages"][1]["content"][1]["image_url"]["url"].endswith("AAAA")
    assert request == original
    assert "session_params" not in fields


@pytest.mark.parametrize(
    "response",
    [
        {"text": "Here is an image prompt: A blue cube"},
        {"text": '```json\n{"kind":"final","text":"done","generation":null}\n```'},
        {"text": '{"kind":"final","kind":"final","text":"done","generation":null}'},
        {
            "text": '{"kind":"final","text":"done","generation":null}',
            "finish_reason": "length",
        },
        {"text": None},
        [],
    ],
)
def test_prose_fences_duplicates_and_truncated_decisions_are_not_handoffs(response):
    with pytest.raises(ValueError):
        Cosmos3UMMAdapter().interpret_reasoner(response)


def test_generation_decision_controls_prompt_and_modality_with_native_tuning(tmp_path):
    adapter = Cosmos3UMMAdapter()
    original = OmniRequest(
        "task",
        {
            "stream": True,
            "diffusion": {"prompt": "stale", "seed": 0, "num_frames": 33},
            "stage_params": {
                "generation": {
                    "prompt": "also stale",
                    "guidance_scale": 0.0,
                    "num_outputs_per_prompt": 9,
                }
            },
        },
    )
    frozen = deepcopy(original)
    request = adapter.generation_request(decision(), original)
    params = build_sampling_params(StagePayload("r", request, None), str(tmp_path))
    assert params["prompt"] == "A blue cube"
    assert params["num_frames"] == 1
    assert params["num_outputs_per_prompt"] == 1
    assert params["seed"] == 0
    assert params["guidance_scale"] == 0.0
    assert original == frozen


def test_video_uses_native_frame_options_and_rejects_image_frame_count():
    adapter = Cosmos3UMMAdapter()
    d = decision(modality="video")
    assert (
        adapter.generation_request(d, OmniRequest("task")).params["diffusion"][
            "num_frames"
        ]
        == 33
    )
    assert (
        adapter.generation_request(
            d, OmniRequest("task", {"diffusion": {"num_frames": 65}})
        ).params["diffusion"]["num_frames"]
        == 65
    )
    with pytest.raises(ValueError, match="more than one frame"):
        adapter.generation_request(
            d, OmniRequest("task", {"diffusion": {"num_frames": 1}})
        )


@pytest.mark.parametrize(
    "change",
    [
        {"path": "/foreign.png"},
        {"kind": "audio"},
        {"mime_type": "image/jpeg"},
        {"size_bytes": True},
        {"size_bytes": INLINE_MEDIA_LIMIT_BYTES + 1},
        {"sha256": "0" * 64},
        {"url": "file:///foreign.png"},
        {"url": "data:image/png;base64,????"},
    ],
)
def test_media_reingestion_rejects_unowned_or_inconsistent_results(change):
    value = media()
    value["media"][0].update(change)
    with pytest.raises(ValueError):
        Cosmos3UMMAdapter().media_segments(value)


def test_media_modality_must_match_the_model_decision():
    with pytest.raises(ValueError, match="modality differs"):
        Cosmos3UMMAdapter().incorporate_media([], decision(), media(kind="video"))


class SavedGenerator:
    local_scheduler_process = None

    def __init__(self, content=b"png fixture", foreign=None, symlink=False, count=1):
        self.content, self.foreign, self.symlink, self.count = (
            content,
            foreign,
            symlink,
            count,
        )

    def generate(self, *, sampling_params_kwargs):
        target = Path(sampling_params_kwargs["output_path"]) / "result.png"
        if self.symlink:
            target.symlink_to(self.foreign)
        elif self.foreign is None:
            target.write_bytes(self.content)
        else:
            target = self.foreign
        result = SimpleNamespace(
            output_file_path=str(target),
            size=(16, 16),
            prompt="cube",
            generation_time=0,
            peak_memory_mb=0,
            metrics={},
        )
        return [result] * self.count

    def shutdown(self):
        pass


def inline_payload():
    return StagePayload(
        "r",
        OmniRequest("cube"),
        None,
        ContinuationToken("session", 0, "generation", "nonce"),
    )


def test_inline_output_is_owned_hashed_and_released_before_forwarding(tmp_path):
    keep = tmp_path / "keep"
    keep.write_text("keep")
    scheduler = NativeGenerationScheduler(SavedGenerator(), str(tmp_path))
    result = scheduler._generate(inline_payload())
    assert list(tmp_path.iterdir()) == [keep]
    assert not scheduler._native_requests
    item = result.data["media"][0]
    assert "path" not in item
    assert item["sha256"] == hashlib.sha256(b"png fixture").hexdigest()
    assert base64.b64decode(item["url"].split(",", 1)[1]) == b"png fixture"
    assert Cosmos3UMMAdapter().media_segments(result.data)[0]["kind"] == "image"


@pytest.mark.parametrize("symlink", [False, True])
def test_inline_generation_rejects_foreign_files_without_removing_them(
    tmp_path, symlink
):
    foreign = tmp_path / "foreign.png"
    foreign.write_bytes(b"retain")
    scheduler = NativeGenerationScheduler(
        SavedGenerator(foreign=foreign, symlink=symlink), str(tmp_path)
    )
    with pytest.raises(ValueError, match="outside its owned request directory"):
        scheduler._generate(inline_payload())
    assert list(tmp_path.iterdir()) == [foreign]
    assert foreign.read_bytes() == b"retain"
    assert not scheduler._native_requests


@pytest.mark.parametrize("content,limit", [(b"1234", 3), (b"", 3)])
def test_inline_byte_bounds_fail_and_release_only_owned_directory(
    tmp_path, content, limit
):
    scheduler = NativeGenerationScheduler(
        SavedGenerator(content), str(tmp_path), inline_media_limit_bytes=limit
    )
    with pytest.raises(ValueError, match="byte limit"):
        scheduler._generate(inline_payload())
    assert list(tmp_path.iterdir()) == []
    assert not scheduler._native_requests


def test_multiple_native_outputs_fail_and_release_directory(tmp_path):
    scheduler = NativeGenerationScheduler(SavedGenerator(count=2), str(tmp_path))
    with pytest.raises(ValueError, match="one output"):
        scheduler._generate(inline_payload())
    assert list(tmp_path.iterdir()) == []


def test_sdk_success_keeps_default_file_delivery(tmp_path):
    scheduler = NativeGenerationScheduler(SavedGenerator(), str(tmp_path))
    result = scheduler._generate(StagePayload("r", OmniRequest("cube"), None))
    assert Path(result.data["media"][0]["path"]).read_bytes() == b"png fixture"
    assert "url" not in result.data["media"][0]
    scheduler.abort("r")
    assert list(tmp_path.iterdir()) == []


def test_umm_variant_keeps_native_engines_and_one_conditional_terminal():
    from sglang_omni.models.cosmos3.config import Cosmos3UMMPipelineConfig, Variants
    from sglang_omni.models.cosmos3.umm import create_umm_scheduler
    from sglang_omni.pipeline.umm import UMMController

    config = Cosmos3UMMPipelineConfig(model_path="checkpoint")
    assert Variants["umm"] is Cosmos3UMMPipelineConfig
    assert config.resolved_entry_stage == "orchestrator"
    assert config.terminal_stages == ["orchestrator"]
    assert config.max_in_flight == 32
    stages = {stage.name: stage for stage in config.stages}
    assert stages["orchestrator"].gpu is None
    assert stages["orchestrator"].next == ["reasoner", "generation"]
    assert stages["orchestrator"].route_fn == "sglang_omni.pipeline.umm.route_umm"
    assert stages["reasoner"].allow_child_processes
    assert stages["generation"].allow_child_processes
    assert stages["reasoner"].gpu != stages["generation"].gpu
    controller = create_umm_scheduler("checkpoint", max_turns=2)
    assert isinstance(controller, UMMController)
    assert controller.limits.max_turns == 2
    assert controller.active_sessions == 0
    controller.stop()


def test_wrong_continuation_phase_rejects_before_native_allocation(tmp_path):
    request = inline_payload()
    request.continuation = ContinuationToken("session", 0, "reasoner", "nonce")
    scheduler = NativeGenerationScheduler(SavedGenerator(), str(tmp_path))
    with pytest.raises(ValueError, match="non-generation continuation"):
        scheduler._generate(request)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["image", "video"])
def test_completed_receipt_binds_model_prompt_and_preserves_history(kind):
    adapter = Cosmos3UMMAdapter()
    prompt = 'A sign reading "你好" on a cube.\nUse the requested lighting.'
    chosen = decision(modality=kind, prompt=prompt)
    generated = media(b"completed result", kind=kind)
    request = OmniRequest("Generate media, inspect it, and revise if needed.")
    history = adapter.start(request)
    original_history = deepcopy(history)
    original_decision = deepcopy(chosen)
    original_result = deepcopy(generated)

    resumed = adapter.incorporate_media(history, chosen, generated)
    receipt_text = resumed[-1]["content"][0]["text"]
    receipt = json.loads(receipt_text.split("Result receipt: ", 1)[1].split("\n", 1)[0])
    assert receipt == {
        "status": "completed",
        "modality": kind,
        "requested_prompt": prompt,
    }
    assert resumed[-1]["content"][1] == {
        "type": f"{kind}_url",
        f"{kind}_url": {"url": generated["media"][0]["url"]},
    }
    native_request = adapter.reasoner_request(resumed, request)
    assert native_request.inputs["messages"][-1] == resumed[-1]
    assert history == original_history
    assert chosen == original_decision
    assert generated == original_result
    assert resumed[: len(history)] == history


@pytest.mark.parametrize("seed", [0, 7])
def test_umm_generation_preserves_sdk_seed(seed):
    request = GenerateRequest(prompt="draw a lake", sampling=SamplingParams(seed=seed))
    native_request = Cosmos3UMMAdapter().generation_request(
        UMMDecision("generate", generation={"modality": "image", "prompt": "a lake"}),
        Client._build_omni_request(request),
    )
    native = build_sampling_params(
        StagePayload("owned", native_request, None), "outputs"
    )
    assert native["seed"] == seed


@pytest.mark.parametrize("mode", ["policy", "inverse_dynamics", " POLICY "])
def test_action_only_override_rejects_before_visual_generation(mode):
    request = OmniRequest(
        {"prompt": "Draw a cube"}, {"diffusion": {"action_mode": mode}}
    )
    with pytest.raises(ValueError, match="action"):
        Cosmos3UMMAdapter().generation_request(decision(), request)


def test_forward_dynamics_requires_a_video_decision():
    options = {"action_mode": "forward_dynamics", "action": [[0.0] * 8] * 4}
    request = OmniRequest({"prompt": "Predict motion"}, {"diffusion": options})
    with pytest.raises(ValueError, match="forward_dynamics requires a video"):
        Cosmos3UMMAdapter().generation_request(decision(modality="image"), request)
    forwarded = Cosmos3UMMAdapter().generation_request(
        decision(modality="video"), request
    )
    assert forwarded.params["diffusion"]["action"] == options["action"]
    assert forwarded.params["diffusion"]["num_frames"] > 1


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "final"},
        {"generation": None},
        {"generation": {"modality": "audio", "prompt": "cube"}},
        {
            "generation": {
                "modality": "image",
                "prompt": "cube",
                "output_path": "/outside",
            }
        },
        {"text": True},
        {"extra": True},
    ],
)
def test_malformed_structured_decisions_fail_before_routing(change):
    response = {
        "kind": "generate",
        "text": "",
        "generation": {"modality": "image", "prompt": "cube"},
    }
    with pytest.raises(ValueError):
        Cosmos3UMMAdapter().interpret_reasoner(
            {"text": json.dumps({**response, **change})}
        )
