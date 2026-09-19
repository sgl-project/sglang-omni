# SPDX-License-Identifier: Apache-2.0
"""Native action payloads must survive the SDK and stage ownership boundary."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

from sglang_omni.client.client import Client
from sglang_omni.models.cosmos3.stages import NativeGenerationScheduler
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.continuation import ContinuationToken


class ActionGenerator:
    local_scheduler_process = None
    supports_cancellation = True
    server_args = SimpleNamespace(
        served_model_name="cosmos3-nano", pipeline_config=Cosmos3Config()
    )

    def __init__(self, values):
        self.calls = []
        self.values = values

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return GenerationResult(
            samples={
                "actions": self.values,
                "action_mode": "policy",
                "domain_id": 8,
                "raw_action_dim": 8,
            },
            size=("action",),
            generation_time=0.25,
            metrics={"denoise": 0.2},
        )

    def shutdown(self):
        pass


def request():
    return StagePayload(
        "action-1",
        OmniRequest(
            {
                "prompt": "Pick up the cube",
                "action_mode": "policy",
                "domain_name": "droid_lerobot",
                "raw_action_dim": 8,
            }
        ),
        None,
    )


@pytest.mark.parametrize("terminal", [True, False])
def test_action_result_keeps_native_structure_without_a_file(tmp_path, terminal):
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    scheduler = NativeGenerationScheduler(ActionGenerator(values), str(tmp_path))
    payload = scheduler._generate(request())
    assert scheduler.claim_result(payload, terminal=terminal)
    chunk = Client._default_result_builder(payload.request_id, payload.data)
    item = chunk.media[0]
    assert item["kind"] == "action"
    assert "path" not in item
    assert item["action"]["values"] == values.tolist()
    assert item["action"]["shape"] == [4, 8]
    assert item["action"]["domain_id"] == 8
    assert item["action"]["raw_action_dim"] == 8
    assert item["metrics"] == {"denoise": 0.2}
    json.dumps(chunk.to_dict(), allow_nan=False)
    scheduler.release_result(payload, delivered=True)
    assert not scheduler._native_requests
    assert not list(tmp_path.iterdir())


def test_action_batch_preserves_input_order(tmp_path):
    values = np.arange(64, dtype=np.float32).reshape(2, 4, 8)
    scheduler = NativeGenerationScheduler(ActionGenerator(values), str(tmp_path))
    payload = scheduler._generate(request())
    assert [item["input_index"] for item in payload.data["media"]] == [0, 1]
    assert [
        item["action"]["values"] for item in payload.data["media"]
    ] == values.tolist()
    scheduler.release_result(payload, delivered=False)
    assert not list(tmp_path.iterdir())


def test_action_continuation_does_not_read_a_media_file(tmp_path):
    scheduler = NativeGenerationScheduler(
        ActionGenerator(np.zeros((4, 8))), str(tmp_path)
    )
    payload = request()
    payload.continuation = ContinuationToken("session", 0, "generation", "nonce")
    result = scheduler._generate(payload)
    assert result.data["media"][0]["kind"] == "action"
    assert result.continuation == payload.continuation
    assert not scheduler._native_requests
    assert not list(tmp_path.iterdir())


def test_action_cancellation_event_reaches_native_runtime(tmp_path):
    generator = ActionGenerator(np.zeros((4, 8)))
    scheduler = NativeGenerationScheduler(generator, str(tmp_path))
    payload = scheduler._generate(request())
    scheduler.abort(payload.request_id)
    assert generator.calls[0]["cancellation_event"].is_set()
    assert not scheduler.claim_result(payload, terminal=True)
    assert not list(tmp_path.iterdir())
