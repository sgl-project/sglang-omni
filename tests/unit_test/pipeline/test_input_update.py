# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import InputUpdateMessage


def test_input_update_message_round_trips_over_control_plane() -> None:
    message = InputUpdateMessage(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        seq_no=3,
        token_ids=[7, 8, 9],
        byte_count=11,
        input_done=True,
    )

    restored = deserialize_message(serialize_message(message))

    assert restored == message
    assert restored.token_ids == (7, 8, 9)
    assert restored.fingerprint == message.fingerprint


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": ""},
        {"session_id": " "},
        {"turn_id": 1},
        {"seq_no": True},
        {"seq_no": -1},
        {"seq_no": 1 << 63},
        {"token_ids": "not-a-token-list"},
        {"token_ids": [True]},
        {"token_ids": [-1]},
        {"token_ids": [1 << 63]},
        {"byte_count": True},
        {"byte_count": -1},
        {"input_done": 1},
        {"token_ids": [], "byte_count": 1},
    ],
)
def test_input_update_message_strict_validation(overrides: dict) -> None:
    values = {
        "request_id": "request-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "seq_no": 0,
        "token_ids": (),
        "byte_count": 0,
        "input_done": False,
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        InputUpdateMessage(**values)
