# SPDX-License-Identifier: Apache-2.0
"""The session variant and the capacity it gives each stage."""

import pytest

from sglang_omni.models.nemotron_voicechat.config import Variants
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    CONTEXT_LENGTH,
    NemotronVoiceChatDuplexPipelineConfig,
)
from sglang_omni.platforms import current_platform


def test_session_variant_serves_duplex_and_needs_realtime() -> None:
    assert Variants["session"] is NemotronVoiceChatDuplexPipelineConfig
    assert NemotronVoiceChatDuplexPipelineConfig.is_realtime_only


@pytest.mark.parametrize("is_npu", [False, True], ids=["default", "ascend"])
def test_duplex_capacity_follows_max_sessions(
    monkeypatch: pytest.MonkeyPatch, is_npu: bool
) -> None:
    monkeypatch.setattr(current_platform, "is_npu", lambda: is_npu)
    config = NemotronVoiceChatDuplexPipelineConfig(model_path="unused", max_sessions=4)
    assert config.stage_factory_kwargs("perception") == {"max_open_sessions": 4}
    assert config.stage_factory_kwargs("code2wav") == {"max_open_sessions": 4}
    for stage in ("thinker", "talker"):
        assert config.stage_factory_kwargs(stage) == {
            "server_args_overrides": {
                "max_running_requests": 5,
                "max_total_tokens": 5 * CONTEXT_LENGTH,
            }
        }


def test_explicit_duplex_engine_capacity_wins() -> None:
    config = NemotronVoiceChatDuplexPipelineConfig(model_path="unused", max_sessions=4)
    config.stage_named("thinker").engine.max_total_tokens = 1234
    config.stage_named("talker").engine.mem_fraction_static = 0.3
    for stage in ("thinker", "talker"):
        assert config.stage_factory_kwargs(stage) == {
            "server_args_overrides": {"max_running_requests": 5}
        }
