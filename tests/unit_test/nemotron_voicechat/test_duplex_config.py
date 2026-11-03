# SPDX-License-Identifier: Apache-2.0
"""The session variant and the capacity it gives each stage."""

from sglang_omni.models.nemotron_voicechat.config import Variants
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    CONTEXT_LENGTH,
    NemotronVoiceChatDuplexPipelineConfig,
)


def test_session_variant_serves_duplex_and_needs_realtime() -> None:
    assert Variants["session"] is NemotronVoiceChatDuplexPipelineConfig
    assert NemotronVoiceChatDuplexPipelineConfig.is_realtime_only


def test_duplex_capacity_follows_max_sessions() -> None:
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
