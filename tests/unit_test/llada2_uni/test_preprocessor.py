from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sglang_omni.models.llada2_uni.components import preprocessor as mod
from sglang_omni.preprocessing import cache_key, resource_connector
from sglang_omni.proto import OmniRequest, StagePayload


def test_llada2_images_follow_the_media_policy_before_the_cache_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    monkeypatch.setenv(resource_connector.ALLOWED_LOCAL_MEDIA_PATH_ENV, str(allowed))
    monkeypatch.setenv(resource_connector.ALLOWED_MEDIA_DOMAINS_ENV, "")
    monkeypatch.setattr(resource_connector, "_global_connector", None)
    reads = []
    monkeypatch.setattr(cache_key, "hash_file_sampled", lambda path: reads.append(path))
    pre = mod.LLaDA2Preprocessor.__new__(mod.LLaDA2Preprocessor)
    pre.validate_messages = lambda messages: None
    inputs = {
        "messages": [{"role": "user", "content": "What color is it?"}],
        "images": [str(outside)],
    }
    payload = StagePayload(
        request_id="llada2-policy", request=OmniRequest(inputs=inputs), data=None
    )

    with pytest.raises(ValueError, match="not within allowed directory"):
        asyncio.run(pre(payload))
    assert reads == []
