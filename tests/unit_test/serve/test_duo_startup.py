"""Realization model work leaves the websocket loop live and closes after GPU work."""

import asyncio
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from scripts import serve_duo_realtime
from sglang_omni.models.duo.live import LiveDuoSessionHooks
from sglang_omni.models.duo.realtime import (
    DuoLiveResources,
    load_duo_live_resources,
    warmup_duo_realtime,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import TimedChunk
from sglang_omni.scheduling.session import SessionContext


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [None, 1])
async def test_realization_warmup_repeats_media_in_one_session_and_releases_failure(
    tmp_path: Path, fail_at: int | None
) -> None:
    """Warmup must advance history while cycling inputs and always release state."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(34):
        value = 32 if index < 17 else 96
        Image.new("RGB", (32, 32), (value, value, value)).save(
            frames / f"{index:03d}.jpg"
        )
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (32, 64)).save(reference)
    samples = 11333
    (tmp_path / "audio.pcm").write_bytes(b"\1\0" * samples + b"\2\0" * samples)
    hooks = Mock(spec=LiveDuoSessionHooks)
    observed: list[tuple[int, float, int, bytes, bool]] = []

    def append(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        observation = chunk.payload
        observed.append(
            (
                chunk.seq,
                chunk.t_start_ms,
                observation["frames_rgb"][0],
                observation["audio_pcm16"][:2],
                chunk.eos,
            )
        )
        if chunk.seq == fail_at:
            raise RuntimeError("warmup failed")
        else:
            return payload

    hooks.append.side_effect = append
    resources = DuoLiveResources(hooks, max_blocks=3)
    outcome = (
        nullcontext()
        if fail_at is None
        else pytest.raises(RuntimeError, match="warmup failed")
    )
    with outcome:
        await warmup_duo_realtime(
            resources,
            sample_directory=tmp_path,
            reference_image=reference,
            blocks=3,
            height=64,
            width=32,
        )
    expected = [
        (0, 0, 32, b"\1\0", False),
        (1, 17000 / 24, 96, b"\2\0", False),
        (2, 34000 / 24, 32, b"\1\0", True),
    ]
    assert observed == (expected if fail_at is None else expected[:2])
    hooks.open.assert_called_once()
    hooks.close.assert_called_once_with(hooks.open.call_args.args[0])
    assert hooks.open.call_args.args[1].inputs["height"] == 64


@pytest.mark.parametrize(
    "workspace_capacity,reference_count,warmup_fails",
    [(None, 1, False), (131072, 2, False), (131072, 2, True)],
)
def test_realization_startup_warmup_precedes_readiness_and_cleans_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warmup_fails: bool,
    reference_count: int,
    workspace_capacity: int | None,
) -> None:
    """Startup hooks must work on the deployed FastAPI lifespan API."""
    hooks = Mock(spec=LiveDuoSessionHooks)
    hooks.planner = Mock()
    hooks.planner.system_prompt_fragments.return_value = ("prefix", "suffix")
    hooks.codecs = Mock()
    hooks.codecs.video_lookahead_blocks = 0
    cleanup = Mock()
    resources = DuoLiveResources(hooks, max_blocks=3, shutdown=cleanup)
    warmup_threads: list[int] = []

    async def warm_model(
        resources: DuoLiveResources,
        *,
        sample_directory: Path,
        reference_image: Path,
        blocks: int,
        height: int,
        width: int,
        description: dict[str, str] | None,
    ) -> None:
        warmup_threads.append(await asyncio.to_thread(threading.get_ident))
        if warmup_fails:
            raise RuntimeError("warmup failed")
        else:
            pass

    warming = AsyncMock(side_effect=warm_model)
    loading = Mock(return_value=resources)
    monkeypatch.setattr(serve_duo_realtime, "load_duo_live_resources", loading)
    monkeypatch.setattr(serve_duo_realtime, "warmup_duo_realtime", warming)
    monkeypatch.setattr(serve_duo_realtime, "freeze_gc", Mock())
    monkeypatch.setattr(serve_duo_realtime.gc, "collect", Mock())
    monkeypatch.setattr(
        "sys.argv",
        [
            "serve_duo_realtime.py",
            "--codec-model-path",
            "model",
            "--intent-model-path",
            "intent",
            "--warmup-sample-directory",
            str(tmp_path),
            "--warmup-reference-image",
            str(tmp_path / "reference.jpg"),
        ]
        + (
            ["--warmup-reference-image", str(tmp_path / "other.jpg")]
            if reference_count == 2
            else []
        )
        + (
            ["--realization-kv-workspace-capacity-tokens", str(workspace_capacity)]
            if workspace_capacity is not None
            else []
        ),
    )

    def serve(application: FastAPI, *, host: str, port: int, log_level: str) -> None:
        with TestClient(application) as client:
            assert warming.await_count == reference_count
            assert [call.kwargs["blocks"] for call in warming.await_args_list] == (
                [40] if reference_count == 1 else [40, 1]
            )
            live_thread = client.portal.call(asyncio.to_thread, threading.get_ident)
            assert warmup_threads == [live_thread] * reference_count
            assert client.get("/duo/configuration").status_code == 200
            cleanup.assert_not_called()

    monkeypatch.setattr(serve_duo_realtime.uvicorn, "run", serve)
    if warmup_fails:
        with pytest.raises(RuntimeError, match="warmup failed"):
            serve_duo_realtime.main()
    else:
        serve_duo_realtime.main()
    assert warming.await_count == (1 if warmup_fails else reference_count)
    cleanup.assert_called_once()
    assert (
        loading.call_args.kwargs["realization_kv_workspace_capacity_tokens"]
        == workspace_capacity
    )
    assert loading.call_args.kwargs["realization_kv_window_tokens"] == 65536


@pytest.mark.parametrize(
    "backend,capacity,message",
    [
        ("native", 65536, "fit the KV window"),
        ("reference", 131072, "requires the native backend"),
    ],
)
def test_realization_workspace_rejects_incompatible_reservation_before_loading_weights(
    backend: Literal["native", "reference"], capacity: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_duo_live_resources(
            model_path="unused",
            intent_model_path="unused",
            intent_adapter_path="",
            raven_lora_path="",
            trainables_path="",
            realization_checkpoint="unused",
            realization_backend=backend,
            intent_backend="native",
            realization_kv_workspace_capacity_tokens=capacity,
        )
