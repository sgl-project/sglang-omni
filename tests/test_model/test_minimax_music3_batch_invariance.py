# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 serving-fidelity checks.

The cookbook promises that the same lyrics, caption, seed and length return
byte-identical audio. Nothing exercised that promise, and nothing said whether
it survives batching: the AR stage runs as an SGLang engine whose logits depend
on batch composition unless deterministic inference is on, the eight-codebook
sampler is composition-invariant by construction (``test_core.py``), and the
acoustic stage decodes one request at a time.

Same shape as ``test_qwen3_tts_batch_invariance.py``: the cookbook's requests
go through a batch-one server serially and a batch-eight server concurrently
under ``enable_deterministic_inference``, the PCM has to match byte for byte,
and ``running_batch_size`` from ``/model_info`` proves the concurrent run was
batched (a music request is a CFG pair, so eight requests are sixteen rows).

Cookbook reference WAV comparison is intentionally outside this test: the
published files are not a stable cross-hardware byte-exact oracle.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import aiohttp
import pytest
import torch
import yaml

from benchmarks.benchmarker.utils import managed_omni_server
from tests.test_model.omni_router_utils import _find_available_port_range

MODEL_PATH = os.environ.get("MINIMAX_MUSIC3_TEST_MODEL", "MiniMaxAI/MiniMax-Music3")
STARTUP_TIMEOUT = 2400
REQUEST_TIMEOUT = 1800
# 250 frames is the cookbook's own audition length (10 s); it keeps a
# batch-eight round well inside a CI budget.
TEST_FRAMES = 250
SAMPLE_RATE = 32000
CHANNELS = 2
# Logical requests admitted together; the engine sees twice as many rows.
BATCH = 8

# docs/cookbook/minimax_music3.md, "Reference outputs": the requests behind the
# five attached WAVs, verbatim.
REFERENCE_REQUESTS: tuple[tuple[str, str, str, int], ...] = (
    (
        "00_lofi_hiphop",
        (
            "[Verse]\nWalking down the empty street at midnight\nStreetlights flicker like a broken dream\n"
            "I've got nothing but the sound of my own heartbeat\nEchoing through the silent concrete stream\n"
            "[Chorus]\nAnd I keep on walking\nTill the morning finds me\nLeave the night behind me"
        ),
        (
            "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, soft vinyl "
            "crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. Intimate bedroom "
            "production, gentle tape saturation, no bright cymbals."
        ),
        1,
    ),
    (
        "01_jpop_bright",
        (
            "[Verse]\nMorning light is spilling through the curtain\nEvery colour waking up with me\n"
            "[Chorus]\nRun into the day and never look back\nEverything we wanted is ahead of us"
        ),
        (
            "A cheerful J-pop song at 128 BPM in C major: bright acoustic piano, chiming electric guitar, "
            "punchy four-on-the-floor drums, and a clear female lead vocal. Polished modern pop production, "
            "wide stereo, energetic and uplifting."
        ),
        2,
    ),
    (
        "02_synthwave_moody",
        "[Intro]\n(instrumental)\n[Outro]\n(instrumental)",
        (
            "A moody synthwave instrumental at 100 BPM in D minor: pulsing analog bass arpeggio, gated "
            "reverb drum machine, wide atmospheric pads, and a soaring lead synth melody. Retro 1980s "
            "production, heavy chorus effect, cinematic and nocturnal."
        ),
        3,
    ),
    (
        "03_acoustic_folk",
        (
            "[Verse]\nI came up on a dirt road, nothing but a name\nCarried all my summers in a canvas bag\n"
            "[Chorus]\nAnd the river keeps on running\nLike it never learned to stay"
        ),
        (
            "A gentle acoustic folk ballad at 76 BPM in G major: fingerpicked steel-string guitar, soft "
            "brushed snare, subtle cello underneath, and a warm male vocal close to the microphone. Sparse "
            "and organic, natural room sound, very little compression."
        ),
        4,
    ),
    (
        "04_orchestral_epic",
        "[Intro]\n(instrumental)\n[Chorus]\nRise above the ashes of the fallen sky\nWe were never meant to say goodbye",
        (
            "An epic cinematic orchestral piece at 90 BPM in E minor: sweeping string ostinato, powerful "
            "brass swells, timpani and taiko percussion, and a distant choir. Wide concert-hall reverb, "
            "dynamic build from restrained to triumphant, no drum kit."
        ),
        5,
    ),
)

# docs/cookbook/minimax_music3.md, "Reproducibility and variations": one prompt,
# three seeds.
VARIATION_REQUESTS: tuple[tuple[str, str, str, int], ...] = tuple(
    (
        f"take_{seed}",
        "[Verse]\nCity lights are calling out my name",
        "A dreamy synthwave track with analog pads and a driving bassline at 110 BPM",
        seed,
    )
    for seed in (1, 2, 3)
)


class MusicServer(NamedTuple):
    base_url: str
    log_file: Path


def _pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == CHANNELS
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())
    assert pcm
    return pcm


def _payload(name: str, lyrics: str, caption: str, seed: int, frames: int) -> dict:
    return {
        "model": MODEL_PATH,
        "input": lyrics,
        "instructions": caption,
        "seed": seed,
        "max_new_tokens": frames,
        "response_format": "wav",
        "stream": False,
        "_name": name,
    }


def _request_body(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


async def _model_info(base_url: str, stop: asyncio.Event) -> int:
    max_batch_size = 0
    timeout = aiohttp.ClientTimeout(total=2)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not stop.is_set():
            try:
                async with session.get(f"{base_url}/model_info") as response:
                    response.raise_for_status()
                    payload = await response.json()
            except TimeoutError:
                pass
            else:
                for result in payload.get("results", []):
                    if result.get("stage") == "minimax_music3_ar":
                        max_batch_size = max(
                            max_batch_size,
                            int(result.get("data", {}).get("running_batch_size", 0)),
                        )
            await asyncio.sleep(0.05)
    return max_batch_size


async def _generate(
    session: aiohttp.ClientSession, base_url: str, payload: dict
) -> bytes:
    async with session.post(
        f"{base_url}/v1/audio/speech", json=_request_body(payload)
    ) as response:
        response.raise_for_status()
        return _pcm(await response.read())


async def _generate_serial(base_url: str, payloads: list[dict]) -> list[bytes]:
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return [await _generate(session, base_url, payload) for payload in payloads]


async def _generate_batch(
    base_url: str, payloads: list[dict]
) -> tuple[list[bytes], int]:
    stop = asyncio.Event()
    poller = asyncio.create_task(_model_info(base_url, stop))
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            outputs = await asyncio.gather(
                *(_generate(session, base_url, payload) for payload in payloads)
            )
    finally:
        stop.set()
        max_batch_size = await poller
    return list(outputs), max_batch_size


@contextmanager
def _music_server(
    tmp_path_factory: pytest.TempPathFactory, name: str
) -> Iterator[MusicServer]:
    config_path = tmp_path_factory.mktemp(f"minimax_music3_{name}") / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "MiniMaxMusic3PipelineConfig",
                "model_path": MODEL_PATH,
                "stages": {
                    "minimax_music3_ar": {
                        "engine": {
                            # Batch-invariant kernels for the backbone; the
                            # Qwen3-TTS test pins the same backend.
                            "enable_deterministic_inference": True,
                            "attention_backend": "triton",
                            "max_running_requests": BATCH,
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    port = _find_available_port_range(1)
    log_file = tmp_path_factory.mktemp(f"minimax_music3_{name}_logs") / "server.log"
    with managed_omni_server(
        model_path=MODEL_PATH,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=str(config_path),
        timeout=STARTUP_TIMEOUT,
    ):
        yield MusicServer(f"http://127.0.0.1:{port}", log_file)


def _test_payloads() -> list[dict]:
    payloads = [
        _payload(name, lyrics, caption, seed, TEST_FRAMES)
        for name, lyrics, caption, seed in REFERENCE_REQUESTS + VARIATION_REQUESTS
    ]
    assert len(payloads) == BATCH
    return payloads


@pytest.mark.benchmark
def test_minimax_music3_deterministic_batch_invariance(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Match fresh batch-one and batch-eight executions, and repeats of one request."""
    if not torch.cuda.is_available():
        pytest.skip("MiniMax Music 3 batch invariance requires CUDA")

    payloads = _test_payloads()
    payload = payloads[0]

    with _music_server(tmp_path_factory, "b1") as server:
        b1 = asyncio.run(_generate_serial(server.base_url, payloads))
        repeated_b1 = asyncio.run(_generate_serial(server.base_url, [payload] * 2))

    with _music_server(tmp_path_factory, "b8") as server:
        mixed_b8, batched_rows = asyncio.run(_generate_batch(server.base_url, payloads))
        repeated_b8, repeated_rows = asyncio.run(
            _generate_batch(server.base_url, [payload] * BATCH)
        )

    # Positive evidence that the concurrent runs were batched: each music
    # request is a conditioned/unconditioned CFG pair, two engine rows.
    assert batched_rows == 2 * BATCH, batched_rows
    assert repeated_rows == 2 * BATCH, repeated_rows

    # Every request is deterministic in its seed across a server restart, and
    # independent of what shares its batch.
    for index, (name, one, eight) in enumerate(
        zip((p["_name"] for p in payloads), b1, mixed_b8, strict=True)
    ):
        assert (
            one == eight
        ), f"{name} (request {index}) differs between batch-one and batch-eight"
    assert all(pcm == b1[0] for pcm in repeated_b1)
    assert all(pcm == b1[0] for pcm in repeated_b8)
    # Different seeds of one prompt are different takes, not the same audio.
    takes = b1[len(REFERENCE_REQUESTS) :]
    assert len({hashlib.sha256(pcm).hexdigest() for pcm in takes}) == len(takes)
