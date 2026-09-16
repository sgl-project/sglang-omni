# SPDX-License-Identifier: Apache-2.0
"""Microphone buffering across delayed acknowledgments, without dropping PCM."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from examples import nemotron_diarization_live as client


class MicrophoneLink:
    class CallbackAbort(Exception):
        pass

    def __init__(self, monkeypatch, stalled_blocks):
        self.stalled_blocks = stalled_blocks
        self.now = 0
        self.active = False
        self.captured = []
        self.sent = []
        self.replies = 0
        monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: self.now))
        monkeypatch.setitem(
            sys.modules,
            "sounddevice",
            SimpleNamespace(RawInputStream=self.open, CallbackAbort=self.CallbackAbort),
        )

    def open(self, **kwargs):
        self.callback = kwargs["callback"]
        return self

    def capture(self, count):
        for _ in range(count):
            pcm = len(self.captured).to_bytes(2, "little") * 1600
            self.captured.append(pcm)
            try:
                self.callback(pcm, 1600, None, False)
            except self.CallbackAbort:
                break

    def __enter__(self):
        self.active = True
        self.capture(1)
        return self

    def __exit__(self, *args):
        self.active = False

    async def send(self, audio):
        self.sent.append(audio)

    async def recv(self):
        # Capture continues while the client awaits the network. Start with a
        # stall, then model 200 ms acknowledgments for 100 ms capture blocks.
        if self.active:
            blocks = self.stalled_blocks if self.replies == 0 else 2
            self.capture(blocks)
            self.now += blocks / 10
        self.replies += 1
        return json.dumps({"type": "audio.ack"})


def test_delayed_acks_batch_and_flush_every_sample_in_order(monkeypatch):
    link = MicrophoneLink(monkeypatch, stalled_blocks=30)
    asyncio.run(client.microphone(link, seconds=5))

    assert b"".join(link.sent) == b"".join(link.captured)
    assert max(map(len, link.sent)) == 32000
    assert all(0 < len(packet) <= 32000 for packet in link.sent)
    assert len(link.sent) < len(link.captured)
    assert not link.active


def test_sustained_stall_reports_overflow_instead_of_dropping_audio(monkeypatch):
    link = MicrophoneLink(monkeypatch, stalled_blocks=51)
    with pytest.raises(RuntimeError, match="buffer reached 5 seconds"):
        asyncio.run(client.microphone(link, seconds=30))
    assert link.sent == link.captured[:1]
    assert not link.active
