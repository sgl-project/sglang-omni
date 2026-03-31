# SPDX-License-Identifier: Apache-2.0
"""Tests for flexible inference path (per-request modality routing)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    DECODE_STAGE,
    TALKER_AR_STAGE,
    thinker_next_flexible,
    thinker_next_speech,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _make_payload(output_modalities: list[str] | None = None) -> StagePayload:
    metadata: dict = {}
    if output_modalities is not None:
        metadata["output_modalities"] = output_modalities
    return StagePayload(
        request_id="test-req",
        request=OmniRequest(inputs={}, metadata=metadata),
        data={},
    )


# ------------------------------------------------------------------
# thinker_next_flexible routing tests
# ------------------------------------------------------------------


class TestThinkerNextFlexible(unittest.TestCase):
    def test_text_only_returns_decode_only(self):
        payload = _make_payload(["text"])
        result = thinker_next_flexible("req-1", payload)
        self.assertEqual(result, DECODE_STAGE)

    def test_text_audio_returns_fanout(self):
        payload = _make_payload(["text", "audio"])
        result = thinker_next_flexible("req-1", payload)
        self.assertEqual(result, [DECODE_STAGE, TALKER_AR_STAGE])

    def test_audio_only_returns_fanout(self):
        payload = _make_payload(["audio"])
        result = thinker_next_flexible("req-1", payload)
        self.assertEqual(result, [DECODE_STAGE, TALKER_AR_STAGE])

    def test_none_modalities_returns_fanout(self):
        """When output_modalities is unset, default to full speech path."""
        payload = _make_payload(None)
        result = thinker_next_flexible("req-1", payload)
        self.assertEqual(result, [DECODE_STAGE, TALKER_AR_STAGE])

    def test_non_stage_payload_returns_fanout(self):
        """Defensive: if output is not a StagePayload, default to full path."""
        result = thinker_next_flexible("req-1", {"some": "dict"})
        self.assertEqual(result, [DECODE_STAGE, TALKER_AR_STAGE])

    def test_original_thinker_next_speech_unchanged(self):
        """Backward compat: the old function still returns full fanout."""
        result = thinker_next_speech("req-1", _make_payload(["text"]))
        self.assertEqual(result, [DECODE_STAGE, TALKER_AR_STAGE])


# ------------------------------------------------------------------
# Coordinator per-request terminal tests
# ------------------------------------------------------------------


class TestCoordinatorPerRequestTerminals(unittest.TestCase):
    """Test the coordinator's per-request terminal stage tracking."""

    def _make_coordinator(self, terminal_stages=None):
        from sglang_omni.pipeline.coordinator import Coordinator

        coord = Coordinator(
            completion_endpoint="inproc://test-comp",
            abort_endpoint="inproc://test-abort",
            entry_stage="preprocessing",
            terminal_stages=terminal_stages,
        )
        # Mock the control plane so we don't need real ZMQ
        coord.control_plane = MagicMock()
        coord.control_plane.submit_to_stage = AsyncMock()
        coord.control_plane.recv_event = AsyncMock()
        coord.control_plane.broadcast_abort = AsyncMock()
        return coord

    def test_effective_terminals_default(self):
        coord = self._make_coordinator(["decode", "code2wav"])
        self.assertEqual(
            coord._effective_terminals("req-1"), {"decode", "code2wav"}
        )

    def test_effective_terminals_with_override(self):
        coord = self._make_coordinator(["decode", "code2wav"])
        coord._request_terminals["req-1"] = {"decode"}
        self.assertEqual(coord._effective_terminals("req-1"), {"decode"})

    def test_submit_text_only_sets_override(self):
        coord = self._make_coordinator(["decode", "code2wav"])
        # Register a dummy entry stage
        coord._stages["preprocessing"] = MagicMock()
        coord._stages["preprocessing"].control_endpoint = "inproc://dummy"

        request = OmniRequest(
            inputs={}, metadata={"output_modalities": ["text"]}
        )
        asyncio.get_event_loop().run_until_complete(
            coord._submit_request("req-text", request)
        )
        self.assertIn("req-text", coord._request_terminals)
        self.assertEqual(coord._request_terminals["req-text"], {"decode"})

    def test_submit_audio_no_override(self):
        coord = self._make_coordinator(["decode", "code2wav"])
        coord._stages["preprocessing"] = MagicMock()
        coord._stages["preprocessing"].control_endpoint = "inproc://dummy"

        request = OmniRequest(
            inputs={}, metadata={"output_modalities": ["text", "audio"]}
        )
        asyncio.get_event_loop().run_until_complete(
            coord._submit_request("req-audio", request)
        )
        self.assertNotIn("req-audio", coord._request_terminals)

    def test_submit_no_modalities_no_override(self):
        coord = self._make_coordinator(["decode", "code2wav"])
        coord._stages["preprocessing"] = MagicMock()
        coord._stages["preprocessing"].control_endpoint = "inproc://dummy"

        request = OmniRequest(inputs={}, metadata={})
        asyncio.get_event_loop().run_until_complete(
            coord._submit_request("req-default", request)
        )
        self.assertNotIn("req-default", coord._request_terminals)

    def test_completion_text_only_resolves_on_decode(self):
        """Text-only request resolves after decode completes, not waiting for code2wav."""
        from sglang_omni.proto import CompleteMessage, RequestInfo, RequestState

        coord = self._make_coordinator(["decode", "code2wav"])
        # Simulate a submitted text-only request
        coord._requests["req-text"] = RequestInfo(
            request_id="req-text", state=RequestState.RUNNING
        )
        coord._request_terminals["req-text"] = {"decode"}
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        coord._completion_futures["req-text"] = future

        loop.run_until_complete(
            coord._handle_completion(
                CompleteMessage(
                    request_id="req-text",
                    from_stage="decode",
                    success=True,
                    result={"text": "hello"},
                )
            )
        )
        # Should be resolved immediately
        self.assertTrue(future.done())
        self.assertEqual(future.result(), {"text": "hello"})
        # Cleanup
        self.assertNotIn("req-text", coord._request_terminals)

    def test_completion_audio_waits_for_both(self):
        """Audio request waits for both decode and code2wav."""
        from sglang_omni.proto import CompleteMessage, RequestInfo, RequestState

        coord = self._make_coordinator(["decode", "code2wav"])
        coord._requests["req-audio"] = RequestInfo(
            request_id="req-audio", state=RequestState.RUNNING
        )
        # No override — uses global terminal_stages
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        coord._completion_futures["req-audio"] = future

        # First terminal completes
        loop.run_until_complete(
            coord._handle_completion(
                CompleteMessage(
                    request_id="req-audio",
                    from_stage="decode",
                    success=True,
                    result={"text": "hello"},
                )
            )
        )
        self.assertFalse(future.done())

        # Second terminal completes
        loop.run_until_complete(
            coord._handle_completion(
                CompleteMessage(
                    request_id="req-audio",
                    from_stage="code2wav",
                    success=True,
                    result={"audio": b"wav-data"},
                )
            )
        )
        self.assertTrue(future.done())
        self.assertEqual(
            future.result(),
            {"decode": {"text": "hello"}, "code2wav": {"audio": b"wav-data"}},
        )

    def test_abort_cleans_up_request_terminals(self):
        from sglang_omni.proto import RequestInfo, RequestState

        coord = self._make_coordinator(["decode", "code2wav"])
        coord._requests["req-abort"] = RequestInfo(
            request_id="req-abort", state=RequestState.RUNNING
        )
        coord._request_terminals["req-abort"] = {"decode"}
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        coord._completion_futures["req-abort"] = future

        loop.run_until_complete(coord.abort("req-abort"))
        self.assertNotIn("req-abort", coord._request_terminals)


if __name__ == "__main__":
    unittest.main()
