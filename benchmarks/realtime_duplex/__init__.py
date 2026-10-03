# SPDX-License-Identifier: Apache-2.0
"""Replayable native full-duplex benchmark harness (issue #2270).

Drives one conversation realtime session with paced audio input and
concurrent output recording, writes a self-contained session log, and
verdicts it offline with deterministic replay checks. CPU-only; the
local mock server speaks the same event protocol as the serve layer.
"""
