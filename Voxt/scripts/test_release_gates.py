#!/usr/bin/env python3
"""Release gates for the short-open latency rows (review 138 Must 4).

Every kept group must show the native min_preview_samples it ran with
(the actual value, or null with an unavailable_reason; a numeric 0, a
missing field and a missing reason are all invalid) and the distinct
text hashes the kept decision was written against.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_loop as loop  # noqa: E402


def row(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "scenario": "cold",
        "text_sha256": "a" * 64,
        "min_preview_samples": 20000,
    }
    base.update(overrides)
    return base


class MinPreviewGateTests(unittest.TestCase):
    def test_numeric_zero_is_invalid(self):
        self.assertFalse(loop.min_preview_samples_valid(row(min_preview_samples=0)))

    def test_missing_field_is_invalid(self):
        broken = row()
        broken.pop("min_preview_samples")
        self.assertFalse(loop.min_preview_samples_valid(broken))

    def test_null_without_reason_is_invalid(self):
        self.assertFalse(loop.min_preview_samples_valid(row(min_preview_samples=None)))
        self.assertFalse(
            loop.min_preview_samples_valid(
                row(
                    min_preview_samples=None,
                    min_preview_samples_unavailable_reason="  ",
                )
            )
        )

    def test_legal_null_passes(self):
        self.assertTrue(
            loop.min_preview_samples_valid(
                row(
                    min_preview_samples=None,
                    min_preview_samples_unavailable_reason="no native moss_preview_timing event for this model or runtime",
                )
            )
        )

    def test_actual_value_passes(self):
        self.assertTrue(loop.min_preview_samples_valid(row(min_preview_samples=1250)))


class TextHashGateTests(unittest.TestCase):
    def test_rows_gate_requires_hashes(self):
        self.assertFalse(loop.latency_rows_valid([row(text_sha256="")]))
        self.assertTrue(loop.latency_rows_valid([row()]))

    def test_distinct_hashes_cold_warm_pair(self):
        same = "b" * 64
        rows = [
            row(scenario="cold", text_sha256=same),
            row(scenario="warm", text_sha256=same),
        ]
        self.assertEqual(loop.distinct_text_hashes(rows), [same])
        rows[1]["text_sha256"] = "c" * 64
        self.assertEqual(loop.distinct_text_hashes(rows), sorted({same, "c" * 64}))


if __name__ == "__main__":
    unittest.main()
