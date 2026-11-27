# SPDX-License-Identifier: Apache-2.0
"""Structured function-call grading for spoken BFCL examples."""

from benchmarks.metrics.bfcl_audio import matches_calls


def test_parallel_calls_match_distinct_references_with_overlapping_answers() -> None:
    references = [
        {"lookup": {"city": ["New York", "NYC"]}},
        {"lookup": {"city": ["NYC"]}},
    ]
    required = {"lookup": {"city"}}
    first = ("lookup", {"city": "NYC"})
    second = ("lookup", {"city": "New York"})
    assert matches_calls([first, second], references, required)
    assert not matches_calls([second, second], references, required)


def test_nested_argument_alternatives_and_optional_omission() -> None:
    references = [{"search": {"filters": [[{"value": [3, 5]}]], "limit": ["", 10]}}]
    required = {"search": {"filters"}}
    assert matches_calls(
        [("search", {"filters": [{"value": 5}]})], references, required
    )
    assert not matches_calls([("search", {"filters": [{}]})], references, required)
    assert not matches_calls(
        [("search", {"filters": [{"value": 5}], "limit": ""})], references, required
    )
