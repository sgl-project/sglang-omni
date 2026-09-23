# SPDX-License-Identifier: Apache-2.0
"""Repeat a measured corpus with distinct request IDs and retained provenance."""

from dataclasses import replace

from benchmarks.dataset.seedtts import SampleInput


def repeat_corpus(
    samples: list[SampleInput], repeats: int
) -> tuple[list[SampleInput], dict[str, str]]:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("corpus_repeats must be a positive integer")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Sample IDs must be unique")
    if repeats == 1:
        return samples, {}
    expanded = []
    sources = {}
    for cycle in range(repeats):
        for index, sample in enumerate(samples):
            request_id = f"restage-repeat-{cycle}-{index}"
            expanded.append(replace(sample, sample_id=request_id))
            sources[request_id] = sample.sample_id
    return expanded, sources
