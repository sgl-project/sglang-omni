# SPDX-License-Identifier: Apache-2.0
"""Speculative CI: serve each draft of the selected model and check it is used.

A draft that loads but proposes the wrong tokens still produces correct output,
only slower, so word error rate alone cannot catch it. The test reads the mean
acceptance length from the server's decode log and fails when it is under the
draft's floor, or when the server logged none, which means the draft never ran.
Word error rate must stay within the model's own calibrated gate. The drafts and
floors live in tests/test_model/spec_ci_config.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 SPEC_CI_MODEL=qwen3-asr pytest -s -x \
        tests/test_model/test_spec_ci.py
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics

import pytest

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.wer import print_asr_speed_summary, print_asr_wer_summary
from sglang_omni.utils import find_available_port
from tests.test_model.spec_ci_config import SpeculativeCiDraft, select_spec_ci_drafts
from tests.utils import MetricCheckCollector

MODEL_NAME, DRAFTS = select_spec_ci_drafts()
SEEDTTS_ASR_EN_SAMPLES = 1088
SEEDTTS_ASR_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts",
    repo_id=DATASETS["seedtts"],
)
STARTUP_TIMEOUT = 900
ACCEPT_LENGTH_PATTERN = re.compile(r"accept len: ([0-9.]+)")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Speculative CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_EN_SAMPLES,
        split="en",
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("draft", DRAFTS, ids=lambda draft: draft.name)
def test_draft_is_accepted_and_keeps_accuracy(
    draft: SpeculativeCiDraft,
    seedtts_en_samples: list[SampleInput],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    require_cuda()
    log_file = tmp_path_factory.mktemp(f"spec_ci_{draft.name}_logs") / "server.log"
    port = find_available_port()
    with managed_omni_server(
        model_path=draft.model_path,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=draft.server_config,
        max_running_requests=draft.max_running_requests,
        timeout=STARTUP_TIMEOUT,
    ):
        results = asyncio.run(
            run_asr_seedtts_once(
                seedtts_en_samples,
                host="127.0.0.1",
                port=port,
                model_path=draft.model_path,
                lang="en",
                concurrency=draft.max_running_requests,
                warmup=draft.max_running_requests * 2,
                disable_tqdm=False,
            )
        )
    summary = results["summary"]
    speed = results["speed"]
    accept_lengths = [
        float(accept_length)
        for accept_length in ACCEPT_LENGTH_PATTERN.findall(log_file.read_text())
    ]
    accept_length_mean = statistics.fmean(accept_lengths) if accept_lengths else None

    print_asr_wer_summary(summary, draft.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL)
    print_asr_speed_summary(speed, draft.model_path, dataset=SEEDTTS_ASR_DATASET_LABEL)
    print(
        f"[Speculative CI] {draft.display_name} mean accept length "
        f"{accept_length_mean} over {len(accept_lengths)} decode log lines"
    )
    results_path = tmp_path_factory.getbasetemp() / f"spec_ci_{draft.name}_results.json"
    results_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "draft": draft.name,
                "accept_length_mean": accept_length_mean,
                "accept_length_log_lines": len(accept_lengths),
                "summary": summary,
                "speed": speed,
            },
            indent=2,
        )
    )

    checks = MetricCheckCollector(f"{draft.display_name} speculative decoding")
    checks.check(
        summary["evaluated"] == summary["total_samples"] == SEEDTTS_ASR_EN_SAMPLES,
        f"{draft.display_name} transcribed {summary['evaluated']} of "
        f"{SEEDTTS_ASR_EN_SAMPLES} samples",
    )
    checks.check(
        accept_length_mean is not None,
        f"{draft.display_name} logged no acceptance length, so the draft never ran",
    )
    checks.check(
        accept_length_mean is None or accept_length_mean >= draft.accept_length_min,
        f"{draft.display_name} mean accept length {accept_length_mean} is under "
        f"{draft.accept_length_min}",
    )
    checks.check(
        summary["corpus_wer"] <= draft.en_corpus_wer_max,
        f"{draft.display_name} EN corpus WER {summary['corpus_wer']:.4f} exceeds "
        f"{draft.en_corpus_wer_max:.4f}",
    )
    checks.assert_all()
