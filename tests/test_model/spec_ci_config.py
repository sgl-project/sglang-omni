# SPDX-License-Identifier: Apache-2.0
"""Drafts that Speculative CI serves, and the floors each one must clear.

CI selects one model per job through the SPEC_CI_MODEL environment variable,
and tests/test_model/test_spec_ci.py serves each of that model's drafts in
turn. Acceptance length and word error rate do not depend on the GPU, so these
floors are measured once rather than calibrated on the CI host. Throughput is
reported, not gated: the model's own stage gates speed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH
from tests.test_model.asr_ci_config import QWEN3_ASR_EN_CORPUS_WER_THRESHOLD


@dataclass(frozen=True, kw_only=True)
class SpeculativeCiDraft:
    name: str
    display_name: str
    model_path: str
    server_config: str
    max_running_requests: int
    accept_length_min: float
    en_corpus_wer_max: float


# note (wilsonzheng0327): a draft that loads but proposes the wrong tokens accepts
# about one token per verify, so each floor sits well under the measured mean and
# well over that.
QWEN3_ASR_DFLASH_ACCEPT_LENGTH_MIN = 5.0  # measured 5.88 and 5.90 on H200


SPEC_CI_DRAFTS: dict[str, tuple[SpeculativeCiDraft, ...]] = {
    "qwen3-asr": (
        SpeculativeCiDraft(
            name="qwen3_asr_dflash",
            display_name="Qwen3-ASR-1.7B DFlash",
            model_path=QWEN3_ASR_MODEL_PATH,
            server_config="examples/configs/qwen3_asr_1_7b_dflash.yaml",
            max_running_requests=8,
            accept_length_min=QWEN3_ASR_DFLASH_ACCEPT_LENGTH_MIN,
            en_corpus_wer_max=QWEN3_ASR_EN_CORPUS_WER_THRESHOLD,
        ),
    ),
}


def select_spec_ci_drafts(
    model_name: str | None = None,
) -> tuple[str, tuple[SpeculativeCiDraft, ...]]:
    selected = model_name or os.environ.get("SPEC_CI_MODEL", "qwen3-asr")
    drafts = SPEC_CI_DRAFTS.get(selected)
    if drafts is None:
        allowed = ", ".join(sorted(SPEC_CI_DRAFTS))
        raise ValueError(
            f"Unsupported SPEC_CI_MODEL={selected!r}; expected one of: {allowed}"
        )
    else:
        pass
    return selected, drafts
