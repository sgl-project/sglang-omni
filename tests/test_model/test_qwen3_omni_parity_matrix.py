# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.qwen3_omni_parity.topdown_matrix import build_topdown_matrix


def test_topdown_matrix_marks_first_failing_contract():
    runtime_capture = {
        "request_id": "runtime-test-1",
        "prompt": "hello",
        "thinker_generated_ids": [1, 2, 3],
        "runtime_delta_text": "ok",
        "runtime_final_text_from_event": "ok",
        "runtime_decode_complete_text": "ok",
        "duration_sec": 10.0,
        "audio_sha256": "runtime-audio",
    }
    hf_capture = {
        "prompt": "hello",
        "generated_ids": [1, 2, 3],
        "generated_text": "ok",
        "wav_duration_sec": 5.0,
        "audio_sha256": "hf-audio",
        "layer0_codes": [11, 22, 44],
        "codec_codes": [
            [11, 22, 44],
            [101, 202, 404],
        ],
    }
    runtime_cp_dump = {
        "layer0_codes": torch.tensor([11, 22, 33]),
        "output_codes": torch.tensor(
            [
                [11, 101],
                [22, 303],
                [33, 505],
            ]
        ),
    }
    runtime_prefill_dump = {
        "input_ids": torch.tensor([1, 2]),
        "input_embeds": torch.ones(2, 4),
        "tts_pad_embed": torch.ones(4),
        "trailing_text_hidden": torch.zeros(0, 4),
    }
    hf_prefill_dump = {
        "input_ids": torch.tensor([1, 2]),
        "input_embeds": torch.ones(2, 4),
        "tts_pad_embed": torch.ones(4),
        "trailing_text_hidden": torch.zeros(0, 4),
    }

    result = build_topdown_matrix(
        runtime_capture,
        hf_capture,
        runtime_cp_dump=runtime_cp_dump,
        runtime_prefill_dump=runtime_prefill_dump,
        hf_prefill_dump=hf_prefill_dump,
    )

    assert result["contracts"]["thinker_text"]["status"] == "pass"
    assert result["contracts"]["talker_prefill"]["status"] == "pass"
    assert result["contracts"]["code_predictor_full_code_rows"]["status"] == "fail"
    assert result["first_failing_contract"] == "code_predictor_full_code_rows"
