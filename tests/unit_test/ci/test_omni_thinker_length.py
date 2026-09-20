# SPDX-License-Identifier: Apache-2.0
"""Model-specific length errors retain the shared HTTP and boundary contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.test_model import test_qwen3_omni_thinker_length as length_ci
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS

MINICPM_OVERLONG = (
    "Request requires more tokens than the thinker KV cache can hold "
    "(input_tokens=10009, max_new_tokens=16, required_tokens=10025, kv_capacity=127)."
)
MINICPM_TOTAL = (
    "Request requires more tokens than the thinker KV cache can hold "
    "(input_tokens=9, max_new_tokens=200, required_tokens=209, kv_capacity=127)."
)
QWEN_OVERLONG = (
    "The input (10009 tokens) is longer than the model's context length (128 tokens)."
)
QWEN_TOTAL = "Requested token count exceeds the model's maximum context length"


@pytest.mark.parametrize(
    ("model", "overlong", "detail", "accepted"),
    [
        ("minicpmo", True, MINICPM_OVERLONG, True),
        ("minicpmo", False, MINICPM_TOTAL, True),
        ("minicpmo", False, "Bad request: unknown model", False),
        (
            "minicpmo",
            False,
            MINICPM_TOTAL.replace("max_new_tokens=200", "max_new_tokens=199"),
            False,
        ),
        (
            "minicpmo",
            False,
            MINICPM_TOTAL.replace("required_tokens=209", "required_tokens=210"),
            False,
        ),
        (
            "minicpmo",
            False,
            MINICPM_TOTAL.replace("kv_capacity=127", "kv_capacity=128"),
            False,
        ),
        ("minicpmo", False, QWEN_TOTAL, False),
        ("qwen3-omni", True, QWEN_OVERLONG, True),
        ("qwen3-omni", False, QWEN_TOTAL, True),
        ("qwen3-omni", False, MINICPM_TOTAL, False),
    ],
)
def test_length_ci_checks_selected_model_error(
    model: str,
    overlong: bool,
    detail: str,
    accepted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        status_code=400, text=detail, json=lambda: {"detail": detail}
    )
    monkeypatch.setattr(length_ci, "_post_chat", lambda *args, **kwargs: response)
    test = (
        length_ci.test_overlong_prompt_returns_400
        if overlong
        else length_ci.test_total_token_overflow_returns_400
    )
    with nullcontext() if accepted else pytest.raises(AssertionError):
        test(OMNI_CI_PRESETS[model], SimpleNamespace(port=8000))
