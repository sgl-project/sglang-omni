# SPDX-License-Identifier: Apache-2.0
"""Static checks for the Ming-Omni cookbook.

These tests avoid live model launches. They lock the public docs contract:
the page must stay discoverable, include supported launchers, and preserve
benchmark caveats that are easy to accidentally overstate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC_PATH = Path("docs/cookbook/ming_omni.md")
INDEX_PATH = Path("docs/index.rst")
README_PATH = Path("README.md")

pytestmark = pytest.mark.docs


def _doc() -> str:
    assert DOC_PATH.exists(), "docs/cookbook/ming_omni.md does not exist"
    return DOC_PATH.read_text(encoding="utf-8")


def test_ming_cookbook_is_in_sphinx_nav() -> None:
    index = INDEX_PATH.read_text(encoding="utf-8")
    assert "cookbook/ming_omni.md" in index


def test_readme_links_to_cookbook() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "docs/cookbook/" in readme


def test_ming_cookbook_has_required_sections() -> None:
    doc = _doc()
    required_headings = [
        "# Ming-Omni",
        "## Prerequisites",
        "## Architecture",
        "## Server Configuration",
        "## Input and Output Examples",
        "## Request Parameters",
        "## Benchmark Results",
        "## Known Limitations",
    ]
    for heading in required_headings:
        assert heading in doc


def test_ming_cookbook_uses_generic_serve() -> None:
    doc = _doc()
    for snippet in [
        "sgl-omni serve",
        "--model-path",
        "inclusionAI/Ming-flash-omni-2.0",
        "--text-only",
        "--thinker-tp-size",
        "--thinker-gpus",
        "--talker-gpu",
        "--cpu-offload-gb",
        "--mem-fraction-static",
        "--stages.2.tp_size",
        "--stages.2.gpu",
    ]:
        assert snippet in doc
    assert "examples/run_ming_omni_server.py" not in doc
    assert "examples/run_ming_omni_speech_server.py" not in doc
    assert "--enable-streaming-tts" not in doc


def test_ming_cookbook_documents_supported_request_shapes() -> None:
    doc = _doc()
    for snippet in [
        "/v1/chat/completions",
        '"modalities": ["text"]',
        '"modalities": ["text", "audio"]',
        '"/path/to/cars.jpg"',
        '"/path/to/question.wav"',
        '"/path/to/demo.mp4"',
        '"stream": true',
    ]:
        assert snippet in doc


def test_ming_cookbook_surfaces_benchmarks() -> None:
    doc = _doc()
    required_snippets = [
        "### Text Thinker (GSM8K)",
        "### Image-Text (MMMU)",
        "### Non-Streaming Talker",
        "### Streaming Talker",
        "### Audio Equivalence",
        "GSM8K",
        "0.615 qps",
        "4.608 qps",
        "0.996 qps",
        "2.02 req/s",
        "0.236 s",
        "0.509 s",
    ]
    for snippet in required_snippets:
        assert snippet in doc


def test_ming_cookbook_preserves_benchmark_caveats() -> None:
    doc = _doc()
    required_phrases = [
        "The streaming measurements are from PR/local-patch evidence",
        "H100-class",
        "not universal guarantees",
        "text-only `stream=true` currently emits an aggregate text chunk",
    ]
    for phrase in required_phrases:
        assert phrase in doc
    assert "sglang_omni_v1" not in doc


def test_ming_cookbook_makes_no_vllm_comparison() -> None:
    doc = _doc().lower()
    assert "vllm" not in doc
    assert "omnicompare" not in doc
