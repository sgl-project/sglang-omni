# Docs Validation Tests

This directory contains pytest-based validation for user-facing workflows that
are documented under `docs/`.

## Relationship to `docs/`

- `docs/` contains the Sphinx source for published documentation.
- `tests/docs/` contains executable checks that validate key documented
  behavior.
- `tests/docs/s2pro/test_docs_tts_s2pro.py` maps the S2-Pro usage guide to
  executable pytest coverage.
- The documentation build verifies that pages render correctly.
- The tests in this directory verify that important documented workflows still
  work correctly end to end.

## Why these tests exist

The S2-Pro documentation is written as Markdown and rendered by Sphinx. It does
not rely on Jupyter notebooks for execution. To keep the documentation accurate,
we use strict functional tests that exercise the underlying workflow directly.

For S2-Pro TTS, that means validating the documented path with:

- end-to-end server startup
- documented non-streaming requests
- documented streaming SSE requests
- documented parameterized requests

These docs-facing tests run in the dedicated workflow
`.github/workflows/test-docs-tts-s2pro.yaml`. They are separate from the
benchmark and WER regression suite in `tests/test_model/test_s2pro_tts_ci.py`.

These tests are intentionally stronger than a docs build. A page can render
successfully while the documented commands or quality expectations have already
regressed. Keeping the tests here makes that contract explicit.
