# Docs Validation Tests

This directory contains pytest-based validation for user-facing workflows that
are documented under `docs/`.

## Relationship to `docs/`

- `docs/` contains the Sphinx source for published documentation.
- `tests/docs/` contains executable checks that validate key documented
  behavior.
- The documentation build verifies that pages render correctly.
- The tests in this directory verify that important documented workflows still
  work correctly end to end.

## Why these tests exist

The S2-Pro documentation is written as Markdown and rendered by Sphinx. It does
not rely on Jupyter notebooks for execution. To keep the documentation accurate,
we use strict functional tests that exercise the underlying workflow directly.

For S2-Pro TTS, that means validating the documented path with:

- end-to-end server startup
- benchmark generation for documented modes
- streaming and non-streaming consistency
- WER-based regression checks

These tests are intentionally stronger than a docs build. A page can render
successfully while the documented commands or quality expectations have already
regressed. Keeping the tests here makes that contract explicit.
