# S2-Pro Docs Test Notes

This directory groups notes for S2-Pro documentation validation.

The docs-mapped executable test lives at
`tests/docs/s2pro/test_docs_tts_s2pro.py`.

The benchmark and WER regression suite remains at
`tests/test_model/test_s2pro_tts_ci.py`.

The relationship is:

- `docs/` contains the published S2-Pro user documentation.
- `tests/docs/s2pro/test_docs_tts_s2pro.py` validates the concrete API examples
  shown in `docs/basic_usage/tts_s2pro.md`.
- `tests/test_model/test_s2pro_tts_ci.py` is the heavier benchmark and quality
  regression guard for the same model path.
- `.github/workflows/test-docs-tts-s2pro.yaml` runs the docs-mapped tests in a
  dedicated GPU CI workflow.

The docs are Markdown rendered by Sphinx, not executable notebooks. To keep the
docs trustworthy, the project uses strict pytest-based functional validation for
the underlying S2-Pro workflow instead of notebook execution.
