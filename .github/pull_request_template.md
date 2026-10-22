<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

<!-- Explain the purpose of this PR and the goals it aims to achieve. -->

## Modifications

<!-- Describe the changes made in this PR. -->

## Related Issues

<!-- Link to any related issues here. e.g. "Fixes #123" or "Closes #456" -->

## Accuracy Test

<!-- If this PR affects model-side code (e.g., kernels, model architecture), please provide accuracy test results. Ref: https://docs.sglang.io/docs/developer_guide/evaluating_new_models -->

## Benchmark & Profiling

<!-- If this PR is expected to impact performance, please provide benchmark and profiling results. Ref: https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling -->

## Checklist

- [ ] Format your code according with pre-commit.
- [ ] Add unit tests.
- [ ] Update documentation / docstrings / example tutorials as needed.
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

Lint, Test Layout, Docs Check, Rust Router and the NPU image build run on GitHub-hosted runners for every PR, and `lint` must pass before merging. The rest of CI runs on self-hosted runners, starts only after lint passes, and requires a maintainer to add CI labels. Once labeled, every subsequent push re-triggers CI as long as the labels remain. Draft PRs are skipped even if labeled.

- `run-all-ci` runs every stage below.
- `run-lint-only` runs lint only. The self-hosted gates succeed and run no tests. Use `/tag-and-rerun-ci lint` to add it. It does not turn off a stage label that is also present.
- A stage label runs one stage: `run-asr-ci`, `run-tts-ci`, `run-mps-ci` and `run-omni-ci` on H100, `run-intel-cpu-ci` (Intel Xeon), `run-intel-xpu-ci` (Intel GPU) and `run-mac-ci` (Apple Silicon). The hardware stages still skip a PR that touches none of their paths.
- Any stage label also runs the H100 unit tests.
- `run-unit-test-ci-only` runs just the H100 unit tests after shared setup when no other stage labels or `run-all-ci` are present. Use `/tag-and-rerun-ci unit-test` to add it and re-run completed CI.
- A model label picks the model of every selected stage that accepts it. It does not select a stage. Without one, a selected stage draws its model at random. Keep at most one model label per stage.
  - ASR: `run-fun-asr-nano-2512`, `run-qwen3-asr-1.7b`, `run-whisper-large-v3`
  - TTS and MPS: `run-higgs-tts-3-4b`, `run-moss-tts-local-transformer-v1.5`. `run-mps-ci` is the only MPS stage label. With both stage labels, one model label selects that model for both. With only `run-tts-ci`, MPS does not run.
  - TTS: `run-qwen3-tts-12hz-1.7b-base`, `run-qwen3-tts-12hz-1.7b-customvoice` (label only, never drawn), `run-fun-cosyvoice3-0.5b-2512`
  - Omni: `run-qwen3-omni-30b-a3b-instruct`, `run-minicpm-o-4.5`

Use `/tag-and-rerun-ci` with short targets, for example `/tag-and-rerun-ci tts asr` or `/tag-and-rerun-ci qwen3-tts qwen3-asr`. Use `/tag-and-rerun-full-ci` for all stages, with optional model targets. Bare `/tag-and-rerun-ci` and full GitHub label arguments are not accepted. Existing stage selections stay enabled. Use `/rerun-failed-ci` to retry failed jobs.

The rules live in `.github/scripts/resolve_ci_labels.py`, and [docs/developer_reference/ci_labels.md](https://github.com/sgl-project/sglang-omni/blob/main/docs/developer_reference/ci_labels.md) describes what every stage covers.
