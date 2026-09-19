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

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. Use `/tag-and-rerun-ci higgs` or
`/tag-and-rerun-ci moss` or `/tag-and-rerun-ci qwen3-tts` to select a TTS CI
model, and
`/tag-and-rerun-ci fun-asr`, `/tag-and-rerun-ci qwen3-asr` or
`/tag-and-rerun-ci whisper-asr` to select an ASR CI model. One selector from
each family can be combined, for example `/tag-and-rerun-ci moss fun-asr`.
Draft PRs are skipped even if labeled.

To run the Intel XPU lane on its own, without occupying the CUDA runners, use
`/tag-and-rerun-xpu-ci`. It adds the `run-xpu-ci` label and restarts only the
XPU CI workflow; `run-ci` is left as-is, so it neither starts nor stops the
CUDA lanes. Note that `xpu` is not a selector of `/tag-and-rerun-ci`: that
command always opts into every lane, so it is refused there.
