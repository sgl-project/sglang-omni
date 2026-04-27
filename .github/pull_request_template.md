<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

<!-- Explain the purpose of this PR and the goals it aims to achieve. -->

## Modifications

<!-- Describe the changes made in this PR. -->

## Related Issues

<!-- Link to any related issues here. e.g. "Fixes #123" or "Closes #456" -->

## Accuracy Test

<!-- If this PR affects model-side code (e.g., kernels, model architecture), please provide accuracy test results. Ref: https://docs.sglang.ai/references/accuracy_evaluation.html -->

## Benchmark & Profiling

<!-- If this PR is expected to impact performance, please provide benchmark and profiling results. Ref: https://docs.sglang.ai/references/benchmark_and_profiling.html -->

## Checklist

- [ ] Format your code according with pre-commit.
- [ ] Add unit tests.
- [ ] Update documentation / docstrings / example tutorials as needed.
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. Draft PRs are skipped even if labeled.
