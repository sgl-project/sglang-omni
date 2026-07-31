<!-- Thank you for your contribution! We appreciate it. The following guidelines will help improve your pull request and facilitate feedback. If anything is unclear, don't hesitate to submit your pull request and ask the maintainers for assistance. -->

## Motivation

Onboarding a new model today requires users to discover the right `config_cls`,
find the official `model_path`, and assemble a correct `sgl-omni serve` command
by reading cookbooks and CI scripts. For the three single-GPU-capable models
**Higgs Audio v3 TTS**, **ZONOS2**, and **Fun-ASR-Nano**, this friction is
unnecessary: they are small enough to run on a single A100-40G / H100, but we
never shipped a copy-paste launcher config for them.

This PR adds declarative example configs plus matching cookbook docs so a user
can go from clone to a running OpenAI-compatible server with one command, with
no need to dig through source or CI for the right class/path pair.

## Modifications

Added three declarative launcher configs under `examples/configs/`, each pinning
the `config_cls` and the official `model_path` (the server auto-detects the rest
of the topology):

- `examples/configs/higgs_tts.yaml` — `HiggsTtsPipelineConfig` /
  `bosonai/higgs-audio-v3-tts-4b` (OpenAI `/v1/audio/speech`)
- `examples/configs/zonos2.yaml` — `Zonos2PipelineConfig` /
  `Zyphra/zonos2` (OpenAI `/v1/audio/speech`)
- `examples/configs/fun_asr.yaml` — `FunASRPipelineConfig` /
  `FunAudioLLM/Fun-ASR-Nano-2512-hf` (OpenAI `/v1/audio/transcriptions`,
  `/v1/audio/translations`)

Each YAML carries a header comment with the exact copy-paste serve command.

Docs:

- `docs/cookbook/higgs_tts.md`, `docs/cookbook/zonos2.md`,
  `docs/cookbook/fun_asr.md` — appended a **"Serve with Example Config"**
  section pointing at the new YAML and the `CUDA_VISIBLE_DEVICES=0 sgl-omni
  serve --config ... --port 8000` command.
- `examples/README.md` — added a **"Single-GPU Example Configs"** index table
  listing the three configs with their workload, endpoint, and cookbook link.

These models already have single-GPU CI coverage
(`test_tts_serving_ci.py` for Higgs, `test_zonos2_tts_ci.py` for ZONOS2,
`test_asr_ci_fun_asr.py` for Fun-ASR). This PR intentionally adds **only
onboarding assets (config + docs)** and introduces no new or duplicated tests.

## Related Issues

None. (This is a docs/onboarding improvement; it does not fix a tracked bug.)

## Accuracy Test

Not applicable — no model-side / kernel / architecture code was changed. The
configs only reference existing, CI-validated pipelines.

## Benchmark & Profiling

Not applicable — no performance-affecting code was changed.

## Checklist

- [x] Format your code according with pre-commit.
- [ ] Add unit tests. (Skipped: CI coverage for these models already exists.)
- [x] Update documentation / docstrings / example tutorials as needed.
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed. (Not needed: no runtime code changed.)
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. Use `/tag-and-rerun-ci higgs` or
`/tag-and-rerun-ci moss` to select a TTS CI model. Draft PRs are skipped even
if labeled.

---

### Suggested PR title

```
docs: add single-GPU example configs for Higgs TTS, ZONOS2, and Fun-ASR
```

### Suggested base branch

`sgl-project/sglang-omni:main`

### Quick copy-paste test for reviewers

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/higgs_tts.yaml --port 8000
```
Replace `higgs_tts.yaml` with `zonos2.yaml` or `fun_asr.yaml` to serve the
other two models.
