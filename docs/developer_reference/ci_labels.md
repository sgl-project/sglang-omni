# CI Workflows and Labels

SGLang-Omni CI has two parts. A few fast checks run on GitHub-hosted runners for every pull request. Everything else runs on self-hosted runners: H100 machines for the GPU stages, an Intel Xeon host, an Intel GPU host and an Apple Silicon Mac.

**A pull request runs none of the self-hosted CI until a maintainer adds CI labels.** Once labeled, every later push re-runs CI for as long as the labels stay. Draft pull requests are blocked from self-hosted CI even when they are labeled.

In this sense, two kinds of labels exist and they combine: a stage label decides whether a part of the self-hosted CI runs, and a model label names only the model. A model label never includes a stage name and never selects a stage by itself. `run-lint-only` is separate: it satisfies the gate and runs no self-hosted job. The full label rules come after the workflows. `.github/scripts/resolve_ci_labels.py` implements them, and every self-hosted workflow resolves its labels through that script.

## Overview

| Workflow | Runner | What it runs | When it runs |
| --- | --- | --- | --- |
| Lint | GitHub-hosted | pre-commit and the realtime playback test | Every pull request |
| Test Layout | GitHub-hosted | Test file location check | Every pull request |
| Docs Check | GitHub-hosted | Sphinx documentation build | Every pull request |
| Rust Router | GitHub-hosted | Rust router format, lint, tests and release build | Every pull request |
| Release Docker Images Nightly (NPU) | GitHub-hosted | NPU configuration checks and an ARM64 image build | Every pull request |
| Omni CI: PR Test | H100 | The unit test suite twice: once with CUDA hidden, once for the tests that need a GPU | Any stage label |
| Omni CI: ASR | H100 | Multi-speaker transcription and diarization with MOSS-Transcribe-Diarize, then SeedTTS English and Chinese word error rate and speed on Fun-ASR-Nano-2512, Qwen3-ASR-1.7B or Whisper-large-v3 | `run-asr-ci` or `run-all-ci` |
| Omni CI: TTS | H100 | Voice cloning on SeedTTS with Higgs-TTS, MOSS-TTS, Qwen3-TTS or Fun-CosyVoice3: non-streaming and streaming speed, word error rate, speaker similarity and UTMOS, streaming consistency, first-audio latency for Qwen3-TTS, and a serving stress test on Higgs-TTS | `run-tts-ci` or `run-all-ci` |
| Omni CI: MPS | H100 | Two Higgs-TTS or MOSS-TTS workers sharing one H100 through CUDA MPS, with the TTS quality assertions | `run-mps-ci` or `run-all-ci` |
| Omni CI: Omni | H100 | Qwen3-Omni or MiniCPM-o 4.5: thinker length limits, SeedTTS voice cloning, accuracy and speed on MMMU, MMSU, Video-MME and Video-AMME with text and spoken answers, a tensor parallel thinker, and Qwen3-Omni process replicas | `run-omni-ci` or `run-all-ci` |
| CPU CI | Intel Xeon (`intel-omni-spr`) | Job `intel-cpu-ci (unit_test/cpu)`: `pytest tests/unit_test/cpu/` | `run-intel-cpu-ci` or `run-all-ci` |
| XPU CI | Intel GPU (`intel-omni-bmg`) | Intel GPU device layer tests | `run-intel-xpu-ci` or `run-all-ci` |
| Voxt Mac CI | Apple Silicon | MLX native runtime and Voxt app tests | `run-mac-ci` or `run-all-ci` |

The Omni CI stages are the core of the project's CI: they serve the main ASR, TTS and omni models end to end on H100 and gate their accuracy and speed.

## Checks On Every Pull Request

These workflows run on GitHub-hosted runners for every pull request, drafts included, with no label.

| Workflow | Job | What it checks |
| --- | --- | --- |
| Lint | `lint` | `pre-commit run --all-files`: formatting, import order, Ruff, YAML and TOML syntax, and the other hooks in `.pre-commit-config.yaml`. |
| Lint | `playback-test` | `node --test` on the realtime playback state in `playground/qwen-omni/realtime/`. |
| Test Layout | `test-layout` | Every Python file under `tests/` lives in `tests/unit_test/`, `tests/test_model/`, `tests/test_ci/` or `tests/utils/`. |
| Docs Check | `build-docs` | The documentation builds with `docs/requirements.txt`. |
| Rust Router | `quality` | `cargo fmt`, a check on the minimum supported Rust version, Clippy with warnings as errors, the tests, the docs, a release build, and `--check-config` on every example config. |
| Release Docker Images Nightly (NPU) | `preprocess`, `build_and_publish_arm64` | The NPU configuration and installer tests without hardware, then an ARM64 build of `docker/npu.Dockerfile`, which copies the whole repository. Pull requests never publish the image, and a new push cancels the build still running for the previous one. |

## The Omni CI workflow

Omni CI runs PR Test and the four H100 model stages. Its preflight job waits for lint, resolves the labels and draws the models once. A shared setup job then builds the virtualenv and downloads model weights. With `run-all-ci`, the stages run in this order:

PR Test → ASR → TTS → MPS → Omni

Only PR Test and the selected stages actually run; a stage that is not selected is skipped at once. With `run-omni-ci` alone, Omni starts as soon as PR Test finishes, and with `run-tts-ci` and `run-mps-ci`, the order is PR Test → TTS → MPS. Each stage waits for the previous one to finish, and a stage that fails does not stop the stages after it. A failed setup job blocks every stage.

### PR Test

| Job | What it runs |
| --- | --- |
| `unit-test-a` | First runtime-balanced partition of `pytest tests/ -m "not benchmark and not accelerator"`, with CUDA hidden |
| `unit-test-b` | Second runtime-balanced partition of the same CPU unit-test suite, with CUDA hidden |
| `unit-test-accelerator` | `pytest tests/ -m "accelerator and not benchmark"`, the unit tests that need a GPU |

The CPU partitions run as independent matrix jobs with fail-fast disabled. Files stay intact and are balanced using `.github/unit_test_durations.json`. Pytest discovers new tests automatically; files without a timing entry receive a one-second estimate, and new cases in an existing file stay in that file's partition. Local pytest remains unpartitioned unless the CI plugin is explicitly loaded.

The initial weights are per-file p90 elapsed estimates from three successful H100 CPU jobs: [118185861718](https://github.com/sgl-project/sglang-omni/actions/runs/39288752772/job/118185861718), [118197139172](https://github.com/sgl-project/sglang-omni/actions/runs/39292142439/job/118197139172) and [118244967308](https://github.com/sgl-project/sglang-omni/actions/runs/39305977181/job/118244967308). Each file has three observations. The estimates aggregate timestamp intervals between completed pytest cases, excluding initial collection; they are not isolated setup/call/teardown measurements. These are PR runs with different revisions, not scheduled main-branch observations. The weights are a committed snapshot, not an automatically refreshed model.

`run-unit-test-ci-only` selects PR Test on H100 without selecting model or hardware stages. It does not run the Intel Xeon job `intel-cpu-ci (unit_test/cpu)`; that job needs `run-intel-cpu-ci` or `run-all-ci`, and the CPU path filter. Other stage labels remain additive, so keep them and `run-all-ci` absent for unit tests only. Shared setup still runs, including model-weight provisioning. PR Test also runs whenever any other stage label selects a stage, the hardware stages included. `run-lint-only` does not run PR Test or setup.

### ASR

| Job | What it covers |
| --- | --- |
| stage 1 - MOSS-TD multi-speaker | Multi-speaker transcription and diarization with MOSS-Transcribe-Diarize, behind a two-worker router: movies800 in offline and streaming mode, AISHELL-4 long-form audio and googletime. It gates character error rate and speed. This job always uses MOSS-Transcribe-Diarize, whatever model the stage draws. |
| stage 2 - SeedTTS ASR | The selected model on the SeedTTS English and Chinese splits. It gates word error rate on both splits, and throughput, latency and real-time factor on the English split. |

The stage label is `run-asr-ci`. A model label does not name the stage. Combine it with `run-asr-ci` or `run-all-ci` to pick the model for stage 2:

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-fun-asr-nano-2512` | `FunAudioLLM/Fun-ASR-Nano-2512-hf` | Yes |
| `run-qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | Yes |
| `run-whisper-large-v3` | `openai/whisper-large-v3` | Yes |

### TTS

| Job | What it covers |
| --- | --- |
| stage 1 - non-streaming | Voice cloning on SeedTTS English at concurrency 16, behind a two-worker router: speed, word error rate, speaker similarity and UTMOS. |
| stage 2 - streaming | The same benchmark with streaming output: speed and word error rate. |
| stage latency - streaming first audio | Median time to first playable audio under Poisson arrivals, with one worker alone on the GPU. Runs only when the stage uses one of the two Qwen3-TTS checkpoints. |
| stage 3 - consistency | Checks the artifacts of stages 1 and 2: both cover all 1088 SeedTTS English samples, every streaming request finished, and streaming and non-streaming audio durations agree. |
| stage 4 - serving stress | A mixed production workload over the REST, streaming and WebSocket endpoints through a two-worker router. This job always uses Higgs-TTS, whatever model the stage draws. |

The stage label is `run-tts-ci`. A model label does not name the stage. Combine it with `run-tts-ci` or `run-all-ci` to pick the model for stages 1, 2, 3 and the latency stage:

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-higgs-tts-3-4b` | `bosonai/higgs-tts-3-4b` | Yes |
| `run-moss-tts-local-transformer-v1.5` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | Yes |
| `run-qwen3-tts-12hz-1.7b-base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Yes |
| `run-qwen3-tts-12hz-1.7b-customvoice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | No |
| `run-fun-cosyvoice3-0.5b-2512` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Yes |

Qwen3-TTS CustomVoice uses a named voice instead of a reference clip, and its thresholds are not yet calibrated on the CI host. It reports its metrics without gating on them, and only its model label selects it.

### MPS

MPS has one job, stage 1 - MPS non-streaming. It runs two TTS router workers on a single H100 that share the GPU through CUDA MPS, and checks three things: the two workers are separate MPS clients on the intended GPU, they actually execute at the same time, and the non-streaming word error rate, speaker similarity, UTMOS and audio integrity match the TTS stage's assertions.

The stage label is `run-mps-ci`. MPS has no model labels of its own. It accepts `run-higgs-tts-3-4b` and `run-moss-tts-local-transformer-v1.5`. A model label does not run MPS. With `run-tts-ci` and `run-mps-ci`, one of these labels selects that model for both stages. With only `run-tts-ci`, MPS does not run. With no Higgs or MOSS label, MPS draws from its own pool.

| Model label | Checkpoint | MPS config | In the random pool |
| --- | --- | --- | --- |
| `run-higgs-tts-3-4b` | `bosonai/higgs-tts-3-4b` | `examples/mps_dp/configs/higgs_h100_dp3.yaml` | Yes |
| `run-moss-tts-local-transformer-v1.5` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | `examples/mps_dp/configs/moss_local_h100_dp2.yaml` | Yes |

### Omni

Every job runs the selected model unless noted otherwise. A job named "Talker" asks for text and audio output, and also gates the word error rate between the text answer and an ASR transcript of the spoken answer.

| Job | What it covers |
| --- | --- |
| stage 1 - thinker length integration | Overlong prompts and prompt plus `max_tokens` overflow return HTTP 400; decoding that hits `max_tokens` returns `finish_reason="length"`. |
| stage 2 - TTS | Voice cloning on SeedTTS: speed, word error rate, speaker similarity and UTMOS. |
| stages 3 and 4 - MMMU, MMMU Talker | Image and text question answering: accuracy and speed. |
| stages 5 and 6 - MMSU, MMSU Talker | Spoken audio understanding: accuracy and speed. |
| stages 7 and 8 - Video-MME, Video-MME Talker | Video understanding: accuracy and speed. |
| stage 9 - Video-AMME | Video with its audio track: accuracy and speed. |
| stage 10 - Video-AMME Talker | Video with audio and spoken answers, with the thinker sharded across two GPUs (tensor parallel size 2). |
| stage 11 - process replicas | A two-GPU deployment with one code2wav replica on each GPU; checks that both replicas start, register and receive requests. Runs only for Qwen3-Omni. |

The stage label is `run-omni-ci`. A model label does not name the stage. Combine it with `run-omni-ci` or `run-all-ci` to pick the model:

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-qwen3-omni-30b-a3b-instruct` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Yes |
| `run-minicpm-o-4.5` | `openbmb/MiniCPM-o-4_5` | Yes |

## Hardware workflows

Each hardware stage is its own workflow. Its gate checks live labels and skips draft blocking, label resolution and lint checks unless its stage label or `run-all-ci` selects it. None of them has model labels. Each keeps a path filter, so a stage label alone does not force a run: a pull request that changes none of the stage's paths skips its tests.

### CPU CI

CPU CI builds the Intel Xeon image from `docker/cpu.Dockerfile` and runs `pytest tests/unit_test/cpu/` in the job `intel-cpu-ci (unit_test/cpu)`. Its tests run when the pull request touches `sglang_omni/` (except Markdown files), `tests/unit_test/cpu/`, `pyproject_cpu.toml`, `scripts/cpu/`, `docker/cpu.Dockerfile` or the workflow itself.

The stage label is `run-intel-cpu-ci`. `run-unit-test-ci-only` does not select this job. When the paths are untouched, or when the pull request selects other stages but not Intel CPU, the test job is skipped and the `CPU CI Status` check still reports success.

### XPU CI

XPU CI pulls or builds the Intel GPU image from `docker/xpu.Dockerfile` and runs the device layer tests in `tests/unit_test/xpu/test_device_layer.py` and the dots.tts flow head and tail tests. Its tests run when the pull request touches `sglang_omni/`, `tests/unit_test/xpu/`, `tests/unit_test/dots_tts/`, `pyproject_xpu.toml`, `scripts/xpu/`, `docker/xpu.Dockerfile` or the workflow itself.

The stage label is `run-intel-xpu-ci`.

### Voxt Mac CI

Voxt Mac CI has two jobs on Apple Silicon. The native runtime job builds the MLX runtime, checks its golden outputs on a frozen corpus for Qwen3-ASR and the other native models, and runs the server API tests. The Voxt unit test job builds the Voxt app with Xcode and runs its Omni and MLX test classes. The workflow starts only when the pull request touches `Voxt/`, `sglang_omni_mlx/` or the workflow itself; otherwise it does not appear on the pull request at all.

The stage label is `run-mac-ci`.

## Label rules

1. A stage runs only when its stage label or `run-all-ci` is on the pull request. `run-all-ci` selects every stage.
2. Self-hosted CI needs `run-all-ci`, a stage label, or `run-lint-only`. Without one, the Omni CI gate fails with `No CI label found`; unselected hardware workflows skip their tests without failing. The checks on GitHub-hosted runners do not depend on labels. `run-lint-only` runs no self-hosted job, so a green lint check leaves no self-hosted failure. It does not turn off a stage label that is also present.
3. Any stage label also runs PR Test. A pull request labeled only `run-mac-ci` still runs PR Test on H100.
4. A model label names a model, not a stage. It picks that model for every selected stage that accepts it, and never selects a stage by itself. Higgs and MOSS labels are accepted by TTS and MPS. If no selected stage accepts the label, the gate prints a warning.
5. A selected stage without a model label draws its model at random from that stage's pool. The draw is seeded by the workflow run ID, so re-running jobs in the same run keeps the same model, while a new push or a newly added label starts a new run and draws again.
6. Each stage takes at most one model label. Two model labels for the same stage fail the gate, even when that stage is not selected.
7. Stage labels combine. `run-tts-ci` together with `run-mps-ci` runs PR Test, TTS and MPS, and nothing else.

| Stage label | Selects |
| --- | --- |
| `run-all-ci` | Every stage |
| `run-lint-only` | No self-hosted job. The gates succeed. |
| `run-unit-test-ci-only` | PR Test on H100. Not `intel-cpu-ci (unit_test/cpu)`. |
| `run-asr-ci` | ASR in Omni CI |
| `run-tts-ci` | TTS in Omni CI |
| `run-mps-ci` | MPS in Omni CI |
| `run-omni-ci` | Omni in Omni CI |
| `run-intel-cpu-ci` | Intel Xeon job `intel-cpu-ci (unit_test/cpu)`, when the CPU paths match |
| `run-intel-xpu-ci` | XPU CI |
| `run-mac-ci` | Voxt Mac CI |

## Examples

The checks on GitHub-hosted runners run in every case below, so the table lists only the self-hosted CI.

| Labels | What runs |
| --- | --- |
| none | Nothing. The Omni CI gate fails with `No CI label found`; hardware tests are skipped. |
| `run-lint-only` | Nothing self-hosted. The gates succeed. |
| `run-unit-test-ci-only` | PR Test only, after shared setup. |
| `run-tts-ci` | PR Test, then TTS on a random model from its pool. |
| `run-tts-ci`, `run-moss-tts-local-transformer-v1.5` | PR Test, then TTS on MOSS-TTS. MPS does not run. |
| `run-mps-ci`, `run-higgs-tts-3-4b` | PR Test, then MPS on Higgs-TTS. TTS does not run. |
| `run-tts-ci`, `run-mps-ci` | PR Test, TTS and MPS, each drawing its own model. |
| `run-tts-ci`, `run-mps-ci`, `run-higgs-tts-3-4b` | PR Test, then TTS and MPS, both on Higgs-TTS. |
| `run-mac-ci` | PR Test on H100 and Voxt Mac CI. No H100 model stage. |
| `run-all-ci` | Every stage. Each model stage draws its own model. |
| `run-all-ci`, `run-minicpm-o-4.5` | Every stage. Omni uses MiniCPM-o 4.5, the other stages draw at random. |
| `run-tts-ci`, `run-minicpm-o-4.5` | PR Test and TTS. The Omni model label is ignored with a warning, because Omni is not selected. |
| `run-minicpm-o-4.5` | Nothing. The gate fails, because a model label does not select a stage. |
| `run-tts-ci`, `run-higgs-tts-3-4b`, `run-moss-tts-local-transformer-v1.5` | Nothing. The gate fails, because TTS and MPS accept two model labels. |

## Label changes, re-runs and manual runs

Adding a label manually, CI or not, starts a new run of Omni CI, CPU CI, XPU CI and Voxt Mac CI (when its paths match) and cancels the run of the same workflow in progress, so the new label set takes effect at once. Models that no label pins are drawn again in the new run.

Removing a label cancels only the workflows that the remaining labels no longer select. Removing `run-mac-ci` while `run-tts-ci` stays cancels Voxt Mac CI and leaves Omni CI running. Removing the last stage label cancels Omni CI, CPU CI, XPU CI and Voxt Mac CI. `run-lint-only` does not keep any of them. A run that is still selected keeps the stages and models it resolved when it started; push again or re-run all jobs to pick up the change.

Re-running jobs reuses the run ID, so every stage keeps the model it drew. The gate reads the live labels from the pull request, so re-running all jobs picks up labels added or removed since the run started.

A maintainer can start Omni CI by hand from the Actions tab or with `gh workflow run`. The `ci_labels` input takes space-separated labels and resolves them exactly like pull request labels; it defaults to `run-all-ci`:

```bash
gh workflow run omni-ci.yaml --ref <branch> \
  -f ci_labels="run-tts-ci run-moss-tts-local-transformer-v1.5"
```

Manual runs of CPU CI, XPU CI and Voxt Mac CI ignore labels and always run their tests.

Maintainers with CI permissions can use `/tag-and-rerun-ci` with one or more short stage or model targets to re-run completed CI workflows:

```text
/tag-and-rerun-ci unit-test
/tag-and-rerun-ci tts asr
/tag-and-rerun-ci tts asr omni
/tag-and-rerun-ci asr qwen3-omni
/tag-and-rerun-ci qwen3-tts
/tag-and-rerun-ci qwen3-tts qwen3-asr
/tag-and-rerun-ci mps-higgs
```

Stage targets are `lint`, `unit-test`, `asr`, `tts`, `mps`, `omni`, `intel-cpu`, `intel-xpu` and `mac`. `lint` adds `run-lint-only`. Model targets are `fun-asr`, `qwen3-asr`, `whisper-asr`, `higgs`, `moss`, `qwen3-tts`, `qwen3-tts-custom-voice`, `cosyvoice3`, `qwen3-omni`, `minicpmo`, `mps-higgs` and `mps-moss`. `higgs` and `moss` add `run-tts-ci` plus `run-higgs-tts-3-4b` or `run-moss-tts-local-transformer-v1.5`. `mps-higgs` and `mps-moss` add `run-mps-ci` plus that same model label. `qwen3-tts` adds `run-tts-ci` plus `run-qwen3-tts-12hz-1.7b-base`. A short model target adds its stage label and its model label. The model label itself never includes the stage name.

Use `/tag-and-rerun-full-ci` to add `run-all-ci` and select all stages. It also accepts short model targets, for example `/tag-and-rerun-full-ci qwen3-tts qwen3-asr qwen3-omni`. `higgs` or `mps-higgs` selects Higgs for every selected stage that accepts it, TTS and MPS included. Model choices not specified by the command keep their existing labels or use the normal random selection. Bare `/tag-and-rerun-ci` and full GitHub label arguments, such as `run-tts-ci` or `run-all-ci`, are not accepted. Manual dispatch still accepts full labels through its `ci_labels` input.

The command replaces a conflicting Higgs or MOSS label for both TTS and MPS, because those labels are shared. It replaces other model labels only in the requested stages, and keeps other labels, including an existing `run-all-ci`. Existing stage labels still apply, so arguments do not narrow an already broader selection. It re-runs all jobs of the newest completed selected parent workflows on the current PR commit, unless GitHub requires workflow approval. If the newest run is active, the command leaves it running rather than retrying an older run. To apply changed labels after preflight has resolved the selection, run the command again after the active workflow finishes. Path filters and PR gates still apply.

`/tag-run-ci-label` accepts the same short targets but only updates labels; without arguments, it adds `run-all-ci`. Labels added by the bot do not start workflows; push a commit or use a tag-and-rerun command to start them. `/rerun-failed-ci` retries failed jobs without changing labels or stopping active runs. PR authors can use it on their own PRs.

## Gate messages

| Message | Meaning |
| --- | --- |
| `No CI label found. Add run-all-ci, run-lint-only, or a stage label: ...` | The pull request has none of those labels. Add one of them. |
| `Keep only one <stage> model label: ...` | Two model labels name the same stage. Remove one of them. |
| `Ignoring <label>: its stage label and run-all-ci are both missing.` | A warning, not a failure. The model label has no effect until its stage is selected. |
| `'lint' check completed with conclusion=failure; skip staged CI.` | Lint failed on this commit. Fix it and push; self-hosted CI starts once lint passes. |
| `PR has merge conflicts with the base branch; skip staged CI.` | Merge or rebase onto the current `main` first. |

A branch created before these labels existed does not contain `resolve_ci_labels.py`. On such a branch, the Omni CI and Voxt Mac CI gates fail until the branch merges the current `main`.
