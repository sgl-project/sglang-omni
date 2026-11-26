# CI Workflows and Labels

SGLang-Omni CI has two parts. A few fast checks run on GitHub-hosted runners for every pull request, though one that changes only the MLX path runs just Lint among them. Everything else runs on self-hosted runners (H100 machines for the GPU stages, an Intel Xeon host and an Intel GPU host), except Mac CI, which runs on GitHub's macOS runners but also needs its label.

**A pull request runs none of the self-hosted CI until a maintainer adds CI labels.** Once labeled, every later push re-runs CI for as long as the labels stay. Draft pull requests are blocked from self-hosted CI even when they are labeled.

Two kinds of labels combine: a stage label runs a part of the self-hosted CI, and a model label picks the model it runs on. An ASR, TTS or Omni model label alone also runs the jobs that test its model. `.github/scripts/resolve_ci_labels.py` implements the [label rules](#label-rules), and every self-hosted workflow resolves its labels through that script.

## Choosing labels

Add the labels that match the change. They combine, and no label runs all of CI.

| The change touches | Add | What runs |
| --- | --- | --- |
| One ASR, TTS or Omni model | That model's label, listed under [ASR](#asr), [TTS](#tts) and [Omni](#omni) | PR Test and the jobs that test that model |
| Several models of one kind, or code they share | `run-asr-ci`, `run-tts-ci` or `run-omni-ci` | PR Test and every job of that stage, on a random model unless a model label pins it |
| CUDA MPS or weight sharing | `run-mps-ci` | PR Test and MPS |
| Full duplex or realtime sessions | `run-native-duplex`, `run-turn-based-duplex` or `run-cascaded-duplex` | PR Test and that Full-Duplex-Bench job |
| Code that only the unit tests cover | `run-unit-test-ci-only` | CPU unit tests on GitHub-hosted VMs and H100 accelerator unit tests. Not `intel-cpu-ci (unit_test/cpu)`. |
| The Intel CPU or Intel GPU path | `run-intel-cpu-ci` or `run-intel-xpu-ci` | PR Test and that workflow's tests, when its paths match |
| The MLX path or the Voxt app | `run-mac-ci` | Mac CI. Not PR Test. |
| Nothing that needs self-hosted CI | `run-lint-only` | No self-hosted job. The gates succeed. |

## Overview

| Workflow | Runner | What it runs | When it runs |
| --- | --- | --- | --- |
| Lint | GitHub-hosted | pre-commit | Every pull request |
| Test Layout | GitHub-hosted | Test file location check | Every pull request except MLX-only ones |
| Docs Check | GitHub-hosted | Sphinx documentation build and the documentation style checker's tests | Every pull request except MLX-only ones |
| Playground Test | GitHub-hosted | The playground frontend tests | Every pull request except MLX-only ones |
| Rust Router | GitHub-hosted | Rust router format, lint, tests and release build | Every pull request except MLX-only ones |
| Release Docker Images Nightly (NPU) | GitHub-hosted | NPU configuration checks and an ARM64 image build | Pull requests that change its Dockerfile, installer, pyproject files or workflows |
| Omni CI: PR Test | GitHub-hosted Linux and H100 | Eight CPU unit-test partitions on Ubuntu VMs and two H100 accelerator partitions | Any stage label except `run-mac-ci`, or a model label |
| Omni CI: ASR | H100 | Multi-speaker transcription and diarization with MOSS-Transcribe-Diarize, and SeedTTS English and Chinese word error rate and speed on ARK-ASR-3B, Fun-ASR-Nano-2512, Qwen3-ASR-1.7B or Whisper-large-v3 | `run-asr-ci` for both jobs, or an ASR model label for the job that tests that model |
| Omni CI: TTS | H100 | Voice cloning on SeedTTS with Higgs-TTS, MOSS-TTS, Qwen3-TTS or Fun-CosyVoice3: non-streaming and streaming speed, word error rate, speaker similarity and UTMOS, streaming consistency, first-audio latency for Qwen3-TTS, and a serving stress test on Higgs-TTS | `run-tts-ci` for every job, or a TTS model label for the jobs that test that model |
| Omni CI: MPS | H100 | External CUDA MPS DP2 non-streaming validation, native MPS lifecycle validation, and native CUDA IPC weight-sharing lifecycle validation | `run-mps-ci` |
| Omni CI: Omni | H100 | Qwen3-Omni or MiniCPM-o 4.5: thinker length limits, SeedTTS voice cloning, accuracy and speed on MMMU, MMSU, Video-MME and Video-AMME with text and spoken answers, a tensor parallel thinker, and Qwen3-Omni process replicas | `run-omni-ci`, or an Omni model label for that model |
| Omni CI: Duplex | H100 | Full-Duplex-Bench v1.0 and v1.5 on three interaction strategies: native full duplex (MiniCPM-o 4.5), turn based (Qwen3-Omni with server VAD) and cascaded (Qwen3-ASR-1.7B, Inkling-2-Mini and Qwen3-TTS Base cloning one Vivian recording). Each gates turn taking, overlap behavior, latency and listening quality against its own baseline | `run-native-duplex`, `run-turn-based-duplex` or `run-cascaded-duplex` |
| CPU CI | Intel Xeon (`intel-omni-spr`) | Job `intel-cpu-ci (unit_test/cpu)`: `pytest tests/unit_test/cpu/` | `run-intel-cpu-ci` |
| XPU CI | Intel GPU (`intel-omni-bmg`) | Intel GPU device layer tests | `run-intel-xpu-ci` |
| Mac CI | GitHub macOS | MLX native runtime build and protocol, the Voxt helper's Swift tests and headless end-to-end tests | `run-mac-ci` |

The Omni CI stages are the core of the project's CI: they serve the main ASR, TTS and omni models end to end on H100 and gate their accuracy and speed.

## Checks On Every Pull Request

These workflows run on GitHub-hosted runners for every pull request, drafts included, with no label. An MLX-only pull request changes nothing outside `sglang_omni_mlx/`, `Voxt/`, `.github/workflows/mac-ci.yaml` and `.github/workflows/voxt*.yaml`. It runs only Lint among these checks and does not start Omni CI: Mac CI and Voxt CI test it, and Mac CI checks its labels.

| Workflow | Job | What it checks |
| --- | --- | --- |
| Lint | `lint` | `pre-commit run --all-files`: formatting, import order, Ruff, YAML and TOML syntax, and the other hooks in `.pre-commit-config.yaml`. |
| Test Layout | `test-layout` | Every Python file under `tests/` lives in `tests/unit_test/`, `tests/test_model/`, `tests/test_ci/` or `tests/utils/`. |
| Docs Check | `build-docs` | The documentation builds with `docs/requirements.txt`. |
| Docs Check | `doc-style-test` | The tests of the documentation style checker in `scripts/`. |
| Playground Test | `playback-test` | `node --test` on the realtime playback state in `playground/qwen-omni/realtime/` and on the frontends in `playground/cascade/frontend/orb/` and `playground/realtime/frontend/orb/`. |
| Rust Router | `quality` | `cargo fmt`, a check on the minimum supported Rust version, Clippy with warnings as errors, the tests, the docs, a release build, and `--check-config` on every example config. |
| Release Docker Images Nightly (NPU) | `preprocess`, `build_and_publish_arm64` | Only when the pull request changes `docker/npu.Dockerfile`, the NPU installer, the pyproject files or these workflows: the NPU configuration and installer tests without hardware, then an ARM64 build of `docker/npu.Dockerfile`. Pull requests never publish the image, and a new push cancels the build still running for the previous one. |

## The Omni CI workflow

Omni CI runs the hosted CPU unit tests (`unit-test-cpu-0` through `unit-test-cpu-7`), PR Test (`unit-test-accelerator-a/b`), and the seven H100 model stages. Its preflight job waits for lint, resolves the labels and draws the models once. After preflight, the eight GitHub-hosted CPU unit-test jobs start immediately in parallel with H100 shared setup (`setup` builds the virtualenv and downloads model weights), without waiting for or occupying H100 runners. With every stage label, the H100 stages run in this order:

PR Test → ASR, TTS, MPS, Omni and Duplex side by side

Only PR Test and the selected stages actually run; a stage that is not selected is skipped at once. ASR, TTS, MPS, Omni and Duplex start independently as soon as PR Test finishes, whether it passed or not, and their jobs spread over whichever H100 runners are free; they do not wait for the hosted CPU partitions. Jobs within a stage run side by side as well. TTS's SeedTTS voice cloning job runs its non-streaming and streaming benchmarks one after the other, because its last check compares their results. MPS uses `github.sha` (the merge commit on a regular `pull_request` run), so it shares the same virtualenv and does not wait for TTS or other model stages. A stage that fails does not stop any other stage. A failed setup job blocks the H100 stages, while hosted CPU unit tests continue independently.

### PR Test

| Job | What it runs |
| --- | --- |
| `unit-test-cpu-0` through `unit-test-cpu-7` | Eight runtime-balanced partitions of `pytest tests/ -m "not benchmark and not accelerator"` on separate GitHub-hosted Ubuntu VMs |
| `unit-test-accelerator-a` | First runtime-balanced partition of `pytest tests/ -m "accelerator and not benchmark"`, the unit tests that need a GPU |
| `unit-test-accelerator-b` | Second runtime-balanced partition of the same accelerator unit-test suite |

Each CPU VM installs the dependencies for the checked-out revision in a fresh environment, runs its partition with retries, and uploads its JUnit report and attempt logs. The CPU and accelerator partitions run as independent matrix jobs with fail-fast disabled. Files stay intact and are balanced using `.github/unit_test_durations.json` for the CPU suite and `.github/unit_test_accelerator_durations.json` for the accelerator suite; a file with both kinds of test has a weight in each. Pytest discovers new tests automatically; files without a timing entry receive a one-second estimate, and new cases in an existing file stay in that file's partition. Local pytest remains unpartitioned unless the CI plugin is explicitly loaded.

The CPU weights are per-file p90 elapsed estimates from three successful H100 A/B runs: [41197670240](https://github.com/sgl-project/sglang-omni/actions/runs/41197670240), [41161315193](https://github.com/sgl-project/sglang-omni/actions/runs/41161315193) and [41140707601](https://github.com/sgl-project/sglang-omni/actions/runs/41140707601). Each file has three observations. The estimates aggregate timestamp intervals between completed pytest cases, excluding initial collection; they are not isolated setup/call/teardown measurements. These are PR runs with different revisions, not scheduled main-branch observations. The weights are a committed snapshot, not an automatically refreshed model.

The accelerator weights are per-file p90 estimates of the same kind, from the first attempt of three H100 accelerator jobs on 2026-11-21: [136957056283](https://github.com/sgl-project/sglang-omni/actions/runs/44939177878/job/136957056283), [136979600046](https://github.com/sgl-project/sglang-omni/actions/runs/44947746078/job/136979600046) and [136987314462](https://github.com/sgl-project/sglang-omni/actions/runs/44948801830/job/136987314462). Each of those attempts stopped at a failure in `tests/unit_test/whisper_asr/test_draft_cuda_graph.py`, so that file and the five after it take their weights from three earlier successful jobs: [136910764501](https://github.com/sgl-project/sglang-omni/actions/runs/44925396141/job/136910764501), [136849929132](https://github.com/sgl-project/sglang-omni/actions/runs/44905131915/job/136849929132) and [136850113353](https://github.com/sgl-project/sglang-omni/actions/runs/44905114597/job/136850113353).

`run-unit-test-ci-only` selects the hosted CPU unit tests and H100 accelerator unit tests without selecting model or hardware stages. It does not run the Intel Xeon job `intel-cpu-ci (unit_test/cpu)`; that job needs `run-intel-cpu-ci` and the CPU path filter. Other stage labels remain additive, so keep them absent for unit tests only. Shared setup still runs for the accelerator unit tests, including model-weight provisioning. Both unit-test suites also run whenever any other stage label selects a stage, the Intel hardware stages included. `run-mac-ci` and `run-lint-only` do not run unit tests or setup.

### ASR

| Job | What it covers |
| --- | --- |
| MOSS-TD multi-speaker | Multi-speaker transcription and diarization with MOSS-Transcribe-Diarize, behind a two-worker router: movies800 in offline and streaming mode, AISHELL-4 long-form audio and googletime. It gates character error rate and speed. |
| SeedTTS ASR | The selected model on the SeedTTS English and Chinese splits. It gates word error rate on both splits, and throughput, latency and real-time factor on the English split. |

The two jobs share no data and run side by side. `run-asr-ci` runs both, and SeedTTS ASR draws its model at random unless a model label pins it. An ASR model label alone runs only the job that tests its model:

| Model label | Job | Checkpoint | In the random pool |
| --- | --- | --- | --- |
| `run-moss-transcribe-diarize` | MOSS-TD multi-speaker | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Its only model |
| `run-ark-asr-3b` | SeedTTS ASR | `AutoArk-AI/ARK-ASR-3B` with DFlash | Yes |
| `run-fun-asr-nano-2512` | SeedTTS ASR | `FunAudioLLM/Fun-ASR-Nano-2512-hf` | Yes |
| `run-qwen3-asr-1.7b` | SeedTTS ASR | `Qwen/Qwen3-ASR-1.7B` | Yes |
| `run-whisper-large-v3` | SeedTTS ASR | `openai/whisper-large-v3` | Yes |

### TTS

| Job | What it covers |
| --- | --- |
| SeedTTS voice cloning | Voice cloning on SeedTTS English at concurrency 16, behind a two-worker router. A non-streaming run checks speed, word error rate, speaker similarity and UTMOS; a streaming run then checks speed and word error rate. Last, a consistency check compares the two runs: both cover all 1088 SeedTTS English samples, every streaming request finished, and streaming and non-streaming audio durations agree. |
| streaming first audio | Median time to first playable audio under Poisson arrivals, with one worker alone on the GPU. Runs only on the two Qwen3-TTS checkpoints. |
| Higgs serving stress | A mixed production workload over the REST, streaming and WebSocket endpoints through a two-worker router, always on Higgs-TTS. |

The first two jobs, the model jobs, test the selected model. `run-tts-ci` runs all three jobs, and the model jobs draw their model at random unless a model label pins it. A TTS model label alone runs the model jobs on its model, and `run-higgs-tts-3-4b` also runs Higgs serving stress. A TTS model label never selects MPS (see [MPS](#mps)).

| Model label | Jobs it runs alone | Checkpoint | In the random pool |
| --- | --- | --- | --- |
| `run-higgs-tts-3-4b` | The model jobs and Higgs serving stress | `bosonai/higgs-tts-3-4b` | Yes |
| `run-moss-tts-local-transformer-v1.5` | The model jobs | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | Yes |
| `run-qwen3-tts-12hz-1.7b-base` | The model jobs | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Yes |
| `run-qwen3-tts-12hz-1.7b-customvoice` | The model jobs | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | No |
| `run-fun-cosyvoice3-0.5b-2512` | The model jobs | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Yes |

Qwen3-TTS CustomVoice uses a named voice instead of a reference clip, and its thresholds are not yet calibrated on the CI host. It reports its metrics without gating on them, and only its model label selects it.

### MPS

`run-mps-ci` runs three sequential validations on H100:

1. External CUDA MPS DP2 non-streaming validation checks that two TTS router workers share one H100 through CUDA MPS and meet the TTS word error rate, speaker similarity, UTMOS and audio integrity assertions.
2. Native MPS lifecycle validation uses Higgs-TTS to check daemon sharing, recovery behavior and clean teardown.
3. Native CUDA IPC weight-sharing lifecycle validation uses MOSS-TTS Local to check shared-weight memory savings, process startup order and leader failure handling.

`run-higgs-tts-3-4b` or `run-moss-tts-local-transformer-v1.5` picks the model for the external CUDA MPS DP2 validation; without one, it draws from its own pool. The two native lifecycle validations always use their fixed models. A model label alone does not run MPS, because MPS tests the CUDA MPS and weight-sharing machinery, not the model: alone, a Higgs or MOSS label runs its TTS jobs. With `run-tts-ci` and `run-mps-ci`, one of these labels pins both the TTS model and the external MPS model.

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

`run-omni-ci` runs these jobs on a model drawn at random unless a model label pins it. An Omni model label alone runs them on its model. Neither runs Duplex, whose jobs each have their own stage label (see [Duplex](#duplex)).

| Model label | Checkpoint | In the random pool |
| --- | --- | --- |
| `run-qwen3-omni-30b-a3b-instruct` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Yes |
| `run-minicpm-o-4.5` | `openbmb/MiniCPM-o-4_5` | Yes |

### Duplex

Each job runs a fixed Full-Duplex-Bench selection on one interaction strategy, the first samples of every v1.5 category and v1.0 subset, then scores it with the [runbook](full_duplex_guide.md#full-duplex-bench-runbook): Parakeet transcripts, the official timing, unified v1.0 metrics, listening quality, and the Qwen3.8-27B behavior and semantic judges. The three strategies behave differently by design, so each job gates only regressions from its own baseline, calibrated on H100. A run must first exercise the v1.5 overlaps: the model has to be speaking when the event begins in at least three of the five pairs of every category and fifteen of the twenty in all, or the overlap scores say nothing about the model. The seven behavior and v1.0 rates score five cases each and are gated together: the run fails when a rate scored fewer than five cases, when it has more than five fewer cases on target than the baseline in all, or when a rate whose baseline met three or more of its five cases meets none. A baseline rate is the mean count over repeated runs, so it may stand for a fraction of a case. The latency metrics, UTMOS and voice similarity each have their own bound. The listening omission rate and the semantic judge's two rates are reported but not gated.

| Stage label | Job | System |
| --- | --- | --- |
| `run-native-duplex` | Full-Duplex-Bench - full duplex | MiniCPM-o 4.5, native full duplex, one server on each of two GPUs |
| `run-turn-based-duplex` | Full-Duplex-Bench - turn based | Qwen3-Omni with server VAD turn detection (`examples/qwen3_omni/session_h100.yaml`) |
| `run-cascaded-duplex` | Full-Duplex-Bench - cascaded | Qwen3-ASR-1.7B, Inkling-2-Mini with DSpark speaking through a `tts_start` tool with its reasoning kept out of the speech, and Qwen3-TTS Base cloning one Vivian recording, with server VAD, on two GPUs (`examples/streaming_cascade/profiles/qwen3-asr-inkling-2-mini-vivian-tts-tool`) |

The three jobs share the pull request's Full-Duplex-Bench assets, so they run one after another, and a failed job does not skip the next. Each stage has one system and no model label.

## Hardware workflows

Each hardware stage is its own workflow. Its gate checks live labels and skips draft blocking, label resolution and lint checks unless its stage label selects it. None of them has model labels. Each keeps a path filter, so a stage label alone does not force a run: a pull request that changes none of the stage's paths skips its tests.

### CPU CI

CPU CI builds the Intel Xeon image from `docker/cpu.Dockerfile` and runs `pytest tests/unit_test/cpu/` in the job `intel-cpu-ci (unit_test/cpu)`. Its tests run when the pull request touches `sglang_omni/` (except Markdown files), `tests/unit_test/cpu/`, `pyproject_cpu.toml`, `scripts/cpu/`, `docker/cpu.Dockerfile` or the workflow itself.

The stage label is `run-intel-cpu-ci`. `run-unit-test-ci-only` does not select this job. When the paths are untouched, or when the pull request selects other stages but not Intel CPU, the test job is skipped and the `CPU CI Status` check still reports success.

### XPU CI

XPU CI pulls or builds the Intel GPU image from `docker/xpu.Dockerfile` and runs the device layer tests in `tests/unit_test/xpu/test_device_layer.py` and the dots.tts flow head and tail tests. Its tests run when the pull request touches `sglang_omni/`, `tests/unit_test/xpu/`, `tests/unit_test/dots_tts/`, `pyproject_xpu.toml`, `scripts/xpu/`, `docker/xpu.Dockerfile` or the workflow itself.

The stage label is `run-intel-xpu-ci`.

### Mac CI

Mac CI has two jobs on GitHub's macOS runners. The native runtime job builds the MLX runtime and a stub server, then tests the protocol every native server shares on the stub server, each server's startup checks, and that the installed runtime links only the system and itself. The Voxt helper tests job builds the helper once, runs its Swift tests, and runs six headless end-to-end tests: the helper's read-only, icon and file-input checks, both text-fit checks, and a session over fake local models. Model outputs are checked by hand per model PR with the tools in `sglang_omni_mlx/native/ci`. The workflow starts when the pull request touches `sglang_omni_mlx/`, `Voxt/`, `.github/workflows/mac-ci.yaml` or `.github/workflows/voxt*.yaml`; otherwise it does not appear on the pull request at all. Unlike the other hardware gates, its gate fails without a CI label, because Omni CI does not start on an MLX-only pull request. On a pull request, the native runtime job runs only when `sglang_omni_mlx/` changes, while the Voxt helper tests job runs for any non-Markdown change under `Voxt/`; a change to the workflow itself runs both, and Voxt Nightly always runs both. A pull request that changes only Voxt documentation runs neither job and passes the gate without a label.

The stage label is `run-mac-ci`. Use it for any change on the MLX path: the SGLang Omni MLX runtime in `sglang_omni_mlx/`, such as a runtime optimization or a new native server, as well as the Voxt app. It runs only this workflow, which waits for lint to pass and then starts the Mac jobs. It does not run PR Test or any other H100 job, because the Mac jobs test the MLX runtime and the Voxt helper, not the `sglang_omni` Python package that PR Test covers.

## Label rules

1. A stage runs only when its stage label is on the pull request. The ASR, TTS and Omni jobs also run with a label for the model they test.
2. No label selects every stage. For a change that really needs all of CI, add `run-asr-ci`, `run-tts-ci`, `run-mps-ci`, `run-omni-ci`, `run-native-duplex`, `run-turn-based-duplex`, `run-cascaded-duplex`, `run-intel-cpu-ci`, `run-intel-xpu-ci` and `run-mac-ci`, or comment `/tag-and-rerun-ci asr tts mps omni full-duplex turn-based cascaded intel-cpu intel-xpu mac`.
3. Self-hosted CI needs a stage label, a model label or `run-lint-only`. Without one, the Omni CI gate fails with `No CI label found`, and so does the Mac CI gate when Mac CI starts; CPU CI and XPU CI skip their tests without failing. An MLX-only pull request does not start Omni CI, so only the Mac CI gate fails.
4. `run-lint-only` satisfies the gates and runs no self-hosted job. It does not turn off a stage label that is also present.
5. Any stage label except `run-mac-ci`, and any model label, also runs PR Test.
6. A model label picks its model for every selected stage that accepts it. When no such stage is selected, it selects the jobs that test its model (see [ASR](#asr), [TTS](#tts) and [Omni](#omni)).
7. A model label never selects MPS. TTS and MPS both accept the Higgs and MOSS labels: alone, one of them runs its TTS jobs, and with `run-mps-ci` it pins the MPS model and adds no TTS job.
8. A selected stage without a model label draws its model at random from that stage's pool. The draw is seeded by the workflow run ID, so re-running jobs in the same run keeps the same model, while a new push or a newly added label starts a new run and draws again.
9. Each stage takes at most one model label. Two model labels for the same stage fail the gate, even when that stage is not selected.

## Examples

The table is for a pull request that is not MLX-only and lists only the self-hosted CI; the checks on GitHub-hosted runners run in every case. An MLX-only pull request starts Mac CI instead of Omni CI: with no label its gate fails, and `run-mac-ci` runs Mac CI alone.

| Labels | What runs |
| --- | --- |
| none | Nothing. The Omni CI gate, and the Mac CI gate when Mac CI starts, fail with `No CI label found`; the other hardware tests are skipped. |
| `run-lint-only` | Nothing self-hosted. The gates succeed. |
| `run-unit-test-ci-only` | Hosted CPU unit tests and H100 accelerator unit tests only. |
| `run-asr-ci` | PR Test, then both ASR jobs, SeedTTS ASR on a random model from its pool. |
| `run-asr-ci`, `run-qwen3-asr-1.7b` | PR Test, then both ASR jobs, SeedTTS ASR on Qwen3-ASR. |
| `run-qwen3-asr-1.7b` | PR Test, then SeedTTS ASR on Qwen3-ASR. MOSS-TD multi-speaker does not run. |
| `run-moss-transcribe-diarize` | PR Test, then MOSS-TD multi-speaker. SeedTTS ASR does not run. |
| `run-tts-ci` | PR Test, then every TTS job, the model jobs on a random model from their pool. |
| `run-qwen3-tts-12hz-1.7b-base` | PR Test, then the TTS model jobs on Qwen3-TTS. Higgs serving stress and MPS do not run. |
| `run-higgs-tts-3-4b` | PR Test, then every TTS job on Higgs-TTS. MPS does not run. |
| `run-tts-ci`, `run-moss-tts-local-transformer-v1.5` | PR Test, then TTS on MOSS-TTS. MPS does not run. |
| `run-mps-ci`, `run-higgs-tts-3-4b` | PR Test, then MPS on Higgs-TTS. TTS does not run. |
| `run-tts-ci`, `run-mps-ci` | PR Test, TTS and MPS, each drawing its own model. |
| `run-tts-ci`, `run-mps-ci`, `run-higgs-tts-3-4b` | PR Test, then TTS and MPS, both on Higgs-TTS. |
| `run-tts-ci`, `run-mac-ci` | PR Test and TTS on H100, and Mac CI. |
| `run-mac-ci` | Mac CI only, once lint passes. Omni CI passes its gate and skips setup and PR Test. |
| `run-omni-ci` | PR Test, then Omni on a random model from its pool. |
| `run-minicpm-o-4.5` | PR Test, then Omni on MiniCPM-o 4.5. Duplex does not run. |
| `run-tts-ci`, `run-minicpm-o-4.5` | PR Test, then TTS on a random model and Omni on MiniCPM-o 4.5. |
| `run-tts-ci`, `run-higgs-tts-3-4b`, `run-moss-tts-local-transformer-v1.5` | Nothing. The gate fails, because TTS and MPS accept two model labels. |

## Label changes, re-runs and manual runs

Adding a label manually, CI or not, starts a new run of Omni CI (unless the pull request is MLX-only), CPU CI, XPU CI and Mac CI (when its paths match) and cancels the run of the same workflow in progress, so the new label set takes effect at once. Models that no label pins are drawn again in the new run.

Removing a label cancels only the workflows that the remaining labels no longer select. Removing `run-mac-ci` while `run-tts-ci` stays cancels Mac CI and leaves Omni CI running. Removing `run-tts-ci` while `run-mac-ci` stays cancels Omni CI and leaves Mac CI running. Removing the last stage label cancels Omni CI, CPU CI, XPU CI and Mac CI, unless a model label remains: it still selects the jobs that test its model, so Omni CI keeps running. `run-lint-only` does not keep any of them. A run that is still selected keeps the stages and models it resolved when it started; push again or re-run all jobs to pick up the change.

Re-running jobs reuses the run ID, so every stage keeps the model it drew. The gate reads the live labels from the pull request, so re-running all jobs picks up labels added or removed since the run started.

A maintainer can start Omni CI by hand from the Actions tab or with `gh workflow run`. The `ci_labels` input takes space-separated labels and resolves them exactly like pull request labels. It is required:

```bash
gh workflow run omni-ci.yaml --ref <branch> \
  -f ci_labels="run-tts-ci run-moss-tts-local-transformer-v1.5"
```

Manual runs of CPU CI, XPU CI and Mac CI ignore labels and always run their tests, and so does the run of Mac CI that Voxt Nightly starts each night.

Maintainers with CI permissions can comment `/tag-and-rerun-ci` with one or more short targets. The command adds each target's labels, then re-runs the selected workflows:

```text
/tag-and-rerun-ci unit-test
/tag-and-rerun-ci tts asr omni
/tag-and-rerun-ci qwen3-tts qwen3-asr
/tag-and-rerun-ci mps-higgs
```

| Target | Labels it adds |
| --- | --- |
| `lint` | `run-lint-only` |
| `unit-test`, `asr`, `tts`, `mps`, `omni`, `full-duplex`, `turn-based`, `cascaded`, `intel-cpu`, `intel-xpu`, `mac` | That stage's label |
| `moss-td-asr`, `ark-asr`, `fun-asr`, `qwen3-asr`, `whisper-asr` | That ASR model's label |
| `higgs`, `moss`, `qwen3-tts`, `qwen3-tts-custom-voice`, `cosyvoice3` | That TTS model's label |
| `qwen3-omni`, `minicpmo` | That Omni model's label |
| `mps-higgs`, `mps-moss` | `run-mps-ci` and the Higgs or MOSS model label |

An ASR, TTS or Omni model target adds only its model label, which runs the jobs for that model. The command needs at least one target and does not accept full label names such as `run-tts-ci`; manual dispatch still takes them through its `ci_labels` input.

The command treats the labels already on the pull request as follows:

- It keeps the stage labels, so targets never narrow a broader selection.
- A model target replaces the other model labels of its stage. A Higgs or MOSS target replaces them for both TTS and MPS, because those two stages share the labels.
- A stage whose model no target names keeps its model label, or draws at random without one.

It then re-runs all jobs of the newest completed run of each selected workflow on the current PR commit, unless GitHub requires workflow approval. If the newest run is still active, the command leaves it running and re-runs nothing older; run the command again once it finishes to apply the changed labels. Path filters and PR gates still apply.

`/tag-run-ci-label` accepts the same short targets but only updates labels; it also requires at least one target. Labels added by the bot do not start workflows; push a commit or use a tag-and-rerun command to start them. `/rerun-failed-ci` retries failed jobs without changing labels or stopping active runs. PR authors can use it on their own PRs.

A bot keeps one comment on a pull request with the `/tag-and-rerun-ci` targets that test what it changes. For a changed model directory it names the targets of the stages a model label selects, and of the Full-Duplex-Bench job of MiniCPM-o, Qwen3-Omni and the streaming cascade. For code that every model shares it names `asr`, `tts` and `omni` until a model stage is selected. The comment follows the labels and commits, and a maintainer still starts every run.

## Gate messages

| Message | Meaning |
| --- | --- |
| `No CI label found. Add run-lint-only, a model label or a stage label: ...` | The pull request has none of those labels. Add one of them. |
| `Keep only one <stage> model label: ...` | Two model labels name the same stage. Remove one of them. |
| `'lint' check completed with conclusion=failure; skip staged CI.` | Lint failed on this commit. Fix it and push; self-hosted CI starts once lint passes. |
| `PR has merge conflicts with the base branch; skip staged CI.` | Merge or rebase onto the current `main` first. |

A branch created before these labels existed does not contain `resolve_ci_labels.py`. On such a branch, the Omni CI and Mac CI gates fail until the branch merges the current `main`.
