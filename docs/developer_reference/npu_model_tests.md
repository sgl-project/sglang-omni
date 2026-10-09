# NPU model CI

The workflow covers Qwen3-TTS 0.6B CustomVoice and Qwen3-ASR 1.7B on one visible
NPU allocated to each test Pod by the `linux-aarch64-a2-2` ARM64 A2 runner platform.
It runs for relevant PR changes,
manually, and nightly at 16:00 UTC.
Missing integration settings fail preflight, not pass or skip hardware checks.
No GPU speed baselines are used.

One workflow runs two independent matrix jobs, one per model, each in a fresh
job container with separate logs and results. Jobs run serially on this runner pool;
a model test failure does not cancel the other model. Both must pass the final
`NPU Model CI Status` check.

## Integration checklist — ask the maintainers

The runner must support Kubernetes container hooks: the workflow selects the
test image through `jobs.models.container.image`, and the platform creates the
test Pod. The test script runs inside that Pod without Docker or BuildKit.
Select a validated ARM64 910B image from the NPU release workflow. A3 results
do not qualify A2.

Set these **repository variables**; the image and data directory are required:

| Variable | Placeholder | Information to request |
|---|---|---|
| `NPU_CI_IMAGE` | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang-omni@sha256:REPLACE_WITH_910B_DIGEST` | Use the final Omni 910B image digest from the release workflow summary, not the SGLang base-image digest |
| `NPU_CI_DATA_DIR` | `/data` | Absolute path inside the test Pod containing the mounted weights, serving configs and fixed ASR fixtures below |

Keep the image digest fixed for reproducible tests; update it when qualifying a
new runtime. Kubernetes manages image pulls and node-local layer caching.
Do not substitute a rolling tag such as `main-cann9.0.0-910b` without a digest.

The 2026-09-29 publication provides this 910B image for runner qualification:

```text
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang-omni@sha256:993fd7fb5a2fceb86f15fabff953fa76b67e63e558954c153aecdccbf72fab9a
```

Also ask an administrator to:

1. Create the `npu-ci` GitHub Environment with **required reviewers**, and allow
   the intended PR refs. Review the exact commit before approving each job.
   The workflow also requires a non-draft PR with `run-ci`, but a label retained
   across pushes is not approval of new code.
2. Enable Kubernetes container hooks for `linux-aarch64-a2-2`, with job workspace
   sharing and cleanup on completion, cancellation and timeout. Configure the
   **test Pod**, not only the runner Pod, with one exclusively allocated NPU,
   compatible driver libraries, model/data mounts and sufficient memory and
   shared memory (8 GiB for `/dev/shm`). Preserve the platform's device visibility
   variables; the serving configs select logical device `0`. Do not mount a
   Docker socket or require Docker-in-Docker. Review workflow changes before
   executing PR code on this infrastructure.
3. Allow registry and PyPI access or configure approved mirrors. Model inference
   uses local weights with `HF_HUB_OFFLINE=1`. Use anonymous SWR pulls if allowed;
   otherwise configure a read-only image pull secret on the test Pods. Do not
   expose image-publishing credentials to PR test code.
4. Grant permission to trigger/rerun Actions and read artifacts. Confirm the
   pre-merge workflow testing route: PR events or a reviewed upstream branch.
   A new manual workflow is not selectable until GitHub knows it on main.

No `NPU_CI_DEVICE` setting is used: the platform owns physical device allocation.
The script preserves `ASCEND_RT_VISIBLE_DEVICES` and other injected device settings.
Both suites require exactly one visible NPU. Simply adding `container.image`
does not configure container hooks, device allocation or data mounts.

## Prepare the Pod mounts

For `NPU_CI_DATA_DIR=/data`, expose this layout inside each test Pod:

```text
/data/
  models/
    qwen3-tts/       # Qwen3-TTS-12Hz-0.6B-CustomVoice, with speech_tokenizer/
    qwen3-asr/       # Qwen3-ASR-1.7B
  configs/
    qwen3-tts.yaml   # approved single-NPU serving config
    qwen3-asr.yaml   # approved single-NPU serving config
  asr/
    cases.json
    english.wav
    chinese.wav
```

Reuse the downloaded checkpoints with read-only mounts; no new download is needed:

| Existing node directory | Test Pod mount |
|---|---|
| `/root/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `/data/models/qwen3-tts` |
| `/root/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1.7B` | `/data/models/qwen3-asr` |

Node-local mounts require scheduling on a node containing those files. Otherwise,
use shared storage. Mount the configs and ASR fixtures read-only as well.
Prepare `configs/qwen3-tts.yaml` from
`examples/configs/qwen3_tts_0_6b_customvoice_npu.yaml` and `configs/qwen3-asr.yaml`
from `examples/configs/qwen3_asr_npu.yaml` in this checkout.

The prevalidation checkpoint revisions were
`85e237c12c027371202489a0ec509ded67b5e4b5` (TTS) and
`7278e1e70fe206f11671096ffdd38061171dd6e5` (ASR). Record and verify the revisions
of the mounted checkpoints before qualifying the runner.

These functional smoke configurations disable engine decode graphs and Torch
compile; they do not qualify graph performance.
Stage placement uses logical device 0. Config paths must be accessible
inside the container. Symlink targets must also reside in the mounted data
directory. Keep revisions in the cache inventory; do not overwrite assets
during CI runs.

`cases.json` uses actual transcripts and per-fixture CER ceilings calibrated
on an approved NPU baseline. Replace all placeholders; no default quality
threshold is silently supplied:

```json
{
  "English": {
    "audio": "english.wav",
    "text": "REPLACE_WITH_EXACT_ENGLISH_TRANSCRIPT",
    "max_cer": "REPLACE_WITH_CALIBRATED_NUMBER_BELOW_1"
  },
  "Chinese": {
    "audio": "chinese.wav",
    "text": "REPLACE_WITH_EXACT_CHINESE_TRANSCRIPT",
    "max_cer": "REPLACE_WITH_CALIBRATED_NUMBER_BELOW_1"
  }
}
```

Use short, licensed audio without personal data: transcripts and generated
audio appear in CI artifacts. CER uses Unicode NFKC/case folding and removes
punctuation/whitespace in both languages; it does not replace corpus WER.

## Run CI

After configuring variables and environment protection, use **NPU Model CI →
Run workflow**, or approve a relevant PR's jobs. `NPU Model CI Status` requires
both models to pass. Only enable it as a required merge check after real Actions
validation; account for path filters in the repository's required-check policy.

For local execution **inside an already provisioned test container**, export
the image reference and mounted data directory above, then select a model:

```bash
NPU_CI_MODEL=qwen3-tts bash scripts/npu/run_model_ci.sh
NPU_CI_MODEL=qwen3-asr bash scripts/npu/run_model_ci.sh
```

If PyPI is unavailable, configure `PIP_INDEX_URL` for an approved mirror in the
test Pod without changing the pinned test dependency versions.

The image supplies runtime dependencies, including `qwen-tts`. The script loads
`$ASCEND_HOME_PATH/set_env.sh` (default: `/usr/local/Ascend/ascend-toolkit/set_env.sh`).
Checked-out source is copied into a temporary directory inside the test Pod so
NPU package metadata does not overwrite the checkout. Before installation,
`install_npu.sh --check` verifies the configured SGLang release and NPU stack
without installing dependencies. The source is then installed with `--no-deps`;
CI tests that checkout, not the source bundled in the image.
Results under the checkout's `npu-ci-results/` include source SHA/status, the
configured image digest, runner log, environment-check log,
serving config, package versions, NPU state, server logs,
outputs and JUnit. Actions uploads `npu-ci-results/` for seven days, including
failed tests when artifact upload can still run. Kubernetes container hooks
remove the test Pod and its temporary source tree after the job.

Each model runs five pytest cases. ASR checks English/Chinese CER, complete SSE
with consistent deltas, two concurrent requests and recovery after rejection.
TTS checks are listed below. Confirm non-skipped test counts, not only a green
job. Before making this a merge gate, validate success, intentional failure,
cancellation/timeout cleanup, artifacts and execution against a changed SHA.

## Run TTS variants directly

In a compatible environment with exactly one reserved visible NPU, install the
source revision under test and use that model's NPU serving config:

```bash
python -m pip install pytest==8.3.2 jiwer==4.0.0 rapidfuzz==3.14.1
export HF_HUB_OFFLINE=1
export OMNI_RUN_NPU_TESTS=1
export OMNI_NPU_TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
export OMNI_NPU_TTS_CONFIG=/path/to/qwen3_tts_0_6b_customvoice_npu.yaml
export OMNI_NPU_TTS_TASK=CustomVoice
export OMNI_NPU_TTS_OUTPUT=/tmp/qwen3-tts-regression
python -m pytest tests/test_model/test_npu_tts.py -v -s \
  --junitxml=/tmp/qwen3-tts-regression.xml
```

`jiwer` / `rapidfuzz` are test-only dependencies required by the existing
`tests.test_model` shared plugin, even when this suite does not run WER. They
need not be added to the inference image's runtime dependency list.

The fixture starts and stops its own server on an available loopback port. Set
`OMNI_NPU_TTS_PORT` to reserve a specific port; an occupied port is an error, not
a server to reuse. Automatic port selection avoids reusing a previous run's
connections while they are still in TIME_WAIT.
Local model and config paths are mandatory. When disabled, tests skip; when
enabled, missing models, unsupported hardware and startup failures fail the run.

For Base, select the corresponding model/config and set:

```bash
export OMNI_NPU_TTS_TASK=Base
export OMNI_NPU_TTS_REFERENCE=/references/voice.wav
export OMNI_NPU_TTS_REFERENCE_TEXT='The exact words spoken in this reference.'
```

The reference must be available to the server. For VoiceDesign, select its
model/config and set `OMNI_NPU_TTS_TASK=VoiceDesign`; no reference is needed.
Run variants sequentially on a reserved card, not concurrently on the same card.

## Coverage and evidence

- English/Chinese non-streaming speech and a fully consumed PCM stream.
- Two concurrent requests (queueing is allowed; this does not prove batching).
- A valid request after rejecting a malformed request.
- Status, sample format, completeness, non-silent output and broad short-prompt
  duration bounds. Durations/first bytes are recorded, not compared to GPU limits.
- Server log, request metadata, WAV/PCM outputs and optional pytest JUnit XML.

This is a functional smoke/regression suite, not a quality benchmark: passing
does not establish intelligibility, speaker similarity or absence of reference
text leakage. WER/speaker-sim evaluation and model-specific performance baselines
must be added with validated assets and thresholds. Network chunk boundaries
are not assumed to equal model chunk boundaries.

No global CUDA/NPU process cleanup is used. The shared `managed_omni_server`
context stops only the server it started. Pod cleanup must terminate remaining
processes on job cancellation or timeout. The script preserves failed test exit
codes and does not retry failed tests.
