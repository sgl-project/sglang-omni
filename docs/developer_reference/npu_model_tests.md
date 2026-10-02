# NPU model CI

The workflow covers Qwen3-TTS 0.6B CustomVoice and Qwen3-ASR 1.7B on one reserved
NPU on the `linux-aarch64-a2-2` ARM64 A2 runner. It runs for relevant PR changes,
manually, and nightly at 16:00 UTC.
Missing integration settings fail preflight, not pass or skip hardware checks.
Qwen3-TTS requires the NPU support from PR #2004. No GPU speed baselines are used.

## Integration checklist — ask the maintainers

The workflow temporarily uses runner `linux-aarch64-a2-2`, physical device `0`,
and `cocoa001/sglang-omni:qwen3-tts-npu-a2-d794efee`, pinned to digest
`sha256:2211349c4664cf792ab989e0f50a615d05d621fb12e6d84dde890c8f39a45823`.
Confirm runner/device allocation before execution and replace the personal image
with a validated official A2 image before production CI. The existing A3 model
results do not qualify this A2 environment or Qwen3-ASR.

Set these **repository variables**; image and device override the temporary
defaults, while the data directory is still required:

| Variable | Placeholder | Information to request |
|---|---|---|
| `NPU_CI_DEVICE` | Temporary default: `0` | Confirm one exclusively reserved physical device, the same on every runner matching the label |
| `NPU_CI_IMAGE` | `lmsysorg/sglang-omni@sha256:REPLACE_WITH_A2_DIGEST` | Compatible ARM64 A2/910B environment image; use the Docker PR's published A2 digest for integrated validation |
| `NPU_CI_DATA_DIR` | `/REPLACE_WITH_HOST_CACHE/npu-ci` | Absolute host directory with the weights, serving configs and fixed ASR fixtures below |

Also ask an administrator to:

1. Create the `npu-ci` GitHub Environment with **required reviewers**, and allow
   the intended PR refs. Review the exact commit before approving each job.
   The workflow also requires a non-draft PR with `run-ci`, but a label retained
   across pushes is not approval of new code.
2. Confirm trusted-runner Docker/privileged access, host networking and Ascend
   mount paths in `scripts/npu/run_model_ci.sh`. This is not a sandbox for
   unreviewed fork code. Workflow changes themselves must be reviewed.
3. Allow registry and PyPI access or configure approved mirrors. Model inference
   uses local weights with `HF_HUB_OFFLINE=1`.
4. Grant permission to trigger/rerun Actions and read artifacts. Confirm the
   pre-merge workflow testing route: PR events or a reviewed upstream branch.
   A new manual workflow is not selectable until GitHub knows it on main.

The runner label is temporarily fixed; no `NPU_CI_RUNNER_LABELS` variable is needed. Confirm
the physical device allocation rather than inferring it from the label suffix.
The workflow serializes model jobs sharing the reserved device. Local runs
must reserve the device separately. Testing A3 does not qualify A2.

## Prepare the data directory

```text
npu-ci/
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

Ask model owners for immutable checkpoint revisions and matching serving
configs. For TTS, start from `examples/configs/qwen3_tts_0_6b_customvoice_npu.yaml`
in #2004. Stage placement uses logical device 0. Config paths must be accessible
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

For local execution, export the three variables above, then select a model:

```bash
NPU_CI_MODEL=qwen3-tts bash scripts/npu/run_model_ci.sh
NPU_CI_MODEL=qwen3-asr bash scripts/npu/run_model_ci.sh
```

The image supplies dependencies; checked-out source is copied and installed in
a disposable container using `--no-deps`. Results include source SHA/status,
image metadata, serving config, package versions, NPU state, server logs,
outputs and JUnit. Actions uploads `npu-ci-results/` for seven days. Only this
run's container is removed; a hard-killed runner may need operator cleanup.

Each model runs five pytest cases. ASR checks English/Chinese CER, complete SSE
with consistent deltas, two concurrent requests and recovery after rejection.
TTS checks are listed below. Confirm non-skipped test counts, not only a green
job. Before making this a merge gate, validate success, intentional failure,
cancellation/timeout cleanup, artifacts and execution against a changed SHA.

## Run TTS variants directly

In a compatible environment with exactly one reserved visible NPU, install the
source revision under test and use that model's NPU serving config:

```bash
python -m pip install pytest==8.3.5 jiwer==4.0.0 rapidfuzz==3.14.1
export ASCEND_RT_VISIBLE_DEVICES=0
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
context stops only the server it started. Prefer a disposable container for CI
so cancellation can also clean up descendants. The CI script records source and
image metadata, archives logs on failure and does not retry failed tests.
