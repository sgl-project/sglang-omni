# SGLang Omni Benchmarks

Benchmark suite for SGLang Omni, covering performance (latency, throughput, RTF)
and accuracy (WER, MMSU, MMMU, Video-MME, Video-AMME) across supported modality
combinations.

## Directory Structure

```
benchmarks/
├── tasks/          # Per-task logic (tts, audio_understanding, visual_understand, video_understanding)
├── metrics/        # Metric computation (performance, accuracy)
├── dataset/        # Dataset loaders + download helpers
├── benchmarker/    # Framework: runner, data structures, utilities
├── eval/           # Entry-point scripts (one per task × model)
├── tts_serving/    # TTS serving harness and Docker contract
├── cache/          # (gitignored) dataset caches
└── results/        # (gitignored) evaluation outputs
```

Nemotron ASR service measurements and reference transcripts: [reproduction guide](eval/nemotron.md).

PersonaPlex reference comparisons: [evaluation setup and limits](eval/personaplex.md).

## Quick Start

```bash
# 0. Prepare dataset (once)
python -m benchmarks.dataset.prepare --dataset seedtts

# 1. Start a server on port 8000 (pick one matching the benchmark below)

# S2-Pro — for sections 2a/2b/2c
python -m sglang_omni.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml --port 8000

# Voxtral-4B-TTS — for section 2d (plain TTS, no voice cloning)
python -m sglang_omni.cli serve \
    --model-path mistralai/Voxtral-4B-TTS-2603 --port 8000

# Higgs TTS — for section 2e (voice cloning via references[])
python -m sglang_omni.cli serve \
    --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
    --port 8000

# MOSS-TTS — for section 2f (voice cloning via references[], duration via token_count)
python -m sglang_omni.cli serve \
    --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
    --config examples/configs/moss_tts.yaml --port 8000

# Qwen3-Omni, speech mode — for section 3 (SeedTTS; multi-GPU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

# Qwen3-Omni, text-only mode — for sections 4 (MMSU) and 5 (MMMU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only --port 8000

# 2a. S2-Pro — full pipeline: generate + WER (server needed for phase 1 only)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 \
    --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8

# 2b. S2-Pro — generate only (speed metrics, no transcription)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only --stream \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 --max-samples 50 --concurrency 8

# 2c. S2-Pro — transcribe only (reuses audio from a prior generate run; no server)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro \
    --output-dir results/s2pro_en --lang en --device cuda:0

# 2d. Voxtral — full pipeline without voice cloning
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model mistralai/Voxtral-4B-TTS-2603 --port 8000 \
    --max-concurrency 16 \
    --output-dir results/voxtral_en --lang en --max-samples 50 \
    --no-ref-audio --voice cheerful_female

# 2e. Higgs TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model boson-sglang/higgs-audio-v3-tts-4b-base --port 8000 \
    --ref-format references \
    --max-concurrency 16 \
    --output-dir results/higgs_tts_en --lang en --max-samples 50

# 2f. MOSS-TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
    --ref-format references --token-count auto \
    --max-concurrency 8 \
    --output-dir results/moss_tts_en --lang en --max-samples 50

# 3a. Qwen3-Omni — full pipeline (generate + transcribe)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3b. Qwen3-Omni — generate only (server required; use in CI to split phases)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3c. Qwen3-Omni — transcribe only (reuses audio; ASR server on --port)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --model qwen3-omni --lang en --port 8000

# 3d. Qwen3-Omni — warm the full speech path with separate references before timing
python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only --voice-clone --stream \
    --meta measured/meta.lst --warmup-meta warmup/meta.lst \
    --warmup 16 --max-concurrency 16 \
    --output-dir results/qwen3_omni_en --model qwen3-omni --port 8000

# 4. Qwen3-Omni — MMSU (audio comprehension)
python -m benchmarks.eval.benchmark_omni_mmsu \
    --model qwen3-omni --port 8000 \
    --modalities text+audio --max-samples 50

# 5a. Qwen3-Omni — MMAU (audio comprehension)
python -m benchmarks.eval.benchmark_omni_mmau \
    --model qwen3-omni --port 8000 --max-samples 50

# 5b. Qwen3-Omni — MMAR (audio reasoning)
python -m benchmarks.eval.benchmark_omni_mmar \
    --model qwen3-omni --port 8000 --max-samples 50

# 6. Qwen3-Omni — MMMU (VLM accuracy, image input)
python -m benchmarks.eval.benchmark_omni_mmmu \
    --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

# 7. Qwen3-Omni — Video-MME (video understanding)
python -m benchmarks.eval.benchmark_omni_videomme \
    --model qwen3-omni --port 8000 --max-samples 50

# 8a. Qwen3-Omni — Video-AMME (video + audio question understanding)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

# 8b. Qwen3-Omni — Video-AMME Talker (text + audio output)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
    --enable-audio --asr-device cuda:0 --asr-concurrency 32

# 9. SocialOmni — fixed-prefix speaker, turn-entry, and response evaluation
python -m benchmarks.eval.benchmark_omni_socialomni \
    --dataset-root /path/to/socialomni \
    --model qwen3-omni --base-url http://localhost:8000 \
    --level both \
    --judge-config benchmarks/configs/socialomni_judges.example.json

# 10a. Offline UTMOS (naturalness MOS prediction) scoring on existing output
# For custom TTS models (e.g. S2-Pro, Voxtral, Higgs TTS):
python -m benchmarks.eval.benchmark_tts_seedtts \
    --utmos-only --output-dir results/s2pro_en --device cuda:0

# For Qwen3-Omni:
python -m benchmarks.eval.benchmark_omni_seedtts \
    --utmos-only --output-dir results/qwen3_omni_en --device cuda:0

# 10b. Offline Speaker Similarity (voice resemblance) scoring on existing output
# For custom TTS models (e.g. S2-Pro, Voxtral, Higgs TTS):
python -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --output-dir results/s2pro_en --device cuda:0

# For Qwen3-Omni:
python -m benchmarks.eval.benchmark_omni_seedtts \
    --similarity-only --output-dir results/qwen3_omni_en --device cuda:0
```

## Eval Scripts

| Script | Task | Model | API |
|--------|------|-------|-----|
| `eval/benchmark_tts_seedtts.py` | TTS speed + WER (unified) | e.g. S2-Pro, Voxtral, Higgs TTS | `/v1/audio/speech` |
| `eval/benchmark_tts_serving.py` | TTS serving contract | OpenAI-compatible TTS models | `/v1/audio/speech`, raw PCM streaming, WebSocket, voice and batch contracts |
| `eval/benchmark_omni_seedtts.py` | TTS speed + WER (unified) | Qwen3-Omni, MiniCPM-o | `/v1/chat/completions` |
| `eval/benchmark_omni_mmsu.py` | MMSU (audio comprehension) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmau.py` | MMAU (audio comprehension) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmar.py` | MMAR (audio reasoning) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmmu.py` | MMMU (VLM accuracy + speed) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videomme.py` | Video-MME (video understanding) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videoamme.py` | Video-AMME (video + audio question understanding) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_socialomni.py` | SocialOmni fixed-prefix speaker, turn-entry, and response evaluation | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_asr_seedtts.py` | ASR concurrency scaling on SeedTTS EN/ZH | Qwen3-ASR, Fun-ASR | `/v1/audio/transcriptions` |
| `eval/benchmark_asr_stt_benchmark.py` | ASR concurrency scaling on the Pipecat STT benchmark set (EN) | Qwen3-ASR, Fun-ASR | `/v1/audio/transcriptions` |
| `eval/benchmark_asr_longform.py` | ASR concurrency scaling on LongLibriHeavy 30/60 s and Meanwhile (EN) | Qwen3-ASR, Fun-ASR | `/v1/audio/transcriptions` |
| `eval/benchmark_asr_realtime.py` | Realtime ASR streaming latency, protocol invariants, and WER on SeedTTS EN | Qwen3-ASR | `/v1/realtime?intent=transcription` |
| `eval/benchmark_duplex.py` | Native full-duplex VoiceChat continuous protocol case | Nemotron VoiceChat | `/v1/realtime` (native) |
| `eval/benchmark_duplex_v15.py` | Full-Duplex-Bench v1.5 paired overlap sessions, event timing and judged behavior | Nemotron VoiceChat | `/v1/realtime` (native) |
| `eval/benchmark_duplex_v10.py` | Full-Duplex-Bench v1.0 pause, turn-taking, interruption and backchannel takeover/latency | Nemotron VoiceChat | `/v1/realtime` (native) |

See [tts_serving/README.md](tts_serving/README.md) for the TTS serving
benchmark design, harness contract, scenario matrix, and Docker usage.

The two `*_seedtts.py` scripts merge the previous `benchmark_*_tts_speed.py`
and `voice_clone_*_wer.py` pairs into a single two-phase pipeline: phase 1
generates + persists WAVs while the TTS server runs, phase 2 transcribes through
an ASR server to avoid GPU contention with the TTS server. Use `--generate-only` or
`--transcribe-only` to run a single phase. For TTS, `--concurrency` and
`--max-concurrency` are equivalent (see `benchmark_tts_seedtts.py`).
`benchmark_tts_seedtts.py` also handles model-specific voice-cloning reference
payloads: the default `--ref-format flat` sends `ref_audio`/`ref_text`, while
`--ref-format references` sends `references=[{audio_path, text}]` for Higgs TTS
and MOSS-TTS. MOSS-TTS additionally supports duration control through
`--token-count`. `--seed`, `--temperature`, `--top-p`, `--top-k`, and
`--repetition-penalty` are recorded in the speed results. Reference audio on
this endpoint is a filesystem path, so it is not client-encoded inside the
request timer. `--concurrencies 1,16 --repeats 5 --generate-only` repeats each
level. One repeat keeps the directory `c<level>`; further repeats write
`c<level>_r<repeat>`. Every row in `concurrency_sweep.json` is an aggregate
of the same speed metrics as the Omni sweep, with `per_repeat` holding each
raw summary.
`--fingerprint` records the client environment and the server `/v1/models`
identity.

Chat-completion speed runs forward `--seed` on the request when it is set
(MMSU, MMAU, and MMAR also use it to shuffle the dataset) and accept
`--fingerprint`. `benchmark_omni_streaming_ttft.py` uses one `--seed` for
warmup and every measured repeat, and records talker sampling knobs.
`benchmark_omni_rollout_stress.py` derives request seed `base + index` from
`--seed` so rollouts differ but stay reproducible. The realtime ASR client
base64-encodes packets before the first-send timestamp.

`benchmark_omni_seedtts.py` documents local vs CI GPU usage in its module
docstring (sequential phases on CI to reduce OOM risk).

Omni warmup runs in the benchmark client after the server is available. By
default it repeats one sample concurrently; `--warmup 0` disables it. Use
`--warmup-meta` with a separate SeedTTS metadata file or dataset to exercise
different reference audio and prompts. Supply at least `--warmup` samples
(the request count defaults to `--max-concurrency`), and choose references and
text outside the measured set to avoid warming its per-sample caches. Use
`--voice-clone --stream` to exercise reference encoding and streaming audio.
Warmup uses normal generation limits and EOS handling; it does not guarantee
that every stage reaches the requested concurrency as one batch.

Separate warmup saves audio and per-request outcomes under `<output-dir>/warmup/`.
All requests must succeed before the measured cohort starts. These outputs
and their wall time are excluded from the main speed results and generated
audio metadata. Apply the same warmup policy to both benchmark revisions;
measure startup-to-ready and the first unconditioned request wave separately
when evaluating production cold starts.

For MiniCPM-o, pass `--voice-clone --reference-audio-field audio.ref_audio`.
The client encodes every reference WAV once before the timed run so file
reads stay out of request latency. `--seed` sends one sampler seed with every
request so generated lengths are reproducible between A/B runs; the seed and
temperature are recorded in the results config.
Seeded sampling is not free: SGLang's seeded sampler hashes every vocabulary
entry per token, which measured about 8% extra latency at concurrency 1 on an
A6000 with identical output. Use the same seed setting in both arms of an A/B
comparison and never compare seeded against unseeded absolute numbers.
`--talker-temperature`, `--talker-top-p`, `--talker-top-k` and
`--talker-repetition-penalty` pin the talker's sampling and are recorded the
same way; unset knobs keep the server defaults. `--fingerprint` records the
client environment and the server's `/v1/models` identity in the results
config, using the same helpers as the ASR sweeps.

`--concurrencies 1,16 --repeats 5 --generate-only` sweeps concurrency levels,
writing each run to `<output-dir>/c<level>_r<repeat>/` and one
`<output-dir>/sweep.json` that aggregates each level's repeats (mean, min,
max, n per metric) with the raw per-repeat summaries, in the same shape as
the ASR sweeps. Tail percentiles need at least 100
measured samples; below that p99 interpolates the two slowest requests and the
benchmark logs a warning. Without `--warmup-meta` the warmup replays the first
measured sample, so its server caches are warm when it is timed.

`benchmark_asr_seedtts.py` is a standalone ASR fan-out sweep (issue #646): it
transcribes the SeedTTS *reference* clips directly against a running Qwen3-ASR
or Fun-ASR router and reports WER + speed + per-worker routing balance per
concurrency level. It reports evaluation coverage and RTFx (successful
input-audio seconds per wall-clock second) alongside the existing RTF. Use it
to measure how ASR concurrency affects capacity, latency, and WER for a given
workload.

Add `--stream` to exercise the transcription SSE path and report text TTFT and
inter-chunk latency while retaining the terminal transcript for WER:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

`benchmark_asr_stt_benchmark.py` runs the same sweep on
[`pipecat-ai/stt-benchmark-data`](https://huggingface.co/datasets/pipecat-ai/stt-benchmark-data):
1000 English utterances, 1 to 16 s each (9.6 s mean), with punctuated, cased
transcripts. Where SeedTTS measures short clean prompts, this set measures
the longer conversational turns a voice agent sees. The script imports the
sweep loop, table, and common arguments from `benchmark_asr_seedtts.py`, so
both result JSONs share one `config`/`results` layout and differ only in the
dataset fields (`repo_id`/`split` instead of `meta`).

This only reuses Pipecat's dataset, not their metric definitions. The upstream
[pipecat-ai/stt-benchmark](https://github.com/pipecat-ai/stt-benchmark)
reports Semantic WER and TTFS (end of speech to final transcript over a
simulated realtime stream); this sweep reports Whisper-normalized WER and
whole-request latency/RTF, and `--stream` uploads the complete file. The
numbers are not comparable to the Pipecat leaderboard.

```bash
python -m benchmarks.dataset.prepare --dataset stt-benchmark
python -m benchmarks.eval.benchmark_asr_stt_benchmark \
  --port 8000 --concurrencies 1,8,32 --repeats 3 --warmup
```

`benchmark_asr_longform.py` registers three canonical long-form English
workloads: the complete LongLibriHeavy `llh_test_30` (1203 samples) and
`llh_test_60` (591 samples) splits, plus all 64 samples in the Meanwhile `test`
split. The LongLibriHeavy names refer to the published splits; the loader does
not apply another duration filter. Likewise, Meanwhile's `begin` and `end`
metadata are not used to crop the already segmented `audio` field.

The loader decodes each source clip and stages it as mono 16 kHz PCM WAV before
the timed sweep. The result JSON uses the same schema and metrics as the
SeedTTS and Pipecat ASR sweeps, and records the dataset alias, repository,
split, pinned revision, expected sample count, and exact input fingerprint.
`--stream` uploads each complete file rather than simulating real-time audio
arrival.

```bash
python -m benchmarks.dataset.prepare --dataset longlibriheavy-30
python -m benchmarks.eval.benchmark_asr_longform \
  --dataset longlibriheavy-30 --port 8000 \
  --concurrencies 1,8,32 --repeats 3 --warmup

python -m benchmarks.dataset.prepare --dataset longlibriheavy-60
python -m benchmarks.eval.benchmark_asr_longform \
  --dataset longlibriheavy-60 --port 8000 \
  --concurrencies 1,8,32 --repeats 3 --warmup

python -m benchmarks.dataset.prepare --dataset meanwhile
python -m benchmarks.eval.benchmark_asr_longform \
  --dataset meanwhile --port 8000 \
  --concurrencies 1,8,32 --repeats 3 --warmup
```

`benchmark_asr_realtime.py` streams SeedTTS reference clips through the
realtime WebSocket endpoint (`--enable-realtime`) at wall-clock pace and
reports client-observed streaming latencies, protocol invariant violations, and
WER of the completed transcript. The client (`benchmarks/realtime_asr/client.py`)
only records timestamps; every metric definition lives in
`benchmarks/realtime_asr/metrics.py` so numbers stay comparable across runs:

- `first_partial_latency_s`: per segment, from the send time of the packet that
  reached the server's first refresh point (`segment_start + decode_interval_ms`)
  to the first partial `transcription.segment`.
- `partial_interval_s`: gaps between consecutive partials of one segment.
- `final_latency_s`: `input_audio_buffer.committed` to the segment's final event.
- `done_to_completed_s`: `transcription.done` sent to `transcription.completed`.

`--mode vad` (default) lets server VAD close turns and pads each clip with
`--trailing-silence-ms` of silence so the last turn closes on VAD; `--mode
manual` disables VAD and commits explicitly. `--http-baseline` transcribes the
same clips over `/v1/audio/transcriptions`; the WER delta is computed only on
samples that succeeded on both paths (`common_evaluated`) and is `null` when
that set is empty.
`--concurrencies` runs one result per level; there is no cross-level report.
The `decode_interval_ms` in effect is read from `session.created` and recorded
in the result `config`.

```bash
python -m benchmarks.eval.benchmark_asr_realtime \
  --port 8000 --max-samples 50 --concurrencies 1,4,8 --http-baseline
```

The CLI also saves input PCM and session observations under `OUTPUT.traces`
(override with `--trace-dir`). Each concurrency run gets a fresh directory with
`manifest.json`, numbered trace JSON files, and `input-SHA256.pcm` files containing
the original mono 16 kHz PCM16 before trailing silence. The manifest lists every
selected sample. Each trace retains packet send times/sample counts, received
event payloads/times, client errors, effective client configuration, and recorder
and grader source hashes. The server revision is explicitly unknown.

Observations are written after the timed collection finishes, before WER or HTTP
baseline scoring. Client timeouts and transport errors retain their partial
traces. Abrupt process termination or an unexpected collection exception can
leave manifest entries without trace files; these are unrecorded attempts, not
passes. Python callers opt in with `run_asr_realtime_once(..., trace_dir=...)`.

Replay one saved trace without connecting to a server or loading a model:

```bash
python -m benchmarks.realtime_asr.replay path/to/run/000000.json
```

Replay recomputes protocol violations and the same client timing metrics; exit
status is 0 for a clean trace and 1 for protocol/client failures. Failed-trace
timings are diagnostic only. It does not rerun WER, infer audio, or verify the
sibling PCM file; the input hash supports separate input verification. Compare
`recorded_source` and `replay_source` to distinguish reproduction from rescoring
with a changed grader. Traces contain audio/transcripts and should be handled as
benchmark data.

The oracle checks required integer event indexes, commit-before-final ordering,
unique session completion, and no transcription segments after completion.
Session control events after completion remain legal. The client stops receiving
at its first completion, so post-completion violations are detected only when
present in the recorded or injected trace. These checks validate the ASR profile,
not native duplex interruption or physical audio playback.

## Native duplex (VoiceChat) protocol benchmark

`benchmark_duplex.py` records a native full-duplex VoiceChat session over
`/v1/realtime` and grades it offline. It runs the `continuous` case and is a
protocol smoke benchmark, not a performance sweep or quality evaluation.
A passing case does not establish product or model readiness.

The native profile is not the transcription profile: PCM16 mono 16 kHz input,
PCM16 22050 Hz output, 80 ms / 2560-byte native units, `tail_policy=pad`, a
240 s session limit, and one admitted connection at a time. `benchmark_asr_realtime.py`
and its oracle do not grade native events, and `benchmarks/duplex/oracle.py`
does not grade transcription events. The recorded manifest pins the profile as
`nemotron-voicechat-pr2188`, the audio profile from
[VoiceChat duplex PR #2188](https://github.com/sgl-project/sglang-omni/pull/2188).
The harness follows the session protocol merged in
[PR #2070](https://github.com/sgl-project/sglang-omni/pull/2070), which removed
output epochs, `response.cancel` and the `session.closed.held` resource receipt.
Use a VoiceChat server that includes this protocol and record its actual revision.

This needs **two checkouts**, because neither side contains the other: the
launcher `examples/run_nemotron_voicechat_duplex.py` exists only in the pinned
target, and `benchmarks/duplex/` exists only here. Run the server from the
target checkout and the harness from this one.

```bash
# Terminal A — in the pinned target checkout: serve the native duplex endpoint
python examples/run_nemotron_voicechat_duplex.py \
    --model-path nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
    --serve --port 8097

# Terminal B — in this repository: record and grade the continuous case
python -m benchmarks.eval.benchmark_duplex \
    --url ws://127.0.0.1:8097/v1/realtime \
    --audio caller-16k.wav \
    --output results/duplex-run-1 \
    --server-revision "$VOICECHAT_SERVER_REVISION" \
    --model nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
    --timeout 180

# Terminal B — replay the saved run offline, with no server and no model
python -m benchmarks.duplex.artifacts results/duplex-run-1
```

Set `VOICECHAT_SERVER_REVISION` to the full commit SHA the server in terminal A
is actually running.

`--audio` must be PCM16 mono 16 kHz. Trailing silence is a property of the
fixture input — it keeps the last speech burst clear of the stream boundary —
and is *not* what ends the turn: under this profile the client ends input with
`sglang.input_audio.end` and waits for `sglang.input_audio.drained`, and the
oracle requires a `completed`/`stop` terminal for each response. Pick a
fixture whose PCM byte length is *not* a multiple of 2560,
otherwise `padding_ms` is always 0 and the tail accounting is never exercised.
`--output` must not already exist; each run directory is immutable.
`--server-revision` and `--model` are required, and `--model-revision` (full
model commit SHA) and `--runtime` (for example a container image digest) are
optional. All four are operator-supplied and recorded as such — the client
cannot read the server's commit, weights or runtime. Before the first session,
the client also records the model IDs the server reports at `/v1/models`, or the
probe error, under `server.served_models`. The IDs cross-check the operator's
claim; they do not identify weights. `source.packages` pins the harness's own
`websockets`, `numpy`, `pydantic` and `soundfile` versions. `--timeout` is the whole-session deadline: it
must exceed the paced input duration and stays at or below 230 s so the client
deadline fires before the server's 240 s session limit.

A run directory holds `manifest.json`, `input.pcm`, one JSONL trace per case,
and the replay-derived `report.json`. Both CLIs exit 0 only when every selected
case passes. Traces carry every send and receive event with client
`time.perf_counter` timestamps, admission denials, and client errors; a failed
or disconnected case keeps its trace and stays in the denominator rather than
disappearing from it.

The oracle recomputes verdicts from the raw records: granted capabilities,
an acceptance receipt for every transmitted append, native unit identity and
accounting, `accepted/consumed/discarded/padding` consistency, response terminal
ordering, input EOS drain, and the close receipt's client event ID and reason.
It also checks output conservation,
`output_samples == ceil(input_samples / 1280) * 1764`.

Replay re-verifies the `input.pcm` digest and re-hashes every transmitted
append against it, so a changed input PCM, a missing trace file or a
protocol-invalid trace fails rather than rescores. The trace bytes themselves
are not hashed: an otherwise-valid edit to recorded server events or timestamps
is re-graded by the oracle, not detected as tampering. Compare `recorded_source`
with `replay_source` in the report to tell reproduction from rescoring with a
changed grader. The append check also fires when a session aborts partway and
sends only a prefix of the fixture, which is an early-stop signature rather than
an integrity alarm — read it together with the case's other violations.
Legacy manifests containing `cancel_resume` require their recorded harness
revision; the current grader rejects that removed scenario. Preserve the original
traces and reports when comparing historical runs.

`report.json` records metrics per case and aggregates only passing cases into
`summary.qualified_metrics`; failed and unexercised cases land in
`diagnostic_metrics`. The timings (`first_audio_packet_s`,
`audio_packet_gap_max_s`, `drain_after_eos_s`, `close_ack_s`) are client packet-receipt and protocol
acknowledgment times. They are not audible speech onset and not acoustic stop
time.

A case that is healthy but did not exercise its scenario grades
`not_exercised`, never `pass`: continuous duplex requires input sent between
observed output audio packets. A second concurrent connection is refused with HTTP 503 and
retried up to three times before any input is sent; that is admission behavior,
not concurrency support.

Declared limits, also written into `manifest.config`: explicit response
cancellation, automatic speech interruption, concurrent native sessions, session
resume and truncate are unsupported by this target; semantic quality, acoustic
speech onset, audible stop time and server resource release are unmeasured.
The close acknowledgment does not prove that GPU or KV resources were released.

The following historical campaigns used the earlier epoch/cancel protocol and
do not validate the current session protocol. The initial H100 campaign (A01)
failed both cases on `buffer_overflow` after
roughly 30 s of paced input. In the later A02 campaign, the same pinned target
passed both cases after a 10 s profile warmup, but needed about 13 s to drain;
a supplemental baseline without that warmup passed one case and failed one.
The separate `fix/nemotron-duplex-realtime` candidate passed both cases twice,
including EOS and tail accounting, but one case still needed 7.20 s to drain.
Stable real-time service, graph/eager numerical equivalence and speech quality
remain unverified. The warmed baseline and candidate used separate boots on the
same GPU/runtime, while candidate repetitions shared one boot. Two lightweight
model-info queries overlapped candidate execution; these are observed workload
timings, not an isolated release benchmark. A01 used a different environment
and is not pooled with A02 for speedup claims.

The benchmark deliberately does not retry a rejected append. The native protocol
does permit resending the same sequence, but retrying would measure client-side
recovery instead of the endpoint's behaviour under real-time pacing, so a
`buffer_overflow` is recorded as a failure and the client stops driving input.

## Native duplex model profiles

Both `benchmark_duplex.py` and `benchmark_duplex_v15.py record` accept
`--profile`. The default remains `nemotron-voicechat-pr2188`; saved manifests
select the same profile during offline replay.

| Profile | Native unit | Output PCM16 | Response completion | Output length |
| --- | --- | --- | --- | --- |
| `nemotron-voicechat-pr2188` | 80 ms | 22,050 Hz, audio | After input EOS | Fixed samples per input unit |
| `minicpmo-native-pr2377` | 1,000 ms | 24,000 Hz, audio required | Natural turn end or input EOS | Variable; silence is valid |

Both profiles check the declared capabilities, causal receipts, complete input
accounting, native units, terminal ordering, EOS drain and session closure.
Only VoiceChat requires continuous output and fixed output-sample conservation.
For MiniCPM-o, `input_output_overlap` remains an observation, not a requirement
that every sample elicit speech. A protocol pass does not establish correct
conversational behavior. WAV reconstruction, duration accounting and optional
Whisper resampling use the selected profile's output rate.
The recorder also saves append-send completion receipts in
`input-send-receipts.json` and waits until the complete input duration has
elapsed before sending EOS, so offline analysis can verify the input window.

Start a compatible MiniCPM-o server using its full-duplex example configuration,
then record all v1.5 pairs:

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile minicpmo-native-pr2377 \
    --dataset-root data/full-duplex-bench-v1.5 \
    --dataset-revision "$DATASET_REVISION" \
    --url ws://127.0.0.1:8097/v1/realtime \
    --server-revision "$MINICPMO_SERVER_REVISION" \
    --model openbmb/MiniCPM-o-4_5 \
    --model-revision "$MODEL_REVISION" \
    --output results/minicpmo-v15 --timeout 90
```

Pin the server and model revisions separately. Sending 80 ms transport packets
does not change MiniCPM-o's one-second native processing unit. The v1.5 scoring
commands below retain their documented measurement scope; adding a model
profile does not change their definitions or imply paper-identical evaluation.

## Full-Duplex-Bench v1.5 overlap scenarios

`benchmark_duplex_v15.py` records the four v1.5 overlap subsets against the same native
endpoint; `benchmark_duplex_reference.py` scores them with the official ASR, timing and
behavior code. Follow the
[Full-Duplex-Bench v1.5 runbook](../docs/developer_reference/full_duplex_bench.md)
for setup, the copy-paste commands, measurement definitions and results.

## Full-Duplex-Bench v1.0 turn-taking tasks (VoiceChat)

`benchmark_duplex_v10.py` runs the five v1.0 subsets against the same native
endpoint, one `continuous` session per sample (no clean variant):

| Subset | Task | Annotation |
|---|---|---|
| `synthetic_pause_handling`, `candor_pause_handling` | pause handling | `pause.json` (one or more pauses) |
| `candor_turn_taking` | turn taking | `turn_taking.json` (one span; its start is the user turn end) |
| `synthetic_user_interruption` | user interruption | `interrupt.json` (span plus `context`/`interrupt` text) |
| `icc_backchannel` | backchannel | none |

It reuses the v1.5 runner, the 80 ms input pacing gate, output reconstruction and
the `transcribe` step. Non-passing samples stay in the selected denominator, and
`record` exits 0 only when every selected sample passes. The dataset is acquired separately.

v1.0 transcribes with Parakeet (`nvidia/parakeet-tdt-0.6b-v2`, the upstream
ASR) by default. Whisper invents words such as "Thank you." on silent output
and stretches their timestamps across the whole file, which turns a model that
correctly stayed quiet into a takeover; pause handling is the task most hurt.
Parakeet needs NeMo, so run `transcribe` in the scoring venv that the
[v1.5 runbook](../docs/developer_reference/full_duplex_bench.md) sets up, with a local `.nemo`
checkpoint. `--asr whisper` remains for diagnostics.

```bash
python -m benchmarks.eval.benchmark_duplex_v10 record \
    --dataset-root data/full-duplex-bench-v1.0 \
    --url ws://127.0.0.1:8097/v1/realtime \
    --output results/fdb10-run \
    --server-revision <full server commit SHA> \
    --model <served model path or ID> \
    --dataset-revision <release or archive digest> \
    --max-per-subset 2

python -m benchmarks.eval.benchmark_duplex_v10 transcribe \
    --run results/fdb10-run --output results/fdb10-asr \
    --model-path /models/parakeet-tdt-0.6b-v2.nemo --device cuda

python -m benchmarks.eval.benchmark_duplex_v10 score \
    --run results/fdb10-run --output results/fdb10-score \
    --transcripts results/fdb10-asr \
    [--backchannel-reference icc_gt_distribution.json]
```

`score` (`fdb-v10-synthetic-v4`) reports per-task takeover rate and latency
from the ASR word timestamps. A takeover is output lasting at least 1 s or
more than 3 words, as upstream. Every task only counts words starting inside
the input duration, matching upstream's equal-length `output.wav`.

- Pause handling: any takeover inside the input is a failure.
- Turn taking and user interruption only count words starting after the user
  turn or interruption ends; latency is the first such word's start minus that
  end, reported only for takeovers. Silero VAD on the output gates both: a
  turn-taking sample whose model is already speaking when the user turn ends is
  `spoke_before_turn_end` (talking over the user is not a response), an
  interruption with no model speech at its onset is `not_exercised`, and an
  interruption the model talks straight through, one VAD segment from before
  its onset to after its end, is `talked_through`. None of these enter the
  takeover rate or latency; their counts are reported. A sample whose
  output speech reaches the input end is flagged `right_censored`.
- Backchannel runs Silero VAD on the output. It reports backchannel rate and,
  with `--backchannel-reference` (upstream `icc_gt_distribution.json`), the
  Jensen-Shannon distance of 0.2 s binned backchannel timing to the human
  reference; 1 when the model produced none. Without the reference, timing is
  unscored.

Selected samples without a record count as `missing`; invalid, unqualified or
untranscribed samples are listed with an `unscored_reason`. This is not the
upstream evaluation code. Upstream has no speaking gates: a model that talks
through the user turn or the interruption scores a near-zero latency there,
and every interruption sample counts. The backchannel classifier differs from upstream
`eval_backchannel.py`: segments after the input end are ignored and others are
clipped to it, any takeover segment marks the sample (upstream keeps the
last segment's verdict and stops at the first segment over 3 s), and a takeover
segment is not also counted as a backchannel (upstream adds every segment up to
3 s to the backchannel list, takeover or not). Interruption
relevance (the upstream GPT-4 rating) is not scored.

Both `*_seedtts.py` scripts also support speech quality and similarity evaluation via UTMOS and WavLM speaker verification metrics. Running with `--utmos-only` or `--similarity-only` loads the respective pre-trained predictor and computes scores on the previously generated audio in the output directory without requiring the TTS/ASR servers to be running.

## SocialOmni

SocialOmni has two levels. Level 1 contains 2,000 four-choice speaker
attribution items. Level 2 classifies whether a target participant should speak
at one annotated time, using only the re-encoded audio-video prefix ending at
that time, and evaluates a generated continuation on gold-positive states.
Neither model prompt receives the reference transcript or continuation.

The Qwen3-Omni server must support `use_audio_in_video` and decode the video's
audio track. Start a text-only server with a context limit sufficient for the
selected videos:

```bash
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --text-only --port 8000 --model-name qwen3-omni \
    --preprocessing.factory.max_seq_len 65536 \
    --thinker.factory.max_seq_len 65536
```

The example sets a 65,536-token context limit for both preprocessing and model
execution. Adjust it and the GPU configuration for the selected videos and
available hardware; see [Qwen3-Omni configuration](../docs/basic_usage/qwen3_omni.md).
Run the benchmark on the server host or use a shared filesystem with identical
media paths. `--base-url` accepts the server root, such as
`http://localhost:8000`, or the same address ending in `/v1` or
`/v1/chat/completions`. Readiness checks use the server's `/health` route.

Level 1 parses the last non-empty response line, matching the prompt's
`Answer: X` format; preceding explanations are ignored. Ambiguous final answers
remain unparseable. A malformed completion response is recorded as a request
failure even when the server returns HTTP 200. Level 1 allows up to 256 output
tokens so a short explanation can precede the final answer. Level 2 turn-entry
decisions allow 32 tokens and must contain only Answer: A or Answer: B; the
continuation budget remains 256 tokens.

Missing or null token usage does not invalidate an otherwise valid completion.
Other non-object usage values and invalid token counts remain response errors.
Unavailable token counts use the shared runner's zero default, so token totals
and token rates are incomplete when usage is absent. The server's finish reason
is retained; length-truncated completions are recorded as failures and counted
in max_token_hits, rather than accepted as complete model or judge answers.

Model requests use `--max-concurrency`; judge concurrency is configured per
endpoint. Environment proxies are disabled by default. Use `--trust-env` to
respect `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` for model, judge, and health
requests; `--no-trust-env` explicitly disables them.
Use `--model-revision` to record the served weight revision in the result
configuration and provenance. This is a user declaration, not a server-verified
identity; `--model` remains the serving name. Prefix preparation failures stay
in per-sample results but are excluded from model request speed statistics.
`--launch-command` records the server command without executing it. Proxy
handling is recorded as `trust_env`; proxy URLs and credentials are not
copied into the result. Each judge result retains the full shared request
record under `request`.
For judges, `request` aggregates one scoring operation, while `attempts` retains
each completion request, including network and score-format retries, with its
own ID, response, error, timing and token counts. The `speed.judges` statistics count
these individual attempts. `speed.judge_scores` summarizes complete scoring
operations, preserving runner slot waits and dispatch lateness once per score.
Retries are not independently scheduled, so their request records do not inherit
the scoring operation's dispatch timing. A successful completion with an invalid score remains
a successful request in speed statistics; score validity is reported separately.
Judge request rates schedule scoring operations, with retries inside each operation;
provenance records this scope as `logical_scores`.

The public dataset downloader pins Hugging Face revision
`3b76009b45090eaa54007454c93a831f3cc8e1e6`.

```bash
python -m benchmarks.dataset.prepare \
    --dataset socialomni --local-dir /path/to/socialomni
```

The judge configuration contains exactly the fixed names `gpt-4o`,
`gemini-2.5-pro`, and `qwen3-omni`. Each entry declares an OpenAI-compatible
endpoint, the environment variable holding its API key, and a concurrency
limit. The output records the environment variable name, never its value. Start
from [`configs/socialomni_judges.example.json`](configs/socialomni_judges.example.json).

```bash
# Deterministic smoke set: both Level 1 visibility strata and Level 2 YES/NO.
python -m benchmarks.eval.benchmark_omni_socialomni \
    --dataset-root /path/to/socialomni --model qwen3-omni \
    --level both --mini --judge-config /path/to/judges.json

# All 2,000 Level 1 items.
python -m benchmarks.eval.benchmark_omni_socialomni \
    --dataset-root /path/to/socialomni --model qwen3-omni \
    --level level1

# All 209 maintained Level 2 items; also emits the first-200 paper view.
python -m benchmarks.eval.benchmark_omni_socialomni \
    --dataset-root /path/to/socialomni --model qwen3-omni \
    --level level2 --judge-config /path/to/judges.json \
    --prefix-cache-dir /path/to/socialomni-prefixes
```

Level 1 reports accuracy, four-position macro-F1, both visibility strata, and
their descriptive accuracy gap. Level 2 reports classification metrics plus
four response quantities: `QGold` is the three-judge mean after forced
generation on every gold-positive state; `QEns` is response quality conditional
on a correct YES decision and a non-empty response; `Cov+` is the fraction of
gold-positive states meeting that condition; and `QEns_joint` is
`Cov+ * QEns`. All three scores in `{0, 25, 50, 75, 100}` are mandatory for
every non-empty eligible response. A judge failure makes the run incomplete.
Omitting `--judge-config` runs model-only diagnostics: model responses, turn-entry
metrics and performance are saved, but `summary.status` remains `incomplete`,
`judge_status.configured` is false, and quality metrics remain null, including
the first-200 view. This mode is not a complete SocialOmni quality evaluation.

Each run writes one JSON file containing its configuration, environment,
per-sample records, failures, performance summary, and paper metrics. The
first-200 paper view uses source order and the same saved records. A model error
or unparseable answer remains in the fixed denominator. Latency percentiles and
throughput are engineering diagnostics, not SocialOmni paper metrics.
Level 2 prepares all video prefixes before timing, then runs the turn-entry
decisions and forced gold-positive responses as separate phases. Each nonempty
model phase repeats its first sample once per concurrent worker for warmup;
`--warmup N` overrides the count and `--warmup 0` disables it. Warmup results
are discarded by the shared runner. As in other benchmarks, a failed warmup
aborts the run before its measured phase. Fixed-denominator accounting applies
to measured requests; use `--warmup 0` to evaluate without this precondition.
Judges use no warmup to avoid duplicate
paid scoring requests. Model wall time sums the two measured runner phases
and excludes media preparation, warmup, and judge scoring.
Prefix preparation failures stay in the sample records and fixed denominator.

`--server-timeout` controls startup readiness independently of the per-request
`--timeout-s`; both default to 300 seconds. `--request-rate` sets the shared
runner's request rate for each model phase and each judge independently. Its
default `inf` sends a new request whenever a concurrency slot is available;
a positive finite value sets the arrival rate in requests per second. The
effective rate is saved in config and provenance. Infinite rates are written
as the JSON string `"inf"`; finite rates remain numbers. Model and judge endpoint
URLs must not contain userinfo, query parameters or fragments. Configure judge
authentication through `api_key_env` instead; URL credentials are rejected
before evaluation results are written.

Dataset preparation defaults to `benchmarks/cache/socialomni/`; `--local-dir`
overrides it. Prefixes are cached in `benchmarks/cache/socialomni-prefixes/`;
result JSON files go to `benchmarks/results/socialomni/`. All three directories
are ignored by Git. The dataset's videos directory may be a symlink to external
storage; individual media paths must stay inside that resolved directory.
Level 2 requires an audio track with decodable samples in every prefix. Missing
audio is reported as a prefix preparation failure. Prefix cache keys include
the required audio mapping, so older optional-audio entries are not reused.
Source video digests are reused within a process while device, inode, size,
modification time and change time remain unchanged. Files changed within the
last second bypass the cache to avoid timestamp collisions. The bounded cache
is rebuilt for each process; a new invocation recomputes source digests before
reusing encoded prefixes.
The result records whether the local metadata matches the pinned revision. This
is a metadata identity check; local media files are validated as samples are
loaded, but are not hashed in full at startup.

## TTS Quality Evaluation

To evaluate the overall quality and vocal resemblance of synthesized speech, the benchmark suite supports offline evaluation using UTMOS (naturalness MOS prediction) and Speaker Similarity (vocal fidelity).

### UTMOS (Naturalness)

UTMOS (UTokyo-Saru Lab MOS Prediction) is a Mean Opinion Score (MOS) predictor model used to evaluate the naturalness and overall quality of synthesized speech. SGLang Omni provides an offline UTMOS evaluator backed by the `balacoon/utmos` JIT model on Hugging Face.

- **Model Weights Cache**: By default, weights (`utmos.jit`) are downloaded from Hugging Face on the first run and cached at `~/.cache/sglang-omni/utmos` (override via `UTMOS_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.utmos --warm-cache
  ```
- **Outputs (`utmos_results.json`)**:
  - `summary`: section for summary metrics
    - `utmos_mean`: Mean predicted MOS score in `[1, 5]` (higher is better).
    - `utmos_median`: Median predicted MOS score.
    - `utmos_p5` / `utmos_p95`: 5th and 95th percentile scores to identify worst-case outliers or top-performing samples.
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

### Speaker Similarity (Vocal Fidelity)

Speaker Similarity evaluates how closely the voice of synthesized speech matches the reference prompt audio.

- **Model Weights Cache**: By default, model weights (`wavlm_large.pt` and `wavlm_large_finetune.pth`) are downloaded from Hugging Face and cached at `~/.cache/sglang-omni/speaker_sim` (override via `SEEDTTS_SIM_CACHE_DIR` environment variable).
- **Warm the Cache (Optional)**:
  ```bash
  python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
  ```
- **Outputs (`similarity_results.json`)**:
  - `summary`: section for summary metrics
    - `speaker_similarity_mean`: Mean cosine similarity score scaled by 100.0 (higher is better).
    - `total_samples`: total number of samples in the dataset.
    - `evaluated`: number of samples that were evaluated.
    - `skipped`: number of samples that were skipped.
  - `config`: evaluation configuration parameters.
  - `per_sample`: list of individual score results for each evaluated sample.

## Adding a New Model or Task

- **New model, same task/API type** (e.g. another OAI-compatible TTS model):
  add an eval script under `eval/` that reuses the existing task helpers
  in `tasks/tts.py` (`make_tts_send_fn`, `run_seedtts_transcribe`, …).
- **New task or API type**: add a task class in the relevant `tasks/*.py`
  file (mirroring `VoiceCloneOmni` in `tasks/tts.py`), expose metric
  helpers, and wire it into a new eval script.

## Datasets

Download helpers live in `benchmarks/dataset/prepare.py`:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts       # full SeedTTS
python -m benchmarks.dataset.prepare --dataset seedtts-mini  # smoke-test subset
python -m benchmarks.dataset.prepare --dataset seedtts-50    # 50-sample subset
python -m benchmarks.dataset.prepare --dataset stt-benchmark # Pipecat STT benchmark set (1000 EN clips)
python -m benchmarks.dataset.prepare --dataset longlibriheavy-30  # LongLibriHeavy llh_test_30 only
python -m benchmarks.dataset.prepare --dataset longlibriheavy-60  # LongLibriHeavy llh_test_60 only
python -m benchmarks.dataset.prepare --dataset meanwhile     # complete Meanwhile test split (64 EN clips)
python -m benchmarks.dataset.prepare --dataset mmmu          # full MMMU (30 subjects)
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50    # MMMU CI subset
python -m benchmarks.dataset.prepare --dataset mmsu          # full MMSU (ddwang2000/MMSU)
python -m benchmarks.dataset.prepare --dataset mmau-mini     # MMAU test_mini split
python -m benchmarks.dataset.prepare --dataset mmar          # MMAR metadata + audio archive
python -m benchmarks.dataset.prepare --dataset videomme-ci-50  # Video-MME CI subset
python -m benchmarks.dataset.prepare --dataset videomme      # full Video-MME
python -m benchmarks.dataset.prepare --dataset videoamme-ci-50  # Video-AMME CI subset
python -m benchmarks.dataset.prepare --dataset socialomni --local-dir /path/to/socialomni
```

Most datasets are pre-warmed into the default Hugging Face cache via
`datasets.load_dataset(repo_id)`. SocialOmni is materialized with its media at
the requested `--local-dir`. SeedTTS Arrow repos stage audio to
process-local tempfiles at load time; no manual `--local-dir` step is needed.

Video-AMME is generated from the Video-MME CI subset by moving the
question/options/instruction into per-sample WAV files. The benchmark request
text only contains routing/format instructions; the actual question content
stays in the dataset WAV files.
