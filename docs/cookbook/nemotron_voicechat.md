# NVIDIA NemotronLabs VoiceChat 11B

[NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
is a frame-locked, full-duplex speech-to-speech model. SGLang-Omni runs it as
four colocated stages on one H100:

```text
16 kHz PCM -> perception (Conformer) -> thinker (Nemotron-H)
           -> talker (EarTTS) -> code2wav (RVQ-VAE) -> 22.05 kHz PCM
```

The realtime endpoint consumes 1,280-sample PCM16 frames (80 ms) continuously.
It does not wait for voice-activity detection or a turn boundary, so microphone
input continues while assistant audio is playing.

## Prerequisites

Use NVIDIA's VoiceChat runtime environment, which supplies the checkpoint's
`nemo.collections.speechlm2` modules, and install SGLang-Omni in that
environment. SGLang-Omni registers its VoiceChat models at runtime and works
with its pinned published SGLang package; no separate SGLang source checkout is
required.

Prepare an absolute data directory outside the repository with this layout:

```text
/absolute/path/to/voicechat-data/
├── checkpoint/
│   ├── config.json
│   └── model.safetensors
└── converted/
    ├── duplex/
    │   ├── config.json
    │   └── model.safetensors.index.json
    └── eartts/
        ├── config.json
        ├── model.safetensors
        └── speaker_latents/
            └── <speaker>.pt
```

`checkpoint` is the unchanged Hugging Face snapshot. Do not point
`--model-path` at only the snapshot or only the converted folder; it must
point at their common `voicechat-data` parent.

Set `VOICECHAT_DATA` to that absolute parent directory, download the snapshot,
and create the two runtime stage directories:

```bash
export VOICECHAT_DATA=/absolute/path/to/voicechat-data

mkdir -p "$VOICECHAT_DATA/checkpoint" "$VOICECHAT_DATA/converted"
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --local-dir "$VOICECHAT_DATA/checkpoint"

python -m sglang_omni.models.nemotron_voicechat.convert_duplex \
  --checkpoint "$VOICECHAT_DATA/checkpoint" \
  --config "$VOICECHAT_DATA/checkpoint/config.json" \
  --output "$VOICECHAT_DATA/converted/duplex"

python -m sglang_omni.models.nemotron_voicechat.convert_eartts \
  --config "$VOICECHAT_DATA/checkpoint/config.json" \
  --model "$VOICECHAT_DATA/checkpoint/model.safetensors" \
  --output "$VOICECHAT_DATA/converted/eartts"
```

The EarTTS conversion instantiates NVIDIA's trained character-aware subword
encoder and should run with a GPU in the VoiceChat runtime environment. Reduce
`--precompute-batch-size` from its default of `256` if needed. Authenticate
with the Hugging Face CLI before downloading if model access requires accepted
terms; do not put access tokens in commands or checked-in files.

## Launch

Pass the absolute `voicechat-data` parent to the provided configuration:

```bash
export VOICECHAT_DATA=/absolute/path/to/voicechat-data

sgl-omni serve \
  --model-path "$VOICECHAT_DATA" \
  --config examples/configs/nemotron_voicechat_h100.yaml \
  --enable-realtime \
  --host 0.0.0.0 \
  --port 18080
```

Startup runs two disposable silent frames through all four stages and closes
that session before accepting traffic. This warms the perception graph,
autoregressive engines, and codec without leaking conversational state into a
real client session.

Connect a realtime WebSocket client to `ws://HOST:18080/v1/realtime`. Send
base64-encoded mono PCM16 through `input_audio_buffer.append`; chunks may contain
any whole number of PCM16 samples because the server assembles exact 1,280-sample
model frames. Assistant audio arrives in `response.audio.delta` events as raw
22.05 kHz PCM16 base64. Send `input_audio_buffer.commit` to drain the current
queue, and `session.close` before disconnecting when possible.

## Live microphone client

The example client captures and resamples microphone audio, streams it while
playing the model response, prints returned text, and saves the response to a
WAV file. Install its optional audio dependency on the client machine:

```bash
# macOS
brew install portaudio
python -m pip install pyaudio websockets

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
python -m pip install pyaudio websockets
```

If the server is on a remote host, forward its realtime port from the client
machine and leave the tunnel running:

```bash
ssh -N -L 18080:127.0.0.1:18080 USER@SERVER_HOST
```

From the SGLang-Omni repository root, list the available audio devices:

```bash
python examples/nemotron_voicechat_client.py --list-devices
```

Then start a full-duplex session. Device indices are optional when the default
input and output devices are correct:

```bash
python examples/nemotron_voicechat_client.py \
  --url ws://127.0.0.1:18080/v1/realtime \
  --input-device-index INPUT_INDEX \
  --output-device-index OUTPUT_INDEX \
  --output-wav response.wav
```

Use headphones to prevent speaker output from feeding back into the microphone.
Speak at any time during the session, including while the assistant is speaking,
and press Enter to stop. Use `--microphone-seconds N` for a fixed-duration run
or `--no-playback` to save the returned audio without playing it.

## Reproduce the latency benchmark

The benchmark uses file input, not microphone input. The reported workload
starts with `/s2s/what_is_your_name.wav` from the pinned NIM image, appends two
seconds of silence, converts it to 137 PCM16 frames at 16 kHz, and sends one
1,280-sample frame every 80 ms. One complete session is discarded before 20
fresh sessions are measured.

Extract the public fixture and prepare the exact input:

```bash
export NIM_IMAGE=nvcr.io/nim/nvidia/nemotron-labs-voicechat@sha256:6e69ff2aac955be2cb65b0de4f5b6d7c0b5e45ca0a1d42a2b153e9b54efb059b
export VOICECHAT_BENCH=/absolute/path/to/voicechat-benchmark

mkdir -p "$VOICECHAT_BENCH"
docker create --name voicechat-nim-files "$NIM_IMAGE"
docker cp voicechat-nim-files:/s2s/what_is_your_name.wav \
  "$VOICECHAT_BENCH/what_is_your_name.wav"
docker rm voicechat-nim-files

python benchmarks/eval/benchmark_nemotron_voicechat.py prepare \
  "$VOICECHAT_BENCH/what_is_your_name.wav" \
  "$VOICECHAT_BENCH/what_is_your_name_137x80ms_16k_pcm16.wav"
```

The source SHA-256 should be
`34137b55b03d6ccd0e054c98f9d973b93ab7e18a2c8f30efef92425cc71e6dfa`;
the prepared WAV SHA-256 should be
`77aa93a6cc36ec7f9371a91c14bb1a4048240a0d6d5b907ea961e84c29fa103d`.
`VOICECHAT_BENCH` is only an output directory; it is not the checkpoint path
used by `--model-path`.

With the SGLang-Omni server running, reproduce the 20 measured sessions:

```bash
python benchmarks/eval/benchmark_nemotron_voicechat.py sglang \
  --url ws://127.0.0.1:18080/v1/realtime \
  --input-wav "$VOICECHAT_BENCH/what_is_your_name_137x80ms_16k_pcm16.wav" \
  --output-dir "$VOICECHAT_BENCH/sglang" \
  --warmup-runs 1 \
  --runs 20
```

The command writes every response WAV and a `raw.json` containing the input
hash, frame schedule, arrival timestamps, transcript, and aggregate summary.

For the matched comparison, create a NIM Triton model repository from the same
published checkpoint. `VOICECHAT_NIM_MODEL_REPO` must be an absolute, empty
output directory; it is not the SGLang `--model-path`:

```bash
export VOICECHAT_NIM_MODEL_REPO=/absolute/path/to/nim-model-repository

mkdir -p "$VOICECHAT_NIM_MODEL_REPO"
docker run --rm --gpus all \
  -v "$VOICECHAT_DATA/checkpoint:/checkpoint:ro" \
  -v "$VOICECHAT_NIM_MODEL_REPO:/data/models" \
  -e NEMO_CHECKPOINT_PATH=/checkpoint \
  -e MODEL_REPOSITORY=/data/models \
  --entrypoint /s2s/deploy_s2s_model.sh \
  "$NIM_IMAGE"
```

Launch the pinned NIM image as described in NVIDIA's
[deployment guide](https://github.com/NVIDIA-NeMo/Speech/blob/nemotron-labs-voicechat/voicechat_realtime_instructions/deploy.md).
This is the command used for the reference results:

```bash
docker run --rm --gpus all --network host --shm-size 8g \
  -v "$VOICECHAT_NIM_MODEL_REPO:/data/models:ro" \
  -e MODEL_REPOSITORY=/data/models \
  -e NIM_HTTP_API_PORT=9000 \
  "$NIM_IMAGE"
```

After `http://127.0.0.1:9000/v1/realtime/health` reports `ok`, send the same
prepared 16 kHz PCM16 frames through its public WebSocket endpoint:

```bash
python benchmarks/eval/benchmark_nemotron_voicechat.py nim \
  --url ws://127.0.0.1:9000/v1/realtime \
  --input-wav "$VOICECHAT_BENCH/what_is_your_name_137x80ms_16k_pcm16.wav" \
  --output-dir "$VOICECHAT_BENCH/nim" \
  --warmup-runs 1 \
  --runs 20
```

Both commands measure from the first audio-frame send to the first audio-event
arrival at their public WebSocket endpoints. They use identical input bytes,
80 ms pacing, prompt, warmup count, and run count. Native output sample rates
and chunk sizes remain unchanged and are recorded in each `raw.json`.

### Reference results

These results were collected on August 21, 2026, with both servers running
separately on the same H100 PCIe. The SGLang-Omni server used this PR with
`sglang==0.5.16`. NIM used the pinned image above. All 20 measured sessions
from both backends returned `I am your AI voice assistant.`.

| Backend | Runs | Mean first audio | Median | p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA NIM 2.0.0 | 20 | 315.59 ms | 315.96 ms | 317.77 ms | 311.91 ms | 322.31 ms |
| SGLang-Omni | 20 | 173.93 ms | 174.59 ms | 175.34 ms | 169.89 ms | 175.44 ms |

The public NIM endpoint delivers two 80 ms audio events back-to-back in each
160 ms emission batch. SGLang-Omni delivers one 80 ms event at a time. The
paired row below groups adjacent SGLang events using their arrival timestamps
to compare the same 160 ms media duration:

| Delivery | Intervals | Mean gap | Median gap | p95 gap | p99 gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| NIM 160 ms emission batch | 1,320 | 160.04 ms | 162.24 ms | 164.98 ms | 166.47 ms |
| SGLang native 80 ms event | 2,720 | 79.47 ms | 79.46 ms | 86.02 ms | 89.60 ms |
| SGLang paired to 160 ms | 1,340 | 159.40 ms | 159.65 ms | 165.89 ms | 168.78 ms |

The NIM cadence row excludes the final batch whose interval crosses
`session.close`. The reference file client deliberately waits two seconds
after the input file before closing, so that last gap measures the shutdown
policy rather than steady-state audio delivery. The benchmark records the
close timestamp and applies this rule when producing
`paired_160ms_interval_ms`.

## Current limits

- The supplied one-H100 configuration admits one full-duplex session.
- Inputs must be mono PCM16 at 16 kHz; clients must resample microphone audio.
- The SGLang thinker uses bfloat16 for latency. Use float32 when producing a
  token-for-token reference comparison.
