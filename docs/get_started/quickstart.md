(api-server-quickstart)=
# Quickstart

Start Qwen3-ASR-1.7B on one CUDA GPU and transcribe a short audio clip.

## Install and start

Use Linux with a compatible NVIDIA GPU and driver. Choose Docker or uv below;
see [CUDA Installation](installation.md) for the requirements and
[other platforms](installation.md#other-platforms) for non-CUDA setups.

**Docker (recommended):** Follow [Docker installation](installation.md#docker-recommended)
to pull the image, start the container, and install SGLang-Omni inside it.
Run the launch command below in that container.

**uv:** With the {ref}`CUDA prerequisites <option-b-manual-install>` installed,
create a Python 3.12 environment on the host:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.2.0"
```

From either environment, start the server:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --host 127.0.0.1 \
  --port 8000
```

The first launch downloads the checkpoint. Wait for the server to finish starting. If startup stalls, check the server log and run `curl -fsS http://localhost:8000/health`.

## Send a request

In a second terminal on the same host, download the sample WAV used in the
[Qwen3-ASR Cookbook](../cookbook/qwen3_asr.md#transcribe-audio) and upload it:

```bash
curl -fL -o query_to_cars.wav \
  https://raw.githubusercontent.com/sgl-project/sglang-omni/7fc1a31850efbc1138e1895b5a3bcbeecf3a8481/tests/data/query_to_cars.wav

curl -fsS http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@query_to_cars.wav \
  -F response_format=json
```

Check the JSON response's `text` field for the spoken question, “How many
cars are there in the picture?”

Keep the server running to try [streaming transcription](examples.md#stream-a-transcript).
