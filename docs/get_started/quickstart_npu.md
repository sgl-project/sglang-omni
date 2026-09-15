# Quickstart — Ascend NPU

Run SGLang-Omni in an ARM64 Ascend container. The image includes Omni source,
SGLang 0.5.19, CANN 9.0.0, PyTorch / `torch_npu` 2.10 and common audio dependencies.
Model weights, host drivers and TorchCodec are not included. Model support on
NPU and optional dependencies must be checked separately; the image does not
enable every Omni model automatically.

## Prerequisites

- An ARM64 Linux host with Ascend A3 or A2 / 910B devices and Docker.
- Compatible host drivers and firmware; see the HDK links in
  [NPU prerequisites](installation_npu.md#prerequisites). Check `npu-smi info`
  on the host before starting.
- Enough device memory and disk space for the selected model. The Qwen3-TTS
  example below uses one device and a model-specific NPU configuration.

The runtime is already installed inside the image. Do not rerun the source
installation or TorchCodec stack-upgrade scripts inside it.

## 1. Prepare the image

Until official images are published, build from the repository root. Choose
**one** build matching your hardware; A2 and A3 operator binaries are not
interchangeable.

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# A3
IMAGE=sglang-omni:npu-a3
docker build -f docker/npu.Dockerfile -t "$IMAGE" .
```

For A2 / 910B, replace the last two commands with:

```bash
IMAGE=sglang-omni:npu-910b
docker build -f docker/npu.Dockerfile \
  --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.19-cann9.0.0-910b@sha256:19beb175fe8b5a636a14c0f8a95aa92686a36e09b826f214be476b2ae22e7188 \
  -t "$IMAGE" .
```

If your operator provides a published image, set `IMAGE` to its matching
registry reference and run `docker pull "$IMAGE"` instead. Use a tested digest
for deployments. A successful build alone does not verify inference on A2 or A3.

## 2. Prepare the model

This example uses Qwen3-TTS 0.6B CustomVoice, which generates speech with built-in
voices without reference audio. Download weights to a host directory using the
image's Hugging Face CLI, or use an existing complete checkpoint, including
`speech_tokenizer/`:

```bash
MODEL_DIR=/mnt/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
mkdir -p "$MODEL_DIR"
docker run --rm \
  -v "$MODEL_DIR:/model" \
  "$IMAGE" hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --revision 85e237c12c027371202489a0ec509ded67b5e4b5 --local-dir /model
```

## 3. Start the service

The example uses physical device 0 and binds the API to host loopback.
Replace the device ID and driver mount paths if your host differs. `--privileged`
grants broad host access: use this quickstart only on a trusted host; production
deployments should use their platform's restricted device-access configuration.

```bash
docker run -d --name omni-npu \
  --privileged --network host --shm-size 8g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_DIR:/model:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=8 -e HF_HUB_OFFLINE=1 \
  "$IMAGE" bash -lc '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    exec sgl-omni serve \
      --model-path /model \
      --config /workspace/sglang-omni/examples/configs/qwen3_tts_0_6b_customvoice_npu.yaml \
      --host 127.0.0.1 --port 8000
  '

docker logs -f omni-npu
```

When selecting another device, update both `/dev/davinci0` and
`ASCEND_RT_VISIBLE_DEVICES`; keep `gpu: 0` in the model configuration, which
refers to the first visible device. Wait for startup to complete, then check
the API from another host terminal:

```bash
curl --fail http://localhost:8000/v1/models
```

## 4. Send a request

Replace `input` with your own text. This example uses the built-in `Vivian`
voice with Chinese; for English, use `language: "English"` and `voice: "Ryan"`.
The response is saved as `speech.wav` on the client for playback.

```bash
curl --fail --show-error --max-time 600 http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，欢迎使用语音合成服务。",
    "language": "Chinese",
    "voice": "Vivian",
    "task_type": "CustomVoice",
    "response_format": "wav",
    "max_new_tokens": 256
  }' --output speech.wav
```

See the [Qwen3-TTS cookbook](../cookbook/qwen3_tts.md) for additional request
options. Base and VoiceDesign checkpoints require their matching NPU
configuration and request fields; changing only the model directory is not
sufficient. For remote access, configure the service binding, authentication
and network access controls before exposing the API.

Stop and remove the container when finished; downloaded host weights remain:

```bash
docker stop omni-npu
docker rm omni-npu
```
