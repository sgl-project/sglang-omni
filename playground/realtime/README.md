# Voice and video playground

Talk to MiniCPM-o through your microphone, or turn on the camera for video chat. An orb follows your voice and the model's, and the model's words appear as captions.

You need a Linux NVIDIA GPU server and Docker with NVIDIA GPU support.
This setup has been tested on one H200. Use a browser on
your own computer; headphones help prevent echo.

## 1. Prepare the server

Run these commands **on the GPU server**:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

docker run -d --name omni-playground \
  --gpus '"device=0"' --ipc=host --network=host \
  -v "$PWD":/workspace/sglang-omni \
  -w /workspace/sglang-omni \
  --entrypoint /bin/bash hongccc/sglang-omni:dev -c 'sleep infinity'

docker exec omni-playground python -m pip install --no-deps -e . 'sglang==0.5.21'
docker exec omni-playground python -m pip install 'onnx==1.23.0' aiohttp soundfile scipy pyyaml pillow

docker exec omni-playground hf download openbmb/MiniCPM-o-4_5 \
  --local-dir /workspace/sglang-omni/models/MiniCPM-o-4_5
```

The first run downloads the container and model, so allow time for both.
If you already have the model, put it at `models/MiniCPM-o-4_5` and skip the download.

## 2. Start the model

In the same terminal:

```bash
docker exec -it omni-playground python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo.yaml \
  --model-path /workspace/sglang-omni/models/MiniCPM-o-4_5 \
  --enable-realtime --host 127.0.0.1 --port 8000
```

Leave this terminal open. In a **second terminal on the server**, wait until
this command returns JSON containing `"native_full_duplex":true`:

```bash
curl --fail http://127.0.0.1:8000/v1/realtime/capabilities
```

Then start the page in that second terminal:

```bash
docker exec -it omni-playground python playground/realtime/app.py \
  --api-base http://127.0.0.1:8000 --port 8080
```

The page first warms up voice, video and speech generation. Wait for the
`Running on` message before opening it; the first startup can take a few minutes.
If warmup fails, the page stays offline and the terminal shows the error.

## 3. Open the page

On **your own computer**, replace `USER` and `SERVER` with your server login:

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@SERVER
```

Keep the SSH terminal open, then visit <http://localhost:8080>.

Click **Start talking** and allow microphone access; the camera button appears once the call starts. The red button ends the conversation. If the server is your own computer, skip the SSH command.

The settings button in the top bar opens the preset (English or Chinese call, with or without video), the system prompt, the voice (the preset's, the model's default or a recording you upload), text-only replies, the microphone and, under **Advanced**, the sampling parameters and camera detail. Changes apply to the next conversation and are remembered by the browser; empty sampling fields keep the server's defaults.

The download button in the top bar saves the last conversation's trace as JSON: every event the page sent and received, including the audio you sent. Attach it when reporting a problem, so the conversation can be replayed exactly.

## Stop or restart

To stop both services, run `docker stop omni-playground` on the server.
To use them again, run `docker start omni-playground`, then repeat steps 2–3.
Your downloaded model stays in the repository's `models/` directory.

## Having trouble?

- **No microphone/camera prompt:** open the `localhost` URL above, not the server's IP address, and check browser permissions.
- **Model service unavailable:** check the model terminal and repeat the `curl` check before starting a conversation.
- **Port 8080 is busy on your computer:** use `ssh -N -L 8081:127.0.0.1:8080 USER@SERVER` and open `http://localhost:8081`.
