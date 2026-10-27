# Cascade Image Playground

The cascade playground supports multi-image upload with a continuous microphone, automatic turn detection and speech-triggered interruption. See the [image cascade guide](../../examples/streaming_cascade/README.md) for Docker setup, model revisions, GPU layout, the script example, limits and real-model acceptance tests.

## Start

From the repository root, start these commands in separate terminals after creating the Docker environment:

```bash
docker exec sglang-omni-d1 sgl-omni serve \
  --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml \
  --enable-realtime --host 0.0.0.0 --port 18020
docker exec sglang-omni-d1 python playground/cascade/app.py \
  --api-base http://127.0.0.1:18020 --host 0.0.0.0 --port 18080
```

Open `http://localhost:18080`, upload one or more images and start the session. The microphone starts after the server acknowledges the images and stays open during answers. Pause for 800 ms to submit a question; speaking over the answer interrupts it and clears buffered playback. The microphone button mutes or unmutes input, and retries capture after a permission failure. Use localhost or HTTPS for browser microphone access from another machine.

## Upload Lifecycle

The picker accepts several JPEG or PNG files at once. Each is validated locally and shown as a thumbnail in upload order; the tray keeps the newest `max_frames_per_unit` images the session grants, which is the number of images the conversation history keeps. Before the session starts the status shows Ready; during a live session new uploads are sent immediately in order and show Sending, then Received after acknowledgment. Only images not yet received by the current session are sent, so earlier pictures are never duplicated. Rejected uploads leave the microphone and session available for another image or a question using retained visual history.

Every image uploaded before an utterance's speech onset is attached to that question, so pictures picked together are asked about together. Images uploaded during speech or an answer apply to the next utterance, including when the earlier transcript is delayed. Earlier pictures stay in the conversation, so a follow-up can compare a new picture with previous ones. Clearing the tray does not retract acknowledged uploads or erase earlier images from conversation history; start a new session for fresh visual context.

Image uploads are shown whenever the deployment grants image input. A deployment without automatic turn detection retains the manual record/submit controls when it grants client commit. Audio-only deployments retain their continuous microphone workflow. The bundled icons are generated from Lucide 1.61.0 and include its ISC license notice.

## Verification

On 2026-10-27, Playwright exercised the actual Docker cascade with Qwen3-ASR-1.7B, Qwen3-VL-4B-Instruct, and Qwen3-TTS-12Hz-1.7B-Base. Chromium captured controlled WebAudio speech through the page's continuous microphone worklet and played nonempty generated audio. Red and blue images produced their respective answers; replacing red with blue during speech preserved the current red answer; speaking over the next blue reply cancelled it and a handle follow-up completed. No client commit or explicit response cancellation was sent. The player reported the interrupted playback position through truncation. Mute, preview removal, session cleanup, and desktop/mobile screenshots also passed, with no page errors, horizontal overflow or caption overlap. On 2026-10-28, Chromium picked red and blue cup images together before starting a session, sent both before opening the microphone, and answered a fake-microphone question with "The cups in all the pictures are red and blue." A 390 px wide page with eight images keeps the tray scrollable without horizontal page overflow.

All 25 cascade browser unit tests pass, including picking several images at once, sending only images the session has not received, and trimming the tray to the granted window. They cover acknowledgment ordering, continuous capture during replies, playback clearing on speech, recoverable rejection, microphone permission failure and image resubmission on retry, disconnect cleanup, pending audio flushing before image upload or manual commit, and existing playback behavior. Physical microphone capture and macOS recording still require checking on the client machine.

```bash
node --test playground/cascade/frontend/orb/*.test.mjs
```
