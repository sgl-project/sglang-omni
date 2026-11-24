# Stable Audio 3

[Stable Audio 3 Medium](https://huggingface.co/stabilityai/stable-audio-3-medium) generates up to 380 seconds of 44.1 kHz stereo music or sound effects from a text prompt. A T5Gemma encoder reads the prompt, a 1.4B diffusion transformer denoises audio latents in 8 ping-pong steps, and the SAME autoencoder decodes them to audio.

The weights are under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-3-medium/blob/main/LICENSE.md). The repository is gated: accept its terms on Hugging Face and set `HF_TOKEN` before the first launch.

## Start the server

Install SGLang-Omni following [Installation](../get_started/installation.md), then start one CUDA worker:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path stabilityai/stable-audio-3-medium \
  --port 8000
```

One GPU stage runs the prompt encoder, the diffusion transformer and the decoder. Like the official inference, the transformer and decoder run in fp16 and the prompt encoder in bf16. Requests that arrive together are generated as one batch of up to `--generation.factory.max_batch_size` requests (default 8). Each request keeps its own duration and seed, so batching does not change its audio. A batch runs in groups of similar length, so a short request is not padded to a long one.

The transformer and decoder blocks are compiled at startup, which takes about 15 seconds the first time, and the prompt encoder and short-clip diffusion steps replay CUDA graphs. `--generation.factory.enable_torch_compile false` and `--generation.factory.enable_cuda_graph false` turn them off.

## Generate audio

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "stabilityai/stable-audio-3-medium",
    "input": "House music that encapsulates the feeling of being at a festival in the sunny weather 124 BPM",
    "seed": 42,
    "stage_params": {"generation": {"duration_seconds": 30}},
    "response_format": "wav"
  }' \
  --output house.wav
```

`input` is the prompt; genre, instruments, mood and tempo all help. `stage_params.generation.duration_seconds` sets the length, up to 380 seconds; the default is 120 seconds, as in the official library. `seed` makes a request reproducible; without it every request draws a new random seed. Voices, streaming, sampling fields and other generation settings are rejected with a 400 error.

## Limitations

- Only Stable Audio 3 Medium is supported. Small Music and Small SFX use a different autoencoder.
- Only text-to-audio is supported. The official library's audio-to-audio editing, inpainting and LoRA loading are not.
- The checkpoint is distilled for 8 sampling steps without classifier-free guidance, so steps and guidance are fixed.

## Validation

```bash
pytest tests/unit_test/stable_audio3 -q
```

The CPU tests cover request validation, the official latent lengths and noise schedule, and that batching leaves every request's audio unchanged.
