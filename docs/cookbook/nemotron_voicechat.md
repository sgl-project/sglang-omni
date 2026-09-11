# NemotronLabs VoiceChat

[NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) is an 11B end-to-end full-duplex speech-to-speech model. It is frame-locked at 12.5 Hz: every 80 ms of caller audio advances a Fast Conformer perception encoder, a Nemotron-H thinker that emits one text token and one function token per frame, an EAR-TTS talker that turns each text token into 31 RVQ codes, and an RVQ-VAE codec that renders the codes to 22.05 kHz audio. The model decides when to speak; there is no external VAD.

SGLang-Omni currently serves it as an **offline** pipeline: one recording in, the agent's reply text and audio out. Live duplex sessions over `/v1/realtime` are tracked in [#1909](https://github.com/sgl-project/sglang-omni/issues/1909) and are not part of this page.

## Prerequisites

The checkpoint is a single 44 GB fp32 `model.safetensors` with a NeMo-style `config.json`. Download it with:

```bash
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
```

The checkpoint ships neither an HF tokenizer nor the backbone `config.json`; both come from the sibling repo named in its config (`model.stt.model.pretrained_llm`, currently `nvidia/NVIDIA-Nemotron-Nano-9B-v2`), about 17 MB of files and no weights. In offline environments, download those files too:

```bash
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 config.json tokenizer.json tokenizer_config.json special_tokens_map.json
```

If they are missing and the Hub is unreachable, start-up fails with an error naming the repo and this command.

The four stages share one GPU. Measured on one H200 (143.8 GB) with the offline example, the run peaks at about 129 GB: the talker engine reserves `mem_fraction_static=0.35` (about 47 GB, most of it KV pool) and the thinker takes what remains (about 73 GB, 17.7 GB of it bf16 weights). Lower `--talker.engine.mem_fraction_static` if the GPU is smaller or shared; the perception encoder and codec need only a few GB each.

## Running the offline example

```bash
python examples/run_nemotron_voicechat.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --audio /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \
  --out reply.wav
```

The example builds `NemotronVoiceChatPipelineConfig`, starts the four stages with `MultiProcessPipelineRunner`, sends one request and writes the reply as 16-bit PCM at 22.05 kHz. Pick the GPU with `CUDA_VISIBLE_DEVICES`. Start-up takes about a minute once the checkpoint is in the page cache; the 41 s `turn_taking.wav` sample renders in roughly real time.

Two input conventions matter:

- The recording is resampled to 16 kHz and **channel 0 is used** (a two-party recording carries the agent on channel 1); it is zero-padded to a multiple of 1280 samples (80 ms).
- Leave trailing silence after the caller's last words. The model answers only while it hears silence, so a clip that ends immediately after the question yields a short or empty reply.

The reply text is the thinker's spoken tokens only; frames where the model is listening carry a marker token and are dropped from the text.

## Request parameters

| Parameter | Effect |
|---|---|
| `temperature`, `top_p`, `top_k` | **Ignored.** The thinker always decodes greedily so that the tokens it streams to the talker are the tokens it committed to; a non-zero `temperature` logs a warning. |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length (one token per 80 ms). |

Audio randomness is governed by the checkpoint's own settings, read from `config.json` (`inference_noise_scale`, `inference_top_p_or_k`): the talker samples its codes with Gumbel-max component selection and Gaussian noise inside the MoG head, so two runs on the same input produce different code sequences and slightly different audio even though the text is identical. There is no `seed` control yet.

## Known limitations

- Offline, single request at a time (`max_running_requests=1` for both engines). No barge-in or interrupt API.
- Single speaker (`Aria`, the checkpoint's baked-in prompt latents).
- The `function_head` tool-call channel is decoded and carried through but nothing acts on it.
- Classifier-free guidance is off (`guidance_scale=0`) although the checkpoint config enables it at 0.2.
- Talker output is not bit-identical to NeMo's offline script even with deterministic sampling on both sides (about 82 % of quantizer cells and 63 of 150 frames agree on a 12 s fixture; transcripts and amplitude envelopes match). Known causes: the backbone and KV cache run in bfloat16 because SGLang's attention backends and `sgl_kernel`'s Gemma RMSNorm have no fp32 path, while NeMo runs the whole talker in fp32; NeMo's offline script enables classifier-free guidance by default; and NeMo's offline recipe walks the system-prompt region before frame 0 (realtime convention here; see `prompt_region_steps`). The thinker is frame-exact with NeMo.
- NeMo decodes the perception encoder's flush row as a 151st frame; this pipeline computes it but does not decode it, so replies are one frame (80 ms) shorter.

## Tests

`tests/unit_test/nemotron_voicechat/` runs on CPU without weights: request frame counts, streaming codec equivalence with whole-utterance decoding, and checkpoint-shim isolation.
