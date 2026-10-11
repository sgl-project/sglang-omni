# Experimental VoXtream2 pipeline

This pipeline serves complete English text with one operator-configured reference
voice. It uses the author's VoXtream2 generator, connects audio to Omni's Stage
and Coordinator messages, and supports streaming PCM and non-streaming audio.
Requests execute serially. Live text ingress and live speaking-rate updates are
not implemented.

## Runtime and assets

The tested author source is `herimor/voxtream` commit
`8ec2d62159dae4716ae7058827244a962d40603c` (package version 0.2.4).
The loader uses its internal construction contract; other versions are untested.
Keep the included `LICENSE-MIT.voxtream` with the adapted construction and cache
code. The generator and model method belong to the upstream authors.

Required local assets:

- The `herimor/voxtream2` checkpoint directory, including `model.safetensors`,
  `config.json`, and `phoneme_to_token.json`.
- Mimi's `tokenizer-e351c8d8-checkpoint125.safetensors` from
  `kyutai/moshiko-pytorch-bf16`.
- The ReDimNet `M-vb2+vox2+cnc-ft_mix.pt` checkpoint and importable ReDimNet source.
- The author's generator and speaking-rate JSON configuration files.
- A local reference WAV selected by the operator.

The worker requires importable VoXtream, ReDimNet, torchtune, torchao, moshi and
their runtime dependencies, plus eSpeak NG. Model imports are deferred until
worker startup. The loader does not download checkpoints or replace hub download
functions. Reference enhancement, VAD, and pickle-based prompt caching are disabled.

## Separate model environment

The author pins conflict with the project's PyTorch and Transformers versions.
Set `model_python` to run the model in a separate Python 3.11 environment. The
Stage stays in the project's environment; no model dependencies are installed
into it. The worker imports this checkout through an explicit source path and
returns float32 audio over local pipes. Client requests cannot choose the
interpreter, source paths, reference audio, or model checkpoints.

Prepare the pinned source checkouts and a model environment on Linux:

```bash
git clone https://github.com/herimor/voxtream.git /path/to/voxtream
git -C /path/to/voxtream checkout 8ec2d62159dae4716ae7058827244a962d40603c
git clone https://github.com/IDRnD/ReDimNet.git /path/to/ReDimNet
git -C /path/to/ReDimNet checkout ce039a624cb99fe127702ceb94c6080090e5032f
uv venv --python 3.11 /path/to/voxtream-env
uv pip install --python /path/to/voxtream-env/bin/python -r sglang_omni/models/voxtream2/requirements-model.in /path/to/voxtream
uv pip check --python /path/to/voxtream-env/bin/python
```

Install eSpeak NG and ensure its executable and shared libraries are visible to
the worker. ReDimNet is supplied as a source directory via `model_source_paths`.
The requirements file pins the complete tested author environment, including
the author's UI dependencies. It also includes `msgspec` for Omni request types
and PyArrow 20 for the pinned datasets package. Version resolution alone does
not establish runtime compatibility; validate imports and real generation too.

The worker environment uses PyTorch/torchaudio 2.8.0, Transformers 4.50.0,
torchtune 0.4.0, torchao 0.9.0, and moshi 0.2.13. Omitting `model_python` starts a child process with the current interpreter and
therefore requires a dependency-compatible custom environment. Model execution
always stays in the child process. Installing the author requirements into the
normal project environment is not supported.

## Configure and launch

Copy `examples/configs/voxtream2.yaml`, replace every asset path, and review the
author generator JSON, particularly `max_audio_length_ms`. The tested value was
20000; longer requests can reach the author's configured generation limit.

```bash
python -m sglang_omni.cli serve --config ./voxtream2.yaml --host 127.0.0.1 --port 8000 --model-name voxtream2
```

Use `voice: "reference"` or `voice: "default"`. This deployment does not accept
uploaded reference voices or arbitrary client filesystem paths.

```json
{
  "model": "voxtream2",
  "input": "The train will arrive at the next station in five minutes.",
  "voice": "reference",
  "language": "English",
  "seed": 101,
  "response_format": "pcm",
  "stream": true,
  "stage_params": {"tts_engine": {"speaking_rate_sps": 3.0}}
}
```

POST this body to `/v1/audio/speech`. For non-streaming WAV, set `stream` to
`false` and `response_format` to `wav`. The SPS parameter sets the author's
conditioning rate for the whole request. The existing HTTP `speed` option remains
post-processing; it is not a live generation control. Explicit per-request
sampling controls other than seed and style instructions are rejected.

`max_pending` counts the active request plus queued requests. Audio/result
buffering is limited by `output_buffer_chunks`; small rejection errors may bypass
that buffer so the Stage receive loop cannot deadlock. Queue overflow reports HTTP 503. Invalid text, rate, seed, language,
style instructions, and unsupported sampling controls report HTTP 400; model
execution and cleanup failures remain server errors. Disconnects cancel
at a generation boundary; running GPU kernels and prompt preparation are not
preempted.

`worker_startup_timeout`, `worker_response_timeout`, and `worker_cancel_timeout`
bound initialization, time between worker responses, and cancellation cleanup.
The output reader buffers one frame in addition to the bounded OS pipe and
scheduler buffer. Messages are limited to 1 MiB. Cancellation acknowledges
generator cleanup before the worker accepts another request. Protocol failure,
worker death, or a timeout closes the owned process and refuses further requests
on that backend; restart the stage to recover. There is no automatic restart or
concurrent model execution. Ordinary invalid requests do not kill the worker.

## Validation scope

Historical tests of the September 18 implementation covered streamed PCM, non-streaming WAV, seeded parity,
disconnect recovery, a six-client burst with capacity four, and service health
after overload. Both streaming and non-streaming HTTP requests were checked
for unsupported sampling controls, style instructions, language, and invalid SPS.
All eight invalid requests returned HTTP 400; two burst rejections returned 503.
The four admitted burst requests ran serially. This is lifecycle
validation, not a throughput optimization or a broad quality benchmark.

The October 11 refresh against current main has not yet repeated these HTTP
or real-model checks. Historical observations do not establish compatibility
with the refreshed implementation.

Code, tests, and this draft were prepared with Codex assistance. The refreshed
code requires human review and listening evaluation before PR submission.
