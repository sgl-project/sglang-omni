# MiniCPM-o Reference Audio

On the speech pipeline, pass an explicit speaker reference in
`audio.ref_audio` on `/v1/chat/completions`:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="unused")
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
```

`stage_params.code2wav.ref_audio` is an alternative, with higher priority than
`audio.ref_audio`. Both accept `prompt_wav` as an alias. The Python pipeline
client can also supply `ref_audio` through `extra_params`. References must be
base64 audio data URIs, inline `{data, media_type}` descriptors, or encoded audio
bytes for the Python client. Paths and HTTP URLs are not fetched by this stage;
read or download the file on the client before sending it.

The reference conditions Token2wav's speaker embedding, prompt tokens, and mel
features. Audio supplied in chat messages remains understanding input and is not
automatically used as the speaker reference. Without an explicit reference,
Token2wav uses the checkpoint's `assets/HT_ref_audio.wav` when available.

The vocoder caches only the most recently used reference by audio content. A
different reference, including switching back to the default, rebuilds the
conditioning. Invalid references fail instead of silently using the default.
Audio output remains non-streaming.

## Whole-Solver CUDA Graphs

The Code2Wav factory accepts `flow_cuda_graph_capture_shapes` and
`flow_cuda_graph_frame_bucket`. An empty shape list (the default) uses eager
execution. Set both options to capture the complete ten-step device Euler solve
at worker startup. For example, these experimental settings cover batches 1, 2,
4, and 8 at a single 256-frame bucket:

```yaml
stages:
  code2wav:
    factory:
      flow_cuda_graph_frame_bucket: 16
      flow_cuda_graph_capture_shapes: [[1, 256], [2, 256], [4, 256], [8, 256]]
```

Frame counts include the reference mel. Selection rounds the batch's maximum
frame count up to the configured quantum and requires an exact batch size and
bucket match. Choose the quantum (for example 16, 32, or 64) and table from a
representative length census; the example is not a deployment recommendation.
Unlisted keys, incompatible inputs, and failed startup capture use eager
execution. Capture publishes all keys together and never runs on requests.

The encoder, prompt preparation, CPU length handling, mel slicing, and HiFT
remain outside the graph. Graphs share a memory pool and run serially through
the Code2Wav scheduler on one CUDA stream. Concurrent use across streams is
unsupported. Replay returns an owned output so subsequent replays
cannot overwrite a previous result.

Capture preserves the native mixed precision inputs: conditioning and the
timestep schedule stay FP32, while noise and speaker embeddings use the model
dtype. Timestep frequencies are cached by input device and dtype during warmup.

Run the isolated real-weight comparison from the repository root:

```bash
PYTHONPATH=. python benchmarks/benchmark_minicpm_flow_cuda_graph.py \
  --assets /path/to/model/assets/token2wav \
  --output /path/to/results --dtype float32 --frame-bucket 16
```

The benchmark covers B1 short/long and B4/B8 mixed lengths, comparing the same
noise and conditioning with shape, finite-value, and numerical checks. It
writes unprofiled median solver wall time, separately profiled kernel counts,
kernel time, CUDA launch API time, graph launch counts, GPU active fraction,
and gaps between kernels, plus Chrome traces. GPU active fraction is summed
kernel time divided by profiled solver wall time. Mixed lengths and conditioning
are synthetic; this does not replace a serving length census or end-to-end
audio evaluation. Profiling changes launch overhead, so compare profiled
metrics together and use unprofiled wall time for speedup. If representative
solver speedup is only 1-2%, stop expanding the table.
