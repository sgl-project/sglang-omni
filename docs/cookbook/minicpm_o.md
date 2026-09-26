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

The vocoder keeps an LRU cache of up to 32 speaker references by default, so switching back
to a cached reference reuses its conditioning. Inline references are keyed by
audio content; file references account for file metadata. Flow inference batches
different references and token lengths together. HiFT groups rows by generated
length to preserve waveform boundaries. Invalid references fail instead of
silently using the default. Audio output remains non-streaming.

Reference preparation uses up to 8 worker threads, preparing each unique
reference once per batch. Set `MINICPMO_REF_WORKERS=1` to prepare references
serially, or set `MINICPMO_PROMPT_CACHE_CAPACITY` to change the cache capacity.
Both settings must be positive integers and are read when Code2Wav is created.
The stage drains reference preparation on shutdown; a closed vocoder rejects
new preparation calls.

## Flow execution options

The Code2Wav stage enables Flow and packed DiT CUDA graph tables by default.
To disable both:

```text
--code2wav.factory.enable_flow_cuda_graph false
```

Two independent CUDA graph tables are configurable at startup:

- `code2wav.factory.flow_cuda_graph_capture_shapes` is keyed by
  `(batch, mel_frames)`. Non-packed keys capture complete Euler steps.
  With variable-length Flow enabled, B2-B8 keys capture a pair of graphs:
  timestep conditioning and dense projection before DiT, then unpacking,
  classifier-free guidance and the Euler state update after DiT.
  The default has 44 batch-1 buckets and 97 B2-B8 mel buckets derived
  from the timed SeedTTS English fits. Frames must be divisible by 16.
- `code2wav.factory.packed_dit_cuda_graph_capture_shapes` captures
  fixed-capacity packing and packed DiT blocks, keyed by `(batch, capacity)`.
  The default has 46 B2-B8 capacities derived from
  the timed SeedTTS English fits. A fit requires
  capacity greater than the CFG-doubled valid frames, with no more than
  one mel-frame width of dummy tokens. The fixed attention maximum is
  1024 frames; longer requests use packed eager DiT. Custom capacities must
  not exceed `(2 * batch + 1) * 1024`, including the dummy sequence.

The two tables compose within each packed Euler step:

```text
Flow pre[B, mel_frames] -> packed DiT[B, capacity] -> Flow post[B, mel_frames]
```

Each batch size has fixed-address boundary buffers shared across mel and
capacity keys. Only active dense rows and the selected packed capacity are
written; the full workspace is not padded or cleared on every step.
Graph preparation and layout installation happen once per Flow solve.
The solver holds both runners' locks in a fixed order, keeps state in
graph-owned buffers and copies out the final cropped result once.
Packed BF16 predictions promote FP16 solver state to FP32, matching eager
Euler updates rather than narrowing the state back to FP16 on every replay.
Timestep scalar updates remain solver-side; this is not a whole-ten-step graph.

Packed graphs capture first to establish the shared workspaces. Each runner
captures its keys largest-first and shares a pool within that runner; the two
runners use separate pools so intermediate graph memory cannot overlap.

Both tables are enabled by default when Flow CUDA graphs are enabled;
`None` selects their built-in tables, while `[]` disables an individual
table. A packed-capacity hit without a Flow key uses the packed graph with
eager pre/post work. Without a packed-capacity hit, variable-length Flow
uses eager execution. Captures may be skipped when less than 3 GiB of GPU
memory remains or CUDA graph capture is unsupported. Packed-graph computation
failures during warmup abort startup rather than being reported as eager
fallback.

The packed-capacity table
`SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES` (46 default keys) lives in
`sglang_omni.models.minicpm_o.components.token2wav.flow_graph_shapes`.
Replay coverage for this path should report composed Flow pre/post steps
separately from packed-only steps, rather than treating every packed replay
as coverage of the full Flow computation.
