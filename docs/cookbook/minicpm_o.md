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

The Code2Wav stage enables separate Flow and packed DiT CUDA graph tables
by default. The batch-1 Flow table spans
mel-frame lengths
`128..1024`, every 16 frames from 272 through 720 and at most 32 frames
apart elsewhere:

```text
--code2wav.factory.enable_flow_cuda_graph false
```

Two independent CUDA graph tables are configurable at startup:

- `code2wav.factory.flow_cuda_graph_capture_shapes` is keyed by
  `(batch, mel_frames)`. Batch-1 keys capture complete Euler steps;
  variable-length batches larger than one capture the Euler epilogue
  after DiT instead. The default has 44 batch-1 buckets and 97 B2-B8
  mel buckets derived from the timed SeedTTS English fits. Frames must
  be divisible by 16.
- `code2wav.factory.packed_dit_cuda_graph_capture_shapes` captures only
  the fixed-capacity packed DiT blocks, keyed by `(batch, capacity)`.
  The mel-width-dependent projection, packing, and unpacking remain
  outside this graph. The default has 46 B2-B8 capacities derived from
  the timed SeedTTS English fits. A fit requires
  capacity greater than the CFG-doubled valid frames, with no more than
  one mel-frame width of dummy tokens. The fixed attention maximum is
  1024 frames; longer requests use packed eager DiT.

Both tables are enabled by default when Flow CUDA graphs are enabled;
`None` selects their built-in tables, while `[]` disables an individual
table. Failed captures and requests outside the captured buckets fall
back to eager execution. Captures may be skipped when less than 3 GiB
of GPU memory remains.

The packed-capacity presets
`SEEDTTS_EN_PACKED_DIT_CUDA_GRAPH_SHAPES` (12 keys) and
`SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES` (46 default keys) live in
`sglang_omni.models.minicpm_o.components.token2wav.flow`. The earlier
H100 experiment captured *different*, whole-Euler-step three-dimensional
graphs; its performance and quality results do not validate these new
packed DiT graphs. Offline application of the two default tables to that
experiment's 294 timed packed fits finds a mel bucket and a capacity for
all 294, with 3.56% mean dummy/valid token ratio. This is *shape-fit*
coverage, not measured capture, replay, performance, or output quality
for the new implementation.
