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

The Code2Wav stage compiles the DiT blocks and captures one Flow Euler step
by default. The graph table covers batch size 1 and mel-frame lengths
`128..1024`, every 16 frames from 272 through 720 and at most 32 frames
apart elsewhere:

```text
--code2wav.factory.enable_dit_torch_compile false
--code2wav.factory.enable_flow_cuda_graph false
```

Compile runs before graph capture when both options are enabled. Graph shapes
are `(batch size, mel frames)` and must have a frame count divisible by 16.
They are captured at startup; each Euler step replays the smallest resident
graph with the same batch size that covers the request's 16-frame rounded
length. Other shapes run the eager solver without request-time capture. A
capture that fails or would leave less than 3 GiB free is skipped while the
remaining buckets stay usable.
Resident graphs consume GPU memory even when requests do not hit them.
Override `flow_cuda_graph_capture_shapes` when a measured serving workload has
different resident shapes. With variable-length DiT enabled, only batch-1
shapes are captured by default; larger batches use packed DiT eagerly. An
experimental packed graph shape can be specified as `(batch, mel frames,
packed capacity)`. The capacity must be 16-aligned and leave one or more
dummy frames after the CFG-doubled valid frames, with no more than one mel
frame width of dummy tokens. Requests outside the captured capacity run packed
DiT eagerly.
