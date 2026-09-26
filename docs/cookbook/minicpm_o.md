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

## Packed DiT compilation

Variable-length Code2Wav compiles the packed DiT blocks by default without
enabling CUDA graphs. To disable compilation:

```text
--code2wav.factory.enable_packed_dit_torch_compile false
```

This compiles `DiTBlock.forward_packed`, not the dense `DiTBlock.forward`.
The packed layout stays dynamic; the attention and convolution operators
remain part of the block, and an unsupported graph break fails rather than
silently falling back. Inductor's own CUDA graphs are disabled so the
compile-only configuration remains distinct from Flow CUDA graphs.
After loading the Flow and HiFT weights, two nonuniform packed batches
materialize the compiled path before Code2Wav reports readiness. Compilation
or warmup failures abort startup; other serving shapes may still specialize.
