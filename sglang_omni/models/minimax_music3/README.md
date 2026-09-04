# MiniMax Music 3

Text-to-music: a Qwen3 backbone with an eight-codebook RVQ frame, followed by a
flow-matching DIT and a DAC decoder. Output is 32 kHz stereo.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000

# Dual GPU
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

Install from a checkout with `uv pip uninstall -y flashinfer-cubin && uv pip install -e .`.
MiniMax Music 3's DIT imports `sglang.multimodal_gen`; those packages are pinned
in `pyproject.toml`. A leftover `flashinfer-cubin` wheel cannot match
`flashinfer-python==0.6.17` (no cubin 0.6.17 exists on PyPI) and will fail the
import. See [the cookbook](../../../docs/cookbook/minimax_music3.md)
for the full request contract.

One visible device colocates both stages; two or more put DIT/DAV on the
second. Only the placement differs — both layouts run the acoustic stage
in FP32. Defaults that are on without further flags: backbone decode
CUDA graph, RVQ depth CUDA graph, compiled DIT blocks, compiled DAV decoder,
and batched seeded sampling.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nHello from SGLang Omni",
    "instructions": "A bright piano pop song with a warm female vocal at 100 BPM",
    "seed": 7,
    "max_new_tokens": 250
  }' \
  --output song.wav
```

`input` carries the lyrics and `instructions` the caption. Structure tags must
sit on their own line: normalization drops whatever follows a tag on the same
line. `max_new_tokens` caps the audio frames at 25 per second, up to 9,000, and
the model may end the song before reaching it.

See [the cookbook](../../../docs/cookbook/minimax_music3.md) for the full
request contract, the parameters this model rejects, and worked examples.

## Development

Two gates cover this model, and both compare against commit `c03ddd56`:

```bash
MINIMAX_MUSIC3_GPUS=<gpu> python test_minimax_music3.py              # AR latents and throughput
CUDA_VISIBLE_DEVICES=<gpu> python test_minimax_music3_acoustic.py  # DIT/DAV output
```

Read the acoustic gate's module docstring before changing its limits. It
measures long-term average spectra rather than waveform distance, because the
DIT solver amplifies any perturbation to a fixed divergence and sample-level
distance cannot separate a harmless kernel change from a harmful one.
