# SenseNova U1

[SenseNova U1](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Interleaved)
is a unified model for image understanding, image generation, and interleaved
text-image generation. The default SGLang-Omni pipeline runs the native
language/MoT tower, native-resolution vision encoder, flow-matching image head,
and interleave state machine in one process.

## Launch

```bash
sgl-omni serve \
  --model-path sensenova/SenseNova-U1-8B-MoT-Interleaved \
  --port 8000
```

The `default` and `interleave` variants return text plus inline PNG images. The
`flow` variant runs T2I or IT2I only.

## Checkpoint Trust

The native model weights are loaded by SGLang, but the tokenizer is created
with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`. Only serve a
checkpoint repository that you trust and have configured intentionally. For a
production deployment, pin the checkpoint revision or use a reviewed local
snapshot rather than an unpinned repository.

## Request Limits

SenseNova U1 validates generation controls before allocating request GPU
tensors. Invalid values return a request error.

| Control | Deployment limit |
|---|---:|
| `width`, `height` | Positive, divisible by 32, each at most 2048 |
| Image pixels | At most 1,048,576 |
| `num_steps` | 1 to 64 |
| Input images | At most 4 |
| `max_images` | 1 to 4 |
| `max_new_tokens` | 1 to 2048 |
| Total prefix, text, and generated-image tokens | At most 4096 |
| Native flow `batch_size` | 1 |

The exact token budget is checked again after CPU tokenization and image
preprocessing, before vision embeddings or flow noise are allocated on the
GPU.

## Runtime Cache Bounds

The exact eager text path keeps a small LRU for repeated prefixes and decode
CUDA graphs. Production defaults are:

| Setting | Default |
|---|---:|
| `eager_prefix_cache_max_entries` | 4 |
| `eager_decode_graph_cache_max_entries` | 2 |
| `eager_decode_graph_max_captures` | 4 |
| `eager_prefix_cache_max_tokens` | 2048 |
| `eager_decode_graph_max_total_tokens` | 1024 |

Entries are evicted least-recently-used, GPU tensor and graph references are
released on eviction, and all entries are cleared when the stage stops. CUDA
graph capture also has a lifetime budget because CUDA graph-private allocator
pools are not reliably reclaimed while the process remains alive. After four
new graph shapes have been captured, an uncached shape uses the exact eager
path; already-cached hot shapes continue to replay. Requests above the
per-entry token limits also run through the exact eager path.

## Current Scope

- The validated production path is single-node and single-GPU.
- The custom U1 hybrid attention mask is implemented for the Triton backend.
- Flow steps execute in-process; cooperative scheduling between a flow step
  and unrelated autoregressive requests remains future work.
