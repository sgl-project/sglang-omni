# EasyMagpie TTS deployment

One command builds the server image and starts EasyMagpie TTS on a GPU:

```bash
./serve.sh /path/to/converted_model
```

The script:

1. checks that the directory is a converted model (`config.json`,
   `model.safetensors`, `codec_native/`, `speaker_embeddings/`);
2. resolves the SGLang-Omni branch to a commit and builds
   `easymagpie-tts:<commit>` if that image does not exist yet;
3. starts the container `easymagpie-tts` with the model mounted read-only at
   `/model` and port 8000 published;
4. waits for `/health` and prints a sample request.

The server runs the tuned defaults from `examples/configs/easymagpie_tts.yaml`
(CUDA graphs, compiled talker, streaming codec schedule, garbage collector
frozen after startup). No flags are needed for the measured performance.

## Requirements

- Linux with an NVIDIA GPU and driver that support CUDA 13.0.
- Docker with the NVIDIA Container Toolkit.
- About 45 GB of disk: 38 GB for the image and 5.4 GB for the compile
  cache, plus the model directory.
- Network access to Docker Hub, GitHub and PyPI for the build.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--port N` | `8000` | Host port |
| `--gpu ID` | `0` | GPU index |
| `--name NAME` | `easymagpie-tts` | Container name; a container of that name started by this script is replaced |
| `--ref REF` | `feat/easymagpie-tts` | SGLang-Omni branch, tag or commit |
| `--rebuild` | off | Rebuild the image even if it exists |
| `-- ARGS...` | none | Extra `sglang_omni.cli serve` arguments |

Set `OMNI_REPO` to build from another SGLang-Omni fork.

## Startup time

The first start compiles the talker and tunes its kernels for every decode
batch size, then captures CUDA graphs. Compiled artifacts are kept in the
Docker volume `easymagpie-tts-cache`, so later starts of the same image skip
the compilation. `docker volume rm easymagpie-tts-cache` clears it.

On an RTX 4080 the first start took 1244 s and a restart with the cache took
282 s.

## Measured performance

RTX 4080, NemotronTTS GRPO Step120 checkpoint, voice `Aria`, streaming PCM,
the same sentence per request (about 8.5 s of audio). Throughput is seconds of
audio per second of wall time, summed over requests.

| Concurrency | First audio P50 | Chunk P50 | Throughput |
|---:|---:|---:|---:|
| 1 | 29.1 to 33.3 ms (3 runs) | 55.0 ms | 11.2 |
| 8 | 50.3 ms | 66.5 ms | 69.6 |
| 16 | 57.8 ms | 74.1 ms | 123.1 |
| 32 | 83.5 ms | 91.7 ms | 196.3 |

## What is in the image

- `lmsysorg/sglang:v0.5.20`, pinned by digest, the same base upstream
  SGLang-Omni uses.
- SGLang-Omni from the chosen ref, with its `pyproject.toml` dependencies.
- `sglang-0.5.20-easymagpie.patch`, applied to the installed SGLang. It is
  the nine commits of the SGLang branch `easymagpie/core-fixes`
  (`5f40f7501d` through `cb28a76895`) backported to the 0.5.20 release:
  Nemotron-H MLP and MoE fixes (relu2 and non-gated experts, the shared-expert
  input, `routed_scaling_factor`), in-place Mamba2 decode state under
  `torch.compile`, Mamba2 prefill inside the prefill CUDA graph, the
  prefill-graph gate for attention-free layers, and PDL constexprs on GPUs
  without PDL. The DeepSeek-V4 file the branch also touches does not exist in
  0.5.20. The build fails if the installed SGLang is not 0.5.20.

Regenerate the patch after changing the SGLang branch by applying its commits
to a v0.5.20 tree and diffing it against the release, with paths relative to
the `sglang` package directory.
