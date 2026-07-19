# Examples

Run these commands from the repository root after installing `sglang-omni`.

## Unified Launcher

`run_omni.py` keeps model and topology choices in reusable presets. Use
`python examples/run_omni.py --help` to list them, then add `--help` after a
preset to inspect its options.

| Preset | Workload |
| --- | --- |
| `qwen3-text-server` | Qwen3-Omni OpenAI server with text output |
| `qwen3-speech-server` | Qwen3-Omni OpenAI server with text and audio output |
| `qwen3-speech` | One offline Qwen3-Omni speech request |
| `ming-text-server` | Ming-Omni OpenAI server with text output |
| `ming-speech-server` | Ming-Omni OpenAI server with text and audio output |
| `ming-speech` | One offline Ming-Omni speech request |
| `ming-text` | One offline Ming-Omni text request |

The older `run_qwen3_omni_*.py` and `run_ming_omni_*.py` paths remain as
compatibility wrappers around these presets.

New model examples should add a model-local module under `examples/launchers/`
that exports its preset map, then register that map in `_omni_launcher.py`.
Keep model defaults, stage mutations, and request schemas in the model-local
module; `_omni_launcher.py` owns only registry and CLI dispatch.

## Qwen3-Omni Server

Text output:

```bash
python examples/run_omni.py qwen3-text-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000 \
  --model-name qwen3-omni
```

Text and audio output:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --port 8000 \
  --model-name qwen3-omni
```

Qwen3-Omni FP8, one-GPU colocated H100/H20:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8000
```

## Ming-Omni Server

Text output:

```bash
python examples/run_omni.py ming-text-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --port 8000 \
  --model-name ming-omni
```

Text and audio output:

```bash
python examples/run_omni.py ming-speech-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --port 8000 \
  --model-name ming-omni
```

Use a different `--port` if you run more than one server at the same time.
