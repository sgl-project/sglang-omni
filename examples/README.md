# Examples

Run these commands from the repository root after installing `sglang-omni`.

## Qwen3-Omni Server

Text output:

```bash
python examples/run_qwen3_omni_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000 \
  --model-name qwen3-omni
```

Text and audio output:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --port 8000 \
  --model-name qwen3-omni
```

## Ming-Omni Server

Text output:

```bash
python examples/run_ming_omni_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --port 8000 \
  --model-name ming-omni
```

Text and audio output:

```bash
python examples/run_ming_omni_speech_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --port 8000 \
  --model-name ming-omni
```

Use a different `--port` if you run more than one server at the same time.
