# Examples

We provide some examples to help you get started with SGLang-Omni. This documentation and examples will be updated when SGLang-integration and API server are ready.

## Two-stage and three-stage pipelines

```bash
python examples/run_two_stage_demo.py
python examples/run_two_stage_demo.py --relay shm

python examples/run_three_stage_demo.py
python examples/run_three_stage_demo.py --relay shm
python examples/run_three_stage_demo.py --relay nixl --gpu-ids 0,1,2
```

## Llama 3 8B pipeline

Requires HF access to the model (e.g., `huggingface-cli login`).

```bash
python examples/run_two_stage_llama_demo.py --prompt "Hello, how are you?"
```

## OpenAI-Compatible Server

The OpenAI adapter uses FastAPI + Uvicorn (installed via `uv pip install -e .`).

```bash
python examples/run_openai_llama_server.py --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
```
