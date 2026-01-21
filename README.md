# SGLang-Omni

## Get Started

```bash
# create a virtual environment
uv venv .venv -p 3.11
source .venv/bin/activate

# install the package
uv pip install -v -e .
```

### Run Demo

You can execute the following commands to run the demos.

```bash
# Use NixlRelay (default)
python examples/run_two_stage_demo.py

# Use NixlRelay (default)
python examples/run_three_stage_demo.py

# Four-stage Llama 3 8B pipeline (template -> tokenize -> engine -> decode)
# Requires HF access to the model (e.g., `huggingface-cli login`)
python examples/run_two_stage_llama_demo.py --prompt "Hello, how are you?"
```

### OpenAI-Compatible Server

The OpenAI adapter uses FastAPI + Uvicorn (installed via `uv pip install -e .`).

```bash
python examples/run_openai_llama_server.py --model-id meta-llama/Meta-Llama-3-8B
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
```
