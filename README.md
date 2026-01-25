# SGLang-Omni

Multi-stage pipeline framework for omni models.

## Quick Start

```bash
uv venv .venv -p 3.11
source .venv/bin/activate
uv pip install -v -e .
```

## Demos

### Two-stage and three-stage pipelines

```bash
python examples/run_two_stage_demo.py
python examples/run_two_stage_demo.py --relay shm

python examples/run_three_stage_demo.py
python examples/run_three_stage_demo.py --relay shm
python examples/run_three_stage_demo.py --relay nixl --gpu-ids 0,1,2
```

### Llama 3 8B pipeline

Requires HF access to the model (e.g., `huggingface-cli login`).

```bash
python examples/run_two_stage_llama_demo.py --prompt "Hello, how are you?"
```

## OpenAI-Compatible Server

The OpenAI adapter uses FastAPI + Uvicorn (installed via `uv pip install -e .`).

```bash
python examples/run_openai_llama_server.py --model-id meta-llama/Meta-Llama-3-8B
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
```

## Config-Driven Pipeline

Minimal example using `PipelineConfig` + `compile_pipeline` + `PipelineRunner`:

```python
import asyncio

from sglang_omni.config import ExecutorConfig, PipelineConfig, PipelineRunner, StageConfig
from sglang_omni.config import compile_pipeline

config = PipelineConfig(
    name="demo_pipeline",
    entry_stage="preprocess",
    stages=[
        StageConfig(
            name="preprocess",
            executor=ExecutorConfig(
                factory="my_project.executors.create_preprocess_executor",
                args={},
            ),
            get_next="my_project.routing.preprocess_next",
        ),
        StageConfig(
            name="thinker",
            executor=ExecutorConfig(
                factory="my_project.executors.create_thinker_executor",
                args={"model_path": "Qwen/Qwen3-Omni"},
            ),
            get_next="my_project.routing.end",
        ),
    ],
)

coordinator, stages = compile_pipeline(config)
runner = PipelineRunner(coordinator, stages)
asyncio.run(runner.run())
```
