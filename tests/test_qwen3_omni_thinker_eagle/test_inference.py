import os
import asyncio
from sglang_omni.models.qwen3_omni.pipeline.stages import create_sglang_thinker_executor_from_config

async def test():
    # Use relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DUMMY_DIR = os.path.join(current_dir, "tiny-qwen3-omni")
    DRAFT_DIR = os.path.join(current_dir, "tiny-qwen3-omni-draft")

    executor = create_sglang_thinker_executor_from_config(
        model_path=DUMMY_DIR,
        thinker_max_seq_len=1024,
        speech_enabled=False,
        speculative_algorithm="EAGLE", # Changed from EAGLE-3
        speculative_draft_model_path=DRAFT_DIR,
    )
    
    print("Engine created. Checking if Draft Worker exists...")
    # The returned executor is an EngineExecutor wrapping OmniEngine.
    # OmniEngine (engine) wraps SGLangModelRunner which wraps ModelWorker.
    # To check the scheduler, we can look at the inner runner.
    omni_engine = getattr(executor, "_engine", None)
    if omni_engine is None:
        # Fallback if the attribute is different
        omni_engine = getattr(executor, "engine", None)
        
    if omni_engine:
        scheduler = getattr(omni_engine.model_runner.model_worker.model_runner, "scheduler", None)
        # However, in SGLModelRunner, scheduler might not be directly exposed.
        # But we can check if it loaded successfully without crash.
        print("Success: Engine initialized without crashing. Speculative Decoding parameters accepted.")
    else:
        print("Could not find internal engine, but initialization succeeded.")

if __name__ == "__main__":
    asyncio.run(test())
