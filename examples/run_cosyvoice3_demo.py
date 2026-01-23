# SPDX-License-Identifier: Apache-2.0
"""
CosyVoice3 Frontend Stage Demo
Stage 1: Frontend processing (Text Normalization -> Tokenization -> Feature Extraction)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Any
import torch

# Add CosyVoice to path
sys.path.append("../CosyVoice")
sys.path.append("../CosyVoice/third_party/Matcha-TTS")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sglang_omni import Coordinator, Stage, Worker
from sglang_omni.executors import FrontendExecutor
from sglang_omni.proto import StagePayload, OmniRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
ENABLE_SAVE_INTERMEDIATE = False

# helpers
def _to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    return x

def save_stage_data(stage: str, data: Any) -> str:
    if not ENABLE_SAVE_INTERMEDIATE:
        return ""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trae_output", "cosyvoice_debug")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stage}_{int(time.time() * 1000)}.pt")
    torch.save(_to_cpu(data), path)
    return path

# Endpoints configuration
STAGE0_ENDPOINT = "tcp://127.0.0.1:18001"
STAGE1_ENDPOINT = "tcp://127.0.0.1:18002"
STAGE2_ENDPOINT = "tcp://127.0.0.1:18003"
STAGE3_ENDPOINT = "tcp://127.0.0.1:18004"
STAGE4_ENDPOINT = "tcp://127.0.0.1:18005"
COORDINATOR_ENDPOINT = "tcp://127.0.0.1:18000"
ABORT_ENDPOINT = "tcp://127.0.0.1:18099"

ENDPOINTS = {
    "frontend": STAGE0_ENDPOINT,
    "llm": STAGE1_ENDPOINT,
    "flow": STAGE2_ENDPOINT,
    "vocoder": STAGE3_ENDPOINT,
    "postprocess": STAGE4_ENDPOINT,
}

def frontend_get_next(request_id: str, output: Any) -> str | None:
    return "llm"

def run_frontend_stage(model_dir: str) -> None:
    import torch
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    # Initialize Frontend
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'
    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')

    with open(hyper_yaml_path, 'r') as f:
        # Note: qwen_pretrain_path might need adjustment depending on where it is relative to model_dir
        # For this demo we assume it's correctly handled or we might need to mock/adjust overrides
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

    frontend = CosyVoiceFrontEnd(
        configs['get_tokenizer'],
        configs['feat_extractor'],
        f'{model_dir}/campplus.onnx',
        f'{model_dir}/speech_tokenizer_v3.onnx',
        f'{model_dir}/spk2info.pt',
        configs['allowed_special']
    )
    sample_rate = configs['sample_rate']
    logger.info("CosyVoice3 Frontend initialized successfully.")

    def processor(payload: StagePayload) -> StagePayload:
        """
        Input payload.request.inputs should be a list containing a dict with:
        - tts_text: str
        - prompt_text: str
        - prompt_audio: str (path to audio file)
        """
        inputs = payload.request.inputs
        if isinstance(inputs, list):
            inputs = inputs[0] # Take the first item
        
        tts_text = inputs.get("tts_text")
        prompt_text = inputs.get("prompt_text")
        prompt_audio = inputs.get("prompt_audio")

        if not tts_text or not prompt_text or not prompt_audio:
             raise ValueError("Missing required inputs: tts_text, prompt_text, prompt_audio")

        logger.info(f"Processing request: tts_text='{tts_text}', prompt_audio='{prompt_audio}'")

        # Text Normalization
        # Note: text_normalize returns a list of chunks. For this demo, we process all chunks.
        # But for 'model input', usually we expect a list of inputs if there are multiple chunks.
        
        normalized_texts = frontend.text_normalize(tts_text, split=True, text_frontend=True)
        model_inputs = []

        # Assuming Zero-Shot mode for now
        # We need to perform normalization on prompt_text as well, similar to CosyVoice.inference_zero_shot
        prompt_text_norm = frontend.text_normalize(prompt_text, split=False, text_frontend=True)

        for text_chunk in normalized_texts:
            logger.info(f"Generating model input for chunk: {text_chunk}")
            model_input = frontend.frontend_zero_shot(
                text_chunk, 
                prompt_text_norm, 
                prompt_audio, 
                sample_rate, 
                zero_shot_spk_id=''
            )
        
            model_inputs.append(model_input)

        payload.data = {"model_inputs": model_inputs}
        # save_stage_data("frontend", payload.data)
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="frontend")

    stage = Stage(
        name="frontend",
        get_next=frontend_get_next,
        recv_endpoint=STAGE0_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "frontend_worker", "gpu_id": None}, # CPU only for now
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


def llm_get_next(request_id: str, output: Any) -> str | None:
    return "flow"


def run_llm_stage(model_dir: str) -> None:
    import torch
    import threading
    import atexit
    import gc
    import signal
    import sys
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
    from vllm import EngineArgs, LLMEngine

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'
    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')

    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

    llm = configs['llm']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    llm.load_state_dict(torch.load(f'{model_dir}/llm.pt', map_location=device), strict=True)
    llm.to(device).eval()
    logger.info("CosyVoice3 LLM (Torch) initialized successfully.")

    vllm_dir = os.path.join(model_dir, "vllm")
    export_cosyvoice2_vllm(llm, vllm_dir, device)
    engine_args = EngineArgs(
        model=vllm_dir,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        gpu_memory_utilization=0.2,
        kv_cache_dtype="auto",
    )
    llm.vllm = LLMEngine.from_engine_args(engine_args)
    llm.lock = threading.Lock()
    try:
        del llm.llm.model.model.layers
    except Exception:
        pass
    logger.info("CosyVoice3 LLM switched to vLLM backend successfully.")

    def _cleanup_vllm():
        try:
            if hasattr(llm, "vllm") and llm.vllm is not None:
                engine = llm.vllm
                for method in ("shutdown", "close", "stop", "exit"):
                    if hasattr(engine, method):
                        try:
                            getattr(engine, method)()
                        except Exception:
                            pass
                try:
                    llm.vllm = None
                except Exception:
                    pass
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.current_stream().synchronize()
            except Exception:
                pass
            try:
                gc.collect()
            except Exception:
                pass

    def _signal_handler(signum, frame):
        _cleanup_vllm()
        try:
            sys.exit(0)
        except SystemExit:
            raise

    atexit.register(_cleanup_vllm)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass

    def processor(payload: StagePayload) -> StagePayload:
        data = payload.data
        if not isinstance(data, dict) or 'model_inputs' not in data:
            raise ValueError("LLM stage expects 'model_inputs' from frontend")
        model_inputs = data['model_inputs']
        if isinstance(model_inputs, list):
            model_input = model_inputs[0]
        else:
            model_input = model_inputs

        text = model_input.get('text', torch.zeros(1, 0, dtype=torch.int32))
        text_len = model_input.get('text_len', torch.tensor([0], dtype=torch.int32))
        prompt_text = model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32))
        prompt_text_len = model_input.get('prompt_text_len', torch.tensor([0], dtype=torch.int32))
        llm_prompt_speech_token = model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
        llm_prompt_speech_token_len = model_input.get('llm_prompt_speech_token_len', torch.tensor([0], dtype=torch.int32))
        llm_embedding = model_input.get('llm_embedding', torch.zeros(0, 192))

        text = text.to(device)
        text_len = text_len.to(device)
        prompt_text = prompt_text.to(device)
        prompt_text_len = prompt_text_len.to(device)
        llm_prompt_speech_token = llm_prompt_speech_token.to(device)
        llm_prompt_speech_token_len = llm_prompt_speech_token_len.to(device)
        llm_embedding = llm_embedding.to(device)

        tokens = []
        for tid in llm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=llm_prompt_speech_token_len,
            embedding=llm_embedding,
        ):
            tokens.append(int(tid))
        tts_speech_token = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
        logger.info(f"LLM generated {tts_speech_token.shape[1]} tokens.")

        # carry forward fields needed for diffusion
        payload.data = {
            "tts_speech_token": tts_speech_token,
            "flow_prompt_speech_token": model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
            "prompt_speech_feat": model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
            "flow_embedding": model_input.get('flow_embedding', torch.zeros(0, 192)),
        }
        # save_stage_data("llm", payload.data)
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="llm")

    stage = Stage(
        name="llm",
        get_next=llm_get_next,
        recv_endpoint=STAGE1_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "llm_worker", "gpu_id": 0 if torch.cuda.is_available() else None},
    )
    stage.add_worker(worker)

    try:
        asyncio.run(stage.run())
    finally:
        _cleanup_vllm()

def flow_get_next(request_id: str, output: Any) -> str | None:
    return "vocoder"


def run_flow_stage(model_dir: str) -> None:
    import torch
    from hyperpyyaml import load_hyperpyyaml

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'
    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')

    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

    flow = configs['flow']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow.load_state_dict(torch.load(f'{model_dir}/flow.pt', map_location=device), strict=True)
    flow.to(device).eval()
    logger.info("CosyVoice3 Flow initialized successfully.")

    def processor(payload: StagePayload) -> StagePayload:
        data = payload.data
        if not isinstance(data, dict) or 'tts_speech_token' not in data:
            raise ValueError("Flow stage expects 'tts_speech_token' from LLM stage")

        token = data['tts_speech_token'].to(device)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(device)

        prompt_token = data.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)).to(device)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(device)

        prompt_feat = data.get('prompt_speech_feat', torch.zeros(1, 0, 80)).to(device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(device)

        embedding = data.get('flow_embedding', torch.zeros(0, 192)).to(device)

        tts_mel, _ = flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
        logger.info(f"Flow produced mel with shape: {tts_mel.shape}")
        payload.data = {"tts_mel": tts_mel}
        # save_stage_data("flow", payload.data)
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="flow")

    stage = Stage(
        name="flow",
        get_next=flow_get_next,
        recv_endpoint=STAGE2_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "flow_worker", "gpu_id": 0 if torch.cuda.is_available() else None},
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())

def vocoder_get_next(request_id: str, output: Any) -> str | None:
    return "postprocess"


def run_vocoder_stage(model_dir: str) -> None:
    import torch
    from hyperpyyaml import load_hyperpyyaml

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'
    if not os.path.exists(hyper_yaml_path):
        raise ValueError(f'{hyper_yaml_path} not found!')

    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None, 'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

    hift = configs['hift']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(f'{model_dir}/hift.pt', map_location=device)
    state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items()}
    hift.load_state_dict(state_dict, strict=True)
    hift.to(device).eval()
    logger.info("CosyVoice3 Vocoder initialized successfully.")

    def processor(payload: StagePayload) -> StagePayload:
        data = payload.data
        if not isinstance(data, dict) or 'tts_mel' not in data:
            raise ValueError("Vocoder stage expects 'tts_mel' from Flow stage")
        tts_mel = data['tts_mel'].to(device)
        tts_speech, _ = hift.inference(speech_feat=tts_mel, finalize=True)
        logger.info(f"Vocoder produced speech with shape: {tts_speech.shape}")
        payload.data = {"tts_speech": tts_speech}
        # save_stage_data("vocoder", payload.data)
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="vocoder")

    stage = Stage(
        name="vocoder",
        get_next=vocoder_get_next,
        recv_endpoint=STAGE3_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "vocoder_worker", "gpu_id": 0 if torch.cuda.is_available() else None},
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())

def postprocess_get_next(request_id: str, output: Any) -> str | None:
    return None


def run_postprocess_stage(model_dir: str) -> None:
    import time
    import torchaudio
    import torch
    from hyperpyyaml import load_hyperpyyaml

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    hyper_yaml_path = f'{model_dir}/cosyvoice3.yaml'
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None, 'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    sample_rate = configs['sample_rate']

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trae_output", "cosyvoice_audio")
    os.makedirs(out_dir, exist_ok=True)

    def processor(payload: StagePayload) -> StagePayload:
        data = payload.data
        if not isinstance(data, dict) or 'tts_speech' not in data:
            raise ValueError("Postprocess stage expects 'tts_speech' from Vocoder stage")
        audio = data['tts_speech']
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] != 1:
            audio = audio[:1]
        ts = int(time.time() * 1000)
        fname = f"cosyvoice_{ts}.wav"
        path = os.path.join(out_dir, fname)
        torchaudio.save(path, audio, sample_rate)
        logger.info(f"Saved audio: {path}")
        payload.data = {"audio_path": path}
        # save_stage_data("postprocess", payload.data)
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="postprocess")

    stage = Stage(
        name="postprocess",
        get_next=postprocess_get_next,
        recv_endpoint=STAGE4_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "postprocess_worker", "gpu_id": None},
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())

async def run_coordinator(args: argparse.Namespace) -> None:
    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage="frontend",
    )

    coordinator.register_stage("frontend", STAGE0_ENDPOINT)
    coordinator.register_stage("llm", STAGE1_ENDPOINT)
    coordinator.register_stage("flow", STAGE2_ENDPOINT)
    coordinator.register_stage("vocoder", STAGE3_ENDPOINT)
    coordinator.register_stage("postprocess", STAGE4_ENDPOINT)

    await coordinator.start()
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    try:
        await asyncio.sleep(15.0) # Wait for stages to connect

        request = OmniRequest(
            inputs=[{
                "tts_text": args.tts_text,
                "prompt_text": args.prompt_text,
                "prompt_audio": args.prompt_audio
            }],
            params={}
        )
        
        logger.info("Submitting request...")
        result = await coordinator.submit("req-cosyvoice-1", request)
        
        logger.info("Request processed successfully!")
        
        if isinstance(result, dict) and "audio_path" in result:
            logger.info(f"Audio saved at: {result['audio_path']}")
        else:
            logger.info(f"Result: {result}")

        await coordinator.shutdown_stages()
        await asyncio.sleep(0.5)
    finally:
        completion_task.cancel()
        try:
            await completion_task
        except asyncio.CancelledError:
            pass
        await coordinator.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice3 Frontend Demo")
    parser.add_argument(
        "--model-dir",
        default="/opt/gpfs/home/tianteng/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice3 model directory",
    )
    parser.add_argument(
        "--tts-text",
        default="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        help="Prompt text",
    )
    parser.add_argument(
        "--prompt-audio",
        # Default to a known existing file found in previous steps
        default="/opt/gpfs/home/tianteng/CosyVoice/asset/zero_shot_prompt.wav", 
        help="Path to prompt audio",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check if prompt audio exists, if not, warn user
    if not os.path.exists(args.prompt_audio):
        logger.warning(f"Prompt audio file not found: {args.prompt_audio}. Please provide a valid path.")
        # Try to find a wav file in CosyVoice dir
        # For now, we proceed, assuming user might provide a valid one via CLI

    stage1_proc = mp.Process(
        target=run_frontend_stage,
        name="FrontendStage",
        args=(args.model_dir,),
    )
    stage2_proc = mp.Process(
        target=run_llm_stage,
        name="LLMStage",
        args=(args.model_dir,),
    )
    stage3_proc = mp.Process(
        target=run_flow_stage,
        name="FlowStage",
        args=(args.model_dir,),
    )
    stage4_proc = mp.Process(
        target=run_vocoder_stage,
        name="VocoderStage",
        args=(args.model_dir,),
    )
    stage5_proc = mp.Process(
        target=run_postprocess_stage,
        name="PostprocessStage",
        args=(args.model_dir,),
    )

    stage1_proc.start()
    stage2_proc.start()
    stage3_proc.start()
    stage4_proc.start()
    stage5_proc.start()
    logger.info(f"Frontend stage started with PID: {stage1_proc.pid}")
    logger.info(f"LLM stage started with PID: {stage2_proc.pid}")
    logger.info(f"Flow stage started with PID: {stage3_proc.pid}")
    logger.info(f"Vocoder stage started with PID: {stage4_proc.pid}")
    logger.info(f"Postprocess stage started with PID: {stage5_proc.pid}")

    try:
        asyncio.run(run_coordinator(args))
    finally:
        stage1_proc.join(timeout=2)
        stage2_proc.join(timeout=2)
        stage3_proc.join(timeout=2)
        stage4_proc.join(timeout=2)
        stage5_proc.join(timeout=2)
        if stage1_proc.is_alive():
            stage1_proc.terminate()
            stage1_proc.join(timeout=1)
        if stage2_proc.is_alive():
            stage2_proc.terminate()
            stage2_proc.join(timeout=1)
        if stage3_proc.is_alive():
            stage3_proc.terminate()
            stage3_proc.join(timeout=1)
        if stage4_proc.is_alive():
            stage4_proc.terminate()
            stage4_proc.join(timeout=1)
        if stage5_proc.is_alive():
            stage5_proc.terminate()
            stage5_proc.join(timeout=1)

if __name__ == "__main__":
    main()
