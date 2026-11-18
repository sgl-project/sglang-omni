# Duo realtime backend

Duo consumes video and speech, produces Intent hidden states with Qwen3-VL, and generates audio/video blocks with a causal Realization DiT. The backend uses the shared `/v1/realtime` endpoint. The browser demo is not included in this integration.

## Dependencies and checkpoints

Install the matching SGLang source checkout containing the Duo Diffusion components after installing Omni. An unmodified SGLang wheel does not include these components. This split retains the validated Omni baseline while moving DiT, physical KV, sampling, and Codec execution into SGLang multimodal_gen; Qwen/H50, realtime sessions, and reset policy remain in Omni. Use a dedicated environment and verify both import paths before serving.

The model-specific tokenizer protocol, checkpoint reader, and causal sampler are provided by the Duo model sources. Obtain the matching Duo inference source bundle from [Duo](https://github.com/inclusionAI/Duo), then set `DUO_MODEL_SOURCE` to the directory containing `torchtitan`, `data`, and `tools/ckpt_to_oss/duo_protocol.py`. For the Duo inference release layout this is `third_party/duo`. Use the model-provider revision matching the checkpoint; this is a model dependency, not a dependency on vLLM.

```bash
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export PYTHONPATH="$PWD:$DUO_MODEL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TORCHTITAN_LIVE=1
pip install -e .
pip install peft tyro
export SGLANG_SOURCE=/path/to/sglang-duo-diffusion
pip install -e "$SGLANG_SOURCE/python[diffusion]"
export PYTHONPATH="$SGLANG_SOURCE/python:$PWD:$DUO_MODEL_SOURCE"
python -c "from sglang.multimodal_gen.runtime.models.dits.duo import native_dcp; print(native_dcp.__file__)"
```

Obtain [Duo-1.0-causal-base](https://huggingface.co/inclusionAI/Duo-1.0-causal-base) and [Duo-1.0-distill](https://huggingface.co/inclusionAI/Duo-1.0-distill). The optional [quantized release](https://huggingface.co/inclusionAI/Duo-1.0-distill-quant) supplies FP8 Linear weights. FP8 Linear does not imply FP8 Attention.

Prepare the checkpoint metadata and Codec layout without modifying tensors. Use the Codec YAML files from the matching Duo model source, not replacement JSON files. The RoPE and AdaLN selections must match the checkpoint's reference inference configuration.

```bash
python scripts/prepare_duo_models.py \
  --release /models/Duo-1.0-distill \
  --base-release /models/Duo-1.0-causal-base \
  --codec-config-dir /path/to/Duo/duo/models/realization/modules/native_vae \
  --output /models/prepared-duo \
  --rope new --adaln-timestep new
python scripts/merge_duo_intent_adapter.py \
  --intent /models/Duo-1.0-causal-base/intent \
  --intent-adapter /models/Duo-1.0-causal-base/intent_adapter \
  --output /models/duo-intent-merged
```

Merge the matching Intent adapter before launching native Qwen TP workers. Keep its tokenizer and protocol manifest with the merged model. Do not substitute an unrelated Qwen checkpoint or adapter when comparing generation quality.

## Deployment

An eight-GPU deployment places Intent TP4 on GPUs 0–3, Realization TP8 on GPUs 0–7, and four-tile Codec decoding on GPUs 4–7. Intent for the next block can overlap Codec for the previous block; Realization waits for shared GPU work before starting. Codec tiling is spatial tile parallelism, not transformer TP.

Start Qwen3-ASR in a separate terminal using the same environment:

```bash
CUDA_VISIBLE_DEVICES=3 python -m sglang_omni.cli serve \
  --model-path /models/Duo-1.0-causal-base/speech/recognizer --model-name duo-asr \
  --asr.engine.max_running_requests 1 --asr.engine.enable_torch_compile false \
  --asr.engine.mem_fraction_static 0.1 --port 18084
```

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
python scripts/serve_duo_realtime.py \
  --codec-model-path /models/prepared-duo/codec \
  --intent-model-path /models/duo-intent-merged \
  --intent-adapter-path /models/Duo-1.0-causal-base/intent_adapter \
  --intent-backend native --intent-tp-size 4 --intent-worker-devices 0 1 2 3 \
  --realization-checkpoint /models/prepared-duo/realization \
  --realization-backend native --realization-stage dmd --steps 4 --flow-shift 5 \
  --realization-tp-size 8 --realization-sp-size 1 \
  --realization-worker-devices 0 1 2 3 4 5 6 7 \
  --realization-optimize-copy-sync --overlap-intent-codec \
  --realization-kv-window-tokens 65536 --realization-reference-window-chunks 14 \
  --realization-kv-workspace-capacity-tokens 131072 \
  --realization-kv-reset-interval-seconds 30 \
  --codec-device cuda:4 --codec-worker-devices 4 5 6 7 \
  --codec-decode-backend solarx --codec-attention-backend torch_cudnn_sdpa \
  --max-context-tokens 32768 --intent-kv-window-tokens 32768 \
  --asr-base-url http://127.0.0.1:18084/v1 --asr-model-name duo-asr \
  --height 768 --width 576 --port 18082
```

The realtime audio/video path requires ASR for input PCM, including silent PCM blocks. ASR is not silently enabled when its endpoint is omitted.

Add `--realization-modelopt-checkpoint /models/Duo-1.0-distill-quant/realization --realization-fp8-direct-kernels` to use released FP8 Linear weights. FP8 Attention additionally requires a compatible built library and calibration profile via the separate attention options. The BF16 Attention path remains available.

## Streaming and state

Send `session.update` selecting audio/video outputs and `sglang.realization.decode="duo_5_17"` with `sglang.max_slice_nums=2`. Each block contains 17 camera frames at 24 FPS and approximately 708.33 ms of 16 kHz mono PCM. The runtime uses sample counts rather than rounded 708 ms windows. Output consists of timestamped video frames and PCM audio. The 1→4 mode is only available with a compatible checkpoint.

The server controls periodic DiT history reset through `--realization-kv-reset-interval-seconds` (default 30; zero disables it). The interval counts generated media, not wall time. With 5→17 blocks the first reset occurs before zero-based block 43, after 30.458 seconds of generated media. Reset happens at a block boundary and preserves the Qwen conversation KV and Codec continuity. Clients do not need a reset button or a session-update counter.

The reset interval and sliding-window limits are different controls: eviction bounds memory continuously, while periodic reset drops accumulated Realization history to limit long-generation drift. Clean commit still computes KV from the final denoised block. Session cancellation drains outstanding model/Codec work before freeing state; a new session starts with its own history.
