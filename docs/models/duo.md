# Duo realtime backend

Duo consumes video and speech, produces Intent hidden states with Qwen3-VL, and generates audio/video blocks with a causal Realization DiT. The backend uses the shared `/v1/realtime` endpoint. The [Duo playground](../../playground/duo/) connects to that endpoint.

## Dependencies and checkpoints

Use the [Duo Diffusion branch](https://github.com/Anmuliar/sglang/tree/duo-0.5.23), based on the SGLang 0.5.23 release required by this Omni branch. Install both in the same environment. An unmodified SGLang wheel does not include these components. This split retains the validated Omni baseline while moving DiT, physical KV, sampling, and Codec execution into SGLang multimodal_gen; Qwen/H50, realtime sessions, and reset policy remain in Omni. Use a dedicated environment and verify both import paths before serving.

The model-specific tokenizer protocol, checkpoint reader, and causal sampler are provided by the Duo model sources. Obtain the matching Duo inference source bundle from [Duo](https://github.com/inclusionAI/Duo), then set `DUO_MODEL_SOURCE` to the directory containing `torchtitan`, `data`, and `tools/ckpt_to_oss/duo_protocol.py`. For the Duo inference release layout this is `third_party/duo`. Use the model-provider revision matching the checkpoint; this is a model dependency, not a dependency on vLLM.

```bash
export DUO_MODEL_SOURCE=/path/to/Duo/third_party/duo
export PYTHONPATH="$PWD:$DUO_MODEL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export TORCHTITAN_LIVE=1
git clone --branch duo-0.5.23 https://github.com/Anmuliar/sglang.git ../sglang-duo
export SGLANG_SOURCE="$(cd ../sglang-duo && pwd)"
git -C "$SGLANG_SOURCE" checkout 55a0c2b8d3d4ed4795b2df64873c0a69ac9a52d2
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23+duo pip install -e "$SGLANG_SOURCE/python[diffusion]"
pip install -e . peft tyro
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

Start the registered pipeline with one command after preparing the weights. Qwen3-ASR runs as a pipeline-managed GPU worker and receives transcription requests through the coordinator.

```bash
sgl-omni serve --model-path /models --enable-realtime --host 127.0.0.1 --port 18082
```

Add `--realization-modelopt-checkpoint /models/Duo-1.0-distill-quant/realization` to use released FP8 Linear weights with direct kernels. The default attention precision is BF16.

## Streaming and state

Send `session.update` selecting audio/video outputs and `sglang.realization.decode="duo_5_17"` with `sglang.max_slice_nums=2`. Each block contains 17 camera frames at 24 FPS and approximately 708.33 ms of 16 kHz mono PCM. The runtime uses sample counts rather than rounded 708 ms windows. Output consists of timestamped video frames and PCM audio. The 1→4 mode is only available with a compatible checkpoint.

Omit `--realization-kv-reset-interval-blocks` to enable manual DiT history reset with a native Realization checkpoint. Set it to a positive block count to enable automatic reset and disable the manual control. With 5→17 decoding, 43 blocks represent about 30.46 seconds. The first automatic reset runs before zero-based block 43. Both modes preserve Qwen conversation KV and Codec state.

The reset interval and sliding-window limits are different controls: eviction bounds memory continuously, while periodic reset drops accumulated Realization history to limit long-generation drift. Clean commit still computes KV from the final denoised block. Session cancellation drains outstanding model/Codec work before freeing state; a new session starts with its own history.
