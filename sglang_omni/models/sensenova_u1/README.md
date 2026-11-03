# SenseNova native image package

This package registers SenseNova-U1.5 with SGLang's native multimodal generation runtime. The model implementation ships with Omni, while SGLang owns request scheduling, worker processes and output saving.

## Run

Use a source installation of Omni and a SGLang version with `register_pipeline` and `SGLANG_EXTERNAL_MODEL_PACKAGE` support.

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_omni.models.sensenova_u1.native
export MODEL_PATH=/path/to/SenseNova-U1.5-8B-MoT

sglang generate \
  --model-path "$MODEL_PATH" \
  --prompt "A red apple on a wooden table, natural window light." \
  --width 512 --height 512 \
  --num-inference-steps 4 --guidance-scale 4 --seed 42 \
  --save-output --output-file-name sensenova.png
```

The external package replaces the in-tree SenseNova pipeline registration while both implementations coexist. The package registers the model classes shipped under `sglang_omni.vendor.sensenova_u1`; it does not import SGLang's SenseNova model implementation. Keep the environment variable set for the launcher and all worker processes.

## Image editing

Single-GPU CUDA editing accepts one or more reference images. Editing requests run individually; multiple outputs use the native runtime's sequential expansion. This path does not enable mask editing, think mode, TP or Cache-DiT.

HTTP editing uses `img_cfg_scale=1.0` and the reference preprocessing described below. Custom image guidance is not exposed through this HTTP endpoint.

```bash
sglang serve --model-path "$MODEL_PATH" --port 30000
```

```bash
curl http://localhost:30000/v1/images/edits \
  -F 'image=@reference.png' \
  -F 'prompt=Change the background to a snowy forest.' \
  -F 'size=1024x1024' \
  -F 'seed=42' \
  -F 'num_inference_steps=50' \
  -F 'response_format=b64_json'
```

The response contains the edited image in `data[0].b64_json`. Repeat the `image` field to supply more references, in prompt order. Set `n=2` to generate two outputs sequentially.

The following model-side settings are available through SenseNovaU1SamplingParams; the HTTP endpoint uses their defaults.

| Parameter | Default | Meaning |
|---|---|---|
| `img_cfg_scale` | `1.0` | Nonnegative reference-image guidance scale. |

Transparent references are composited onto white. An explicit `size` determines the output dimensions. Without it, the sampling defaults remain 2048×2048. Reference preprocessing follows the historical native model's pixel budget: a minimum of 512² and a maximum of min(2048², 4096² / reference count), with dimensions aligned to 32 pixels. Editing uses the default timestep shift and CFG controls; CFG normalization and think mode are not supported on this path.
