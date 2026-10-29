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

HTTP editing uses the default image guidance and reference preprocessing settings. Passing custom `img_cfg_scale`, `input_max_pixels` and `do_resize` values through HTTP requires a separate SGLang API change and is deferred.

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
| `input_max_pixels` | 262144 pixels per reference | Reference resize budget, at least 262144 pixels. Omitting it uses a 512² pixel budget per reference to reduce editing memory use. |
| `do_resize` | `true` | Resize references to the budget and patch grid. With `false`, reference dimensions must already be divisible by 32; the resize budget is not applied. |

Transparent references are composited onto white. An explicit `size` determines the output dimensions. Without it, the first prepared reference determines the output aspect ratio at the default 2048² pixel budget, rounded to the patch grid. Editing uses the default timestep shift and CFG controls; CFG normalization and think mode are not supported on this path.
