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

Single-GPU CUDA editing accepts one or more reference images. Editing requests run individually; multiple outputs use the native runtime's sequential expansion. This path does not enable mask editing, think mode or TP.

HTTP editing uses `img_cfg_scale=1.0` and the reference preprocessing described below. Custom image guidance is not exposed through this HTTP endpoint.

Use the same explicit shared-stage configuration as #3260. Save the following as sensenova.yaml and replace the model path with your checkpoint directory:

```yaml
config_cls: Cosmos3PipelineConfig
model_path: /path/to/SenseNova-U1.5-8B-MoT
stages:
  generation:
    factory:
      output_dir: outputs/sensenova
      server_args_overrides:
        enable_torch_compile: false
```

Cosmos3PipelineConfig supplies the shared native generation stage; the model path and external package select SenseNova. Automatic Omni model-path discovery is tracked in #3277.

With the external package environment variable above set, start the server:

```bash
python -m sglang_omni.cli serve \
  --config sensenova.yaml \
  --host 127.0.0.1 --port 30000 \
  --skip-server-warmup
```

The same server handles text-to-image requests at /v1/images/generations and editing requests at /v1/images/edits.

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

Editing accepts at most 64 references and rejects an edit whose estimated prefix memory exceeds the GPU's free memory plus the memory PyTorch has reserved but not allocated, before encoding its references. The SenseNova worker enables PyTorch expandable segments as it loads the model unless `PYTORCH_CUDA_ALLOC_CONF` or `PYTORCH_ALLOC_CONF` is set. On devices that share host memory, startup sets `PYTORCH_CUDA_ALLOC_CONF` to expandable segments when that variable is unset, and it takes precedence over `PYTORCH_ALLOC_CONF`, so opt out there through `PYTORCH_CUDA_ALLOC_CONF`.

## Cache-DiT

Cache-DiT reuses the output of the middle decoder layers on denoising steps where that output barely changes, trading some image quality for speed. It is off by default and covers both text-to-image and editing. Only the denoising steps go through the cache; the prompt and reference prefix pass always runs every layer.

The server default comes from the same environment variables that SGLang's native image pipelines read. Set them in the shell, or for the generation stage in the Omni configuration:

```yaml
stages:
  generation:
    env:
      SGLANG_CACHE_DIT_ENABLED: "true"
      SGLANG_CACHE_DIT_RDT: "0.24"
```

| Variable | Default | Setting |
|---|---|---|
| `SGLANG_CACHE_DIT_ENABLED` | `false` | Server switch |
| `SGLANG_CACHE_DIT_FN` | `1` | `Fn_compute_blocks`, the leading layers that always run |
| `SGLANG_CACHE_DIT_BN` | `0` | `Bn_compute_blocks`, the trailing layers that always run |
| `SGLANG_CACHE_DIT_WARMUP` | `4` | `max_warmup_steps`, the first steps that never reuse the cache |
| `SGLANG_CACHE_DIT_RDT` | `0.24` | `residual_diff_threshold`, between 0 and 1; higher reuses more |
| `SGLANG_CACHE_DIT_MC` | `3` | `max_continuous_cached_steps`; -1 means no limit |

SenseNova runs plain DBCache only. TaylorSeer, step computation masks (SCM), DMD and secondary settings are not supported.

A request overrides the server default with `enable_cache_dit` and `cache_dit_params`. Left out or null, `enable_cache_dit` keeps the server switch; true or false replaces it for that request; any other value, such as the JSON string `"false"`, is rejected. Each key in `cache_dit_params` replaces the matching variable, and the keys a request leaves out keep the server values. The keys are the five setting names in the table.

```bash
curl http://localhost:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red apple on a wooden table.", "size": "1024x1024",
       "enable_cache_dit": true,
       "cache_dit_params": {"residual_diff_threshold": 0.12, "max_warmup_steps": 8}}'
```

The edits endpoint takes the same fields as form fields, with `cache_dit_params` as a JSON object: `-F 'enable_cache_dit=true' -F 'cache_dit_params={"residual_diff_threshold": 0.12}'`.

With Cache-DiT on, a request is rejected with status 400 when its settings use an unknown key, a TaylorSeer, SCM or DMD key or `secondary`, fall outside the ranges above, or set `Fn_compute_blocks` plus `Bn_compute_blocks` above the decoder layer count. The cache cannot follow two guidance configurations: editing with three guidance branches, where `img_cfg_scale` is neither 1 nor equal to `guidance_scale`, and text-to-image with `guidance_scale` above 1 and a `cfg_interval` other than `(0, 1)`. A request with either one runs without Cache-DiT when the switch comes from `SGLANG_CACHE_DIT_ENABLED`, though its `cache_dit_params` are still checked as above, and is rejected when it sets `enable_cache_dit` to true. Only SenseNovaU1SamplingParams, as with `sglang generate`, can set these; the HTTP image routes accept neither `img_cfg_scale` nor `cfg_interval`. A request with Cache-DiT off ignores its `cache_dit_params`. Startup fails for out-of-range variables, for `SGLANG_CACHE_DIT_DMD=true` or `SGLANG_CACHE_DIT_TAYLORSEER=true`, for an `SGLANG_CACHE_DIT_SCM_PRESET` other than `none`, for either SCM bins variable, and for `SGLANG_CACHE_DIT_FN` plus `SGLANG_CACHE_DIT_BN` above the decoder layer count.

The cache stays mounted between requests. A request with the same settings and guidance branches refreshes it, a request with different ones mounts it again, and a request with Cache-DiT off removes it.

With dynamic batching on (`--batching-max-size` above 1), text-to-image requests with the same Cache-DiT switch and settings can share a batch, as in SGLang's other image pipelines. The cache then decides for the whole batch at each step whether to reuse the middle layers, so a request's output can depend on the requests batched with it.
