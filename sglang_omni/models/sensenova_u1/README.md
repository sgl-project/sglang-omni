# SenseNova native T2I package

This package registers SenseNova-U1.5 with SGLang's native multimodal generation runtime. The model implementation ships with Omni, while SGLang owns request scheduling, worker processes and output saving.

## Run

Use a source installation of Omni and a SGLang version with `register_pipeline` and `SGLANG_EXTERNAL_MODEL_PACKAGE` support. The extraction baseline is SGLang commit `e4d4a9a7de5454cd8b854b21af723c2155a55a3d`.

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

## Scope

This first migration preserves native single-GPU T2I execution and sequential multi-output generation. Image editing, native tensor parallelism and Cache-DiT integration are separate follow-ups. Existing native restrictions on parallelism, LoRA and offload remain in place.

The package has been tested with Omni's shared native generation stage using an explicit pipeline configuration. Automatic `--model-path` selection for native generation depends on Omni support.

## Tests

```bash
python -m pytest tests/unit_test/sensenova/test_native_package.py -q
```

The unit tests cover fresh-process registration, local checkpoint discovery, discovery with in-tree SenseNova model and pipeline imports blocked, generation parameter/output conversion and unsupported parallelism. SGLang still eagerly imports its built-in configuration modules during registry initialization; their removal belongs to the corresponding SGLang cleanup. GPU qualification compares decoded image pixels against the in-tree pipeline with identical model weights, seeds, dimensions and sampling parameters.
