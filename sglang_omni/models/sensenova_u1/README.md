# SenseNova native T2I package

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

## Devices smaller than the checkpoint

The bf16 checkpoint needs 32.7 GiB because the MoT backbone carries a second expert branch for image tokens. When the launch device cannot hold it, the loader places the backbone across as many consecutive devices as the weights need and leaves the rest free; a device that holds the checkpoint on its own keeps single-device loading. On Intel XPU with 24 GiB Arc Pro B60 cards that means two cards, so the launcher needs at least two visible devices:

```bash
export ZE_AFFINITY_MASK=0,1
```
