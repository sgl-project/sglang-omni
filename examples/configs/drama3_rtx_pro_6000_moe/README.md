# Drama 3 FP8 experts on RTX PRO 6000 Blackwell Server Edition

Set the config directory on the server process before launching Drama 3 with SGLang and Triton 3.8.0.

```bash
export SGLANG_MOE_CONFIG_DIR="$PWD/examples/configs/drama3_rtx_pro_6000_moe"
```

The config selects a 16-row tile for one through sixteen tokens routed to the 256 block-FP8 experts. Both projections use the same tile. The entry at seventeen tokens selects the runtime's 64-row default for larger batches because SGLang looks up the nearest token count. The filename selects the server-edition GPU, expert dimension and quantization block shape. Unset the variable and restart to use the runtime's config directory.
