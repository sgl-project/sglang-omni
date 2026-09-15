# Cosmos 3 Super visual generation

SGLang-Omni serves the full `nvidia/Cosmos3-Super` checkpoint by delegating
model loading, diffusion execution, scheduling, and media encoding to native
SGLang. This initial qualified slice is BF16 text-to-image generation through
the Omni SDK and `POST /v1/images/generations`.

## Support matrix

| Checkpoint | Operation | Interface | Status |
| --- | --- | --- | --- |
| `nvidia/Cosmos3-Super` | Text to image | Omni SDK | Qualified on the profile below |
| `nvidia/Cosmos3-Super` | Text to image | `/v1/images/generations` | Qualified on the profile below |
| `nvidia/Cosmos3-Super` | T2V, I2V, V2V | SDK / HTTP | Native capability; not yet qualified in Omni |
| `nvidia/Cosmos3-Super` | Reasoner, sound, actions | SDK / HTTP | Not included in this support slice |
| Specialized Super checkpoints | Visual generation | SDK / HTTP | Not yet qualified in Omni |

Support is deliberately operation- and topology-specific. The presence of a
modality in checkpoint metadata does not make it part of this profile.

## Reproducible environment

The Super profile is pinned to these revisions:

| Component | Revision |
| --- | --- |
| SGLang-Omni adapter base | `bd864f7393c9df4c0bfdb474ab65521247442858` (#2050 + #2048) |
| Native SGLang | `5298d85218a7f5d56370afe1c611a4a5c152fe58` |
| `nvidia/Cosmos3-Super` | `fe77b66696d645f663b8f27e942b3b43e4629e23` |
| PyTorch / CUDA | `2.13.0` / `13.0` |
| Transformers / Diffusers | `5.12.1` / `0.37.0` |
| SGLang kernel / FlashInfer | `0.4.7` / `0.6.18` |

Install the pinned native runtime before installing this checkout. Rust
extensions are unrelated to Cosmos generation and can be omitted when a Rust
toolchain is unavailable.

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 5298d85218a7f5d56370afe1c611a4a5c152fe58
SGLANG_BUILD_RUST_EXTS=none pip install -e "./sglang/python[diffusion]"
pip install ninja
pip install --no-deps -e .
```

Cosmos guardrails are a separate optional native dependency. The commands
below disable them only to reproduce the isolated generation qualification.
Install `cosmos-guardrail==0.3.1` and omit the environment variable for a
guardrail-enabled deployment.

## One-A100 BF16 profile

[`cosmos3_super_t2i_1xa100_bf16.yaml`](https://github.com/sgl-project/sglang-omni/blob/main/examples/configs/cosmos3_super_t2i_1xa100_bf16.yaml)
loads the full 64B checkpoint on one A100 80 GB. Because the BF16 DiT is larger
than device memory, native layerwise offload keeps 24 layers resident in each
of the two 64-layer transformer groups and streams the remaining layers from
host storage. The checkpoint itself is about 124 GiB, so use fast local
storage and leave substantial host memory for loading and the page cache. A
multi-GPU profile is preferable when available.

The checked-in request uses a fixed prompt and explicit 640x640 dimensions,
seed, guidance, step count, and flow shift. Run the direct native reference
first:

```bash
SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
python examples/cosmos3_super_t2i.py direct \
  --output-dir outputs/cosmos3_super/direct
```

Then exercise the same request through the full Omni SDK path. `--repeats 2`
also checks that back-to-back requests keep independent request state and
owned media paths.

```bash
SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
python examples/cosmos3_super_t2i.py sdk \
  --repeats 2 \
  --output-dir outputs/cosmos3_super/sdk
```

Each generated PNG is decoded with Pillow, checked for its requested shape and
non-uniform content, and accompanied by a JSON manifest containing the exact
request, latency, byte count, entropy, channel standard deviations, file
SHA256, decoded-pixel SHA256, and native metrics.

## Qualification evidence

The profile was exercised on 2026-09-15 with one NVIDIA A100-SXM4 80 GB. The
native runtime reported a 117.85 GB BF16 transformer, 48 of 128 transformer
layers resident (24 in each group), and a 52,458 MiB maximum peak CUDA
reservation. These are qualification observations, not general performance
claims; layer-streaming latency depends heavily on local storage and host page
cache behavior.

| Path | Startup / request observations | Result |
| --- | --- | --- |
| Direct native | 1,286.8 s generation; 52,456 MiB peak | Valid 640x640 PNG |
| Omni SDK | 656.8 s startup; 1,025.2 s cold request; 1,108.2 s back-to-back request | Two independent result paths |
| Image API | Startup exceeded the default 600 s, hence the explicit 7,200 s setting below; 1,164.2 s corrected request; 52,458 MiB peak | HTTP 200 with inline base64 PNG |

Direct, both SDK requests, and the corrected HTTP request produced exactly the
same decoded RGB pixels at seed 0: SHA256
`4896fe07ac4724004e7d23f6e3be6ae10d0228071c9bf64ea187a3c3dd4c520d`.
The HTTP PNG file hash differs because the API re-encodes the same pixels.
During qualification, `/health` also returned HTTP 200 while generation was in
flight. Normal SDK and HTTP shutdowns released the scheduler, GPU allocation,
and listening ports; a forced startup timeout also reclaimed CUDA memory
before a successful restart.

## HTTP generation

Start the server with the same pinned profile:

```bash
SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
SGLANG_OMNI_STARTUP_TIMEOUT=7200 \
sgl-omni serve \
  --config examples/configs/cosmos3_super_t2i_1xa100_bf16.yaml \
  --model-name nvidia/Cosmos3-Super \
  --host 127.0.0.1 \
  --port 30010
```

In another shell, submit and verify the matching OpenAI-compatible image
request:

```bash
python examples/cosmos3_super_t2i.py http \
  --server-url http://127.0.0.1:30010 \
  --output-dir outputs/cosmos3_super/http
```

The endpoint returns inline base64 image data. The example decodes it to PNG
and applies the same verification and manifest generation used by the SDK and
direct-native modes.

The real-checkpoint regression is opt-in because it downloads about 124 GiB
and each one-GPU request can take many minutes:

```bash
SGLANG_COSMOS3_SUPER_E2E_MODE=sdk \
pytest -q tests/test_model/test_cosmos3_super_generation.py
```

## Scope and lifecycle

Generated SDK files are request-owned until terminal delivery. HTTP inline
content is copied by the client before the response completes. A normal server
shutdown stops the native scheduler before releasing the Omni stage process;
restart qualification must use a fresh server process and repeat the same
health and generation request. Unsupported operations are not claimed by this
profile even when the native checkpoint exposes them.
