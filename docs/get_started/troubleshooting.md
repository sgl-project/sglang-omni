# Troubleshooting

## Regional PyPI mirrors

A configured PyPI mirror may be missing a required package version or lag
behind PyPI. For example, if the mirror does not provide
`mooncake-transfer-engine-cuda13>=0.3.10`, installation fails to resolve the
requirement in [`pyproject.toml`](../../pyproject.toml). This is an index
availability issue; the mirror does not have the required version.

Download that one wheel from PyPI, then use `--find-links` while keeping your
existing uv mirror configuration for the other packages. With the Python 3.12
virtual environment from the [installation guide](./installation.md#install-from-source)
active, install pip into it so the downloaded wheel matches that environment:

```bash
uv pip install pip
python -m pip download --no-deps --index-url https://pypi.org/simple \
    "mooncake-transfer-engine-cuda13>=0.3.10" -d ./wheels
uv pip install --prerelease=allow --find-links ./wheels -e .   # or "sglang-omni==<version>" in place of -e .
```

## Expected output of uv pip check

`uv pip check` and `python -m pip check` compare the requirements every installed distribution declares with the versions installed next to it. The CUDA install paths leave a few declared ranges unmet on purpose, so the lines in this table are expected and the code they concern runs. Any other line is a real conflict and worth reporting.

| Install path | Reported line | Why the runtime still works |
| --- | --- | --- |
| Every CUDA path | `grpcio-health-checking` and `grpcio-reflection` require `protobuf>=7.35.1,<8.0.0`, protobuf 6.x is installed | SGLang pulls both through `smg-grpc-servicer`, and `tool.uv.override-dependencies` in `pyproject.toml` keeps protobuf in the 6.x series, so `import grpc_health.v1.health_pb2` fails with a protobuf `VersionError` (gencode 7.35.1, runtime 6.33.6). Neither SGLang-Omni nor any module of the installed `sglang` package imports `grpc_health` or `grpc_reflection`. |
| After `uv pip install --no-deps qwen-tts==0.1.1` | `qwen-tts` requires `transformers==4.57.3` | The project runs Transformers 5.19.0, and `sglang_omni/models/qwen3_tts/compat.py` shims the API differences, see the [Qwen3-TTS cookbook](../cookbook/qwen3_tts.md#prerequisites). |
| Same | `qwen-tts` requires `accelerate==1.12.0` | `qwen-tts` 0.1.1 never imports `accelerate`. |
| Same | `qwen-tts` requires `onnxruntime`, which is not installed | The CUDA path installs `onnxruntime-gpu`, which provides the `onnxruntime` module the speech tokenizer imports. |
| `[fishaudio]` extra and the CI image | `descript-audiotools` requires `protobuf<3.20,>=3.9.2` | The override keeps protobuf 6.x. `audiotools` reaches protobuf only through `torch.utils.tensorboard`, which the installed TensorBoard supports, and the CI import probe imports `dac` on every run. |
| Every CUDA path | `decord 0.6.0 is not supported on this platform` (pip) or `The package decord was built for a different platform` (uv) | The wheel is published as `py3-none-manylinux2010_x86_64`, but its WHEEL file records the tag `cp36-cp36m`. decord loads `libdecord.so` through ctypes, and the MiniCPM-o video frontend decodes with it on Python 3.12. |

The two tools word the lines differently, and `uv pip check` exits with status 1 whenever it prints one, so its exit status is not a gate. A source checkout has a script that runs the same comparison with the interpreter it is started with and fails only on a line outside this table. The PyPI package does not include it, so run it from the root of the checkout, for example after [installing from source](./installation.md#install-from-source), with the environment's Python:

```bash
cd sglang-omni   # the root of the source checkout
python .github/scripts/check_omni_installed_requirements.py
```

`--prerelease=allow` is required because SGLang 0.5.22 pins `cuda-tile==1.6.0rc5`. The flag also lets uv take a pre-release of any other package when one is newer than the latest release on PyPI, which is why `uv pip freeze` can list development or release-candidate versions of packages such as `einops`, `safetensors` or `grpcio`.

SGLang 0.5.22 ships `manylinux_2_34` wheels, so the CUDA path needs glibc 2.34 or newer, as on Ubuntu 22.04. On an older system uv stops with `sglang==0.5.22 has no wheels with a matching platform tag`.
