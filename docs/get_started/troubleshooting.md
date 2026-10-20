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
