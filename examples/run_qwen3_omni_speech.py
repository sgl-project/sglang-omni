# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the qwen3-speech preset."""

try:
    from examples import _omni_launcher as _launcher
except ModuleNotFoundError:
    import _omni_launcher as _launcher


def parse_args():
    return _launcher.parse_preset_args("qwen3-speech")


def main() -> None:
    _launcher.run_preset("qwen3-speech")


if __name__ == "__main__":
    main()
