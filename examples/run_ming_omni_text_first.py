# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the ming-text preset."""

try:
    from examples import _omni_launcher as _launcher
except ModuleNotFoundError:
    import _omni_launcher as _launcher


def parse_args():
    return _launcher.parse_preset_args("ming-text")


def main() -> None:
    _launcher.run_preset("ming-text")


if __name__ == "__main__":
    main()
