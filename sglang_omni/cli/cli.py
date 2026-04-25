from __future__ import annotations

import warnings

from . import app

warnings.warn(
    "`sglang_omni.cli.cli` is deprecated and will be removed after the "
    "SGLang Omni V1 release. "
    "Use `sglang_omni.cli` instead.",
    DeprecationWarning,
    stacklevel=1 if __name__ == "__main__" else 2,
)


if __name__ == "__main__":
    app()
