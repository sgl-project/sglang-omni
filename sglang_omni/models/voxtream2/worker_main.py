# SPDX-License-Identifier: Apache-2.0
"""Model-process entrypoint with dependencies imported at module scope."""

import importlib.metadata
import sys
import traceback

from sglang_omni.models.voxtream2.backend import VoXtreamBackend
from sglang_omni.models.voxtream2.loader import load_generator
from sglang_omni.models.voxtream2.wire import PROTOCOL_VERSION, read_frame, write_frame
from sglang_omni.models.voxtream2.worker import serve


def main() -> int:
    source, target = sys.stdin.buffer, sys.stdout.buffer
    # Author progress prints belong to logs, not the framed protocol.
    sys.stdout = sys.stderr
    try:
        init = read_frame(source)
        if init.get("type") != "init":
            raise RuntimeError("Expected VoXtream2 worker initialization")
        else:
            pass
        config = dict(init["config"])
        reference = config.pop("reference_audio")
        backend = VoXtreamBackend(load_generator(**config), reference)
        versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchaudio", "transformers", "torchao", "voxtream")
        }
        write_frame(
            target,
            {
                "type": "ready",
                "version": PROTOCOL_VERSION,
                "sample_rate": backend.sample_rate,
                "max_samples": backend.max_samples,
                "runtime": {"python": sys.executable, "versions": versions},
            },
        )
        serve(backend, source, target)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            write_frame(target, {"type": "fatal", "error": str(exc)})
        except (BrokenPipeError, OSError):
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
