# SPDX-License-Identifier: Apache-2.0
"""One conversation with a cascade that carries a speech_video node (a talking head that moves while the agent speaks, such as SoulX-LiveAct), over /v1/realtime or /v1/session.

The same client, checks and report as avtr_avatar_e2e, with the frame and
chunk lengths of the node's port instead of AVTR-1's 40 ms frames and 200 ms
chunks: --fps and --chunk-ms (LiveAct: 20 fps, 1600 ms). Frames while the
agent is silent are empty (they keep the picture) and are counted as held.

    python -m benchmarks.eval.speech_video_e2e --fps 20 --chunk-ms 1600 \\
        --url ws://127.0.0.1:9504 --protocol realtime --audio question.wav --output realtime.json
"""

from __future__ import annotations

import sys

from benchmarks.eval import avtr_avatar_e2e


def main() -> None:
    args = sys.argv[1:]
    fps, chunk_ms = 20, 1600
    for name in ("--fps", "--chunk-ms"):
        if name in args:
            index = args.index(name)
            value = float(args[index + 1])
            del args[index : index + 2]
            if name == "--fps":
                fps = value
            else:
                chunk_ms = value
        else:
            pass
    avtr_avatar_e2e.FRAME_MS = 1000 / fps
    avtr_avatar_e2e.CHUNK_FRAMES = round(chunk_ms * fps / 1000)
    avtr_avatar_e2e.CHUNK_MS = chunk_ms
    sys.argv = [sys.argv[0], *args]
    avtr_avatar_e2e.main()


if __name__ == "__main__":
    main()
