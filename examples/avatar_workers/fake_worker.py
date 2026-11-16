# SPDX-License-Identifier: Apache-2.0
"""A chunk worker without a model, for wiring checks: each frame is a flat picture whose brightness follows the speech level and whose hue follows the listening level, or a still skeleton pose.

python examples/avatar_workers/fake_worker.py --port 8120 --fps 25 --chunk-frames 10 --lookahead-ms 0 --listens --render-ms 50
"""

from __future__ import annotations

import time

import numpy as np
from chunk_server import (
    SAMPLE_RATE_HZ,
    SessionModel,
    base_parser,
    jpeg_of,
    serve,
    setup_logging,
    warm_up,
)


class FakeModel(SessionModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.info = {
            "model": "fake",
            "fps": args.fps,
            "chunk_frames": args.chunk_frames,
            "lookahead_samples": round(args.lookahead_ms * SAMPLE_RATE_HZ / 1000),
            "listens": args.listens,
            "media": args.media,
        }
        if args.media == "jpeg_frame":
            self.info |= {"width": args.width, "height": args.height}
        else:
            self.info["features"] = 3 + 4 * args.joints

    def start(self):
        return {"chunks": 0}

    def step(self, session, speech, listen):
        time.sleep(self.args.render_ms / 1000)
        session["chunks"] += 1
        frames = []
        n = self.args.chunk_frames
        per = len(speech[: self.chunk_samples()]) // n
        for index in range(n):
            level = float(np.abs(speech[index * per : (index + 1) * per]).mean())
            heard = (
                0.0
                if listen is None
                else float(np.abs(listen[index * per : (index + 1) * per]).mean())
            )
            if self.args.media == "jpeg_frame":
                rgb = np.zeros((self.args.height, self.args.width, 3), np.uint8)
                rgb[..., 0] = min(255, int(level * 2000))
                rgb[..., 2] = min(255, int(heard * 2000))
                rgb[..., 1] = 40
                frames.append(jpeg_of(rgb))
            else:
                row = np.zeros(3 + 4 * self.args.joints, np.float32)
                row[3::4] = 1.0
                row[1] = level
                frames.append(row.tobytes())
        return frames


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--lookahead-ms", type=float, default=0)
    parser.add_argument("--listens", action="store_true")
    parser.add_argument(
        "--media", default="jpeg_frame", choices=["jpeg_frame", "bvh_motion"]
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--joints", type=int, default=4)
    parser.add_argument("--render-ms", type=float, default=20)
    args = parser.parse_args()
    setup_logging()

    def load():
        model = FakeModel(args)
        warm_up(model, args.warmup_chunks)
        return model

    serve(load, args.host, args.port)


if __name__ == "__main__":
    main()
