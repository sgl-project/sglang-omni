# SPDX-License-Identifier: Apache-2.0
"""A SoulX-FlashHead (Lite) speech-avatar worker: a talking head from one portrait, a chunk of frame_num - 9 frames at 25 fps per request.

Run it in FlashHead's own environment from anywhere; it changes into the
repository, whose code reads its configs by relative path:

    python flashhead_worker.py --repo SoulX-FlashHead --port 8031 \
        --ckpt-dir SoulX-FlashHead/models/SoulX-FlashHead-1_3B \
        --wav2vec-dir SoulX-FlashHead/models/wav2vec2-base-960h \
        --cond-image portrait.png

Each chunk follows FlashHead's streaming mode (generate_video.py
--audio_encode_mode stream): wav2vec2 reads the last cached_audio_duration
(8 s) of speech, the model generates frame_num frames whose first
motion_frames_num (9) repeat the previous chunk's last frames, and those 9 are
dropped. The state is the motion latents of the previous chunk's last frames,
so a Session's next chunk continues from its last picture.
"""

from __future__ import annotations

import io
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_server import (  # noqa: E402
    BadRequest,
    ChunkModel,
    base_parser,
    serve,
    setup_logging,
    warm_up,
)

LATENT_HEADER = struct.Struct("<4I")


class FlashHeadModel(ChunkModel):
    def __init__(self, args) -> None:
        os.chdir(args.repo)
        sys.path.insert(0, args.repo)
        import torch
        from flash_head import inference
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.inference = inference
        inference.infer_params["frame_num"] = args.frame_num
        self.pipeline = inference.get_pipeline(
            world_size=1,
            ckpt_dir=args.ckpt_dir,
            model_type="lite",
            wav2vec_dir=args.wav2vec_dir,
        )
        inference.get_base_data(
            self.pipeline,
            cond_image_path_or_dir=args.cond_image,
            base_seed=args.seed,
            use_face_crop=False,
        )
        params = inference.get_infer_params()
        self.params = params
        fps = params["tgt_fps"]
        self.motion_frames = params["motion_frames_num"]
        self.chunk_frames = params["frame_num"] - self.motion_frames
        cached = params["sample_rate"] * params["cached_audio_duration"]
        chunk_samples = self.chunk_frames * params["sample_rate"] // fps
        self.audio_end = params["cached_audio_duration"] * fps
        self.audio_start = self.audio_end - params["frame_num"]
        self.first_latents = self.pipeline.ref_img_latent[:, :1].clone()
        self.latent_dtype = self.first_latents.dtype
        self.jpeg_quality = args.jpeg_quality
        self.encoder = ThreadPoolExecutor(max_workers=8)
        reference = self.pipeline.original_color_reference[0, :, 0]
        picture = (
            ((reference.float() + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        self.idle = self.encode(picture)
        self.info = {
            "kind": "video",
            "model": "flashhead-lite",
            "fps": fps,
            "chunk_frames": self.chunk_frames,
            "context_samples": cached - chunk_samples,
            "lookahead_samples": 0,
            "width": params["width"],
            "height": params["height"],
        }
        warm_up(self, args.warmup_chunks)

    def encode(self, frame: np.ndarray) -> bytes:
        out = io.BytesIO()
        self.Image.fromarray(frame).save(out, format="JPEG", quality=self.jpeg_quality)
        return out.getvalue()

    def idle_jpeg(self) -> bytes:
        return self.idle

    def latents_of(self, state: bytes):
        torch = self.torch
        if not state:
            return self.first_latents.clone()
        elif len(state) < LATENT_HEADER.size:
            raise BadRequest("not a FlashHead state")
        else:
            shape = LATENT_HEADER.unpack_from(state)
        data = state[LATENT_HEADER.size :]
        if (
            len(data) != 2 * int(np.prod(shape))
            or shape[0] != self.first_latents.shape[0]
        ):
            raise BadRequest(f"not a FlashHead state of this checkpoint: {shape}")
        else:
            values = torch.frombuffer(bytearray(data), dtype=torch.int16).view(
                self.latent_dtype
            )
        return values.reshape(shape).to(self.pipeline.device)

    def state_of(self, latents) -> bytes:
        torch = self.torch
        latents = latents.detach().to(self.latent_dtype).contiguous().cpu()
        return (
            LATENT_HEADER.pack(*latents.shape)
            + latents.view(torch.int16).numpy().tobytes()
        )

    def render(self, state: bytes, speech: np.ndarray):
        torch = self.torch
        pipeline = self.pipeline
        pipeline.latent_motion_frames = self.latents_of(state)
        with torch.no_grad():
            embedding = self.inference.get_audio_embedding(
                pipeline, speech, self.audio_start, self.audio_end
            )
            video = self.inference.run_pipeline(pipeline, embedding)
            frames = video[self.motion_frames :].to(torch.uint8).cpu().numpy()
        next_state = self.state_of(pipeline.latent_motion_frames)
        return next_state, list(self.encoder.map(self.encode, frames))


def main() -> None:
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--wav2vec-dir", required=True)
    parser.add_argument("--cond-image", required=True)
    parser.add_argument(
        "--frame-num",
        type=int,
        default=33,
        help="Frames per generation, 1 + 8k: 33 (the release) makes 24 new frames a chunk, 17 makes 8.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    for name in ("repo", "ckpt_dir", "wav2vec_dir", "cond_image"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    setup_logging()
    serve(lambda: FlashHeadModel(args), args.host, args.port)


if __name__ == "__main__":
    main()
