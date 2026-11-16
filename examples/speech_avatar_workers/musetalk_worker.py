# SPDX-License-Identifier: Apache-2.0
"""A MuseTalk 1.5 speech-avatar worker: the mouth of a prepared avatar redrawn from speech, --chunk-frames frames at 25 fps per request.

Run it in MuseTalk's own environment from anywhere; it changes into the
repository, whose code loads its models by relative path:

    python musetalk_worker.py --repo MuseTalk --port 8032 --avatar-dir <prepared avatar>

The avatar directory is what MuseTalk's realtime mode prepares
(scripts/realtime_inference.py, preparation: True): full_imgs/, mask/,
coords.pkl, mask_coords.pkl and latents.pt. Each chunk follows the realtime
mode's inference: Whisper-tiny encoder features of the speech window (the
chunk, --context-ms before it and 80 ms after it, since a frame's features
reach two frames to each side), one UNet step per frame on the avatar
frame's latent, the VAE decoder, and the face blended back into the avatar
frame. The state is the index of the next avatar frame, so the avatar's own
loop continues from chunk to chunk.

The picture shown while the agent is silent (/idle) is, by default, the
avatar's first frame with its mouth redrawn from silence, so a reply's first
chunk does not jump from the source picture's mouth to a generated one; a
Session's first chunk also fades from it over --fade-in-frames frames. With
--blend-at-output-size (the default) the avatar is scaled to the output
width once at load and each frame is blended at that size.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
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

FPS = 25
SAMPLE_RATE = 16000
# Whisper features run at 50 per second; a frame's features are the 10 around it (audio_padding_length_left = right = 2).
FEATURES_PER_S = 50
PADDING_LEFT = 2
PADDING_RIGHT = 2
LOOKAHEAD_SAMPLES = PADDING_RIGHT * SAMPLE_RATE // FPS
FRAME_INDEX = struct.Struct("<q")


class MuseTalkModel(ChunkModel):
    def __init__(self, args) -> None:
        os.chdir(args.repo)
        sys.path.insert(0, args.repo)
        import cv2
        import torch
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.utils import load_all_model
        from PIL import Image
        from transformers import WhisperModel

        self.torch = torch
        self.cv2 = cv2
        self.Image = Image
        self.blend = get_image_blending
        self.device = torch.device("cuda")
        vae, unet, pe = load_all_model(
            unet_model_path="models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="models/musetalkV15/musetalk.json",
            device=self.device,
        )
        self.pe = pe.half().to(self.device)
        vae.vae = vae.vae.half().to(self.device)
        unet.model = unet.model.half().to(self.device)
        self.vae, self.unet = vae, unet
        self.timesteps = torch.tensor([0], device=self.device)
        self.audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
        self.whisper = (
            WhisperModel.from_pretrained("models/whisper")
            .to(device=self.device, dtype=unet.model.dtype)
            .eval()
        )
        self.whisper.requires_grad_(False)
        avatar = args.avatar_dir
        names = sorted(
            os.listdir(os.path.join(avatar, "full_imgs")),
            key=lambda name: int(os.path.splitext(name)[0]),
        )
        self.frames = [cv2.imread(os.path.join(avatar, "full_imgs", n)) for n in names]
        self.masks = [cv2.imread(os.path.join(avatar, "mask", n)) for n in names]
        with open(os.path.join(avatar, "coords.pkl"), "rb") as source:
            self.coords = pickle.load(source)
        with open(os.path.join(avatar, "mask_coords.pkl"), "rb") as source:
            self.mask_coords = pickle.load(source)
        self.latents = torch.load(os.path.join(avatar, "latents.pt"))
        if (
            not len(self.frames)
            == len(self.masks)
            == len(self.coords)
            == len(self.latents)
        ):
            raise ValueError(f"{avatar} is not a complete prepared MuseTalk avatar")
        else:
            pass
        height, width = self.frames[0].shape[:2]
        self.size = (args.width, round(height * args.width / width))
        if args.blend_at_output_size and self.size[0] != width:
            self.scale_avatar(args.width / width)
        else:
            pass
        if args.compile:
            torch.backends.cudnn.benchmark = True
            self.unet_forward = torch.compile(unet.model)
            self.vae_decode = torch.compile(vae.vae.decode)
        else:
            self.unet_forward = unet.model
            self.vae_decode = vae.vae.decode
        self.fade_in_frames = args.fade_in_frames
        self.chunk_frames = args.chunk_frames
        self.context_samples = round(args.context_ms * SAMPLE_RATE / 1000)
        self.jpeg_quality = args.jpeg_quality
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.info = {
            "kind": "video",
            "model": "musetalk-1.5",
            "fps": FPS,
            "chunk_frames": self.chunk_frames,
            "context_samples": self.context_samples,
            "lookahead_samples": LOOKAHEAD_SAMPLES,
            "width": self.size[0],
            "height": self.size[1],
        }
        self.idle_frame = (
            self.frames[0] if args.idle == "source" else self.silent_frame()
        )
        self.idle = self.encode(self.idle_frame)
        warm_up(self, args.warmup_chunks)

    def scale_avatar(self, scale: float) -> None:
        """Scale the avatar's frames, masks and boxes by `scale` once, so each frame blends and encodes at the output size."""
        cv2 = self.cv2

        def box(coords):
            return [round(value * scale) for value in coords]

        size = (
            round(self.frames[0].shape[1] * scale),
            round(self.frames[0].shape[0] * scale),
        )
        self.frames = [
            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for frame in self.frames
        ]
        self.coords = [box(coords) for coords in self.coords]
        self.mask_coords = [box(coords) for coords in self.mask_coords]
        self.masks = [
            cv2.resize(
                mask,
                (
                    crop[2] - crop[0],
                    crop[3] - crop[1],
                ),
                interpolation=cv2.INTER_AREA,
            )
            for mask, crop in zip(self.masks, self.mask_coords)
        ]

    def encode(self, frame_bgr: np.ndarray) -> bytes:
        picture = self.Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        if picture.size != self.size:
            picture = picture.resize(self.size, self.Image.BILINEAR)
        else:
            pass
        out = io.BytesIO()
        picture.save(out, format="JPEG", quality=self.jpeg_quality)
        return out.getvalue()

    def silent_frame(self) -> np.ndarray:
        """The avatar's first frame with its mouth redrawn from silence."""
        window = self.window_samples()
        faces = self.faces(0, np.zeros(window, dtype=np.float32))
        return self.blended(0, faces[0])

    def idle_jpeg(self) -> bytes:
        return self.idle

    def blended(self, index: int, face: np.ndarray) -> np.ndarray:
        cycle = index % len(self.frames)
        x1, y1, x2, y2 = self.coords[cycle]
        face = self.cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
        return self.blend(
            self.frames[cycle].copy(),
            face,
            [x1, y1, x2, y2],
            self.masks[cycle],
            self.mask_coords[cycle],
        )

    def compose(self, item) -> bytes:
        index, face, weight = item
        frame = self.blended(index, face)
        if weight < 1:
            frame = self.cv2.addWeighted(frame, weight, self.idle_frame, 1 - weight, 0)
        else:
            pass
        return self.encode(frame)

    def features(self, speech: np.ndarray):
        """The 10 Whisper feature steps around each frame of the chunk, as the UNet reads them."""
        torch = self.torch
        mel = self.audio_processor.feature_extractor(
            speech, return_tensors="pt", sampling_rate=SAMPLE_RATE
        ).input_features.to(self.device, dtype=self.unet.model.dtype)
        hidden = self.whisper.encoder(mel, output_hidden_states=True).hidden_states
        hidden = torch.stack(hidden, dim=2)[0]
        per_frame = FEATURES_PER_S // FPS
        first = self.context_samples * FEATURES_PER_S // SAMPLE_RATE
        clips = []
        for frame in range(self.chunk_frames):
            center = first + frame * per_frame
            clips.append(
                hidden[
                    center
                    - per_frame * PADDING_LEFT : center
                    + per_frame * (PADDING_RIGHT + 1)
                ]
            )
        clips = torch.stack(clips)
        return clips.reshape(clips.shape[0], -1, clips.shape[-1])

    def faces(self, index: int, speech: np.ndarray) -> np.ndarray:
        """The chunk's generated faces (BGR uint8), its frames starting at avatar frame `index`."""
        torch = self.torch
        with torch.no_grad():
            audio = self.pe(self.features(speech))
            latents = torch.cat(
                [
                    self.latents[(index + k) % len(self.latents)]
                    for k in range(self.chunk_frames)
                ]
            ).to(device=self.device, dtype=self.unet.model.dtype)
            predicted = self.unet_forward(
                latents, self.timesteps, encoder_hidden_states=audio
            ).sample
            # MuseTalk's VAE.decode_latents, through the (compiled) decoder.
            latents = predicted.to(dtype=self.vae.vae.dtype) / self.vae.scaling_factor
            image = (self.vae_decode(latents).sample.float() / 2 + 0.5).clamp(0, 1)
            image = (image * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
            return image.cpu().numpy()[..., ::-1]

    def render(self, state: bytes, speech: np.ndarray):
        if not state:
            index = 0
        elif len(state) == FRAME_INDEX.size:
            (index,) = FRAME_INDEX.unpack(state)
        else:
            raise BadRequest("not a MuseTalk state")
        faces = self.faces(index, speech)
        # A Session's first chunk fades in from the idle picture the client shows until then.
        fade = self.fade_in_frames if index == 0 else 0
        jpegs = list(
            self.pool.map(
                self.compose,
                [
                    (index + k, face, min(1.0, (k + 1) / (fade + 1)))
                    for k, face in enumerate(faces)
                ],
            )
        )
        return FRAME_INDEX.pack(index + self.chunk_frames), jpegs


def parse_args(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--avatar-dir", required=True)
    parser.add_argument("--chunk-frames", type=int, default=5)
    parser.add_argument(
        "--context-ms",
        type=float,
        default=1000.0,
        help="speech before the chunk the Whisper encoder hears",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="output width; the height keeps the avatar's aspect ratio",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--idle",
        choices=["generated", "source"],
        default="generated",
        help="the picture while the agent is silent: the first avatar frame with its mouth redrawn from silence, or the avatar frame as it is",
    )
    parser.add_argument(
        "--fade-in-frames",
        type=int,
        default=3,
        help="a Session's first chunk fades from the idle picture over this many frames",
    )
    parser.add_argument(
        "--blend-at-output-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale the avatar to the output width once at load and blend each frame at that size",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the UNet and the VAE decoder (compiled during the warm-up)",
    )
    args = parser.parse_args(argv)
    args.repo = os.path.abspath(args.repo)
    args.avatar_dir = os.path.abspath(args.avatar_dir)
    return args


def main() -> None:
    args = parse_args()
    setup_logging()
    serve(lambda: MuseTalkModel(args), args.host, args.port)


if __name__ == "__main__":
    main()
