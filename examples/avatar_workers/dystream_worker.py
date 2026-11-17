# SPDX-License-Identifier: Apache-2.0
"""Chunk worker of DyStream (dyadic talking and listening head; the release has no LICENSE file): 25 fps, 512x512, ten frames per chunk.

Runs in the model's own environment from its checkout (it imports the
release's app.py for model loading and image processing):

  cd DyStream && python /path/to/dystream_worker.py --port 8123 --avatar ref.png

The authors released offline inference only. It already generates one frame
per autoregressive step from the 94 previous motion latents and the audio
features of both parties up to that frame, so the loop streams as it is; what
does not stream is the audio front end, wav2vec2 over the whole clip. Here:

- each chunk runs both parties' wav2vec2 once over a window of 106 frames
  (4.24 s) ending `lookahead` frames past the chunk, the way training
  encoded each 96-frame window on its own (normalized per window, 80 zero
  samples appended); a frame's features therefore hear at most `lookahead`
  frames of future audio instead of the whole clip;
- the per-frame step is the release's `one_clip_only_inference` with those
  features; its own two wav2vec2 passes per frame, whose result the release
  discards, are skipped;
- the 94 past latents start as the portrait's latent, and audio before the
  Session is silence, as the release pads its input;
- frames render against the Session's first generated latent, as the
  release renders against its first output frame, in one batch per chunk.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_server import (  # noqa: E402
    SAMPLE_RATE_HZ,
    SessionModel,
    base_parser,
    logger,
    serve,
    setup_logging,
    warm_up,
)

FPS = 25
FRAME_SAMPLES = SAMPLE_RATE_HZ // FPS
PAST = 94
FEATURE_WINDOW = PAST + 12


class DyStream(SessionModel):
    def __init__(self, args) -> None:
        super().__init__()
        os.chdir(args.repo)
        sys.path.insert(0, str(Path(args.repo).resolve()))
        sys.argv = sys.argv[:1]
        import app
        import torch
        import torch.nn.functional as F
        from diffusers import FlowMatchEulerDiscreteScheduler
        from omegaconf import OmegaConf
        from PIL import Image

        self.torch, self.F, self.app = torch, F, app
        self.args = args
        app.load_dystream_model()
        app.load_visualization_model()
        model = app._dystream_model
        if app._dystream_ema is not None:
            app._dystream_ema.to(app.DEVICE)
            app._dystream_ema.copy_to(model.parameters())
        model.cfg_audio, model.cfg_audio_other = args.cfg_audio, args.cfg_audio_other
        model.cfg_anchor, model.cfg_all = args.cfg_anchor, args.cfg_all
        # The release recomputes both parties' features every frame and then uses the precomputed ones.
        model.get_audio2face_fea = lambda *unused: None
        model.get_audio2face_fea_other = lambda *unused: None
        self.model = model
        self.scheduler_kwargs = OmegaConf.to_container(
            app._dystream_cfg.noise_scheduler_kwargs, resolve=True
        )
        self.Scheduler = FlowMatchEulerDiscreteScheduler
        resized, _, motion_latent = app.process_image(Image.open(args.avatar))
        self.anchor = motion_latent.to(app.DEVICE).reshape(1, 1, -1)
        vis = app._vis_ctx
        with torch.no_grad():
            self.face_feat = vis["face_encoder"](
                vis["transform"](resized.convert("RGB")).unsqueeze(0).to(app.DEVICE)
            )
        self.lookahead = args.lookahead_frames
        self.chunk = args.chunk_frames
        self.info = {
            "model": "dystream",
            "fps": FPS,
            "chunk_frames": self.chunk,
            "lookahead_samples": self.lookahead * FRAME_SAMPLES,
            "listens": True,
            "media": "jpeg_frame",
            "width": 512,
            "height": 512,
        }
        self.timings: list = []
        self.dtype = torch.bfloat16 if args.bf16 else None
        self.gpt = self.head = None
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def start(self):
        torch = self.torch
        history = (PAST + self.lookahead) * FRAME_SAMPLES
        return {
            "past_motion": self.anchor.repeat(1, PAST, 1).clone(),
            "speech": np.zeros(history, np.float32),
            "user": np.zeros(history, np.float32),
            "first": None,
            "scheduler": self.Scheduler(**self.scheduler_kwargs),
            "generator": torch.Generator(device="cuda").manual_seed(self.args.seed),
        }

    def features(self, encoder, audio: np.ndarray):
        """The release's wav2vec2 front end over one window: processor normalization, 80 zero samples, 50 to 25 Hz."""
        torch, F, model = self.torch, self.F, self.model
        inputs = model.audio_processor(
            [audio], sampling_rate=16000, return_tensors="pt", padding=True
        ).to("cuda")
        values = torch.cat(
            [inputs.input_values, torch.zeros([1, 80], device="cuda")], dim=-1
        )
        high = encoder(values)["high_level"]
        return F.interpolate(
            high.transpose(1, 2),
            scale_factor=(FPS / 50),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def step(self, session, speech, listen):
        torch, model = self.torch, self.model
        start = time.perf_counter()
        chunk_samples = self.chunk * FRAME_SAMPLES
        # The window: PAST frames before the chunk, the chunk and the look-ahead, of each party.
        keep = PAST * FRAME_SAMPLES
        speech_window = np.concatenate([session["speech"][-keep:], speech])
        session["speech"] = np.concatenate([session["speech"], speech[:chunk_samples]])[
            -keep:
        ]
        user_window = np.concatenate(
            [
                session["user"][-keep:],
                listen[-chunk_samples:],
                np.zeros(self.lookahead * FRAME_SAMPLES, np.float32),
            ]
        )
        session["user"] = np.concatenate([session["user"], listen[-chunk_samples:]])[
            -keep:
        ]
        latents = []
        autocast = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.dtype
            else torch.autocast("cuda", enabled=False)
        )
        with torch.no_grad(), autocast:
            own = self.features(model.audio_encoder_face, speech_window)
            other = self.features(model.audio_encoder_face_other, user_window)
            if own.shape[1] < FEATURE_WINDOW or other.shape[1] < FEATURE_WINDOW:
                raise RuntimeError(
                    f"features of {own.shape[1]} and {other.shape[1]} frames, need {FEATURE_WINDOW}"
                )
            torch.manual_seed(
                int(
                    torch.randint(
                        0,
                        2**31 - 1,
                        (1,),
                        generator=session["generator"],
                        device="cuda",
                    )
                )
            )
            for index in range(self.chunk):
                # Frame f = chunk start + index sits at window index PAST + index; the release's window for it runs from f - 94 to f + 1.
                low = index
                step_frame = (
                    self.fast_frame if self.args.fast else model.one_clip_only_inference
                )
                out = step_frame(
                    per_compute_audio_feature=own[:, low : low + PAST + 2].float(),
                    per_compute_audio_other_feature=other[
                        :, low : low + PAST + 2
                    ].float(),
                    audio_self=torch.zeros(1, 1, device="cuda"),
                    past_audio_self=None,
                    audio_other=None,
                    past_audio_other=None,
                    anchor_latent=self.anchor,
                    past_motion=session["past_motion"],
                    gen_frames=1,
                    noise_scheduler=session["scheduler"],
                    num_inference_steps=self.args.steps,
                ).float()
                session["past_motion"] = torch.cat(
                    [session["past_motion"], out], dim=1
                )[:, -PAST:]
                latents.append(out[:, 0])
            latents = torch.cat(latents, dim=0)
            if session["first"] is None:
                session["first"] = latents[0:1].clone()
            generated = time.perf_counter()
            pixels = self.draw_frames(session["first"], latents)
        jpegs = [self.encode(frame) for frame in pixels]
        self.timings.append((generated - start, time.perf_counter() - generated))
        return jpegs

    def fast_frame(
        self,
        *,
        per_compute_audio_feature,
        per_compute_audio_other_feature,
        anchor_latent,
        past_motion,
        noise_scheduler,
        num_inference_steps,
        **unused,
    ):
        """The release's `one_clip_only_inference` for one frame, without the guidance branches whose scale is 0 (the anchor one at the release's settings): 4 rows instead of 5, the same arithmetic per row."""
        torch, model = self.torch, self.model
        own = per_compute_audio_feature[:, 1:]
        other = per_compute_audio_other_feature[:, 1:]
        seq_len = own.shape[1]
        t = model.inpainting_length
        a = model.audio_proj(own)
        o = model.audio_other_proj(other)
        fuse = model.audio_audioother_fusion
        hidden = {
            "none": fuse(torch.cat([a * 0, o * 0], dim=-1)),
            "audio": fuse(torch.cat([a * 1, o * 0], dim=-1)),
            "other": fuse(torch.cat([a * 0, o * 1], dim=-1)),
            "all": fuse(torch.cat([a * 1, o * 1], dim=-1)),
        }
        anchor = model.anchor_embed(anchor_latent)
        # (audio rows, anchor scale, guidance weight) per branch; the first is the unconditional one.
        branches = [
            ("none", 0, None),
            ("none", 1, model.cfg_anchor),
            ("audio", 0, model.cfg_audio),
            ("other", 0, model.cfg_audio_other),
            ("all", 1, model.cfg_all),
        ]
        branches = [b for b in branches if b[2] is None or b[2] != 0]
        n = len(branches)
        mask = model.generate_causal_mask(seq_len, own.device)
        cross = model.generate_cross_causal_mask(seq_len, own.device)
        x = model.face_embed(past_motion)[:, :t].repeat(n, 1, 1)
        audio_in = torch.cat([hidden[name][:, :t] for name, _, _ in branches], dim=0)
        anchor_in = torch.cat([anchor * scale for _, scale, _ in branches], dim=0)
        if self.gpt is None:
            masks = (mask[:t, :t].clone(), cross[:t, :t].clone())

            def gpt_core(x, audio_in, anchor_in):
                for block in model.blocks:
                    x = block(x, audio_in, anchor_in, *masks)
                return model.output_proj(model.output_norm(x[:, -1:]))

            def head(latent, gpt, temb):
                return model.diffusion_head(latent, gpt, temb=temb)

            compile_ = (
                (lambda f: torch.compile(f, mode=self.args.compile_mode))
                if self.args.compile
                else (lambda f: f)
            )
            self.gpt, self.head = compile_(gpt_core), compile_(head)
        gpt = self.gpt(x, audio_in, anchor_in).clone()
        noise_scheduler.set_timesteps(num_inference_steps, device=own.device)
        latent = torch.randn_like(gpt[:1])
        for timestep in noise_scheduler.timesteps:
            t_batch = torch.full((1,), timestep, device=own.device, dtype=torch.long)
            out = (
                self.head(latent, gpt, model.time_embed(t_batch).unsqueeze(1))
                .clone()
                .chunk(n, dim=0)
            )
            pred = out[0] + sum(
                weight * (row - out[0])
                for (_, _, weight), row in zip(branches[1:], out[1:])
            )
            if noise_scheduler.step_index is None:
                noise_scheduler._init_step_index(timestep)
            sigma = noise_scheduler.sigmas[noise_scheduler.step_index].to(
                device=own.device
            )
            latent = noise_scheduler.step(
                (latent - pred) / (sigma + 1e-9), timestep, latent, return_dict=False
            )[0]
        return latent

    def draw_frames(self, first, latents):
        torch = self.torch
        vis = self.app._vis_ctx
        n = latents.shape[0]
        try:
            target = vis["flow_estimator"](first.repeat(n, 1), latents)
            pictures = vis["face_generator"](
                target,
                (
                    self.face_feat.repeat(n, *[1] * (self.face_feat.dim() - 1))
                    if torch.is_tensor(self.face_feat)
                    else self.face_feat
                ),
            )
        except Exception:
            pictures = torch.cat(
                [
                    vis["face_generator"](
                        vis["flow_estimator"](first, latents[i : i + 1]), self.face_feat
                    )
                    for i in range(n)
                ]
            )
        return (
            ((pictures.float().permute(0, 2, 3, 1) + 1) / 2 * 255)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def encode(self, rgb: np.ndarray) -> bytes:
        import cv2

        ok, data = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(rgb[:, :, ::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        assert ok
        return data.tobytes()


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--lookahead-frames", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg-audio", type=float, default=0.5)
    parser.add_argument("--cfg-audio-other", type=float, default=0.5)
    parser.add_argument("--cfg-anchor", type=float, default=0.0)
    parser.add_argument("--cfg-all", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="with --fast: torch.compile the per-frame transformer and diffusion head (static shapes)",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the guidance branches of weight 0 (4 rows a step instead of 5)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="autocast the motion model and the renderer to bfloat16",
    )
    parser.add_argument(
        "--tf32", action="store_true", help="let fp32 matmuls and convolutions use TF32"
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    args.avatar = str(Path(args.avatar).resolve())
    setup_logging()

    def load():
        model = DyStream(args)
        times = warm_up(model, args.warmup_chunks)
        logger.info("warm-up chunk times (ms): %s", [round(t) for t in times])
        return model

    serve(load, args.host, args.port)


if __name__ == "__main__":
    main()
