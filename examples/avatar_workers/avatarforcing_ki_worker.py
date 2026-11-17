# SPDX-License-Identifier: Apache-2.0
"""Chunk worker of Avatar Forcing (TaekyungKi/AvatarForcing, CC BY-NC 4.0): a talking and listening head at 25 fps, 512x512, ten frames per chunk.

Runs in the model's own environment from its checkout (it imports the
release's `models` package):

  cd AvatarForcing && python /path/to/avatarforcing_ki_worker.py --port 8122 --avatar data/rumi.jpg

The release generates offline: wav2vec2 over the whole clip, a 50-frame first
block, then 10-frame blocks with a KV cache. A Session here runs the same
blocks one chunk at a time:

- its first block is the release's 50-frame block over silence, so the KV
  cache and the two context latents are primed before the first chunk;
- each chunk is one release block: the frames [t-2, t+10) of both parties'
  wav2vec2 features come from a rolling 2 s window of each party's audio (the
  clip length the model was trained on), the agent's ending `lookahead` past
  the block, the user's ending at the newest audio the node sent;
- the user's video is not sent: the user motion is the model's null
  condition (zeros), which classifier-free guidance was trained with;
- the last denoising step, at t = 0, is skipped: the release's solver
  returns its input unchanged there, so the forward pass it runs is unused;
- RoPE tables are extended past the release's 1024 positions (41 s), so a
  Session may run for about 40 minutes.
"""

from __future__ import annotations

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
BLOCK = 10
FIRST_BLOCK = 50
CONTEXT = 2
FRAME_SAMPLES = SAMPLE_RATE_HZ // FPS
CHUNK_SAMPLES = BLOCK * FRAME_SAMPLES
WINDOW_FRAMES = 50
WINDOW_SAMPLES_AUDIO = WINDOW_FRAMES * FRAME_SAMPLES
ROPE_POSITIONS = 65536


def normalized(audio: np.ndarray) -> np.ndarray:
    """Wav2Vec2FeatureExtractor's zero-mean unit-variance normalization of one window; silence stays silence."""
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)


class AvatarForcingKI(SessionModel):
    def __init__(self, args) -> None:
        super().__init__()
        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, str(Path(args.repo).resolve()))
        import inference as release

        self.torch = torch
        self.args = args
        self.device = "cuda:0"
        opt = OmegaConf.merge(
            OmegaConf.load(str(Path(args.repo) / "configs/inference.yaml")),
            OmegaConf.create(
                {
                    "mae_ckpt_path": str(
                        Path(args.repo) / "pretrained_dir/motion_autoencoder.pth"
                    ),
                    "ckpt_path": str(
                        Path(args.repo) / "pretrained_dir/flow_transformer.pth"
                    ),
                    "wav2vec_model_path": str(
                        Path(args.repo) / "pretrained_dir/wav2vec2-base-960h"
                    ),
                    "rank": 0,
                    "ngpus": 1,
                }
            ),
        )
        self.opt = opt
        torch.manual_seed(args.seed)
        # The release's InferenceAgent reads its checkpoints from the script's global `opt`.
        release.opt = opt
        agent = release.InferenceAgent(opt)
        self.G = agent.G
        self.G.eval()
        self.extend_rope(ROPE_POSITIONS)
        self.lookahead_frames = args.lookahead_frames
        lookahead = self.lookahead_frames * FRAME_SAMPLES
        self.info = {
            "model": "avatarforcing-ki",
            "fps": FPS,
            "chunk_frames": BLOCK,
            "lookahead_samples": lookahead,
            "listens": True,
            "media": "jpeg_frame",
            "width": opt.input_size,
            "height": opt.input_size,
        }
        face = agent.data_processor.preprocess_face(args.avatar)
        ref = (
            agent.data_processor.transform(image=face)["image"]
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            s_r, r_s_lambda, s_r_feats = self.G.encode_image_into_latent(ref)
            self.s_r = s_r
            self.r_s = self.G.motion_autoencoder.dec.direction(r_s_lambda)
            self.s_r_feats = [f.repeat_interleave(BLOCK, dim=0) for f in s_r_feats]
        self.steps = np.linspace(opt.num_train_timestep, 0, args.nfe - 1).tolist()
        self.G.denoising_step_list = torch.tensor(self.steps)
        self.timings: list = []
        if args.compile:
            self.G.flow_transformer.forward_with_precomputed = torch.compile(
                self.G.flow_transformer.forward_with_precomputed, dynamic=False
            )
            self.G.motion_autoencoder.dec = torch.compile(
                self.G.motion_autoencoder.dec, dynamic=False
            )

    def extend_rope(self, positions: int) -> None:
        from models.avatarforcing import flow_transformer

        for module in self.G.modules():
            if hasattr(module, "freqs_cis") and hasattr(module, "head_dim"):
                module.freqs_cis = flow_transformer.precompute_freqs_cis(
                    module.head_dim, positions
                ).to(module.freqs_cis.device)

    def features(self, audio: np.ndarray):
        """wav2vec2 features of one 2 s window, one per frame."""
        torch = self.torch
        a = torch.from_numpy(normalized(audio)).float().unsqueeze(0).to(self.device)
        return self.G.audio_encoder.inference(a, seq_len=WINDOW_FRAMES)

    def start(self):
        torch = self.torch
        G = self.G
        generator = torch.Generator(device=self.device).manual_seed(self.args.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            G.initialize_kv_cache(batch_size=1, dtype=torch.float32, device=self.device)
            silence = self.features(np.zeros(WINDOW_SAMPLES_AUDIO, np.float32))
            motion = torch.zeros(1, FIRST_BLOCK, self.opt.dim_w, device=self.device)
            x_t = torch.randn(
                1, FIRST_BLOCK, self.opt.dim_w, device=self.device, generator=generator
            )
            c, wr, adaln = G.prepare_cfg_condition(
                silence, silence, motion, self.r_s, seq_len=FIRST_BLOCK, context_len=0
            )
            x_t = self.denoise(x_t, c, wr, adaln, start_pos=0, use_kv_cache=False)
        return {
            "kv": G.kv_cache,
            "x_t": x_t[:, -CONTEXT:].clone(),
            "t": FIRST_BLOCK,
            "speech": np.zeros(WINDOW_SAMPLES_AUDIO, np.float32),
            "user": np.zeros(WINDOW_SAMPLES_AUDIO, np.float32),
            "generator": generator,
        }

    def denoise(self, x_t, c, wr, adaln, *, start_pos, use_kv_cache):
        """The release's sampling loop over one block, skipping the t = 0 step it leaves unchanged, then its KV cache update."""
        G = self.G
        for index, timestep in enumerate(self.steps):
            if timestep == 0:
                continue
            x_t = G.solve_cfg(
                B=1,
                index=index,
                current_timestep=timestep,
                x_t=x_t,
                precomputed_c=c,
                precomputed_wr=wr,
                precomputed_adaLN=adaln,
                start_pos=start_pos,
                context_len=CONTEXT if use_kv_cache else 0,
                use_kv_cache=use_kv_cache,
                a_cfg_scale=self.args.a_cfg,
                u_cfg_scale=self.args.u_cfg,
            )
        G.update_kv_cache(
            final_latents=x_t,
            precomputed_c=c,
            precomputed_wr=wr,
            precomputed_adaLN=adaln,
            start_pos=start_pos,
        )
        return x_t

    def step(self, session, speech, listen):
        torch = self.torch
        G = self.G
        start = time.perf_counter()
        lookahead = self.lookahead_frames * FRAME_SAMPLES
        keep = WINDOW_SAMPLES_AUDIO - len(speech)
        speech_window = np.concatenate([session["speech"][-keep:], speech])
        session["speech"] = np.concatenate([session["speech"], speech[:CHUNK_SAMPLES]])[
            -WINDOW_SAMPLES_AUDIO:
        ]
        session["user"] = np.concatenate([session["user"], listen[-CHUNK_SAMPLES:]])[
            -WINDOW_SAMPLES_AUDIO:
        ]
        G.kv_cache = session["kv"]
        t = session["t"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            last = WINDOW_FRAMES - self.lookahead_frames
            wa = self.features(speech_window)[:, last - BLOCK - CONTEXT : last]
            wa_user = self.features(session["user"])[:, -(BLOCK + CONTEXT) :]
            motion = torch.zeros(1, BLOCK + CONTEXT, self.opt.dim_w, device=self.device)
            noise = torch.randn(
                1,
                BLOCK,
                self.opt.dim_w,
                device=self.device,
                generator=session["generator"],
            )
            x_t = torch.cat([session["x_t"], noise], dim=1)
            c, wr, adaln = G.prepare_cfg_condition(
                wa, wa_user, motion, self.r_s, seq_len=BLOCK, context_len=CONTEXT
            )
            x_t = self.denoise(
                x_t, c, wr, adaln, start_pos=t - CONTEXT, use_kv_cache=True
            )
            r_d = x_t[:, -BLOCK:]
            frames = G.decode_block(
                r_d, self.s_r.unsqueeze(1), self.s_r_feats, BLOCK, 1
            )
            pixels = (
                ((frames.float().clamp(-1, 1) + 1) * 127.5)
                .round()
                .to(torch.uint8)
                .reshape(BLOCK, 3, self.opt.input_size, self.opt.input_size)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
        generated = time.perf_counter()
        session["x_t"] = x_t[:, -CONTEXT:].clone()
        session["t"] = t + BLOCK
        session["kv"] = G.kv_cache
        jpegs = [self.encode(frame) for frame in pixels]
        self.timings.append((generated - start, time.perf_counter() - generated))
        return jpegs

    def encode(self, rgb: np.ndarray) -> bytes:
        import cv2

        ok, data = cv2.imencode(
            ".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality]
        )
        assert ok
        return data.tobytes()


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument(
        "--repo", default=".", help="The AvatarForcing checkout with pretrained_dir/"
    )
    parser.add_argument("--avatar", required=True, help="The avatar's portrait")
    parser.add_argument("--nfe", type=int, default=10)
    parser.add_argument("--a-cfg", type=float, default=2.0)
    parser.add_argument("--u-cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument(
        "--lookahead-frames",
        type=int,
        default=2,
        help="Frames of the agent's audio past a block its features hear",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    setup_logging()

    def load():
        model = AvatarForcingKI(args)
        times = warm_up(model, args.warmup_chunks)
        logger.info("warm-up chunk times (ms): %s", [round(t) for t in times])
        return model

    serve(load, args.host, args.port)


if __name__ == "__main__":
    main()
