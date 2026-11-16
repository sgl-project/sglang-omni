# SPDX-License-Identifier: Apache-2.0
"""A StreamTalk speech-avatar worker: SMPL-X co-speech motion at 30 fps, one block of the last --block-frames frames of a 2 s window per request.

Run it in StreamTalk's own environment (Python 3.9, torch 2.0):

    python streamtalk_worker.py --repo StreamTalk --port 8033 \
        --ckpt StreamTalk/checkpoints/pretrained/streamtalk_speaker2_combined_e0946_cfg3.pt \
        --wavlm-model <wavlm-libri-clean-100h-large> --smplx-model StreamTalk/assets/SMPLX_NEUTRAL_2020.npz \
        --retrieval-data-folder <BEAT2>/smplxflame_30 --retrieval-csv-file <BEAT2>/train_test_split.csv

Each block runs StreamTalk's per-window step (Scripts/FM/TestFixedExpressions_keyposes.py):
WavLM features of the 2 s of speech ending at the block's end, the facial
expression model, a first flow-matching pass, the key pose retrieved from
the speaker's BEAT2 motion for the window's last 8 frames, and the refining
pass anchored on it. The release slides its 60-frame window by 52 frames
with an 8-frame seed; this worker slides it by --block-frames (15 by
default, 0.5 s) and seeds the window's first 60 - block frames with the
motion it generated before, so a block needs no speech past its end. The
state is that seed (global 6D rotations and root translation). A Session's
first block is seeded with the example clip's first frames, as the release
seeds its first window.

Sliding by 15 frames rather than 52 makes the motion move about half again
as much from frame to frame, so by default the worker (a) retrieves and
anchors a key pose once per 60 frames rather than every block, as the
release does about once per window (blocks without one skip the refining
pass), and (b) runs a causal One-Euro filter (2 Hz) over the parent-relative
rotations and root translation, the jaw excepted, seeding the next window
with the filtered frames. It looks at no frame ahead; the body trails the
unfiltered motion by about 65 ms, the jaw not at all. --keypose-every 0
--smooth-min-cutoff 0 --no-seed-smoothed is the unsmoothed worker. The state
also carries the frames since the last key pose and the filter's last frame
and derivative.
"""

from __future__ import annotations

import argparse
import os
import sys

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

FPS = 30
WINDOW_FRAMES = 60
WINDOW_SAMPLES = 2 * 16000
POSE6D = 55 * 6
TRANS = 3
# The release retrieves a key pose for one of the window's last 8 frames.
KEY_POSE_FRAMES = 8
FEATURES6D = POSE6D + TRANS
# The jaw's 6D rotation (joint 22); the smoother leaves it alone so the mouth keeps up with the speech.
JAW6D = slice(22 * 6, 23 * 6)


class OneEuro:
    """A causal One-Euro filter (Casiez et al., CHI 2012) over the per-frame motion features (parent-relative 6D rotations and the root translation), one frame at a time.

    Its state is the last filtered frame and its filtered derivative, so a
    worker keeps none of its own: the Session's state carries them.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.keep = np.ones(FEATURES6D, dtype=bool)
        self.keep[JAW6D] = False

    @staticmethod
    def alpha(cutoff, rate: float = FPS):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * rate)

    def run(self, frames: np.ndarray, last: np.ndarray, derivative: np.ndarray):
        """The filtered frames and the filter's next (last, derivative), from the raw [n, FEATURES6D] frames."""
        out = np.empty_like(frames)
        alpha_d = self.alpha(self.d_cutoff)
        for index, frame in enumerate(frames):
            raw_derivative = (frame - last) * FPS
            derivative = alpha_d * raw_derivative + (1 - alpha_d) * derivative
            cutoff = self.min_cutoff + self.beta * np.abs(derivative)
            alpha = self.alpha(cutoff)
            last = np.where(self.keep, alpha * frame + (1 - alpha) * last, frame)
            out[index] = last
        return out, last, derivative


class StreamTalkModel(ChunkModel):
    def __init__(self, args) -> None:
        sys.path.insert(0, os.path.join(args.repo, "Scripts"))
        sys.path.insert(0, os.path.join(args.repo, "Scripts", "FM"))
        os.chdir(os.path.join(args.repo, "Scripts", "FM"))
        import TestFixedExpressions_keyposes as release
        import torch
        from Common.SMPLX_Util import (
            LocalToGlobalAA,
            global_to_local_batch,
            local_to_global_batch,
        )
        from Common.SMPLXConfig import GetBodyPartIndices
        from Common.Util import (
            NormalizeRot6d,
            axis_angle_to_rotation_6d,
            matrix_to_rotation_6d,
            rotation_6d_to_matrix,
            rotation_matrix_to_axis_angle,
        )
        from FindNearest import MakeKDTree
        from Net import (
            DiffusionDITNetPartsFixedExpressions2PostNormInteraction2,
            SimpleSpeechModel,
        )

        self.torch = torch
        self.NormalizeRot6d = NormalizeRot6d
        self.rotation_6d_to_matrix = rotation_6d_to_matrix
        self.matrix_to_rotation_6d = matrix_to_rotation_6d
        self.rotation_matrix_to_axis_angle = rotation_matrix_to_axis_angle
        self.global_to_local_batch = global_to_local_batch
        self.local_to_global_batch = local_to_global_batch
        if not 0 < args.block_frames <= WINDOW_FRAMES - KEY_POSE_FRAMES:
            raise ValueError(
                f"--block-frames must be in 1..{WINDOW_FRAMES - KEY_POSE_FRAMES}"
            )
        else:
            self.block = args.block_frames
        self.seed_frames = WINDOW_FRAMES - self.block
        self.cfg_scale = args.cfg_scale
        # 0: retrieve a key pose for every block, as the release does for every window.
        self.keypose_every = args.keypose_every or self.block
        self.smoother = (
            OneEuro(args.smooth_min_cutoff, args.smooth_beta)
            if args.smooth_min_cutoff > 0
            else None
        )
        self.seed_smoothed = args.seed_smoothed
        release.setup_seed(args.seed)
        self.indices = {
            "hand_indices": GetBodyPartIndices("hand", "rot6d"),
            "upper_indices": GetBodyPartIndices("upper", "rot6d"),
            "lower_indices": GetBodyPartIndices("lower", "rot6d"),
            "face_joint_indices": GetBodyPartIndices("face", "rot6d"),
            "trans_indices": np.array([330, 331, 332]),
        }
        self.speech_model = SimpleSpeechModel().cuda().requires_grad_(False).eval()
        self.speech_model.load_state_dict(
            torch.load(release.BUNDLED_FACE_CHECKPOINT, map_location="cpu")
        )
        self.net = (
            DiffusionDITNetPartsFixedExpressions2PostNormInteraction2()
            .cuda()
            .requires_grad_(False)
            .eval()
        )
        self.net.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
        self.smplx = release.EvaluateSMPLX(args.smplx_model)
        kdtree = MakeKDTree(
            pid=args.pid,
            data_folder=args.retrieval_data_folder,
            csv_file=args.retrieval_csv_file,
            smplx_model_path=args.smplx_model,
        )
        self.key_poses = axis_angle_to_rotation_6d(
            torch.from_numpy(LocalToGlobalAA(kdtree["poses"])).float()
        ).cuda()
        self.key_joints = torch.from_numpy(kdtree["joints"]).float().cuda().half()
        self.pid = torch.LongTensor([int(args.pid.split("_")[0])]).cuda()
        example = np.load(args.example_motion_npz, allow_pickle=True)
        self.betas = torch.from_numpy(example["betas"].astype(np.float32)).cuda()
        poses = example["poses"].astype(np.float32)[: self.seed_frames]
        first = np.concatenate(
            [
                axis_angle_to_rotation_6d(LocalToGlobalAA(poses)).reshape(
                    self.seed_frames, POSE6D
                ),
                example["trans"].astype(np.float32)[: self.seed_frames],
            ],
            axis=1,
        )
        self.first_seed = np.ascontiguousarray(first, dtype=np.float32)
        self.state_bytes = self.first_seed.nbytes + 4 * (1 + 2 * FEATURES6D)
        self.first_last = self.feature_row(
            *self.local_features(torch.from_numpy(self.first_seed[-1:]).cuda())
        )
        self.audio = (
            release.AudioFeatureExtractor(args.wavlm_model)
            .requires_grad_(False)
            .cuda()
            .eval()
        )
        self.info = {
            "kind": "motion",
            "model": "streamtalk",
            "fps": FPS,
            "chunk_frames": self.block,
            "context_samples": WINDOW_SAMPLES - self.block * 16000 // FPS,
            "lookahead_samples": 0,
            "features": 271,
        }
        warm_up(self, args.warmup_chunks)

    def state_of(self, state: bytes):
        """The seed, the frames since the last key pose, the last frame sent and the smoother's derivative a state holds."""
        if not state:
            seed = self.first_seed
            return (
                seed,
                self.keypose_every,
                self.first_last.copy(),
                np.zeros(FEATURES6D, np.float32),
            )
        elif len(state) != self.state_bytes:
            raise BadRequest(
                f"not a StreamTalk state of {self.seed_frames} seed frames"
            )
        else:
            values = np.frombuffer(state, dtype=np.float32)
            seed = values[: self.first_seed.size].reshape(self.first_seed.shape)
            tail = values[self.first_seed.size :]
            return (
                seed,
                int(tail[0]),
                tail[1 : 1 + FEATURES6D].copy(),
                tail[1 + FEATURES6D :].copy(),
            )

    def pack_state(self, seed, since_key_pose, last, derivative) -> bytes:
        return b"".join(
            np.ascontiguousarray(part, dtype=np.float32).tobytes()
            for part in (seed, [since_key_pose], last, derivative)
        )

    def model_kwargs(
        self,
        hubert,
        expressions,
        expressions_uncond,
        pose_prompt,
        trans_prompt,
        prompt_mask,
    ):
        torch = self.torch
        trans_mask = torch.full((1, WINDOW_FRAMES), True).cuda()
        trans_mask[:, : self.seed_frames] = False
        kwargs = {
            "pid": self.pid,
            "poses_prompt": pose_prompt,
            "prompt_mask": prompt_mask,
            "trans_prompt": trans_prompt,
            "trans_mask": trans_mask,
            "hubert": hubert,
            "expressions": expressions,
            **self.indices,
        }
        if expressions_uncond is not None:
            kwargs["expressions_uncond"] = expressions_uncond
        else:
            pass
        return kwargs

    def render(self, state: bytes, speech: np.ndarray):
        torch = self.torch
        normalize = self.NormalizeRot6d
        seed, since_key_pose, last, derivative = self.state_of(state)
        # The last frame the client got, which the root velocity starts from.
        shown = last.copy()
        with torch.no_grad():
            (hubert,) = self.audio.extract(speech.astype(np.float32))
            pose_prompt = torch.zeros((1, WINDOW_FRAMES, POSE6D)).cuda()
            trans_prompt = torch.zeros((1, WINDOW_FRAMES, TRANS)).cuda()
            pose_prompt[0, : self.seed_frames] = torch.from_numpy(
                seed[:, :POSE6D].copy()
            ).cuda()
            trans_prompt[0, : self.seed_frames] = torch.from_numpy(
                seed[:, POSE6D:].copy()
            ).cuda()
            prompt_mask = torch.full((1, WINDOW_FRAMES), True).cuda()
            prompt_mask[:, : self.seed_frames] = False
            expressions = self.speech_model(hubert, self.pid)
            expressions_uncond = (
                self.speech_model(hubert, torch.zeros_like(self.pid))
                if self.cfg_scale != 1.0
                else None
            )
            noise = torch.randn((1, WINDOW_FRAMES, FEATURES6D)).cuda()
            kwargs = self.model_kwargs(
                hubert,
                expressions,
                expressions_uncond,
                pose_prompt,
                trans_prompt,
                prompt_mask,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xt_list = self.net.sample(
                    noise, normalize_func=normalize, cfg_scale=self.cfg_scale, **kwargs
                )
                xt = xt_list[-1]
            if since_key_pose + self.block >= self.keypose_every:
                xt, anchor = self.refine_on_key_pose(
                    xt, xt_list[5], noise, kwargs, pose_prompt, prompt_mask
                )
                since_key_pose = WINDOW_FRAMES - 1 - anchor
            else:
                since_key_pose += self.block
            xt = xt.float()
            xt[..., :POSE6D] = normalize(xt[..., :POSE6D])
            local, trans = self.local_features(xt[0, self.seed_frames :])
            if self.smoother is not None:
                features = torch.cat(
                    [self.matrix_to_rotation_6d(local).reshape(-1, POSE6D), trans],
                    dim=1,
                )
                features, last, derivative = self.smoother.run(
                    features.cpu().numpy(), last, derivative
                )
                features = torch.from_numpy(features).cuda()
                local = self.rotation_6d_to_matrix(
                    normalize(features[:, :POSE6D]).reshape(-1, 55, 6)
                )
                trans = features[:, POSE6D:]
            else:
                last = self.feature_row(local[-1:], trans[-1:])
            poses = self.rotation_matrix_to_axis_angle(local).reshape(-1, 55 * 3)
            previous = torch.from_numpy(shown[POSE6D:]).cuda()[None]
            velocity = torch.diff(torch.cat([previous, trans]), dim=0) * FPS
            rows = torch.cat(
                [poses, expressions[0, self.seed_frames :].float(), trans, velocity],
                dim=1,
            )
            next_seed = xt[0, -self.seed_frames :].contiguous().cpu().numpy()
            if self.seed_smoothed and self.smoother is not None:
                world = self.matrix_to_rotation_6d(self.local_to_global_batch(local))
                next_seed[-self.block :, :POSE6D] = (
                    world.reshape(-1, POSE6D).cpu().numpy()
                )
                next_seed[-self.block :, POSE6D:] = trans.cpu().numpy()
        next_state = self.pack_state(next_seed, since_key_pose, last, derivative)
        return next_state, [rows.cpu().numpy().astype("<f4").tobytes()]

    def local_features(self, rows):
        """Each joint's rotation relative to its parent ([n, 55, 3, 3]) and the root translation of [n, FEATURES6D] rows in the model's global 6D layout."""
        world = self.rotation_6d_to_matrix(rows[:, :POSE6D].reshape(-1, 55, 6))
        return self.global_to_local_batch(world), rows[:, POSE6D:]

    def feature_row(self, local, trans) -> np.ndarray:
        """One frame as the smoother sees it: parent-relative 6D rotations, then the root translation."""
        row = self.torch.cat(
            [self.matrix_to_rotation_6d(local).reshape(-1, POSE6D), trans], dim=1
        )
        return row[0].cpu().numpy().astype(np.float32)

    def refine_on_key_pose(self, xt, xt_half, noise, kwargs, pose_prompt, prompt_mask):
        """The release's second pass: anchor the nearest BEAT2 pose of the speaker at one of the window's last frames and resample from halfway."""
        torch = self.torch
        normalize = self.NormalizeRot6d
        xt = xt.clone()
        xt[..., :POSE6D] = normalize(xt[..., :POSE6D])
        local = self.rotation_6d_to_matrix(xt[..., :POSE6D].reshape(-1, 55, 6))
        local = self.global_to_local_batch(local)
        local = self.rotation_matrix_to_axis_angle(local).reshape(1, -1, 55 * 3)
        joints = self.smplx(betas=self.betas, poses=local.reshape(-1, 55 * 3)).reshape(
            (WINDOW_FRAMES, 1, -1)
        )
        distances = torch.abs(self.key_joints - joints.half()).sum(dim=-1)
        dists, nearest_ids = torch.min(distances, dim=-1)
        dists = dists[-KEY_POSE_FRAMES:]
        nearest_ids = nearest_ids[-KEY_POSE_FRAMES:]
        best = int(torch.argmin(dists))
        anchor = WINDOW_FRAMES - KEY_POSE_FRAMES + best
        nearest = self.rotation_6d_to_matrix(
            self.key_poses[nearest_ids[best]].reshape(1, 55, 6)
        )
        orientation = self.rotation_6d_to_matrix(xt[0, anchor, :6].reshape(1, 6))
        nearest = torch.einsum("bij,bnjk->bnik", orientation, nearest)
        nearest = self.matrix_to_rotation_6d(nearest).reshape(1, POSE6D)
        prompt_mask = prompt_mask.clone()
        prompt_mask[:, anchor] = False
        pose_prompt = pose_prompt.clone()
        pose_prompt[:, anchor] = nearest
        xt[:, anchor, :POSE6D] = nearest
        kwargs = dict(kwargs, poses_prompt=pose_prompt, prompt_mask=prompt_mask)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            xt = self.net.sample_with_kp_bk(
                noise,
                xt_half,
                5,
                normalize_func=normalize,
                cfg_scale=self.cfg_scale,
                **kwargs,
            )
        return xt, anchor


def parse_args(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--wavlm-model", required=True)
    parser.add_argument("--smplx-model", required=True)
    parser.add_argument("--retrieval-data-folder", required=True)
    parser.add_argument("--retrieval-csv-file", required=True)
    parser.add_argument(
        "--example-motion-npz",
        default=None,
        help="default: the repository's Data/Test/2_scott_0_1_1.npz",
    )
    parser.add_argument("--pid", default="2_scott_0")
    parser.add_argument("--block-frames", type=int, default=15)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keypose-every",
        type=int,
        default=60,
        help="retrieve and anchor a BEAT2 key pose once this many frames passed since the last one (the release: once per 52-frame window); 0: every block",
    )
    parser.add_argument(
        "--smooth-min-cutoff",
        type=float,
        default=2.0,
        help="a causal One-Euro filter on the body, hands and root (not the jaw) with this minimum cutoff in Hz; 0: off",
    )
    parser.add_argument("--smooth-beta", type=float, default=0.0)
    parser.add_argument(
        "--seed-smoothed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="seed the next window with the smoothed frames rather than the model's",
    )
    args = parser.parse_args(argv)
    args.repo = os.path.abspath(args.repo)
    args.example_motion_npz = os.path.abspath(
        args.example_motion_npz
        or os.path.join(args.repo, "Data/Test/2_scott_0_1_1.npz")
    )
    for name in ("ckpt", "smplx_model", "retrieval_data_folder", "retrieval_csv_file"):
        setattr(args, name, os.path.abspath(getattr(args, name)))
    return args


def main() -> None:
    args = parse_args()
    setup_logging()
    serve(lambda: StreamTalkModel(args), args.host, args.port)


if __name__ == "__main__":
    main()
