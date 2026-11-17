# SPDX-License-Identifier: Apache-2.0
"""Chunk worker of SentiAvatar (non-commercial license): body motion of a 63-joint BVH skeleton at 30 fps, 24 frames (0.8 s) per chunk, published as bvh_motion rows.

Runs in the model's own environment from the release's motion_generation
directory:

  cd SentiAvatar/motion_generation && python /path/to/sentiavatar_worker.py --port 8124

The release plans a whole utterance at once: Chinese HuBERT features at
10 fps, a Qwen2-0.5B LLM that writes a keyframe every 4 audio tokens (served
by vLLM over HTTP), a mask transformer that fills the frames between, and an
RVQ-VAE decoder with a receptive field of about 3 s on each side. Nothing
carries over between utterances. A Session here runs that pipeline once per
chunk over a window of the agent's audio, `past` before the chunk and
`lookahead` after it, and keeps the shown motion continuous:

- the LLM runs in-process (vllm.LLM with the release server's settings), the
  models load once and HuBERT stays resident;
- a chunk's rows cross-fade (quaternion slerp, linear root) from what the
  previous chunk's window predicted for the same frames to what this window
  predicts, so each boundary joins the motion already shown;
- the root keeps moving from where the previous chunk left it; its height is
  the window's own;
- a window without speech is planned with the listening label
  ("动作：认真聆听") instead of the talking one;
- hands are the release's placeholder clip and the face is not generated.

Rows are float32: the root position (cm) then each of the template's 63
joints' local rotation as a unit quaternion (w, x, y, z), in the template's
joint order, exactly what the release's save_quat_motion_to_bvh writes to a
.bvh after process_batch_data: Y up, the character facing +z, its left on +x
(examples/avatar_workers/bvh_skeleton.py writes the skeleton the stage
publishes from the same template; give it --up y).
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

FPS = 30
CHUNK_FRAMES = 24
CHUNK_SAMPLES = CHUNK_FRAMES * SAMPLE_RATE_HZ // FPS
SILENCE_RMS = 0.003


def slerp(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-row slerp of unit quaternions [..., 4] with weights [rows]."""
    dot = (a * b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.abs(dot).clip(-1, 1)
    w = w.reshape(-1, *([1] * (a.ndim - 1)))
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    wa = np.where(small, 1 - w, np.sin((1 - w) * theta) / np.where(small, 1, sin))
    wb = np.where(small, w, np.sin(w * theta) / np.where(small, 1, sin))
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class SentiAvatar(SessionModel):
    def __init__(self, args) -> None:
        super().__init__()
        os.chdir(args.repo)
        sys.path.insert(0, str(Path(args.repo).resolve()))
        import joblib  # noqa: F401
        import pipeline_infer
        import torch
        from actions.postprocess import MotionPostprocesser, process_batch_data
        from infer import load_config_from_checkpoint, load_model
        from reconstruct_from_tokens import decode_body_tokens
        from single_case_infer import ApplyKmeans, resample_fps_1d
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        from vllm import LLM, SamplingParams

        self.torch = torch
        self.args = args
        self.pipeline_infer = pipeline_infer
        self.decode_body_tokens = decode_body_tokens
        self.process_batch_data = process_batch_data
        self.resample = resample_fps_1d
        project = Path(args.repo).resolve().parent
        checkpoints = project / "checkpoints"
        self.device = torch.device("cuda:0")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(checkpoints / "chinese-hubert-base")
        )
        self.hubert = (
            HubertModel.from_pretrained(str(checkpoints / "chinese-hubert-base"))
            .to(self.device)
            .eval()
        )
        self.kmeans = ApplyKmeans(str(checkpoints / "hubert_kmeans/model.mdl")).to(
            self.device
        )
        self.mask_model = pipeline_infer.load_mask_transformer(
            str(checkpoints / "mask_transformer"), device=str(self.device)
        )
        rvq_ckpt = str(checkpoints / "rvqvae/model/epoch_30.pth")
        self.rvq = load_model(
            rvq_ckpt, load_config_from_checkpoint(rvq_ckpt), self.device
        )
        meta = Path(args.repo) / "meta/mta_gen_demo"
        self.mean = torch.tensor(np.load(meta / "mean.npy")).to(self.device)
        self.std = torch.tensor(np.load(meta / "std.npy")).to(self.device)
        self.placeholder = np.load(
            Path(args.repo) / "meta/xiu_joint_quat_vecs/Daiji_A_001_V001.npy",
            allow_pickle=True,
        ).item()
        self.post = MotionPostprocesser()
        self.llm = LLM(
            model=str(checkpoints / "llm"),
            gpu_memory_utilization=args.llm_memory,
            max_model_len=1600,
            max_num_batched_tokens=1600,
            max_num_seqs=1,
            enable_chunked_prefill=True,
        )
        self.sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=1024,
            stop="<|im_end|>",
            skip_special_tokens=False,
        )
        self.client = self.InProcessPlanner(self)
        self.past = round(args.past_s * SAMPLE_RATE_HZ)
        self.lookahead = round(args.lookahead_s * SAMPLE_RATE_HZ)
        self.joints = len(self.post.anim.names)
        self.info = {
            "model": "sentiavatar",
            "fps": FPS,
            "chunk_frames": CHUNK_FRAMES,
            "lookahead_samples": self.lookahead,
            "listens": False,
            "media": "bvh_motion",
            "features": 3 + 4 * self.joints,
        }
        self.timings: list = []

    class InProcessPlanner:
        """The release's VLLMClient answered by the in-process LLM, so its parsing of the plan applies unchanged."""

        def __init__(self, owner) -> None:
            self.owner = owner

        def predict_motion_plan(self, text_list, **kwargs):
            owner = self.owner
            start = time.perf_counter()
            outputs = owner.llm.generate(text_list, owner.sampling, use_tqdm=False)
            cleaned = [
                o.outputs[0]
                .text.strip()
                .split("<|im_end|>")[0]
                .replace("<unk>", "")
                .replace(" ", "")
                for o in outputs
            ]
            owner.llm_s = time.perf_counter() - start

            class Answer:
                status_code = 200
                text = ""

                @staticmethod
                def json():
                    return {"motion_sequence_list": list(zip(text_list, cleaned))}

            requests = owner.pipeline_infer.requests
            post = requests.post
            requests.post = lambda *unused, **unused_kw: Answer
            try:
                return owner.pipeline_infer.VLLMClient.predict_motion_plan(
                    self, text_list, **kwargs
                )
            finally:
                requests.post = post

        base_url = "in-process"

    def window_rows(self, audio: np.ndarray, talking: bool):
        """The release's pipeline over one window of 16 kHz audio: rows [frames at 30 fps, 3 + 4 * joints], or None when the LLM planned fewer than two keyframes."""
        torch = self.torch
        values = self.extractor(
            audio, return_tensors="pt", sampling_rate=SAMPLE_RATE_HZ
        ).input_values.to(self.device)
        with torch.no_grad():
            layer9 = (
                self.hubert(values, output_hidden_states=True)
                .hidden_states[8][0]
                .float()
                .cpu()
                .numpy()
            )
        features = self.resample(layer9, src_fps=50.0, tgt_fps=10.0).astype(np.float32)
        tokens = self.kmeans.feat2token(torch.tensor(features, device=self.device))
        result = self.pipeline_infer.run_pipeline_single(
            self.client,
            self.mask_model,
            action_text=self.args.talk_label if talking else self.args.listen_label,
            audio_tokens=tokens,
            audio_features=features,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            generate_steps=self.args.generate_steps,
        )
        if result is None:
            return None
        motion = self.decode_body_tokens(
            self.rvq,
            result["dense_tokens"],
            self.placeholder,
            self.mean,
            self.std,
            self.device,
            src_fps=20.0,
            tgt_fps=30.0,
        )
        quats, root = self.process_batch_data(
            torch.as_tensor(motion["quat"])[None],
            torch.as_tensor(motion["offset"])[None],
            self.post.anim,
            self.post.skel.src_joint_dict,
            shape="wxyz",
        )
        # process_batch_data already returns (w, x, y, z) and the root as (x, z, y), i.e. Y up.
        wxyz = quats[0].float().cpu().numpy()
        root = (
            root[0].float().cpu().numpy().reshape(-1, 3)
            if root is not None
            else np.zeros((len(wxyz), 3), np.float32)
        )
        return root, wxyz

    def start(self):
        return {
            "audio": np.zeros(self.past, np.float32),
            "next": None,
            "root": None,
            "last": None,
            "height_axis": None,
        }

    def step(self, session, speech, listen):
        started = time.perf_counter()
        audio = np.concatenate([session["audio"][-self.past :], speech])
        session["audio"] = np.concatenate([session["audio"], speech[:CHUNK_SAMPLES]])[
            -self.past :
        ]
        talking = float(np.sqrt(np.mean(speech**2))) > SILENCE_RMS
        self.llm_s = 0.0
        planned = self.window_rows(audio, talking)
        if planned is None:
            logger.warning("the LLM planned fewer than two keyframes: holding the pose")
            session["next"] = None
            self.timings.append((time.perf_counter() - started, self.llm_s))
            return [b""] * CHUNK_FRAMES
        root, quats = planned
        first = round(self.past * FPS / SAMPLE_RATE_HZ)
        if len(quats) < first + CHUNK_FRAMES:
            pad = first + CHUNK_FRAMES - len(quats)
            quats = np.concatenate([quats, np.repeat(quats[-1:], pad, 0)])
            root = np.concatenate([root, np.repeat(root[-1:], pad, 0)])
        chunk_q = quats[first : first + CHUNK_FRAMES]
        if session["next"] is not None:
            previous_q = session["next"]
            weights = (np.arange(CHUNK_FRAMES) + 1) / CHUNK_FRAMES
            chunk_q = slerp(previous_q, chunk_q, weights)
        session["next"] = (
            quats[first + CHUNK_FRAMES : first + 2 * CHUNK_FRAMES]
            if len(quats) >= first + 2 * CHUNK_FRAMES
            else None
        )
        if session["height_axis"] is None:
            session["height_axis"] = int(np.argmax(np.abs(root.mean(0))))
        up = session["height_axis"]
        base = root[first - 1] if first > 0 else root[0]
        moves = root[first : first + CHUNK_FRAMES] - base
        origin = session["root"] if session["root"] is not None else root[first]
        chunk_root = origin + moves * (np.arange(3) != up) * self.args.root_scale
        chunk_root[:, up] = root[first : first + CHUNK_FRAMES, up]
        session["root"] = chunk_root[-1] * (np.arange(3) != up) * self.args.root_decay
        rows = np.concatenate(
            [chunk_root, chunk_q.reshape(CHUNK_FRAMES, -1)], axis=1
        ).astype("<f4")
        self.timings.append((time.perf_counter() - started, self.llm_s))
        return [row.tobytes() for row in rows]


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument(
        "--repo", default=".", help="The release's motion_generation directory"
    )
    parser.add_argument("--past-s", type=float, default=3.2)
    parser.add_argument("--lookahead-s", type=float, default=1.2)
    parser.add_argument("--talk-label", default="动作：说话")
    parser.add_argument("--listen-label", default="动作：认真聆听")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.2)
    parser.add_argument("--generate-steps", type=int, default=6)
    parser.add_argument("--llm-memory", type=float, default=0.2)
    parser.add_argument("--root-scale", type=float, default=1.0)
    parser.add_argument("--root-decay", type=float, default=0.98)
    args = parser.parse_args()
    setup_logging()

    def load():
        model = SentiAvatar(args)
        times = warm_up(model, args.warmup_chunks)
        logger.info("warm-up chunk times (ms): %s", [round(t) for t in times])
        return model

    serve(load, args.host, args.port)


if __name__ == "__main__":
    main()
