# SPDX-License-Identifier: Apache-2.0
"""A Live Avatar (Wan2.2-S2V-14B, 4-step) speech-avatar worker on five GPUs in the authors' TPP layout: four DiT ranks each run one denoising step of every block and the fifth decodes it.

Run it with torchrun from the Live Avatar repository, in its own environment:

    cd LiveAvatar && ENABLE_COMPILE=true torchrun --nproc_per_node=5 \
        /path/to/liveavatar_worker.py --port 9440 --image examples/anchor.jpg \
        --prompt "..." --ckpt-dir ckpt/Wan2.2-S2V-14B/

The worker speaks the chunk_server.py protocol (/health, /info, /idle,
/render) from rank 0. A chunk is one block of the model: 3 latent frames, 12
video frames at 25 fps, 480 ms. Rank 0 encodes the chunk's speech window with
wav2vec2 and runs step 1; the block's latents, its audio features and a
header travel down the ranks; rank 4 decodes the block, encodes JPEGs and
hands them back to rank 0, which answers the request.

The Session's state lives on the GPUs (each DiT rank's rolling KV cache, the
decoder's stream cache), so the state that goes over HTTP is a handle: a
16-byte Session key and the block's index. An empty state starts a Session
with a new key; a request whose key is not the current Session's, or whose
index is 0, starts that Session afresh. The worker renders the blocks of one
Session at a time, in index order, and keeps up to /info's pipeline_depth of
them in flight: a client that knows the next handle (key, index + 1) sends the
next block before the previous one returns, which is what fills the
pipeline. The handle each answer carries is that next one.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_server import (  # noqa: E402
    STATE_LENGTH,
    BadRequest,
    parse_render_body,
)

SAMPLE_RATE_HZ = 16000
FPS = 25
LATENT_FRAMES_PER_BLOCK = 3
FRAMES_PER_BLOCK = 4 * LATENT_FRAMES_PER_BLOCK
BLOCKS_PER_CLIP = 4
CHUNK_SAMPLES = FRAMES_PER_BLOCK * SAMPLE_RATE_HZ // FPS
DIT_RANKS = 4
VAE_RANK = DIT_RANKS
WORLD_SIZE = DIT_RANKS + 1
# wav2vec2 features come at 50 Hz and Live Avatar reads them resampled to 30 Hz.
AUDIO_FEATURE_HZ = 30
HANDLE = struct.Struct("<16sI")
HEADER_LENGTH = 8
OP_BLOCK = 1
# A request waits this long for its block's turn (an earlier index still on its way) and for its frames.
ORDER_TIMEOUT_S = 10.0
RESULT_TIMEOUT_S = 60.0

logger = logging.getLogger("liveavatar_worker")


@dataclass
class Job:
    job_id: int
    new_session: bool
    speech: np.ndarray
    enqueued: float
    done: threading.Event = field(default_factory=threading.Event)
    payloads: Optional[List[bytes]] = None
    timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--image", required=True, help="The avatar's reference picture.")
    parser.add_argument("--prompt", required=True, help="What the video shows, as the authors' examples word it.")
    parser.add_argument("--ckpt-dir", default="ckpt/Wan2.2-S2V-14B/")
    parser.add_argument("--lora", default="Quark-Vision/Live-Avatar")
    parser.add_argument("--training-config", default="liveavatar/configs/s2v_causal_sft.yaml")
    parser.add_argument("--size", default="720*400", help="The pixel area (the authors' real-time setting); the picture keeps the reference's aspect.")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--steps", type=int, default=DIT_RANKS)
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--context-ms", type=float, default=1000.0, help="Speech before the block that wav2vec2 hears.")
    parser.add_argument("--lookahead-ms", type=float, default=240.0, help="Speech after the block that wav2vec2 hears.")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--pipeline-depth", type=int, default=WORLD_SIZE, help="Blocks a client keeps in flight, announced in /info.")
    parser.add_argument("--warmup-blocks", type=int, default=12, help="Blocks of a warm-up Session (then a second, short one) rendered before /health answers 200, so compilation happens first.")
    parser.add_argument("--timing-log", default=None, help="Rank 0 appends one JSON line per block here.")
    args = parser.parse_args()
    if args.steps != DIT_RANKS:
        parser.error("the TPP layout runs one step per DiT rank: --steps must be 4")
    return args


def samples_of(ms: float) -> int:
    return int(round(ms * SAMPLE_RATE_HZ / 1000))


class Engine:
    """One rank's share of the model and of the Session it renders."""

    def __init__(self, args: argparse.Namespace, rank: int) -> None:
        import torch
        import torch.distributed as dist
        from torchvision import transforms
        from PIL import Image

        from diffusers import FlowMatchEulerDiscreteScheduler
        from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from liveavatar.utils.args_config import (
            parse_args_for_training_config as training_config_parser,
        )

        self.torch = torch
        self.dist = dist
        self.args = args
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        cfg = WAN_CONFIGS["s2v-14B"]
        self.dtype = cfg.param_dtype
        settings = training_config_parser(args.training_config)
        pipe = WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=rank,
            rank=rank,
            convert_model_dtype=True,
            single_gpu=False,
        )
        self.pipe = pipe
        if rank < DIT_RANKS:
            pipe.noise_model = pipe.add_lora_to_model(
                pipe.noise_model,
                lora_rank=settings["lora_rank"],
                lora_alpha=settings["lora_alpha"],
                lora_target_modules=settings["lora_target_modules"],
                init_lora_weights=settings["init_lora_weights"],
                pretrained_lora_path=args.lora,
                load_lora_weight_only=False,
            )
            if not args.no_fp8:
                from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

                replace_linear_with_scaled_fp8(
                    pipe.noise_model,
                    ignore_keys=[
                        "text_embedding",
                        "time_embedding",
                        "time_projection",
                        "head.head",
                        "casual_audio_encoder.encoder.final_linear",
                    ],
                )
            else:
                pass
        else:
            # The decoding rank needs no DiT; it shares its GPU with other services.
            del pipe.noise_model
            torch.cuda.empty_cache()
        height, width = pipe.get_gen_size(
            size=None,
            max_area=MAX_AREA_CONFIGS[args.size],
            ref_image_path=args.image,
            pre_video_path=None,
        )
        self.height, self.width = height, width
        self.frame_seq_length = height // 8 * width // 8 // 4
        self.lat_motion_frames = (pipe.motion_frames + 3) // 4
        reference = Image.open(args.image).convert("RGB")
        picture = transforms.CenterCrop((height, width))(
            transforms.Resize(min(height, width))(reference)
        )
        self.idle_picture = picture
        pixels = transforms.ToTensor()(picture).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        pixels = pixels.to(dtype=pipe.vae.dtype, device=pipe.vae.device)
        with torch.no_grad():
            self.first_ref_latents = torch.stack(pipe.vae.encode(pixels))
            self.motion_latents = torch.stack(
                pipe.vae.encode(pixels.repeat(1, 1, pipe.motion_frames, 1, 1))
            )
        latent_shape = (16, LATENT_FRAMES_PER_BLOCK, height // 8, width // 8)
        self.latent_shape = latent_shape
        self.audio_shape = (1, 25, 1024, FRAMES_PER_BLOCK)
        self.cond_states = torch.zeros(
            (1, *latent_shape), dtype=self.dtype, device=self.device
        )
        if rank < DIT_RANKS:
            with torch.no_grad():
                self.context, _ = pipe.encode_prompt(args.prompt, None, offload_model=True)
            del pipe.text_encoder
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=3
            )
            scheduler.set_timesteps(args.steps, device=self.device)
            self.scheduler = scheduler
            self.timesteps = scheduler.timesteps
            self.sigmas = scheduler.sigmas
            pipe._initialize_crossattn_cache(
                batch_size=1, dtype=self.dtype, device=self.device
            )
            self.kv_cache_size = (
                BLOCKS_PER_CLIP * LATENT_FRAMES_PER_BLOCK * self.frame_seq_length
            )
        else:
            del pipe.text_encoder
        if rank == 0:
            pipe.audio_encoder.model.to(device=self.device, dtype=self.dtype)
            pipe.audio_encoder.model.requires_grad_(False)
            pipe.audio_encoder.model.eval()
        else:
            pass
        self.context_samples = samples_of(args.context_ms)
        self.lookahead_samples = samples_of(args.lookahead_ms)
        self.window_samples = self.context_samples + CHUNK_SAMPLES + self.lookahead_samples
        # wav2vec2 feature index of each of the block's frames inside the window.
        self.feature_index = [
            int(round((self.context_samples / SAMPLE_RATE_HZ + frame / FPS) * AUDIO_FEATURE_HZ))
            for frame in range(FRAMES_PER_BLOCK)
        ]
        self.gloo = dist.new_group(backend="gloo", timeout=timedelta(days=7))
        self.session_tag = 0
        self.ref_latents = self.first_ref_latents
        self.ref_sends: list = []
        self.has_pending_ref = False

    # -------- shared --------

    def header(self, values: List[int]):
        header = self.torch.zeros(HEADER_LENGTH, dtype=self.torch.int64, device=self.device)
        header[: len(values)] = self.torch.tensor(values, dtype=self.torch.int64)
        return header

    def reset(self, tag: int) -> None:
        """Start Session `tag`: fresh caches and the reference picture as the attention sink."""
        torch = self.torch
        if self.rank < DIT_RANKS:
            if self.rank < DIT_RANKS - 1 and self.has_pending_ref:
                # The previous Session's block 0 sent a sink this rank never needed; take it off the wire.
                self.receive_ref(self.session_tag)
            else:
                pass
            for work, _ in self.ref_sends:
                work.wait()
            self.ref_sends = []
            self.pipe._initialize_kv_cache(
                batch_size=1,
                dtype=self.dtype,
                device=self.device,
                kv_cache_size=self.kv_cache_size,
            )
        else:
            vae = self.pipe.vae.model
            vae.first_decode = True
            vae.clear_cache()
            torch.cuda.empty_cache()
        self.session_tag = tag
        self.ref_latents = self.first_ref_latents
        self.has_pending_ref = False

    def receive_ref(self, tag: int) -> None:
        cpu = self.torch.empty(
            (1, 16, 1, self.height // 8, self.width // 8), dtype=self.dtype
        )
        self.dist.recv(cpu, src=DIT_RANKS - 1, group=self.gloo, tag=tag % (1 << 30))
        self.ref_latents = cpu.to(self.device)
        self.has_pending_ref = False

    # -------- DiT ranks --------

    def block_args(self, audio, sink: bool) -> dict:
        args = {
            "context": self.context[0:1],
            "seq_len": None,
            "cond_states": self.cond_states,
            "motion_latents": self.motion_latents,
            "ref_latents": self.ref_latents,
            "audio_input": audio,
            "motion_frames": [self.pipe.motion_frames, self.lat_motion_frames],
            "drop_motion_frames": False,
        }
        if sink:
            args["sink_flag"] = True
        else:
            pass
        return args

    def clip_noise(self, clip: int):
        torch = self.torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.seed + clip)
        shape = self.latent_shape
        return torch.randn(
            16,
            BLOCKS_PER_CLIP * LATENT_FRAMES_PER_BLOCK,
            shape[2],
            shape[3],
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )

    def denoise(self, latents, audio, block: int):
        """This rank's step of the Session's block `block`, after the sink prefill the first two clips start with."""
        torch = self.torch
        clip, index = divmod(block, BLOCKS_PER_CLIP)
        if index == 0 and clip in (0, 1):
            if clip == 1 and self.rank < DIT_RANKS - 1:
                self.receive_ref(self.session_tag)
            else:
                pass
            noise = self.clip_noise(clip)[:, :LATENT_FRAMES_PER_BLOCK]
            self.pipe.noise_model(
                [noise],
                t=torch.zeros([1, LATENT_FRAMES_PER_BLOCK], device=self.device, dtype=self.dtype),
                **self.block_args(audio, sink=True),
                kv_cache=self.pipe.kv_cache1,
                crossattn_cache=self.pipe.crossattn_cache,
                current_start=0,
                current_end=LATENT_FRAMES_PER_BLOCK * self.frame_seq_length,
            )
        else:
            pass
        t = self.timesteps[self.rank]
        timestep = torch.tensor([t] * LATENT_FRAMES_PER_BLOCK).to(self.device).unsqueeze(0)
        start = block * LATENT_FRAMES_PER_BLOCK * self.frame_seq_length
        prediction = self.pipe.noise_model(
            [latents],
            t=timestep,
            **self.block_args(audio, sink=False),
            kv_cache=self.pipe.kv_cache1,
            crossattn_cache=self.pipe.crossattn_cache,
            current_start=start,
            current_end=start + LATENT_FRAMES_PER_BLOCK * self.frame_seq_length,
            mask=None,
        )
        scheduler = self.scheduler
        scheduler.timesteps = self.timesteps
        scheduler.sigmas = self.sigmas
        scheduler._step_index = self.rank
        scheduler._begin_index = 0
        latents = scheduler.step(
            prediction[0].unsqueeze(0), t, latents.unsqueeze(0), return_dict=False
        )[0].squeeze(0)
        if self.rank == DIT_RANKS - 1 and block == 0:
            # The first block's clean first frame becomes the sink from the second clip on.
            ref = latents.unsqueeze(0)[:, :, 0:1].contiguous()
            cpu = ref.cpu()
            for peer in range(DIT_RANKS - 1):
                self.ref_sends.append(
                    (self.dist.isend(cpu, dst=peer, group=self.gloo, tag=self.session_tag % (1 << 30)), cpu)
                )
            self.pending_own_ref = ref
        elif self.rank < DIT_RANKS - 1 and block == 0:
            self.has_pending_ref = True
        else:
            pass
        if self.rank == DIT_RANKS - 1 and block == BLOCKS_PER_CLIP - 1:
            self.ref_latents = self.pending_own_ref
        else:
            pass
        return latents

    def encode_speech(self, speech: np.ndarray):
        """The block's wav2vec2 features [1, 25 layers, 1024, 12 frames] from its speech window."""
        torch = self.torch
        features = self.pipe.audio_encoder.extract_audio_feat_from_array(
            speech, return_all_layers=True, dtype=self.dtype
        )
        index = torch.tensor(
            [min(i, features.shape[1] - 1) for i in self.feature_index],
            device=features.device,
        )
        return features[:, index].permute(0, 2, 1).unsqueeze(0).to(self.device, self.dtype).contiguous()

    def send_down(self, header, latents, audio) -> None:
        dist = self.dist
        dist.send(header, self.rank + 1)
        dist.send(latents.contiguous(), self.rank + 1)
        if self.rank + 1 < DIT_RANKS:
            dist.send(audio, self.rank + 1)
        else:
            pass

    def receive_up(self):
        torch = self.torch
        dist = self.dist
        header = torch.empty(HEADER_LENGTH, dtype=torch.int64, device=self.device)
        dist.recv(header, self.rank - 1)
        latents = torch.empty(self.latent_shape, dtype=self.dtype, device=self.device)
        dist.recv(latents, self.rank - 1)
        audio = None
        if self.rank < DIT_RANKS:
            audio = torch.empty(self.audio_shape, dtype=self.dtype, device=self.device)
            dist.recv(audio, self.rank - 1)
        else:
            pass
        return header.tolist(), latents, audio

    def run_first(self, jobs: "queue.Queue[Job]", timing_log: Optional[str]) -> None:
        """Rank 0: take blocks from the HTTP side, encode their speech and run step 1."""
        torch = self.torch
        block = 0
        noise = None
        events = []
        with torch.amp.autocast("cuda", dtype=self.dtype), torch.no_grad():
            while True:
                job = jobs.get()
                job.timings["start"] = time.time()
                if job.new_session:
                    self.reset(self.session_tag + 1)
                    block = 0
                else:
                    pass
                clip, index = divmod(block, BLOCKS_PER_CLIP)
                if index == 0:
                    noise = self.clip_noise(clip)
                else:
                    pass
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                audio = self.encode_speech(job.speech)
                latents = noise[:, index * LATENT_FRAMES_PER_BLOCK : (index + 1) * LATENT_FRAMES_PER_BLOCK].contiguous()
                latents = self.denoise(latents, audio, block)
                end.record()
                header = self.header([OP_BLOCK, self.session_tag, block, job.job_id, int(job.new_session)])
                self.send_down(header, latents, audio)
                events.append((job.job_id, block, begin, end))
                block += 1
                if len(events) > 8:
                    self.log_events(events[:-8], timing_log)
                    events = events[-8:]
                else:
                    pass

    def log_events(self, events, timing_log: Optional[str]) -> None:
        if timing_log is None:
            return
        else:
            lines = []
        for job_id, block, begin, end in events:
            end.synchronize()
            lines.append(json.dumps({"rank": self.rank, "job": job_id, "block": block, "gpu_ms": round(begin.elapsed_time(end), 2)}))
        with open(timing_log, "a") as log:
            log.write("\n".join(lines) + "\n")

    def run_middle(self, timing_log: Optional[str]) -> None:
        """Ranks 1-3: steps 2-4 of each block as it comes down."""
        torch = self.torch
        events = []
        with torch.amp.autocast("cuda", dtype=self.dtype), torch.no_grad():
            while True:
                header, latents, audio = self.receive_up()
                _, tag, block, job_id, new_session = header[:5]
                if new_session:
                    self.reset(tag)
                else:
                    pass
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                latents = self.denoise(latents, audio, block)
                end.record()
                self.send_down(self.header(header), latents, audio)
                events.append((job_id, block, begin, end))
                if len(events) > 8:
                    self.log_events(events[:-8], timing_log)
                    events = events[-8:]
                else:
                    pass

    # -------- decoding rank --------

    def run_decoder(self, result_url: str, jpeg_quality: int) -> None:
        """Rank 4: decode each block, encode its JPEGs and hand them to rank 0."""
        torch = self.torch
        from PIL import Image

        encoder = ThreadPoolExecutor(max_workers=12)
        poster = ThreadPoolExecutor(max_workers=2)

        def encode(frame: np.ndarray) -> bytes:
            out = io.BytesIO()
            Image.fromarray(frame).save(out, format="JPEG", quality=jpeg_quality)
            return out.getvalue()

        def deliver(job_id: int, frames: np.ndarray, timings: dict) -> None:
            try:
                jpegs = list(encoder.map(encode, frames))
                timings["encoded"] = time.time()
                meta = json.dumps({"job": job_id, "lengths": [len(j) for j in jpegs], "timings": timings}).encode()
                body = struct.pack("<I", len(meta)) + meta + b"".join(jpegs)
                request = urllib.request.Request(result_url, data=body, method="POST")
                with urllib.request.urlopen(request, timeout=30) as answer:
                    answer.read()
            except Exception:
                logger.error("delivering job %d failed:\n%s", job_id, traceback.format_exc())

        with torch.amp.autocast("cuda", dtype=self.dtype), torch.no_grad():
            while True:
                header, latents, _ = self.receive_up()
                _, tag, block, job_id, new_session = header[:5]
                received = time.time()
                if new_session:
                    self.reset(tag)
                else:
                    pass
                vae = self.pipe.vae
                if block == 0:
                    vae.stream_decode(self.motion_latents[:, :, :7])
                else:
                    pass
                image = torch.stack(vae.stream_decode(latents.unsqueeze(0)))[:, :, -FRAMES_PER_BLOCK:]
                if block == 0:
                    # The authors drop the first block's first 3 frames; the chunk keeps its 12 by holding the first kept frame.
                    image = image[:, :, 3:]
                    image = torch.cat([image[:, :, :1].repeat(1, 1, 3, 1, 1), image], dim=2)
                else:
                    pass
                frames = (
                    ((image[0].float() + 1) * 127.5)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 3, 0)
                    .cpu()
                    .numpy()
                )
                decoded = time.time()
                poster.submit(deliver, job_id, frames, {"received": received, "decoded": decoded, "decode_ms": (decoded - received) * 1000})


class Front:
    """Rank 0's HTTP side: orders a Session's blocks, queues them for the ranks and answers each with its frames."""

    def __init__(self, engine: Engine, args: argparse.Namespace) -> None:
        self.engine = engine
        self.args = args
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.lock = threading.Condition()
        self.waiting: Dict[int, Job] = {}
        self.next_job = 1
        self.key: Optional[bytes] = None
        self.expected = 0
        self.ready = False
        out = io.BytesIO()
        engine.idle_picture.save(out, format="JPEG", quality=args.jpeg_quality)
        self.idle = out.getvalue()
        self.info = {
            "kind": "video",
            "model": "liveavatar-wan2.2-s2v-14b-4step-tpp",
            "fps": FPS,
            "chunk_frames": FRAMES_PER_BLOCK,
            "context_samples": engine.context_samples,
            "lookahead_samples": engine.lookahead_samples,
            "width": engine.width,
            "height": engine.height,
            "pipeline_depth": args.pipeline_depth,
            "sessions": 1,
        }

    def submit(self, state: bytes, speech: np.ndarray) -> tuple:
        """Queue one block of a Session in index order; returns the job and the next handle."""
        if state:
            if len(state) != HANDLE.size:
                raise BadRequest("a Live Avatar state is a 16-byte key and a 4-byte block index")
            else:
                key, index = HANDLE.unpack(state)
        else:
            key, index = uuid.uuid4().bytes, 0
        deadline = time.monotonic() + ORDER_TIMEOUT_S
        with self.lock:
            while True:
                if index == 0:
                    new_session = True
                    break
                elif key == self.key and index == self.expected:
                    new_session = False
                    break
                elif key == self.key and index < self.expected:
                    raise BadRequest(f"block {index} of the Session was rendered already")
                else:
                    remaining = deadline - time.monotonic()
                if remaining > 0:
                    # An earlier block of the Session is still on its way.
                    self.lock.wait(remaining)
                elif key == self.key:
                    raise TimeoutError(f"block {self.expected} of the Session never came before block {index}")
                else:
                    # A Session this worker does not know past its start (the worker restarted): start it afresh here.
                    new_session = True
                    break
            job = Job(self.next_job, new_session, speech, time.time())
            self.next_job += 1
            self.key = key
            self.expected = index + 1
            self.waiting[job.job_id] = job
            self.jobs.put(job)
            self.lock.notify_all()
        return job, HANDLE.pack(key, index + 1)

    def deliver(self, body: bytes) -> None:
        (meta_length,) = struct.unpack_from("<I", body)
        meta = json.loads(body[4 : 4 + meta_length])
        payloads = []
        offset = 4 + meta_length
        for length in meta["lengths"]:
            payloads.append(body[offset : offset + length])
            offset += length
        with self.lock:
            job = self.waiting.pop(meta["job"], None)
        if job is not None:
            job.payloads = payloads
            job.timings.update(meta["timings"])
            job.timings["delivered"] = time.time()
            job.done.set()
        else:
            pass

    def render(self, state: bytes, speech: np.ndarray) -> tuple:
        job, next_state = self.submit(state, speech)
        if not job.done.wait(RESULT_TIMEOUT_S):
            with self.lock:
                self.waiting.pop(job.job_id, None)
            raise TimeoutError(f"block job {job.job_id} got no frames in {RESULT_TIMEOUT_S} s")
        else:
            pass
        timings = job.timings
        timings = {
            "queue_ms": (timings["start"] - job.enqueued) * 1000,
            "render_ms": (timings["delivered"] - job.enqueued) * 1000,
            "pipeline_ms": (timings["received"] - timings["start"]) * 1000,
            "decode_ms": timings["decode_ms"],
            "encode_ms": (timings["encoded"] - timings["decoded"]) * 1000,
        }
        if self.args.timing_log:
            with open(self.args.timing_log, "a") as log:
                log.write(json.dumps({"rank": "front", "job": job.job_id, "new_session": job.new_session, **{k: round(v, 2) for k, v in timings.items()}}) + "\n")
        else:
            pass
        return next_state, job.payloads, timings

    def warm_up(self) -> None:
        """Two Sessions of silence and noise through the whole pipeline, the first long enough to roll the KV cache."""
        rng = np.random.default_rng(0)
        window = self.engine.window_samples
        for blocks in (self.args.warmup_blocks, 3):
            state = b""
            pending = []
            for index in range(blocks):
                speech = np.zeros(window, np.float32) if index % 2 == 0 else (0.1 * rng.standard_normal(window)).astype(np.float32)
                job, state = self.submit(state, speech)
                pending.append(job)
            for job in pending:
                start = job.enqueued
                if not job.done.wait(3600):
                    raise TimeoutError("the warm-up got no frames")
                else:
                    logger.info("warm-up job %d: %.0f ms after it was queued", job.job_id, (job.timings["delivered"] - start) * 1000)

    def serve(self) -> None:
        front = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def answer(self, code: int, body: bytes, content_type: str = "application/octet-stream", headers: Optional[dict] = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/health":
                    self.answer(200 if front.ready else 503, b"ok" if front.ready else b"warming up")
                elif path == "/info":
                    self.answer(200, json.dumps(front.info).encode(), "application/json")
                elif path == "/idle":
                    self.answer(200, front.idle, "image/jpeg")
                else:
                    self.answer(404, b"not found")

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = urlparse(self.path).path
                if path == "/_result":
                    if self.client_address[0] != "127.0.0.1":
                        self.answer(403, b"forbidden")
                    else:
                        front.deliver(body)
                        self.answer(200, b"ok")
                    return
                elif path != "/render":
                    self.answer(404, b"not found")
                    return
                elif not front.ready:
                    self.answer(503, b"warming up")
                    return
                else:
                    pass
                try:
                    state, speech = parse_render_body(body, front.engine.window_samples)
                    next_state, payloads, timings = front.render(state, speech)
                except BadRequest as error:
                    self.answer(400, str(error).encode())
                    return
                except Exception:
                    logger.error("render failed:\n%s", traceback.format_exc())
                    self.answer(503, traceback.format_exc().encode())
                    return
                self.answer(
                    200,
                    next_state + b"".join(payloads),
                    headers={
                        "X-State-Length-Bytes": str(len(next_state)),
                        "X-Payload-Lengths": ",".join(str(len(p)) for p in payloads),
                        "X-Render-Ms": f"{timings['render_ms']:.1f}",
                        "X-Timings": json.dumps({k: round(v, 1) for k, v in timings.items()}),
                    },
                )

        server = ThreadingHTTPServer((self.args.host, self.args.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        if self.args.host not in ("127.0.0.1", "0.0.0.0", ""):
            internal = ThreadingHTTPServer(("127.0.0.1", self.args.port), Handler)
            internal.daemon_threads = True
            threading.Thread(target=internal.serve_forever, daemon=True).start()
        else:
            pass
        logger.info("listening on %s:%d", self.args.host, self.args.port)


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise SystemExit(f"the TPP layout runs on {WORLD_SIZE} GPUs, not {world_size}")
    else:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank{rank} %(name)s %(levelname)s %(message)s",
    )
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(days=7),
    )
    start = time.time()
    engine = Engine(args, rank)
    logger.info("loaded in %.0f s: %dx%d", time.time() - start, engine.width, engine.height)
    dist.barrier()
    timing_log = args.timing_log
    if timing_log and rank != 0:
        timing_log = f"{timing_log}.rank{rank}"
    else:
        pass
    if rank == 0:
        front = Front(engine, args)
        front.serve()

        def warm() -> None:
            began = time.time()
            front.warm_up()
            front.ready = True
            logger.info("ready after %.0f s of warm-up: %s", time.time() - began, front.info)

        threading.Thread(target=warm, daemon=True).start()
        engine.run_first(front.jobs, timing_log)
    elif rank < DIT_RANKS:
        engine.run_middle(timing_log)
    else:
        engine.run_decoder(f"http://127.0.0.1:{args.port}/_result", args.jpeg_quality)


if __name__ == "__main__":
    main()
