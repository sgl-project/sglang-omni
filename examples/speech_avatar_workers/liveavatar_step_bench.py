# SPDX-License-Identifier: Apache-2.0
"""Time Live Avatar's DiT on its own: the blocks of one Session through one stage's steps, on 1..N sequence-parallel ranks.

    torchrun --nproc_per_node=N liveavatar_step_bench.py --stages 1 --sp N [worker args] \
        --port 0 --image ... --prompt ... [--bench-blocks 12] [--profile-block 9] [--save-latents out.pt]

No decoder rank: the world is the stage's ranks. Prints one JSON line per block
(GPU ms of the stage's steps) from rank 0 and, with --profile-block, the top
CUDA kernels of that block. --save-latents keeps every block's output for a
numerical comparison between layouts (same seed, same speech features).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    extra = {"--bench-blocks": "12", "--profile-block": "-1", "--save-latents": "", "--repeat": "1"}
    argv = []
    rest = sys.argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] in extra:
            extra[rest[i]] = rest[i + 1]
            i += 2
        else:
            argv.append(rest[i])
            i += 1
    sys.argv = [sys.argv[0]] + argv
    import liveavatar_worker as lw
    import torch
    import torch.distributed as dist

    args = lw.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(lw.device_of(rank))
    dist.init_process_group(backend="nccl", rank=rank, world_size=int(os.environ["WORLD_SIZE"]), timeout=timedelta(hours=2))
    assert args.stages == 1 and int(os.environ["WORLD_SIZE"]) == args.sp, "one stage, no decoder"
    engine = lw.Engine(args, rank)
    blocks = int(extra["--bench-blocks"])
    profile_block = int(extra["--profile-block"])
    generator = torch.Generator(device=engine.device).manual_seed(7)
    audio = [torch.randn(engine.audio_shape, generator=generator, device=engine.device, dtype=engine.dtype) for _ in range(blocks)]
    saved = []
    for repeat in range(int(extra["--repeat"])):
        engine.reset(repeat + 1)
        noise = None
        with torch.amp.autocast("cuda", dtype=engine.dtype), torch.no_grad():
            for block in range(blocks):
                clip, index = divmod(block, engine.blocks_per_clip)
                if index == 0:
                    noise = engine.clip_noise(clip)
                latents = noise[:, index * engine.lfb : (index + 1) * engine.lfb].contiguous()
                prof = None
                if block == profile_block and repeat == int(extra["--repeat"]) - 1:
                    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                out = engine.denoise(latents, audio[block], block)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - begin) * 1000
                if prof is not None:
                    prof.__exit__(None, None, None)
                    if rank == 0:
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
                        prof.export_chrome_trace(f"/tmp/la_step_trace_sp{args.sp}.json")
                if rank == 0:
                    print(json.dumps({"repeat": repeat, "block": block, "ms": round(ms, 1), "steps": len(engine.my_steps), "sp": args.sp, "block_frames": engine.lfb}), flush=True)
                if repeat == int(extra["--repeat"]) - 1:
                    saved.append(out.float().cpu())
    if extra["--save-latents"] and rank == 0:
        torch.save(torch.stack(saved), extra["--save-latents"])
    dist.barrier(group=engine.gloo)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
