# MiniCPM-o duplex gate harness

Scripts that set up a GPU node and run the gates used for the MiniCPM-o native duplex optimization PRs: unit tests, the aligned-start session ladder, the low-concurrency serving gate, the 727-sample agreement gate, perception-stage timing, and a session-count sweep that finds the operating point and names the stage to optimize next.

| file | what it is |
|---|---|
| `config.sh` | node layout: paths, image tag, cards, container names, client core ranges, model revision, dataset digest, recorder tree; every value can be overridden from the environment |
| `setup_node.sh` | prepares a node (image, checkpoint with digest check, one container per card, Full-Duplex-Bench v1.0 with digest check) |
| `gate.sh` | pushes trees to the node and runs the gates |
| `report.py` | turns a results directory into the markdown tables below (standard library only) |
| `fdbconc_metrics.py` | unit lag, miss and session accounting from recorder traces |
| `timing_hook/sitecustomize.py` | measurement-only stage timing hook, active only with `STAGE_TIMING_DIR` |
| `hostload.sh` | host CPU sampler (`HOSTLOAD=1`) |
| `ids96.txt` | the 96 Full-Duplex-Bench v1.0 samples of the ladders, mixed across the five subsets |
| `ref_units.json` | per-unit speak decisions of the official offline inference loop on all 727 v1.0 samples |
| `frame_hook/sitecustomize.py`, `frames/` | recorder-side video probe: with `--frames-per-unit N` every recorder attaches N of the four 640×480 JPEGs to each 1 s unit (`sglang.input_image.append`, rotating per unit; the dataset audio is unchanged) |

## 1. Set up a node

From a checkout that has the commits (the node needs no credentials; trees travel as `git archive` tarballs):

```bash
export NODE=lijrjyan@47.74.68.185
benchmarks/duplex/gates/gate.sh push base:8eee6add1 P9:8792c85bc   # also pushes $BENCH_TREE
ssh $NODE 'setsid nohup bash ~/omni/src/harness/setup_node.sh base > ~/omni/setup.log 2>&1 < /dev/null &'
ssh $NODE 'tail -3 ~/omni/setup.log'                                  # wait for SETUP_DONE (about 15 minutes)
```

`setup_node.sh base` installs the dependencies of the pushed tree `base` into the first container, commits it and clones it for the other cards. Cards, container names and client cores come from `CARDS`, `CONTAINERS` and `CLIENT_CPUS_LIST` (defaults: cards 0–2, `sglang-omni-junnan`, `-1`, `-2`). It is idempotent: rerun it after a partial failure.

The recorder and serving clients (`benchmarks/eval/benchmark_duplex_v10.py`, `benchmarks/duplex/serving.py`) are not on main yet; they come from `BENCH_TREE` (default `bench:88ad7b44d`, branch `bench/minicpmo-duplex-serving`), which `push` sends along.

## 2. Run a gate

On the node host (`~/omni/src/harness/gate.sh`). Each command starts detached work in the containers and returns at once; results land in `~/omni/logs/gates/<name>` (`/logs/gates/<name>` inside the containers). A tree is `<tag>:<sha>` of a pushed tree.

| command | what it runs |
|---|---|
| `gate.sh unit P9:8792c85bc` | CPU unit tests `tests/unit_test/minicpm_o tests/unit_test/scheduling tests/unit_test/test_stage_device_contract.py`, pinned to `UNIT_CPUS`, `CUDA_VISIBLE_DEVICES=` (`--full`: all of `tests/unit_test`; `--gpu`: `tests/unit_test/minicpm_o` on a card with `MINICPMO_CHECKPOINT`) |
| `gate.sh ladder P0:8eee6add1,P9:8792c85bc 16 3` | aligned-start recorder wave, 16 sessions, 3 runs per tree, interleaved over the cards (16 sessions use 48 samples, 32 or more use 96; `--samples N`, `--speech default`) |
| `gate.sh perception P9:8792c85bc 48 3` | the ladder with the default speech settings, reported as perception-stage timing |
| `gate.sh serving P9:8792c85bc 1,2,4,8 3` | `benchmarks/duplex/serving.py`, realtime then lockstep pacing, 3 serial server starts, `max_sessions` = the largest c, own parity yaml (`--graphs` removes `disable_cuda_graph`) |
| `gate.sh agree P9:8792c85bc` | all 727 v1.0 samples, `max_sessions: 4`, four shards at once, no MPS, no hook; compared per subset with `ref_units.json` |
| `gate.sh sweep P9:8792c85bc` | the ladder at 1, 2, 4, 8, 16, 32, 48 and 64 sessions, 2 runs each (`--sessions 40,44,48,52,56` to zoom in, `--runs`, `--runs-above-target 1` to run fewer repetitions above the first unsolved level, `--miss-threshold 1.0`, `--stop-miss 30`, `--frames-per-unit 1` for the video probe) |
| `gate.sh status <name>`, `gate.sh report <name>`, `gate.sh stop` | progress, the report again, stop everything |

Every ladder-type gate (ladder, perception, sweep) puts one worker per card on a shared queue of runs. Each worker first does one throwaway warm-up run (`warm-c<card>`, excluded from the reports; `--no-warm` skips it), starts MPS in its container, and then for each run:

- writes the server yaml (below), starts the server with the timing hook, and records the GPU memory every second;
- waits until no other card started a client wave in the last `WAVE_GAP` (30) seconds, then starts one recorder per session at once: each recorder is a single-threaded process (`OMP/OPENBLAS/MKL_NUM_THREADS=1`) pinned with `taskset` to the card's client core range; recorder k takes samples k, k+N, k+2N … of the run's sample list;
- after the clients finish, counts server tracebacks, kills the server's process session, and appends one line to `chain.log`.

When the last worker finishes it writes `report.md` and `DONE`. To pull a results directory back without the recorder traces (they carry the output audio and are large; `TRACES=1` keeps them): `NODE=… gate.sh pull <name> <dest>`; `report.py` then reads each run's `run.json`.

### The sweep and the current target level

The sweep exists to give every optimization PR one number to move. It runs the ladder at each session count with `max_sessions` = N, the tuned speech block and graphs on, and stops early once a level fails to start or misses more than `--stop-miss` percent. The report names:

- a level is **solved** when every run started, no session ended with an error, the server logged no traceback, and the mean miss over its runs is at most the threshold;
- the **operating point**: the largest solved N with every smaller N also solved;
- the **current target level**: the smallest N that is not solved (miss above the threshold, session errors, or failed to start), with the stage ranking that says which stage to optimize; when the target failed to start, the report prints the start error and the ranking at the operating point instead.

`report.py sweep <dir> <dir>…` merges several sweeps of the same tree by session count, so a zoom-in or extra repetitions add to the first sweep.

**Every optimization PR is measured at the current target level, and the ladder below it is the regression check.** Once the target level is solved, the next level up becomes the target. A later sweep can zoom in around the target with an explicit list, for example `--sessions 40,44,48,52,56`. One run that misses far more than its siblings at a low, lightly loaded level is a stall, not a capacity limit: repeat that level before trusting the target (see the video example below).

Sample count per run: 2 per session, at least 16 and at most 96 (so 1 session plays 16 samples one after another, 48 sessions play 96).

## 3. Read the report

`report.md` in the results directory (or `python report.py <kind> <dir>`, which also runs locally on a pulled copy). The numbers:

| number | meaning |
|---|---|
| miss | share of units whose lag exceeds 1 s; per run, then min / median / max / mean over the runs |
| lag p50 / p95 | per unit: receive time of `sglang.unit.done` minus the send time of the 80 ms input packet holding the unit's media end; median over runs |
| peak card memory | largest `nvidia-smi` memory.used during the run (all stage processes plus MPS) |
| startup | seconds from server launch to the first answer of `/v1/realtime/capabilities` |
| sessions with error / errors | samples whose trace has an `error` event / number of such events |
| tracebacks | `Traceback` lines in the server log |
| units | completed units / expected units (expected = ceil(accepted input ms / 1000) per sample) |
| busy, first 10 s / whole run | share of wall time the stage's process spends in recorded work (union of call intervals, garbage-collector pauses included) between the first client send and the last client receive |
| calls over budget | share of the stage's main calls (session hooks for perception and speech, the omni scheduler batch for thinker and talker) longer than `--budget-ms` (default 1000 ms, one unit of audio) |
| worst call | the longest main call, its time after the first client send, and its batch size |
| gen-2 GC | generation-2 collections over 5 ms inside the window, and the longest |
| peak reserved | `torch.cuda.max_memory_reserved` of the stage process |
| thinker prefill / decode tokens per unit | new prompt tokens of the thinker's extend batches and its decode steps over the run, divided by the completed units: the per-session context fill rate; "units to fill 8192 tokens" is 8192 divided by their sum |
| largest context seen | the longest thinker request (prefix plus new tokens) in any batch |
| perception call: audio / image / other | mean time per perception call in the streaming audio encoder, in image encoding (`encode_image`, preprocessing included), and in the rest of the call |

The limiting stage is the stage with the highest busy share over the run; when another stage is busier in the first 10 s, the report says so, since a saturated first wave is the usual cause of the early misses.

## Server yaml conventions

- Ladder, perception and sweep runs: the tree's `examples/full_duplex/minicpmo-parity.yaml`, minus `disable_cuda_graph` on the thinker and talker (graphs on), with `max_sessions` = N and the tuned speech block `speech: {dtype: float32, enable_dit_torch_compile: true, n_timesteps: 5}`. `--speech default` leaves the speech block out; trees before the speech settings were exposed need it.
- Agreement: the same parity yaml minus `disable_cuda_graph`, `max_sessions: 4`, no speech block.
- Serving: the tree's own parity yaml (`--graphs` removes `disable_cuda_graph`), `max_sessions` = the largest concurrency.
- `model_path` is always `/models/MiniCPM-o-4_5`.

## Known traps

- One container per card, created with `--device nvidia.com/gpu=<UUID>`: a single container with `CUDA_VISIBLE_DEVICES=N` breaks stage platform resolution.
- A fresh container's first server is cold (JIT; the first decode call takes about 5 s); the warm-up run absorbs it.
- The Drive folder of Full-Duplex-Bench also holds v1.5 and v3.0, whose download stalls; `setup_node.sh` stops gdown once the five v1.0 zips are complete and checks the digest.
- `--server-revision` of the recorder must be the full 40-character sha; `gate.sh` resolves abbreviated shas against the pushed tree.
- Never `pkill -f` a pattern that matches your own command line; `gate.sh stop` runs the patterns from inside the script.
- Unpinned, multi-threaded recorders produce a burst of over 1,000 runnable threads at client start, which shows up as first-wave misses that are not the server's. Keep the client cores away from `UNIT_CPUS` when unit tests run at the same time.
- Wait for a gate with a background loop on `DONE`, not a foreground sleep.
- The node can be wiped at any lease change; pull results as each gate finishes.

## Example: audio and video sweeps of the speech-flow-graphs tree

2026-10-07, node 47.74.68.185, H200 cards 0–2, image `lmsysorg/sglang:v0.5.21-cu130`. Tree `P9` = `perf/minicpmo-speech-flow-graphs` at `8792c85bc` (eight commits on main `8eee6add1`). Audio: `gate.sh sweep P9:8792c85bc` (doubling ladder, 2 runs per level) plus a zoom `--sessions 52,56,60`. Video: `--frames-per-unit 1 --sessions 1,2,4,8,16,32,48 --runs 3 --runs-above-target 1`, plus a repeat `--sessions 8,32 --runs 3`. Both with the tuned speech block and graphs on; the tables merge each sweep with its follow-up (`report.py sweep <sweep> <zoom>`).

| sessions | audio: runs | audio: miss mean / max | audio: lag p95 | audio: peak memory | video: runs | video: miss mean / max | video: lag p95 | video: peak memory |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 0.0 / 0.0 % | 202 ms | 31 GiB | 3 | 0.0 / 0.0 % | 231 ms | 31 GiB |
| 2 | 2 | 0.0 / 0.0 % | 235 ms | 32 GiB | 3 | 0.0 / 0.0 % | 251 ms | 32 GiB |
| 4 | 2 | 0.0 / 0.0 % | 236 ms | 35 GiB | 3 | 0.4 / 1.2 % | 280 ms | 35 GiB |
| 8 | 2 | 0.0 / 0.0 % | 305 ms | 44 GiB | 6 | 1.9 / 10.2 % (median 0.2 %) | 506 ms | 44 GiB |
| 16 | 2 | 0.0 / 0.0 % | 330 ms | 61 GiB | 2 | 0.0 / 0.0 % | 614 ms | 62 GiB |
| 32 | 2 | 0.0 / 0.1 % | 482 ms | 90 GiB | 4 | 0.9 / 1.5 % | 766 ms | 91 GiB |
| 48 | 2 | 0.3 / 0.5 % | 664 ms | 125 GiB | 1 | **40.9 %** | 3399 ms | 134 GiB |
| 52 | 2 | **3.6 / 7.0 %** | 927 ms | 135 GiB | | | | |
| 56 | 2 | 9.4 / 18.8 %, 20 sessions CUDA OOM | 1979 ms | 140 GiB | | | | |
| 60 | 2 | 46 sessions CUDA OOM | 556 ms | 140 GiB | | | | |
| 64 | 2 | failed to start: thinker KV needs 73.1 GiB > 0.52 of the card | | | | | | |

Startup was 60–100 s at every level; no traceback below 56 sessions.

**Audio.** Operating point **48** sessions, current target level **52** (one of two runs missed 7.0 %). At 52 sessions the thinker is the busiest stage (38 % of the run, 46 % of the first 10 s), and the speech stage had one 1,528 ms call at 6.7 s (batch 14), which is what the missing run paid for. Above 52 the card runs out of memory (peak 140 GiB of 140 GiB; 56 and 60 sessions end sessions with CUDA OOM), and at 64 the thinker KV no longer fits its 0.52 share.

**Video, one 640×480 frame per unit.** By the rule the target is **8** sessions (operating point 4), but that comes from one run of six that missed 10.2 % while every stage was under 25 % busy; the other five missed 0.0–0.9 %, and 16 and 32 sessions are solved. The capacity knee is between 32 (0.9 % over four runs) and **48 (40.9 %)**. At 48 sessions the limiting stage is **perception**: 90 % busy in the first 10 s and 54 % over the run, worst call 743 ms (batch 33), against 31 % / 35 % for the thinker. Image encoding is one frame at a time: 20 ms per perception call at 1 session, 31 ms at 32, 51 ms at 48, while the batched audio encoder stays at 7 ms. Batching image encoding across sessions is the next optimization for video.

**Context fill.** The thinker takes 16–18 new tokens per unit with audio only (plus about 1.3 decoded tokens) and 82–83 with one frame per unit (plus about 1.4 decoded), so an 8,192-token context fills after about 450 units (7.5 minutes) with audio and about 98 units (1.6 minutes) with one frame per second.
