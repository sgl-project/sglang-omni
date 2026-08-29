# Prefill/Decode Disaggregation

> **TL;DR** — PD splits one stage into a prefill process and a decode process, so a prefill no longer stalls the decoding already in flight. Where each half runs and how much of a card it gets are separate, continuous choices, and they decide what the split returns. Start with both halves on one GPU. Turn CUDA graph on for both halves before you measure anything.

A colocated replica runs prefill and decode on one card and one scheduler thread, so a prefill blocks the decode steps of every request already running. On Qwen3-Omni that cost is large, and it is worst on images, where a 6127-token prompt cannot be chunked.

PD moves prefill into its own process. The prompt KV and the request's continuation state transfer once over CUDA IPC, and decoding continues on the other half.

## Before you measure: turn CUDA graph on

PD sets `disable_cuda_graph` on neither half. Both inherit the stage's engine arguments, so whatever the rest of your deployment does, PD does.

This matters because it is easy to compare a PD arm with the graph off against a colocated baseline with it on, and read the result as a cost of splitting. On Qwen3-Omni thinker, 8128-token prompts, `max_running_requests=4`, one H200 per half:

| Two-card PD | req/s | TPOT p50 |
| --- | ---: | ---: |
| graph off, c16 | 1.95 | 0.2514 |
| graph on, c16 | **5.58** | **0.0083** |

Thirty times the inter-token latency, from one setting. With the graph on, the same PD pair also beats a colocated replica measured the same way (5.58 against 4.55 req/s).

## Turning it on

Declare `pd_disaggregation` on the stage you want split. The compiler replaces it with `<stage>_prefill` and `<stage>_decode` and rewrites the edges around it; nothing else in the graph changes.

```yaml
stages:
  thinker:
    pd_disaggregation:
      prefill:
        gpu: 0
        memory_fraction: 0.30
      decode:
        gpu: 0
        memory_fraction: 0.47
```

A full example is `examples/configs/qwen3_omni_pd_h200.yaml`.

The stage's factory must declare `pd_capable`. Qwen3-Omni and Ming do; a model that does not is rejected at config time, naming the stage, before any process starts.

## Both halves on one GPU

This is the placement to start from, and it is not a degenerate case. What PD needs is the process split, and a prefill step leaves the decode scheduler thread whichever card it runs on. A second card is an additional decision about hardware.

**Both halves must declare `memory_fraction` when they share a card**, and the config is rejected if either does not. The launcher sums *declared* fractions per GPU and skips a stage that declares none, so two undeclared halves would each size against the whole card, and whichever won the startup lock would take the memory while the other failed to start.

The split between them matters less than you would expect. Sweeping the prefill share across a 27-fold range of KV budget moved throughput by 1.5%:

| prefill / decode share | prefill KV | decode KV | c8 req/s |
| --- | ---: | ---: | ---: |
| 0.47 / 0.47 | 22,547 | 643,731 | 2.797 |
| 0.58 / 0.32 | 191,238 | 413,697 | 2.778 |
| 0.45 / 0.50 | 613,060 | 68,554 | 2.761 |
| 0.40 / 0.55 | 536,382 | 145,232 | 2.803 |

So size the two halves for the memory each needs, and do not expect the ratio itself to be a throughput knob.

### Sharing one copy of the weights

On one card the two halves load the same weights twice. `share_weights: true` has one half export its copy over CUDA IPC and the other point its parameters at it, returning 56.94 GiB to the pool on an H200.

It is off by default because it needs the model's cooperation, and because adopting does not lower *peak* memory — the adopting half still loads its own copy before it can point at the peer's. On one card the publisher is already holding weights and pool by then, so the shares have to leave a whole extra copy free at that moment. A share that cannot is rejected with the numbers, before the loader raises a CUDA OOM from three frames down.

Which half publishes is decided from the declared shares, not by whichever wins the startup lock. The publisher holds the weights on top of its own KV, so at prefill 0.30 / decode 0.62 the smaller half cannot hold the copy — and letting a race decide meant the same config came up one time and failed the next.

## What the split trades

Same-GPU PD moves latency from the first token to the gaps between tokens. Against a colocated replica on the same card, c8, one fresh server per cell:

| prompt tokens | colo req/s | PD req/s | colo TTFT p50 | PD TTFT p50 | colo TPOT | PD TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 7.515 | 7.035 | 0.659 | **0.123** | 0.0128 | 0.0309 |
| 1024 | 6.766 | 6.873 | 0.751 | **0.172** | 0.0129 | 0.0317 |
| 4096 | 3.433 | **4.153** | 1.661 | **0.630** | 0.0201 | 0.0397 |

First token is 2.6 to 5.4 times faster at every length; the inter-token interval is about twice as slow. Throughput crosses over around 1024 tokens.

Under open-loop load the first-token difference widens sharply. At 42-token prompts and offered rate 16, colocated's p95 first-token time was 7.48 and 8.84 s across two rounds against PD's 0.123 and 0.125 s — while colocated completed 13.29 req/s against PD's 10.61. The two arms are not trading the same quantity there, so read both numbers.

## Sizing beyond one prefill and one decode

A 1:1 pair cannot beat two colocated replicas on throughput, by arithmetic: a card doing both jobs has the harmonic mean of its prefill-only and decode-only capacities, a pair has the minimum of the two, and the minimum never exceeds the harmonic mean. Parity is 1:1's ceiling. Choosing the ratio is where the gain is.

Replicate the decode process to get one prefill and N decode halves:

```yaml
processes:
  thinker_decode:
    num_replicas: 2
    replica_devices: [1, 2]
```

Which replica a request goes to is the coordinator's admission binding — chosen once and carried on the message envelope — so the prefill half does not choose again.

Measured on three H200s:

| cell | 1:1 | 1:2 | change |
| --- | ---: | ---: | ---: |
| 1024 / 32, c8 | 13.925 | 20.375 | **+46%** |
| 1024 / 32, c16 | 15.328 | 25.431 | **+66%** |
| 8128 / 32, c8 | 4.835 | 4.819 | −0.3% |

At 8128 tokens the single prefill card runs at 100% for the whole cell, so a second decode half adds nothing. Short prompts are where 1:N pays.

## Overload arrives as latency, not rejection

A colocated replica throttles admission by contention. Splitting removes that and nothing replaces it. Measured on two H200s at offered rate 16, where both arms admitted every request:

| | decode in flight | total time p50 | admission |
| --- | ---: | ---: | ---: |
| colocated replica | ~18 | 2.29 s | 100% |
| PD pair | ~437 | 40.96 s | 100% |

Admission reads 100% from a healthy request to a 41-second one, which is why this does not look like a failure. Two knobs bound it:

- `max_queued_requests` on the decode half rejects beyond a bound. Unset, the decode half logs at startup that its queue is unbounded. `models/qwen3_tts` sets 16 for its generation stage.
- `max_inflight_handoffs` bounds how many handoffs send at once. Each one holds its request's prompt KV on the prefill card until decode acknowledges the copy. Note that this bounds concurrent *sends*, not pinned KV: the lease exists before the send is queued.

## Limits today

- `tp_size` must be 1. The KV send is rank-addressed and the decode half has no cross-rank join, so it would admit on a partial copy.
- `page_size` must be 1.
- RadixCache must be disabled. With a prefix tree the handed-off pages are not solely the request's, and the continuation carries neither `extra_key` nor `cache_salt`, so the decode half would insert under a key that collides with a different request holding the same token ids.
- Prefix reuse across the handoff is therefore unavailable. This is temporary.

Each is checked when the config is read, naming the stage and what has to be built for the restriction to go.
