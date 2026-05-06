# Omni Router Design

The Omni Router is an external HTTP process that routes client traffic across
complete SGLang-Omni V1 server replicas. It is not a V1 pipeline stage router,
and it does not inspect `PipelineConfig`, stage graphs, tensor-parallel ranks,
or future data-parallel internals.

The request path is:

`Client / benchmark` -> `SGLang-Omni Router` -> `complete Omni V1 server replica`

## Key Files

| File | Role |
| --- | --- |
| `sglang_omni_router/config.py` | Router and worker config validation |
| `sglang_omni_router/worker.py` | Worker state, active request counts, health counters |
| `sglang_omni_router/selector.py` | `round_robin`, `least_request`, and `random` worker selection |
| `sglang_omni_router/health.py` | Active worker health probes with failure and success thresholds |
| `sglang_omni_router/proxy.py` | Raw byte forwarding, SSE relay, route diagnostics |
| `sglang_omni_router/app.py` | FastAPI routes and app lifecycle |
| `sglang_omni_router/launcher.py` | `python -m sglang_omni_router.serve.omni_router` CLI |

## Launch

```bash
python -m sglang_omni_router.serve.omni_router \
  --port 8000 \
  --policy round_robin \
  --health-failure-threshold 3 \
  --health-success-threshold 2 \
  --worker-urls http://127.0.0.1:8101 http://127.0.0.1:8102 \
  --model qwen3-omni \
  --route-log-path /tmp/qwen3-router/routes.jsonl
```

The installed CLI also exposes the same router launcher through:

```bash
sgl-omni router ...
```

## Routing Policies

`round_robin` is the default. It is deterministic and is the right policy for CI
because route JSONL can prove every configured replica served traffic.

`least_request` chooses the healthy worker with the lowest active request count.
Streaming requests remain active until the upstream stream closes, so this policy
is useful for audio workloads with uneven request durations.

`random` is available for smoke tests and fault-injection. It is not a CI
default because worker utilization is probabilistic.

## Health Surfaces

| Route | Meaning |
| --- | --- |
| `/live` | Router process/event-loop is alive |
| `/ready` | Router config is loaded and the health loop is running |
| `/health` | Aggregate pool health; 200 only when at least one worker is routable |
| `/workers` | Per-worker diagnostics for CI and operators |

CI should not overload `/health` with the stricter "all workers used" invariant.
Instead it should check `/workers` and route JSONL.

## Request Handling

The router forwards model requests as raw bytes. It may parse a small JSON copy
only for route metadata such as `request_id`, `model`, and `stream`, but it must
not reserialize the forwarded request body. This preserves Omni-specific fields
such as `audio`, `audios`, `images`, `videos`, `stage_sampling`, and
`stage_params`.

Streaming requests use `httpx.AsyncClient.send(..., stream=True)` and relay
upstream bytes exactly as received. The router does not synthesize or duplicate
SSE frames.

## DP Compatibility

The router is replica-level. Current Omni V1 exposes `tp_size` for tensor
parallel stage processes, but no public `dp_size` or `data_parallel_rank`
request contract. Future DP remains compatible under two rules:

- If each DP group is exposed as a complete HTTP server URL, the router treats
  those URLs as ordinary workers.
- If DP is internal to one Omni V1 server, the router remains unaware and does
  not inject `data_parallel_rank` or mutate request JSON.

Future capacity differences can be represented with worker metadata such as
`capacity_weight`; GPU count and rank semantics should not leak into the router.

## Future Hardening

Do not add these to the first router unless they are implemented and tested:

- circuit breaker / passive ejection;
- automatic retries;
- bounded router queues;
- `power_of_two` load balancing;
- modality-aware `cost_aware` routing;
- manual affinity headers.
