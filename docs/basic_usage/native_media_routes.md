# Native Media Routes

Pipelines whose generation stage runs native SGLang `multimodal_gen`, such as Cosmos3, serve the native SGLang image, video and action routes from the Omni server. These requests go straight to the native scheduler that the generation stage owns, so their request and response formats are those of native SGLang.

| Routes | Purpose |
| --- | --- |
| `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/images/{id}/content` | Image generation and editing |
| `POST /v1/videos`, `GET /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}` | Video jobs |
| `POST /v1/actions/generations`, `GET /v1/actions/metadata` | Action generation, when the checkpoint supports it |

```bash
sgl-omni serve --model-path /path/to/Cosmos3-Nano --port 8000
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red cube on a wooden table", "size": "640x640", "response_format": "b64_json"}'
```

## Behavior

- Only the routes above are served. Native weights, memory, LoRA, rollout, mesh, realtime, server info and stats routes are not, so use the Omni admin routes, which require the admin key when one is set.
- The native scheduler listens on `127.0.0.1` whatever host the server binds. Startup fails if the generation stage's `server_args_overrides` set another `host`.
- The routes run in the generation stage's process, which already holds native SGLang, and start with that stage. The Omni server relays them to the stage over a local socket and imports no native module itself, so the first request to them pays no start and the server process keeps no native memory.
- If the routes cannot start, for example because the prompt enhancer config cannot be read, the generation stage fails to start and the server reports the cause.
- If the native scheduler stops, requests in flight and new requests on these routes get 503 at once. If the generation stage stops, these routes answer 503, and the server stops when the generation stage fails.

## Server warmup

With `warmup_mode: server` in the generation stage's `server_args_overrides`, the generation stage runs native SGLang's server warmup before it reports ready, with the requests native SGLang's own server sends, so `/health` answers 200 only after it and the first request finds the generation stage warm. As in native SGLang, a warmup that fails without explicit `warmup_resolutions` is logged as a warning and startup continues. The warmup input image is removed afterwards. The warmup is off by default, and then the generation stage instead sends one step of the same synthetic request in the background once it serves, so readiness does not wait for it and a request that arrives meanwhile queues behind that step and then runs warm. Two series of paired launches of Cosmos3 Nano on one H200, each launch with the warmup paired with one without it, measured these costs as the mean difference and its 95% interval:

| Cost against warmup off | First series | Second series, busier host |
| --- | --- | --- |
| Time until `/health` answers 200 | +16.2 s (+13.1 to +19.3 s) | +24.1 s (-17.9 to +66.2 s) |
| Energy of the generation GPU until ready | +9.1 kJ (+8.7 to +9.4 kJ) | +10.0 kJ (+5.4 to +14.6 kJ) |
| Energy of the other visible GPU until ready, idle during the longer startup | +1.3 kJ (+1.0 to +1.5 kJ) | +1.9 kJ (-1.4 to +5.1 kJ) |
| GPU memory in use from readiness until the first request | +5568 MiB in every launch | +5568 MiB in every launch |
| Native worker memory at readiness | PSS +873 MiB (+677 to +1069 MiB), RSS +859 MiB (+858 to +860 MiB) | PSS +745 MiB (+721 to +768 MiB), RSS +859 MiB (+856 to +862 MiB) |
| Native worker memory after the first request | PSS +52 MiB (-18 to +121 MiB), RSS +80 MiB (+80 to +81 MiB) | PSS +115 MiB (-67 to +297 MiB), RSS +81 MiB (+78 to +83 MiB) |

Native SGLang's three warmup requests take about 14 s of the readiness difference, 11 s of them its automatic residency probe. Readiness, and with it the energy used until ready, also varies with how fast the weights load, which depends on the page cache and on other load on the host. In the second series the transformer load took 12 to 38 s per launch, and the readiness difference of single pairs ranged from -14.5 s to +44.7 s. GPU memory returns to its level without the warmup with the first request.

In return the first request after readiness is faster: 1.2 s instead of 2.8 s in the first series (1.6 s less, 95% interval 1.1 to 2.2 s) and 0.9 s instead of 2.8 s in the second (1.9 s less, 1.5 to 2.3 s).

## Media policy

`--allowed-local-media-path` and `--allowed-media-domain` apply to every media path and URL in a native route request, including native extra parameters and action observations. With `--allowed-local-media-path`, local paths and file URLs must resolve inside that directory, including the values native SGLang decodes from JSON encoded strings. Native SGLang opens a file URL as written, as a path relative to the server working directory, so send local media as plain paths. While `--allowed-media-domain` is set, remote URLs are refused on these routes, because native SGLang fetches them itself and follows redirects without checking each one, so send such media as an upload or a data URL. Without either option, media references are read as on other routes. Requests can never set server paths such as `output_path`, `perf_dump_path` or model paths. Native SGLang names a saved upload after its file name, and an action upload after its `request_id` or `id` field too, so an upload whose name contains `/` is refused.

## Retention

Generated images and videos stay in the generation stage output directory (`stages.generation.factory.output_dir`, `outputs` by default), and native SGLang keeps one job record per request in server memory until the server stops. Native SGLang has no expiry for either. `DELETE /v1/videos/{id}` only removes the job record, so it neither stops a running job nor deletes its file. Remove old files from the output directory on a long running server. Uploaded inputs are kept only for their request or video job.
