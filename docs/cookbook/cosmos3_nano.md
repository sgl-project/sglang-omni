# Cosmos3 Nano serving

Nano reuses native SGLang `Engine` and `OpenAIServingChat` for understanding and `multimodal_gen.DiffGenerator` for generation. Omni owns request routing, continuation state and delivery. Native SGLang owns model loading, scheduling, KV caches, codecs and parallelism.

## SGLang prerequisites

Nano needs an SGLang build that provides the SRT Cosmos3 model and native chat serving APIs, cooperative SDK cancellation, scheduler owner-failure propagation, video job settlement, the Nano modality contracts and native `Engine` cleanup at shutdown. Recent SGLang releases, including the pinned one, provide the model and chat APIs, but the other capabilities come from coordinated SGLang changes that no published release contains yet, so public packages alone cannot reproduce this setup.

Startup stops with an error that names the missing capability when it can detect one. The generation stage requires cooperative cancellation (`DiffGenerator.supports_cancellation`), because without it an aborted request cannot stop its native generation and a request in flight when the generation worker dies waits indefinitely. Even with a capable build, a configuration that cannot cancel cooperatively, such as more than one native GPU, logs a warning at startup and keeps that cost, and so does a request with more than one output per prompt. The native media routes require scheduler owner-failure support (`worker_failure in AsyncSchedulerClient.initialize`), because without it a request in flight when the generation worker dies waits for its client deadline. Startup does not detect the other capabilities, so confirm them in the installed build. Audiovisual generation requires the full Nano checkpoint and FFmpeg with AAC support. Policy-DROID requires its separate checkpoint. Keep checkpoint configuration and tokenizer files alongside the weights.

## Interleaved understanding and generation

Save this as `cosmos3.yaml`. The default placement uses visible GPU 0 for generation, visible GPU 1 for the Reasoner and CPU for orchestration, so it needs two visible GPUs. With one visible GPU, startup stops with a GPU id out of range error for the reasoner stage. Set `stages.reasoner.gpu` and `stages.generation.gpu`, or pass `--reasoner.gpu` and `--generation.gpu`, to choose other devices. Placement refuses both native stages on one GPU unless each declares `gpu_memory_fraction` or `total_reserve_bytes`, and that sharing has not been qualified.

```yaml
config_cls: Cosmos3UMMPipelineConfig
model_path: /path/to/Cosmos3-Nano
stages:
  generation:
    factory:
      output_dir: outputs/cosmos3
```

```bash
sgl-omni serve --config cosmos3.yaml --host 127.0.0.1 --port 8000 --model-name cosmos3
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Create an image of a red cube, inspect it, then create a blue cube and compare them.",
    "stream": true,
    "return_logprob": false,
    "sampling_params": {"temperature": 0, "max_new_tokens": 1024},
    "stage_params": {"generation": {"width": 640, "height": 640, "num_inference_steps": 35, "seed": 7}}
  }'
```

`/generate` streams each accepted text or media segment once through `event: segment`, then emits `event: complete` and `[DONE]`. The complete event carries the concatenated `text`, `finish_reason` and `segments` as `segment_index` and `kind` references without data. Failure emits `event: error` with the error `message`, `type` and status `code`. Set `return_logprob=false` for streaming. A non-streaming request returns `segments` with their data, concatenated `text` and `media`.

`/v1/chat/completions` exposes the same ordered extension as `choices[0].message.segments` or streamed `choices[0].delta.segment`. SDK results expose `CompletionResult.segments` and `GenerateChunk.segment`. The SDK reads the terminal result as segments only when its `type` is `umm_result`, as the controller sets it. Use a fresh request ID for every execution and close unfinished SDK iterators to propagate cancellation.

Each segment has `session_id`, a contiguous `segment_index`, `kind` and `data`. Media carries inline bytes, MIME type, SHA-256 and byte count. The generator reads only its owned output, releases its temporary directory and sends those bytes into the next native image or video chat turn. The Reasoner chooses `generate` or `final` through a validated JSON decision. Internal model calls are non-streaming.

The controller defaults to 8 generation turns, 17 segments (one message and one media segment per turn plus the final text), 32 sessions, 64 MiB retained JSON per session and a 3600-second deadline. Override these under `stages.orchestrator.factory` using `max_turns`, `max_segments`, `max_sessions`, `max_context_bytes` and `timeout_s`. Without an explicit `max_segments`, the segment limit follows `max_turns`. Once no generation turn remains, the Reasoner can only choose `final`. A request that exceeds `max_turns`, `max_segments` or `max_context_bytes` fails with status 400 naming the limit and its override, and a request that outlives `timeout_s` fails with an error naming it. A non-streaming request returns the status as its HTTP status. A streaming request has already received HTTP 200, so its error carries the status as `code`.

The top-level `max_in_flight` (default 32) caps the requests the pipeline admits, and `max_sessions` caps live controller sessions. Keep `max_in_flight` at or below `max_sessions` and raise them together, so excess requests get status 503 at admission, reported the same way, before they reach the controller. Use native backend limits for model memory. Retained JSON limits do not measure peak host or GPU memory.

## Direct generation

For direct native generation, change `config_cls` to `Cosmos3PipelineConfig`. This uses visible GPU 0 and exposes native media routes.

| Operation | HTTP route | Native options |
| --- | --- | --- |
| Image generation | `/v1/images/generations` | Prompt and native image sampling options |
| Text, image or video to video | `/v1/videos` | Native input references and video sampling options |
| Video with sound | `/v1/videos` | `generate_sound=true` and `sound_duration` matching the requested video duration |
| Action-conditioned video | `/v1/videos` | `action_mode=forward_dynamics`, initial image and known actions |
| Action prediction | `/v1/actions/generations` | `action_mode=policy` or `inverse_dynamics`, matching observation and domain |

Video routes return jobs. Poll `/v1/videos/{id}` with a deadline, retrieve `/v1/videos/{id}/content`, and delete finished jobs with `DELETE /v1/videos/{id}`. SDK generation returns caller-owned saved files or structured action values. Native HTTP retention and SDK file ownership are separate contracts.

The Policy-DROID checkpoint uses `raw_action_dim=8`. Match observation layout, units and normalization to the checkpoint. Actions and audiovisual codecs remain native model responsibilities. Image and video feedback are supported by this interleaving adapter. Sound in a returned video does not establish audio understanding, and action feedback loops require a separate adapter contract.

## Reusing the serving pipeline

Implement `UMMAdapter` from `sglang_omni.pipeline.umm` for another compatible model to prepare native Reasoner and generation requests, interpret decisions, incorporate generated media and expose segments. Its orchestrator factory returns `UMMController(adapter, limits=UMMLimits(...))`, as `create_umm_scheduler` does for Cosmos3. Keep model prompts, media conversion and modality restrictions in the adapter. Reuse the existing stage scheduler, transport and client delivery. The shared controller relies on these contracts:

- The orchestrator stage is terminal, routes with `route_umm` and lists exactly the stages named `reasoner` and `generation` in `next`. Startup rejects other names.
- Both native stages set `next` to the orchestrator and return the input `StagePayload`, or a result that copies its `continuation`. A result without it fails the request.
- Internal turns never stream. The controller sends them with `stream` false and streams accepted segments itself.
- Generation returns inline media bytes rather than stage-local files, because the controller keeps media after the stage releases its outputs.
- `reasoner_request` receives `remaining_generation_turns` and allows only a final decision when it is 0.
- UMM pipelines refuse session operations.
- `max_in_flight` stays at or below `max_sessions`.

Qualify actual generated bytes in the following understanding turn, stale results, cancellation, deadlines and native ownership before enabling a new model. Interface reuse does not establish checkpoint quality or performance. H100 and robot task quality require separate qualification.
