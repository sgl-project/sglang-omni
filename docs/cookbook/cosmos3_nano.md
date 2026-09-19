# Cosmos3 Nano serving

Nano reuses native SGLang `Engine` and `OpenAIServingChat` for understanding and `multimodal_gen.DiffGenerator` for generation. Omni owns request routing, continuation state and delivery. Native SGLang owns model loading, scheduling, KV caches, codecs and parallelism.

The native prerequisites are not yet published as an installable revision. The pinned `sglang==0.5.19` package is insufficient. These launch instructions require the coordinated SGLang changes for cooperative SDK cancellation, video job settlement, scheduler failure propagation and Nano modality contracts. A tested public native revision and its installation instructions remain required before this setup can be reproduced from public packages. Audiovisual generation requires the full Nano checkpoint and FFmpeg with AAC support. Policy-DROID requires its separate checkpoint. Keep checkpoint configuration and tokenizer files alongside the weights.

## Interleaved understanding and generation

Save this as `cosmos3.yaml`. The default placement uses visible GPU 0 for generation, visible GPU 1 for the Reasoner and CPU for orchestration.

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

`/generate` streams complete accepted text and media through `event: segment`, then emits `event: complete` and `[DONE]`. Failure emits `event: error`. The final snapshot repeats accepted segments, so render each segment once. Set `return_logprob=false` for streaming. A non-streaming request returns `segments`, concatenated `text` and `media`.

`/v1/chat/completions` exposes the same ordered extension as `choices[0].message.segments` or streamed `choices[0].delta.segment`. SDK results expose `CompletionResult.segments` and `GenerateChunk.segment`. Use a fresh request ID for every execution and close unfinished SDK iterators to propagate cancellation.

Each segment has `session_id`, a contiguous `segment_index`, `kind` and `data`. Media carries inline bytes, MIME type, SHA-256 and byte count. The generator reads only its owned output, releases its temporary directory and sends those bytes into the next native image or video chat turn. The Reasoner chooses `generate` or `final` through a validated JSON decision. Internal model calls are non-streaming.

The controller defaults to 8 generation turns, 16 segments, 32 sessions, 64 MiB retained JSON per session and a 3600-second deadline. Override these under `stages.orchestrator.factory` using `max_turns`, `max_segments`, `max_sessions`, `max_context_bytes` and `timeout_s`. Set `max_in_flight` for pipeline admission and use native backend limits for model memory. Retained JSON limits do not measure peak host or GPU memory.

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

Implement `UMMAdapter` for another compatible model to prepare native Reasoner and generation requests, interpret decisions, incorporate generated media and expose segments. Configure the existing native stages to return to the shared controller through `route_umm`. Keep model prompts, media conversion and modality restrictions in the adapter. Reuse the existing stage scheduler, transport and client delivery.

Qualify actual generated bytes in the following understanding turn, stale results, cancellation, deadlines and native ownership before enabling a new model. Interface reuse does not establish checkpoint quality or performance. H100 and robot task quality require separate qualification.
