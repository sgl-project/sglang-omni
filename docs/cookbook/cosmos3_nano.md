# Cosmos3 Nano serving

Nano reuses native SGLang `Engine` and `OpenAIServingChat` for understanding and `multimodal_gen.DiffGenerator` for generation. Omni owns request routing, interleaving state and delivery. Native SGLang owns model loading, scheduling, KV caches, codecs and parallelism.

## Requirements

The pinned SGLang release serves Nano. The generation stage provides the Omni generation plugin to the native workers it starts, so the plugin needs no installation step and no other SGLang process loads it. The plugin lets an aborted or disconnected request stop its native generation within about one denoising step, and lets a request fail promptly when the native generation worker dies. If `SGLANG_PLUGINS` is set, it must list `omni_generation_cancellation`. Without the plugin, or with more than one native GPU for generation, startup logs a warning that names the reason, and an aborted request keeps the generation stage busy until its native generation finishes. A later SGLang that cancels natively is used instead of the plugin.

Native Cosmos3 guardrails run when the `cosmos-guardrail` package is installed. Set `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` to turn them off, for example on hosts without network access to the guardrail checkpoint. Audiovisual generation requires the full Nano checkpoint and FFmpeg with AAC support. Keep checkpoint configuration and tokenizer files alongside the weights.

## Interleaved understanding and generation

Save this as `cosmos3.yaml`. The default placement uses visible GPU 0 for generation, visible GPU 1 for the Reasoner and CPU for orchestration, so it needs two visible GPUs. Set `stages.reasoner.gpu` and `stages.generation.gpu`, or pass `--reasoner.gpu` and `--generation.gpu`, to choose other devices.

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
    "stage_params": {"generation": {"width": 640, "height": 640, "seed": 7}}
  }'
```

The Reasoner answers each turn with a structured decision: generate an image or a video from a prompt, or give the final answer. Generated media returns to the Reasoner in its next turn, until it answers final or eight generations ran. Answers and prompts may contain any text, including newlines and quotes. A blank final answer ends the request without a text segment, and a blank generation prompt is not run: the Reasoner gets a receipt saying so and decides again within the same eight generations. A generation that native SGLang ends without media fails the request with a server error, since native reports a guardrail block, running out of GPU memory and other failures alike as no output, and the generation stage log gives its reason. Malformed, oversized or extra media fail the request. Video generations default to 33 frames, and `stage_params.generation.num_frames` sets another count. With `action_mode` `forward_dynamics`, the Reasoner may only request videos. Other generation options in `stage_params.generation` apply to every generation of the request and are checked before the first turn, so a field the native image and video routes do not accept, such as `output_path`, is refused with 400 before any model work.

Every accepted output is a segment with `type` `segment`, `execution_id`, a contiguous `segment_index`, `kind` (`text`, `image` or `video`) and `data`. Text data is a string. Media data carries `kind`, `mime_type`, `size_bytes`, `sha256` and `url`, a data URL with the bytes.

- `/generate` streams each segment once as `event: segment`, then `event: complete` with `finish_reason` and the segments as references without data, then `[DONE]`. Failure emits `event: error` with the error `message`, `type` and status `code`. Without `stream`, the response carries the joined text segments as `text` and `segments` with their data. Leave `return_logprob` unset: interleaved results carry no logprobs, so the default is dropped and setting it or another rollout artifact to true returns 400. Other pipelines refuse `stream=true` on `/generate`.
- `/v1/chat/completions` returns `choices[0].message.segments` beside the joined text, or streams each segment as `choices[0].delta.segment`, text segments also as `delta.content`.
- The SDK returns `UMMCompletionResult.segments` from `Client.completion` and yields `UMMGenerateChunk.segment` from `Client.generate` with `stream=True`, both from `sglang_omni.client.types`.

Decisions sample greedily. A sampling field the client sets, at the top level, in `stage_sampling.reasoner` or in `stage_params.reasoner`, applies to every decision. An SDK request sets a top level field by giving it a value other than its default, as on the native chat stage. Length and stop settings never apply to decisions, since the decision format bounds their length. Chat media and generation `image_path` and `video_path` options follow `--allowed-local-media-path` and `--allowed-media-domain` and are refused with 400 before any model work when the policy refuses them. Each Reasoner turn sends the whole conversation again, with the client's chat media and every image and video generated so far, so the Reasoner reads them again on every turn. Generated media travel inline from the orchestrator to the Reasoner stage: with eight videos at the 8 MiB inline limit, the ninth Reasoner turn carries about 89.5 MB and one request relays about 402.7 MB to the Reasoner in total. Client chat media are fetched again on every turn.

The orchestrator bounds each request under `stages.orchestrator.factory`: `max_dispatches` (17, eight generations and the Reasoner turns around them), `max_outputs` (17), `max_active` (32 requests at once, which also caps admission, so excess requests get 503), `max_retained_bytes` (128 MiB) and `timeout_s` (3600). A request may retain up to `max_retained_bytes` of host memory in the orchestrator process, so 32 active requests hold at most 4 GiB. `max_active` bounds that memory and Reasoner concurrency. Generation capacity follows the measured generation time instead: the generation stage serves one request at a time, the oldest request first, and the orchestrator plans for every active request the generations it may still run so that each ends before its deadline at the longest generation and Reasoner turn times measured in the last `timeout_s`, a queued generation at the longest time of its modality. A new request plans as many of its eight generations as end before its deadline behind the generations planned for the active requests, and is refused with 503 when they leave it fewer than an idle server would. When a longer time is measured, including the first one after startup, the orchestrator plans again in the generation stage's order. A request left without a planned generation is offered only a final answer, and a generation it already chose returns to the Reasoner as a receipt saying no time is left. A request that cannot finish even its queued generation and a final answer, with that generation at the shortest time measured for its modality, fails at once with 503 and a message saying why and that it can be retried later, and its generation is cancelled, instead of failing when `timeout_s` passes. A queued generation that fits only at that shortest time keeps its place, since the longest time of its modality may come from a longer generation of another request. A request whose plan was lowered gets a generation back when the requests after it keep theirs. Before the first generation after startup nothing is measured, so requests are admitted and the plan at the first measurement settles them, unless that first generation alone outlasts `timeout_s`. The largest output 17 dispatches can produce, eight videos at the 8 MiB inline limit and their turns with answers up to 8192 characters and prompts up to 4096, takes about 87 MiB, and the rest bounds the client's chat history. An answer longer than 8192 characters or a prompt longer than 4096 exceeds these server limits and fails the request with a server error. A request whose history leaves too little room is refused with 400 before any turn. Configuration loading refuses a `max_retained_bytes` or `max_outputs` that cannot hold the largest output of `max_dispatches`. A request that outlives `timeout_s` fails with an error naming it. A non-streaming request returns the status as its HTTP status. A streaming request has already received HTTP 200, so its error carries the status as `code`. Use a fresh request ID for every request, and close unfinished SDK iterators so the request is aborted.

## Direct generation

For native generation without interleaving, change `config_cls` to `Cosmos3PipelineConfig`. This uses visible GPU 0 and serves the native image, video and action routes described in [native media routes](../basic_usage/native_media_routes.md). SDK generation returns caller-owned saved files. Actions are served by the native action route.

The Policy-DROID checkpoint uses `raw_action_dim=8` in the joint position action space, as NVIDIA's reference policy server does. Match observation layout, units and normalization to the checkpoint.

## Reusing the serving pipeline

The interleaving controller is model agnostic. Another unified model implements `UMMAdapter` from `sglang_omni.pipeline.umm` with `start`, `advance`, `usage` and `close`, and subclasses `UMMPipelineConfig`, as `Cosmos3UMMAdapter` and `Cosmos3UMMPipelineConfig` do. Keep prompts, decision formats, media conversion and modality rules in the adapter. The developer reference describes the controller contract.

Qualify generated bytes in the following understanding turn, cancellation, deadlines and native ownership before enabling a new model. Interface reuse does not establish checkpoint quality or performance.
