# Cache Plane

SGLang-Omni keeps cache ownership split across the layer that owns the bytes:
upstream SGLang owns AR KV cache, relay owns transfer blobs, and model stages
own model-specific artifacts such as encoder outputs or uploaded-speaker
features. The cache plane is a model-neutral metadata layer over those owners.

## What It Owns

`sglang_omni.cache` records:

- stable `CacheKey` identity: namespace, artifact kind, digest, model/version,
  stage, and optional session
- `CacheOwner`: stage/process/GPU/device metadata for locality decisions
- `ArtifactHandle`: owner-specific pointer, such as a stage-output key
- lifecycle metadata: size, timestamps, TTL, ready/building state, and pin count
- local eviction, invalidation, and single-flight build coordination
- session-to-owner bindings and owner affinity ranking for locality-aware routing

It does not own tensor bytes, relay blobs, or SGLang KV blocks.

## Current Integrations

`StageOutputCache` can publish entries into a `LocalCachePlane` when constructed
with a `cache_namespace`. Qwen3-Omni image and audio encoder output caches use
this path, so repeated media outputs remain local to the stage cache while their
owner and size metadata are visible through the cache plane.

Qwen3-Omni preprocessing also keeps a small session media registry. When a
request carries `metadata.session_id` and `metadata.reuse_session_media=true`,
the preprocessor can reuse the previous media cache key for that session. If the
cache plane shows the encoder artifact is still ready, preprocessing builds only
the multimodal placeholder prompt and sends a cache-only encoder request. The
encoder stage then reads the artifact from `StageOutputCache` instead of
decoding pixels or rerunning the encoder. If the artifact has been evicted and
raw media references are still retained, preprocessing falls back to the normal
media path; otherwise the request must resend media.

`SpeakerArtifactCache` uses the same metadata path for uploaded-speaker
artifacts. It still stores artifacts in its bounded LRU and keeps the
voice-level invalidation contract.

## Application Scenario

The high-value scenario is a multimodal chat session: a user uploads an image or
video once, then asks several follow-up questions in the same session. The first
turn builds the encoder output and publishes the artifact.
Later turns can send only the new text plus:

```json
{
  "metadata": {
    "session_id": "conversation-123",
    "reuse_session_media": true
  }
}
```

This avoids redundant image/video encoder work. For Qwen3-Omni, the session
registry also keeps lightweight multimodal shape metadata such as
`image_grid_thw`, so follow-up turns can skip media decoding while still
expanding the correct placeholder tokens for M-RoPE and SGLang prefix/radix
identity. The cache entry is the encoder artifact keyed by media content and
preprocessing parameters, not the raw image bytes.

## Locality API

`LocalCachePlane.bind_session(session_id, owner)` records an explicit affinity
between a logical session and the owner that should be preferred for follow-up
work. Entries whose `CacheKey.session_id` is set also update that binding when
they are published or registered as in-flight builds.

`LocalCachePlane.rank_owners(selector, session_id=...)` returns `CacheAffinity`
scores grouped by owner. This is the hook a scheduler or router can use to pick
the worker/process/GPU with the best existing artifacts before falling back to
normal load balancing. The current stage integrations publish metadata only; no
scheduler route is changed by this API yet.

## Boundaries

The cache plane is intentionally thin:

- Stage runtime does not branch on cache behavior.
- Schedulers still own batching and request lifecycle.
- Model code still decides whether an artifact is cacheable and how to compute
  content identity.
- SGLang radix/KV cache stays upstream-owned; the cache plane can observe or
  route around it later, but does not mutate SGLang internals.

Future distributed implementations should preserve the same schema and owner
contract while replacing the local registry backend.
