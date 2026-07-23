# Reference Encode Service

`ReferenceEncodeService` owns reusable mechanics for ad-hoc TTS reference
encoding:

- cache-keyed lookup;
- byte-bounded LRU storage;
- same-key single-flight while an encode is in progress;
- failure propagation to waiters without caching failures;
- artifact store/load conversion and caller-owned return values;
- basic cache statistics.

The service is for ad-hoc request references. Registered or uploaded voices
continue to use `SpeakerArtifactCache`, because they have a different lifetime,
key space, and invalidation path.

## API Shape

The implementation lives in `sglang_omni/scheduling/reference_encoder.py`.

```python
@dataclass(frozen=True)
class ReferenceEncodeKey:
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str
    input_key: str
    options_key: str = ""


class ReferenceEncodeHook(Generic[InputT, ArtifactT, StoredT]):
    def normalize_input(self, raw_input: Any) -> InputT: ...
    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None: ...
    def encode_one(self, item: InputT) -> ArtifactT: ...
    def store_artifact(self, artifact: ArtifactT) -> StoredT: ...
    def load_artifact(self, stored: StoredT) -> ArtifactT: ...
    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool: ...


class KeyedReferenceEncodeHook(ReferenceEncodeHook[InputT, ArtifactT, StoredT]):
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str

    def input_key(self, item: InputT) -> str | None: ...
    def options_key(self, item: InputT) -> str: ...


class TensorReferenceEncodeHook(
    KeyedReferenceEncodeHook[InputT, Tensor, Tensor]
):
    storage_dtype: torch.dtype | None
    output_dtype: torch.dtype | None


class ReferenceEncodeService(Generic[InputT, ArtifactT, StoredT]):
    def get_or_encode(self, raw_input: Any, *, desc: str | None = None) -> ArtifactT: ...
    def stats(self) -> dict[str, int]: ...
```

`ReferenceEncodeService` is synchronous and thread-first. Existing TTS
preprocessing and encoder stages already run synchronous model code inside
`SimpleScheduler` or `ThreadedSimpleScheduler`, so adding an async surface would
force nested event-loop management without changing the underlying work.

## Responsibility Split

The service owns mechanics:

- `_inflight` single-flight map;
- `StageOutputCache` access under a service-owned lock;
- cache insertion, byte budget, and LRU eviction;
- follower waits, timeout handling, and exception fanout;
- no-poison-on-failure behavior;
- stats for hits, misses, merges, failures, uncacheable inputs, entries, bytes,
  and evictions.

The hook owns model semantics:

- request-specific input normalization;
- cacheability;
- model/checkpoint/config key parts;
- `encode_one`;
- artifact device and dtype policy;
- store/load conversion;
- revalidation for mutable local files.

Hooks with structured identity should inherit `KeyedReferenceEncodeHook`. They
provide key metadata, `input_key`, and `encode_one`, plus input normalization
when the service does not already receive the typed item. The default builds
`ReferenceEncodeKey` and rechecks input and option keys before insertion.
Tensor-producing encoders should inherit `TensorReferenceEncodeHook`, which
also stores a cache-owned CPU clone and returns a caller-owned clone in
`output_dtype`. Structured artifacts such as nested prompt dictionaries keep
their own store/load policy on top of `KeyedReferenceEncodeHook`.

## Cache-Key Contract

`ReferenceEncodeKey` must include every input that can change the encoded
artifact identity:

- model family or checkpoint identity;
- model or encoder revision;
- encoder implementation and config hash;
- artifact kind;
- normalized reference content identity;
- encode options that affect the artifact.

Local reference files should use
`reference_path_cache_key(path, trust_stat=False)` and revalidate before cache
insert. Bytes and data-URI payloads should key by the bytes or original payload
actually consumed by the model hook. Remote URLs should not be cached by URL
string alone unless an upstream fetch layer has already materialized immutable
content identity.

## Artifact Policy

Hooks should store cache-owned artifacts, usually detached CPU tensors or a
small CPU dictionary of detached tensors. `load_artifact` must return a
caller-owned object, commonly by cloning and moving to the expected dtype or
device. `TensorReferenceEncodeHook` provides this policy through
`storage_dtype` and `output_dtype`. The service enforces the byte budget on the
stored representation.

If a stored artifact is larger than `max_bytes`, the leader request and all
same-key followers still receive a result, but the artifact is not inserted into
the LRU.

## Failure And Waiters

For a cacheable key:

1. Cache hits return `hook.load_artifact(stored)`.
2. If another request is already encoding the same key, followers wait on the
   leader future.
3. The leader encodes once, stores the artifact representation, optionally
   inserts it into the LRU, resolves waiters, and removes the in-flight entry.

Leader failures are propagated to waiters and are not cached. The next request
can retry as a new leader. A follower timeout does not remove the leader's
in-flight entry.

## M4a And M4b Boundary

This document covers **M4a only**: the ad-hoc reference cache and same-key
single-flight that ships today.

**M4b (different-key batch coalescing) is not implemented and is a non-goal
here**; it is described only to mark the scope boundary. Do not add M4b runtime
code until profiling proves it is worth the extra scheduling surface.

Before building M4b, run cold-cache workloads with different reference audio per
request at concurrency 8 and 16 for FishAudio S2-Pro, Qwen3-TTS, and
MOSS-TTS Local. Track preprocessing/reference-encode p50/p95, end-to-end TTFA
and latency, throughput, cache hit/miss/merge counts, and GPU/CPU utilization.
Build M4b for a model only if different-key reference encode remains a top
bottleneck and batching gives at least 15% p95 latency reduction or 20%
throughput improvement versus M4a.

If M4b is built later, it should be an opt-in extension of
`ReferenceEncodeService`, not the default path:

- add explicit hook capability, for example `can_encode_batch()` defaulting to
  `False`, and call `encode_batch(items)` only for hooks that opt in;
- add service knobs such as `max_batch_size=1` and `max_batch_wait_ms=0`;
- preserve M4a ordering: normalize input, compute cache key, check cache,
  merge same-key inflight work, and only then enqueue distinct cache-miss
  leaders for different-key batching;
- use an internal queue that drains up to `max_batch_size` or
  `max_batch_wait_ms`, calls one hook batch encode, and then stores,
  revalidates, cache-inserts, and resolves each item independently;
- on a batch failure, retry per item so one bad reference does not fail the
  whole batch.

Model rollout should stay evidence-driven. MOSS-TTS Local already has its own
batched reference encoder and should not be migrated just to fit this generic
surface. FishAudio S2-Pro is the first plausible candidate only if profiling
shows real benefit; its batched path would decode/resample each reference, pad
waveforms, call the codec once, and split outputs while preserving parity with
`encode_one`. Qwen3-TTS should remain M4a-only unless the upstream wrapper
exposes a safe batch primitive for `create_voice_clone_prompt`.
