# OmniModelArtifacts

`OmniModelPreparation.prepare(from:to:replaceExisting:validate:checkCancellation:)`
explicitly copies an installed model under a source reader into exclusive sibling
staging, dereferences regular-file links, structurally validates and atomically
publishes through the existing publication API. Original weights are unchanged;
replacement is opt-in and destination readers exclude it. Failed hidden staging
is retained, not automatically resumed. Budgets are 100,000 entries and 128 GiB
of file bytes; the initial directory listing is not memory-bounded by that entry
check. This is cooperating preparation, not hostile namespace, immutable blob,
model compatibility, checksum, recovery or power-loss certification.

A dependency-free, Foundation-only Swift 6.2 package for **structural** validation of local safetensors directories. It does not import MLX or instantiate models. The macOS minimum is 10.15 for throwing FileHandle APIs; Linux is also supported.

```swift
import OmniModelArtifacts

try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory)
```

The synchronous public enum and its API are `nonisolated`; callers should schedule file I/O off the main actor. Failures throw public `OmniModelArtifactValidationError`, a `LocalizedError`, `Equatable`, and `Sendable` enum with directory, missing-weights, and file-with-reason cases.

## What is checked

- An exact root entry named `config.json`, containing a UTF-8 JSON object. Duplicate fields and trailing commas are rejected; signed/fractional/exponent configuration values remain supported. No configuration keys or architecture-specific semantics are required.
- At least one top-level entry whose `pathExtension` is exactly lowercase `safetensors`, matching the native loaders' selection. Hidden entries are included; nested files and uppercase extensions are not selected. Every selected file is checked, not just the first shard.
- Files resolve to regular, nonempty files and can be opened for reading. Internal and external symlinks, including copied Hugging Face cache links to blobs, are allowed. Broken links and nonregular targets fail.
- An eight-byte little-endian unsigned header length, followed by a UTF-8 JSON object. Config reads are limited to 16 MiB; **prefix plus header reads** are limited to 16 MiB per safetensors file (the JSON header limit is 16 MiB minus eight bytes). Lengths are bounded before allocation or conversion to `Int`.
- Each file has at least one tensor descriptor, containing exactly `dtype`, `shape`, and `data_offsets`. Optional `__metadata__` is a string-to-string object. Duplicate header keys, including escaped equivalent spellings, and trailing array/object commas are rejected.
- Shapes and the two offsets use unsigned integer JSON tokens, not booleans, strings, fractional or exponent notation. Foundation's permissive numeric conversions are not relied upon for this check.
- Shape products and dtype byte lengths use checked `UInt64` arithmetic. Rank-zero scalars and zero-element tensors are supported. Arithmetic overflow is rejected.
- Byte offsets are relative to the payload, ordered, in bounds, and agree with shape/dtype byte counts. Sorted ranges cover the entire payload with no overlaps, holes, or trailing unreferenced bytes. Truncated headers and payload lengths fail.

Only the prefix and header are read from weight files; tensor payloads are never loaded or memory-mapped. File length comes from the opened handle. JSON decoding and directory enumeration allocate memory; this is a per-file read bound, not a total directory-size, shard-count, CPU-time, or process-memory quota.

## Dtype compatibility

Byte widths follow the byte-aligned dtypes in the [safetensors v0.6.2 format implementation](https://github.com/huggingface/safetensors/blob/v0.6.2/safetensors/src/tensor.rs):

| Bytes | Dtypes |
| --- | --- |
| 1 | `BOOL`, `I8`, `U8`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0` |
| 2 | `I16`, `U16`, `F16`, `BF16` |
| 4 | `I32`, `U32`, `F32` |
| 8 | `I64`, `U64`, `F64` |

Unknown dtypes and packed sub-byte types (`F4`, `F6_E2M3`, `F6_E3M2`) are rejected rather than assigned a guessed width. Header/config size limits, closed tensor descriptors, and requiring tensors in each shard are deliberate restrictions, not claims to accept every valid safetensors file. Structural dtype acceptance does not imply runtime support.

## Boundaries

Success does **not** establish tensor-name/model compatibility, tokenizer validity, configuration semantics, a complete shard set from an index, cross-shard key consistency, numerical payload validity, GPU behavior, or successful inference. Shard-index files and unrelated artifacts are not interpreted. Empty JSON configurations and arbitrary structurally valid tensor names can pass.

This is not an integrity or authorization boundary. Payload bytes and hashes are not verified. Symlink targets need not stay inside the supplied directory; consumer path authorization is separate. Enumeration, attribute checks, opening, and later consumption are not one atomic snapshot. Files may change during or after validation. Filesystem snapshot, TOCTOU, checksum/integrity, and atomic-publication acceptance remain pending/separate responsibilities.

## Tests

```sh
swift test -c debug
swift test -c release
```

The tests import the public module without `@testable`. They write authentic binary fixtures (little-endian prefixes, padded JSON headers, tensor payload bytes), malformed variants, internal/external symlinks, inclusive read-limit fixtures, and an 8 GiB sparse file. No model download, GPU, Python library, or network is needed. A privileged runner that can read mode-000 files skips the permission-rejection test explicitly. Package tests do not certify macOS app/Xcode integration or real model compatibility.
