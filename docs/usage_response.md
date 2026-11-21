# Images response and usage

Scope: `/v1/images/generations` and `/v1/images/edits`. The shared response builder follows the field semantics in SenseNova commit `f3a2befe0902ce25d5784bdc540ff66ba70bcc3c`.

- Each image has `output_format`, determined from its actual encoded bytes.
- `usage.image_count` counts returned images.
- `usage.width` and `usage.height` describe the actual common output size.
- `usage.input_tokens_details` contains text and reference-image token counts.
- `input_tokens` and `total_tokens` equal text tokens only, excluding image tokens.
- Missing model token statistics remain unavailable; output pixels are not tokens.
- Repeated outputs for the same prompt do not multiply its input token count.

An ordinary result carries one image in `image_b64`, several in `images_b64`, and producer statistics in `image_usage`. The client preserves string images and lists containing only strings. The existing `image` result field is unchanged.

A UMM result reports these statistics in one `usage` segment whose data contains `image_usage` and the producing `stage_name`. Non-streaming final results expose these fields on the client chunk. A streamed request already received the segment data, so its final result leaves these fields unset; the streamed usage chunk identifies the producing stage.

UMM image segments can use `InlineImage` data, whose byte count and SHA-256 must match its inline data URL. Producer token statistics in ordinary results and non-streaming UMM usage segments are validated strictly and cannot be negative.

This module supplies the shared response builder and client usage carriers. Model integrations supply token statistics and register the Images endpoints. No model implementation is included.

Existing Chat and `/generate` token usage behavior is unchanged. Metrics are outside this scope.
