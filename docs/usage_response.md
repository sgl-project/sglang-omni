# Images response and usage

Scope: `/v1/images/generations` and `/v1/images/edits`.
The shared response builder matches SenseNova commit
`f3a2befe0902ce25d5784bdc540ff66ba70bcc3c`.

- Each image has `output_format`, determined from its actual encoded bytes.
- `usage.image_count` counts returned images.
- `usage.width` and `usage.height` describe the actual common output size.
- `usage.input_tokens_details` contains text and reference-image token counts.
- `input_tokens` and `total_tokens` equal text tokens only, excluding image tokens.
- Missing model token statistics remain unavailable; output pixels are not tokens.
- Repeated outputs for the same prompt do not multiply its input token count.

The main-based branch supplies the response builder and image usage carrier.
The SenseNova validation branch supplies the model producer and mounts both APIs.
No SenseNova model implementation is imported into the main-based branch.

The previous Chat usage extensions, streaming usage options, and `/generate`
cached-token mapping are withdrawn. Existing base token usage is unchanged.
Metrics remain a separate branch and are outside this change.
