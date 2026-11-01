# LLM protocol fixtures

Golden transcripts of the llm protocol (`docs/llm-protocol.md`), one JSON object per line:

```json
{"to":"llm","msg":{…}}   // app → voxtnew-llm (a request)
{"to":"app","msg":{…}}   // voxtnew-llm → app (a response or a notification)
```

Both sides decode every line:

- `go test ./internal/llmrt/wire` decodes each line strictly with the Go types (unknown fields fail), checks that it encodes back to the same JSON, and checks the sequencing: one response per request, a job's `chat.delta` before its response, `job.cancel` answered after the job it ends. `TestFixturesCoverMethods` fails when a method has no fixture; `TestHelloListsEveryRequest` when the handshake does not list every request.
- `cd llm && swift test` (FixtureTests) decodes each line with the Swift types and checks that it encodes back to the same JSON: a field one side lacks, or with another optionality, fails.

The fake process (`internal/llmrt/llmrttest`) speaks the same lines.

Change a fixture together with the Go types, the Swift types and the spec.
