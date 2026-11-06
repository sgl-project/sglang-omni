# Helper protocol fixtures

Golden transcripts of the helper protocol (`docs/helper-protocol.md`), one JSON object per line:

```json
{"to":"helper","msg":{…}}   // app → helper (a request)
{"to":"app","msg":{…}}      // helper → app (a response or a notification)
```

The format is AudioSmith's engine fixtures', with the helper in the engine's place. Both sides decode every line:

- `go test ./internal/helper/proto` decodes each line strictly with the Go types (unknown fields fail), checks that it encodes back to the same JSON, and checks the sequencing rules: one response per request, a job's notifications before its response and after its `capture.started`, frames without gaps in `seq` and `offset` and counted by the result, `job.cancel` and `capture.stop` answered after the job they end, valid hotkey bindings and events. `TestFixturesCoverMethods` fails when a method has no fixture; `TestHelloListsEveryRequest` when the handshake does not list every request.
- `cd helper && swift test` (FixtureTests) decodes each line with the Swift types and checks that it encodes back to the same JSON: a field one side lacks, or with another optionality, fails.

The fake helper (`internal/helper/helpertest`) can replay a fixture (`replay=PATH`).

`audio/tone-22050-stereo.wav` (0.4 s of 440 Hz, 22.05 kHz, 16-bit stereo: the left channel at 0.5, the right at 0.25) is the test-only file input's (`VOXTNEW_HELPER_CAPTURE_FILE`): `CaptureFileTests` and `TestMacHelperCaptureFile` check that it comes out resampled to 16 kHz and mixed to mono at 0.375.

Change a fixture together with the Go types, the Swift types and the spec.
