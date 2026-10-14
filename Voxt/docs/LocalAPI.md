# Local API

Send recorded audio to Voxt from a script, a shortcut or another app, and get back the text Voxt would have typed: the same transcription model, enhancement prompt and dictionary you chose in Settings. Nothing is pasted on the Mac.

## Try it

1. Open **Settings → General → Local API** and turn on **Allow other apps to dictate through Voxt**.
2. Click **Copy** next to **Access token**.
3. Send a recording:

```sh
curl -H "Authorization: Bearer $VOXT_TOKEN" -H "Content-Type: audio/wav" \
  --data-binary @recording.wav http://127.0.0.1:47910/v1/dictations
```

```json
{"id":"5214EBF0-…","text":"Please send Alice fifty dollars by Friday.","rawText":"Please send Alice fifty dollars by Friday.","enhanced":false,"asrMs":531,"enhanceMs":0}
```

`text` is the result after your enhancement prompt and dictionary; `rawText` is the transcript before them. This example used Qwen3-ASR 0.6B with enhancement off. The result is also saved to History with the source **Local API**.

## Where it listens

Voxt binds `127.0.0.1` only, on port 47910 by default (change it in the same card). Other computers and phones cannot connect directly. To use it from another device, put your own tunnel in front of it, for example `tailscale serve --https=47911 http://127.0.0.1:47910`. Use a different public port: macOS refuses Voxt's bind when the tunnel already holds the same port.

Every request needs `Authorization: Bearer <token>`. **Regenerate** issues a new token and disconnects existing clients.

## Endpoints

| Request | Response |
| --- | --- |
| `GET /v1/health` | `{"ok": true, "app": "Voxt", "version": "…"}` |
| `GET /v1/capabilities` | API version, endpoints, accepted audio types, size limit, and whether the selected transcription and text models are ready |
| `POST /v1/dictations` | Body: audio bytes. `Content-Type`: `audio/wav`, `audio/x-m4a`, `audio/mp4`, `audio/aac`, `audio/mpeg` or `audio/x-caf`. Add `?enhance=0` to skip the text model. Returns `id`, `text`, `rawText`, `enhanced`, `asrMs`, `enhanceMs` |

`capabilities` example:

```json
{"api":{"major":1,"minor":0},"app":"Voxt","version":"1.15.0",
 "endpoints":["health","capabilities","dictations"],
 "audio":{"contentTypes":["audio/aac","audio/m4a","audio/mp4","audio/mpeg","audio/wav","audio/wave","audio/x-caf","audio/x-m4a","audio/x-wav"],"maxBodyBytes":26214400},
 "models":{"transcription":"ready","text":"off"}}
```

Model states are `ready`, `missing` (selected but not downloaded or not configured), `off` (no text model) and `unsupported` (Apple Speech cannot transcribe recorded files).

`/v1` only gains fields and endpoints. Clients should ignore fields they do not know. A breaking change would use `/v2`.

## Errors

Errors return `{"error": "…"}`.

| Status | When |
| --- | --- |
| 400 | Empty body or audio that cannot be read |
| 401 | Missing or wrong token |
| 404 / 405 | Unknown path or wrong method |
| 411 | Chunked request body; send `Content-Length` |
| 413 | Body larger than 25 MB |
| 415 | Unsupported audio type |
| 422 | No speech detected |
| 503 | Voxt is recording or in a meeting, or another API dictation is still running |
| 500 | The selected model failed, for example a remote provider could not be reached |

If the text model fails, the request still succeeds with the raw transcript and `"enhanced": false`.
