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
{"id":"AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE","text":"Please send Alice fifty dollars by Friday.","rawText":"Please send Alice fifty dollars by Friday.","enhanced":false,"asrMs":531,"enhanceMs":0}
```

`text` is the result after your enhancement prompt and dictionary; `rawText` is the transcript before them. This example used Qwen3-ASR 0.6B with enhancement off. The result is also saved to History with the source **Local API**.

## Where it listens

Voxt binds `127.0.0.1` only, on port 47910 by default (change it in the same card). Other computers and phones cannot connect directly. To use it from another device, put your own tunnel in front of it, for example `tailscale serve --https=47911 http://127.0.0.1:47910`. Use a different public port: macOS refuses Voxt's bind when the tunnel already holds the same port.

Every request needs `Authorization: Bearer <token>`.

Voxt Connector, the menu bar app that carries dictations from Voxt Mobile on an iPhone, does not need the token copied by hand. When it is signed by the same team as Voxt, it asks for the port and token over a Unix socket at the top of Voxt's sandbox container, `~/Library/Containers/com.voxt.Voxt/Data/connector.sock` (`~/Library/Application Support/Voxt/connector.sock` for an unsandboxed build), and Voxt asks you once: **Always Allow**, **Allow Once** or **Don't Allow**. Voxt checks the Connector's code signature (Voxt's own team, the bundle ID `com.voxt.VoxtConnector` or the one in `VoxtConnectorBundleIdentifier` of Voxt's Info.plist, and the hardened runtime) before it answers; an ad-hoc or unsigned Voxt does not open the socket.

In the App Sandbox (the release build), macOS lets Voxt inspect another process's code only when that code is somewhere the sandbox may read, which for an app is `/Applications`. So **install Voxt Connector in /Applications**. Anywhere else the look-up is refused, Voxt rejects the Connector, and the Connector shows its manual entry: copy the port and token from this card. Voxt checks both the Connector's running code and its bundle on disk, so a bundle changed after signing is refused too. After **Always Allow**, a Connector that passes these checks gets the token without a prompt.

## Endpoints

| Request | Response |
| --- | --- |
| `GET /v1/health` | `{"ok": true, "app": "Voxt", "version": "…"}` |
| `GET /v1/capabilities` | API version, endpoints, accepted audio types, size limit, and whether the selected transcription and text models are ready |
| `POST /v1/dictations` | Body: audio bytes. `Content-Type`: `audio/wav`, `audio/x-m4a`, `audio/mp4`, `audio/aac`, `audio/mpeg` or `audio/x-caf`. Add `?enhance=0` to skip the text model, and `?language=auto` (detect) or `?language=English` (a language name) to override your language settings for this recording. Returns `id`, `text`, `rawText`, `enhanced`, `asrMs`, `enhanceMs` |
| `POST /v1/history/sync` | Exchanges text history with Voxt Mobile when **Sync history with Voxt Mobile** and History are both on (403 otherwise). JSON body: `storeID` (from the previous answer, absent the first time), `cursor`, `limit`, `items` (`id`, `text`, `createdAt`, optional `mode`, `elapsedSeconds`) and `deletedIDs`. Returns `storeID`, the Mac's `items` and `deletedIDs` changed after `cursor`, the next `cursor`, and `hasMore` when another page follows; the Mac keeps a revision for each entry and a tombstone for each deletion. Audio and meeting transcripts never leave the Mac |

| `POST /v1/compose` | JSON body: `audio` (base64), `contentType` (one of the audio types above), optional `language` (as for dictations) and `options`. Runs the selected text model with an instruction for `options.mode`. Returns the dictation fields plus `mode`, and `status: "not_applied"` with the unchanged selection when an edit could not be applied |
| `POST /v1/corrections` | JSON body `{"original": "…", "edited": "…"}`, each 1 to 2,000 characters and different. When the user fixed one misheard word or name in a dictated passage, returns `{"correction": {"original": "…", "replacement": "…"}}` (a pair that is in both passages), otherwise `{"correction": null}`. Uses the selected text model; nothing is stored, and the caller decides whether to add the pair to its dictionary |
| `GET /v1/compose/prompts` | `{"prompts": {"dictate": "…", "translate": "…", "edit": "…", "write": "…"}}`: the built-in instruction of each mode |

`options` of a compose request:

| Field | Meaning |
| --- | --- |
| `mode` | `dictate` (clean up the speech), `translate` (needs `targetLanguage`, a BCP-47 tag such as `zh-Hans`), `edit` (apply the spoken instruction to `selectedText`) or `write` (draft from the spoken request) |
| `selectedText`, `contextBefore`, `contextAfter` | Text at the caller's cursor: up to 8,000 characters selected and 2,000 on each side |
| `style` | `automatic`, `formal`, `casual` or `concise` |
| `vocabulary` | Up to 200 preferred spellings, each up to 100 characters |
| `languageVariants` | At most one per language: `en-US`, `en-GB`, `en-AU`, `en-CA`, `zh-Hans`, `zh-Hant`, `es-ES`, `es-MX`, `fr-FR`, `fr-CA`, `pt-PT`, `pt-BR` |
| `styleExamples` | Up to 8 `{"before", "after"}` wording pairs of 1 to 120 characters that keep the same digits; used only with `style: automatic` and not for `translate` |
| `customPrompt` | Replaces the built-in instruction of the mode for this request, up to 4,000 characters |

Every compose mode needs a working text model; without one the request fails with 503 instead of returning the raw transcript.

`capabilities` example:

```json
{"api":{"major":1,"minor":0},"app":"Voxt","version":"1.15.0",
 "endpoints":["health","capabilities","dictations","compose","compose/prompts"],
 "audio":{"contentTypes":["audio/aac","audio/m4a","audio/mp4","audio/mpeg","audio/wav","audio/wave","audio/x-caf","audio/x-m4a","audio/x-wav"],"maxBodyBytes":26214400},
 "models":{"transcription":"ready","text":"off"},
 "locations":{"transcription":"local","text":"local"}}
```

`locations` says where the selected models run: `local`, `remote` (a cloud provider) or `unknown` (Apple Speech without on-device recognition, or anything Voxt cannot vouch for). A client that must keep audio and text on the Mac sends `X-Voxt-Model-Policy: local-only` with a dictation or compose request; Voxt then answers 403 unless both are `local`.

Model states are `ready`, `missing` (selected but not downloaded or not configured), `off` (no text model) and `unsupported` (Apple Speech cannot transcribe recorded files, or the local runtime cannot run the selected model).

Example requests, responses and errors live in [`Voxt/Mobile/contract`](../Mobile/contract): `LocalAPIContractTests` checks them against the real router, and Voxt Mobile's tests decode the same files.

`/v1` only gains fields and endpoints. Clients should ignore fields they do not know. A breaking change would use `/v2`.

## Errors

Errors return `{"error": "…"}`.

| Status | When |
| --- | --- |
| 400 | Empty body, audio that cannot be read, or a `language` that is not `auto` or a language name |
| 401 | Missing or wrong token |
| 403 | `X-Voxt-Model-Policy: local-only`, and a selected model is not local |
| 404 / 405 | Unknown path or wrong method |
| 411 | Chunked request body; send `Content-Length` |
| 413 | Body larger than 25 MB |
| 415 | Unsupported audio type |
| 422 | No speech detected |
| 503 | Voxt is recording or in a meeting, another API dictation is still running, or a compose request has no working text model |
| 500 | The selected model failed, for example a remote provider could not be reached |

If the text model fails, the request still succeeds with the raw transcript and `"enhanced": false`.
