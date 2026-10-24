# Loopback end-to-end run

What this checks: Voxt built from this tree answers every Local API request the iPhone app sends except history sync (`POST /v1/history/sync`), on 127.0.0.1, with real models, and its answers have the shapes in [`Voxt/Mobile/contract`](../../../contract).

How it was hosted: the unattended run served the Local API from the XCTest host, through `Voxt/VoxtTests/LocalAPILoopbackServeTests.swift`, which turns the Local API on with the real app delegate and keeps it up. That uses the same router, pipeline, transcription and text models as the app; only the launch path differs, because launching Voxt.app itself (the path in the Voxt Mobile README) asks macOS for Accessibility at startup, and that prompt needs a person to answer it. A run against a launched Voxt.app is added separately once it has been done with someone present.

## Run it

```sh
python3 Voxt/Mobile/docs/evidence/loopback-e2e/run.py --models <folder with mlx-audio and mlx-llm> --runtime <runtime>/bin/qwen3_asr_server
```

- Local transcription runs only on the Native Omni runtime, which a test host built by `xcodebuild` does not contain, so `--runtime` (or `VOXT_OMNI_RUNTIME`) names the `qwen3_asr_server` of a runtime built with `sglang_omni_mlx/native/scripts/build_runtime.sh`, and the run passes it to the test host as `VOXT_OMNI_RUNTIME`.
- Builds the TestDebug test bundle with `CODE_SIGNING_ALLOWED=NO`, `-skipPackagePluginValidation` and `-packageAuthorizationProvider netrc`, so no Keychain is used for packages.
- Runs the test host with its own home (`CFFIXED_USER_HOME`) whose model folder links to the given models, a random token in `VOXT_LOCAL_API_TOKEN` (read only by Debug builds) and a free loopback port; secure storage stays in memory under XCTest, and the Voxt Connector link does not open in an unsigned build.
- Once `/v1/health` answers, writes `build/loopback-e2e/local-api.env` (mode 600) with `VOXT_E2E_URL` and `VOXT_E2E_TOKEN` for other clients such as the iPhone simulator run, and the file `build/loopback-e2e/ready`; both are removed when the run ends.
- Sends the requests below with `curl` and keeps the Local API up for `--serve-seconds` (default 900).

`--host app` prints the build command and the steps for the same requests against a launched Voxt.app.

## Checks

| Check | Request | Compared with the contract |
| --- | --- | --- |
| health | `GET /v1/health` | keys, `app` and `ok` of `health.json` |
| capabilities | `GET /v1/capabilities` | equal to `capabilities.json` apart from `version` |
| dictation-jfk | `POST /v1/dictations?language=auto`, `testdata/jfk.wav` | keys of `dictation-response.json`; the text contains "country" |
| dictation-zh | `POST /v1/dictations?language=auto`, `testdata/zh.wav` | the text is in Chinese characters |
| compose-translate | `POST /v1/compose` with `compose-request.json` and jfk.wav | keys of `compose-response.json`; mode translate, text in Chinese |
| compose-prompts | `GET /v1/compose/prompts` | one built-in prompt per mode |
| correction | `POST /v1/corrections` with `correction-request.json` | the pair or null, as in `correction-response.json` (only when the contract lists corrections) |
| error-unauthorized, error-notFound, error-wrongMethod | wrong token, unknown path, `GET /v1/dictations` | equal to `errors.json` |
| dictation-after-idle | `POST /v1/dictations?language=auto`, `testdata/jfk.wav`, sent after both models unloaded on idle (about two minutes after the last request), which also starts Voxt's idle memory reclamation | answers within 90 s with the text of jfk.wav; a stall behind the reclamation shows as no answer |

Model output is not deterministic, so text is compared by what it must contain, and everything else exactly. The checker was itself checked against a fake server that answers from the contract: 10 of 10 checks pass with the right token and 1 of 10 (the unauthorized check) with a wrong one; that fake server does not unload models, so dictation-after-idle is checked against Voxt itself: it fails on a build that stalls behind the reclamation and passes on one that does not.

## Results

`results/summary.md` and one JSON file per check, written by the run (ids replaced by the sample id, no token or local path).
