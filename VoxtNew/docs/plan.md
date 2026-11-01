# VoxtNew plan (summary)

*The full plan is kept with the project's notes (draft v0, 2026-10-16); this is what the code needs to know. Milestones M0–M4 here are VoxtNew's. The M1–M6 that comments and older docs mention (`settings.go`, behavior.md) are Auralink's milestones, from before the copy.*

## What VoxtNew is

A lean macOS voice input app on **local models only**: 转录 Transcribe, 翻译 Translate, 转写 Rewrite, 应用增强 App enhancement and the 词典 Dictionary, with the history, overlay, shortcuts, menu bar, setup guide, settings and updates. No remote models, meetings, notes, file transcription or sync. It is Auralink (the same author's Go/MyGo app on remote models) with its remote layer replaced, plus the Omni behaviour of the old Swift Voxt (`../Voxt`).

## Runtimes

- **Speech recognition and VAD**: sglang-omni's native MLX servers (`sglang_omni_mlx/native`, C++, no Python): `qwen3_asr_server` (also `--model-kind silero_vad`) and its siblings. HTTP `/v1/audio/transcriptions`, WebSocket `/v1/realtime`, `/v1/vad/stream`, `/v1/vad/speech_timestamps`; models pinned by revision and sha256 (`ci/models.json`). One process per model on a loopback port, started on first use, ready handshake on stdout, unloaded when idle, restarted after a crash, all stopped on quit (Voxt's `OmniASRRuntime.swift`).
- **Language models**: mlx-swift-lm, plus a few GGUF models (llama.cpp), in a Swift process of its own, `voxtnew-llm` (separate from the helper: model memory and crashes stay apart, the helper stays small and stateless), streaming over loopback HTTP, supervised from Go like the speech servers.
- **Models**: a few best choices, tuned by us; the UI describes what each is good at and which languages it speaks, not its family, size or quantisation. First picks (to be measured on mac-air): Qwen3-ASR 1.7B 8-bit (default) and 0.6B 4-bit; silero-vad-v6 (built in); Qwen3.5 4B OptiQ 4-bit (default), Qwen3.5 2B 4-bit, Hy-MT2 1.8B Q4_K_M GGUF (translation).

## Milestones

| | What | Where it plugs in |
|---|---|---|
| M0 | The skeleton: Auralink copied and renamed, its remote layer removed behind `internal/local` (nothing installed), the Local models page as an empty state; tests, vet and archtest green; the app builds and starts on the Mac | done |
| M1 | Local transcription: `internal/omni` (supervisor; Final over HTTP multipart, Live over `/v1/realtime`, Silero VAD; each model's request policy from Voxt: language hint, the dictionary as context bias, token budget, chunking, in `omni/models.go` and `policy.go`), `internal/models` (catalogue, resumable Hugging Face downloads, checks, storage, delete, disk use); Silero gates and trims the recording (the energy VAD is the fallback); the runtime packaged in the app (`scripts/build-omni.sh`); Transcribe end to end on the Mac | done: `localOpeners().ASR` (models_omni.go) opens `asr.FileTranscriber` + `asr.Streamer` with their VAD |
| M2 | Local language models: `voxtnew-llm` and its downloads; Translate, Rewrite and Transcribe's clean-up | done: `localOpeners().LLM` (llm_link.go) returns a `local.Chat` (an `llm.Client` over `internal/llmrt`'s backend; the process speaks the llm protocol on stdio, docs/llm-protocol.md) |
| M3 | Real-model quality: the eval set, the language-hint policy, the clean-up's example turn, the dictionary measured, the Mac regression (App enhancement came from Auralink; Silero VAD and the dictionary as context bias with M1) | done: docs/eval.md, `scripts/mac-regress.sh` |
| M4 | The Local models page in full (a loaded model's memory), settings and setup-guide polish, packaging (the helper, `omni/bin/*` and `llm/*` in the app: `scripts/release.sh`), signing every nested binary with the hardened runtime (no sandbox: it starts processes), notarisation prepared, updates from actnow.dev, the system proxy for downloads, CI at the repository root | done: docs/release.md; what needs the certificate or actnow.dev is listed there |

After M0, the Omni client and supervisor, the model downloader and the LLM runtime can go in parallel, each with fake-server tests on Linux, then come together on the Mac.

## Test Mac

`mac-air` (M4, 24 GB, macOS 26.7, Xcode 26.6; about 32 GB of disk free): `scripts/mac.sh`, `MAC_DIR=code/VoxtNew`. Delete large build trees and downloaded models after a test.
