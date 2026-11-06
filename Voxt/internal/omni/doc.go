// Package omni runs VoxtNew's local speech models on sglang-omni's native
// MLX runtime (sglang_omni_mlx/native): C++ servers, one process per
// model, spoken to over loopback HTTP and WebSockets. It supervises them
// and implements internal/asr's FileTranscriber and Streamer for the
// curated Qwen3-ASR models, and internal/vad's Model for Silero VAD.
//
// # Models
//
// Only a few models, each with parameters tuned for it (models.go): the
// speech models Qwen3-ASR 1.7B 8-bit (the default, accurate) and 0.6B
// 4-bit (fast), and Silero VAD v6. They are keyed by the catalogue's
// stable ids (ASRAccurate "asr.accurate", ASRFast "asr.fast", VADSilero
// "vad.silero"); the Hugging Face repository and its pinned revision are
// recorded beside them. Downloads, checksums and install state are
// internal/models'; this package gets a model's folder. An ASR may carry
// the VAD (WithVAD): its VAD method is local.VoiceActivity, how a session
// finds the Silero model beside the speech model.
//
// # Processes (Runtime)
//
// A Runtime knows where the binaries are (Locate: $VOXTNEW_OMNI_RUNTIME,
// else VoxtNew.app/Contents/Resources/omni) and owns the servers. A
// server starts on first use: qwen3_asr_server --supervised on 127.0.0.1
// port 0, which prints {"event":"ready","port":…} on stdout once its model
// is loaded; a server that says nothing within StartTimeout is killed.
// Uses hold a lease; a server with no lease for IdleTimeout stops (idle
// unload; SetIdleTimeout changes it while servers run; Unload stops a
// model's server, before its files are deleted).
// One speech model runs at a time (a switch stops the other once
// its uses end, so two copies of weights are never resident, as Voxt);
// Silero has a server of its own, shared by every detector. A server that
// dies (or hangs: a Final whose stream stops for StallTimeout) is replaced
// at the next use. Close stops them all: {"command":"shutdown"} on stdin,
// then SIGTERM, then SIGKILL; a server whose stdin closes (VoxtNew
// crashed) stops by itself. Config.OnModel hears each model loaded,
// busy, idle and unloaded (internal/resident: the family's runtime file),
// and Config.BeforeLoad runs before a server loads its model (the
// family's memory budget).
//
// # Recognition (Voxt's Qwen3-ASR policy)
//
// Final (POST /v1/audio/transcriptions, SSE): the language hint (the
// user's main language when the model knows it), the dictionary's terms
// as the context bias (one per line, the most recent first, at most 500
// terms and 8000 characters), a token budget from the audio's length
// (28 tokens a second + 64, within 256…8192), Voxt's stop rules and audio
// layout, and recordings over 60 s (Voxt: 1200 s; the encoder's memory
// grows with a request's audio) cut at their quietest 100 ms, the
// chunks sharing the budget and the first chunk's detected language
// forced on the rest. On audio without speech the model writes its
// context out: a request's text that ends in a sentence of the context's
// terms is checked against the same audio recognised without it, and
// the terms it did not hear go (echo.go).
//
// Live (WS /v1/realtime): a manual-turn session with the language hint;
// the server decodes 100 ms after the first audible audio, then once a
// second (launch flags). Its segments make the partial text. At Finish
// the socket is dropped and the Final above runs on everything sent, as
// Voxt: the Final, with its context bias, is the text that counts. The
// stream's events are partials, then exactly one final or error, then the
// channel closes.
//
// # Voice activity (Silero)
//
// VAD.Speech posts a recording to /v1/vad/speech_timestamps; VAD.Open
// streams it over /v1/vad/stream in 512-sample chunks, one probability
// each. The session uses them where the energy detector was (gating a
// stream, trimming a whole recording); the energy detector stays the
// fallback.
//
// Everything here is goroutine-safe and blocks: call it off the main
// thread. Errors are *asr.Error values (provider "omni").
//
// Imports: asr, lang, resident, vad.
package omni
