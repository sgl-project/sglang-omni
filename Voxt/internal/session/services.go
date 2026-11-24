package session

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// Services are the speech and language models of one session.
type Services struct {
	// ASR is nil when no speech model is ready (the session fails with
	// ProblemNoASR when it needs one).
	ASR     asr.Provider
	ASRInfo Model
	// BuiltinHints: the speech model also gets the built-in words' hints
	// (dict.BuiltinHints). Only the on-device model they were measured on
	// does (eval.md "Built-in vocabulary"); a remote service or another
	// model gets the user's terms alone.
	BuiltinHints bool
	// LLM is nil when the feature has no language model ready.
	LLM     LLM
	LLMInfo Model
	// Fallback is Translate's fallback model, or with the Language models
	// page's switch another language model, asked once when LLM fails;
	// nil: none.
	Fallback     LLM
	FallbackInfo Model
	// VAD is the voice activity model (Silero) that finds the speech in
	// the recording; nil: the energy detector of internal/vad does, as it
	// does when the model fails (vad_model.go).
	VAD vad.Model
}

// Resolver opens the models a session of kind uses with settings s; it
// may block (it may start a runtime). speech false: the session hears
// nothing (a selection to translate, a typed text), so no speech model
// is opened and ASR stays nil (#3838). A model that is not ready is nil
// in Services (an error wrapping ErrNoASR works too).
type Resolver func(ctx context.Context, kind Kind, s settings.Settings, speech bool) (Services, error)
