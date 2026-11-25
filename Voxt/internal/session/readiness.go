package session

import (
	"context"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

// ASRReadiness is the speech-side state a session needs before recognition.
type ASRReadiness struct {
	ASR     asr.Provider
	ASRInfo Model
	VAD     vad.Model
	Err     error
}

// LLMOutcome is the completed state of a cleanup model resolution.
type LLMOutcome string

const (
	LLMNotRequested LLMOutcome = "not_requested"
	LLMUnavailable  LLMOutcome = "unavailable"
	LLMAvailable    LLMOutcome = "available"
)

// LLMReadiness is the cleanup-side state a session needs after recognition.
type LLMReadiness struct {
	LLM          LLM
	LLMInfo      Model
	Fallback     LLM
	FallbackInfo Model
	Outcome      LLMOutcome
	Err          error
}

// ASRResolver resolves speech state without opening cleanup providers.
type ASRResolver func(ctx context.Context, kind Kind, s settings.Settings) ASRReadiness

// LLMResolver resolves cleanup state without opening speech providers.
type LLMResolver func(ctx context.Context, kind Kind, s settings.Settings) LLMReadiness

type runInitialization struct {
	split                bool
	asrDone              chan ASRReadiness
	llmDone              chan LLMReadiness
	termsDone            chan []store.Term
	snippetsDone         chan []snippet.Snippet
	llmStarted           bool
	llmReady             bool
	llmOutcome           LLMOutcome
	cleanupDecisionReady bool
	cleanupRequested     bool
	termsReady           bool
	snippetsReady        bool
	sceneReady           bool
	cleanupReady         bool
}
