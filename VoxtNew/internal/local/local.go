// Package local is the seam between a session and the models on this
// Mac: which speech and language models are installed, and the clients a
// session uses (internal/asr's FileTranscriber and Streamer, and Chat).
//
// Library is the Services of the app: the models internal/models
// installed, opened by the runtimes' openers (Openers: the Omni native
// MLX servers for speech, the voxtnew-llm process for language). Without
// a runtime an open fails with ErrNoRuntime. NotInstalled lists nothing
// and fails every open with ErrNotInstalled.
//
// Installed reads memory only, so that views may call it while they
// build; ASR and LLM may block (they may start a runtime and load a
// model): call them off the main thread.
//
// A speech model's asr.Provider may also implement VoiceActivity: the
// voice activity model (Silero VAD, installed with the speech models)
// sessions use to find speech, where the energy detector would.
//
// Imports: asr, llm, models, vad.
package local

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// Kind is a kind of model: speech recognition or a language model.
type Kind string

const (
	ASR Kind = "asr"
	LLM Kind = "llm"
)

// Model is one installed model, as a feature chooses it.
type Model struct {
	Kind Kind
	// ID is the model's stable id in the catalogue; settings.Feature's
	// ASR and LLM hold it ("" there: the first installed).
	ID string
	// Name is what the UI shows: a plain description of what it is good
	// at, not its family, size or quantisation (the plan's model policy).
	Name string
	// Features are the only features it serves ("translate"), nil for
	// every one (ForFeature).
	Features []string
}

// For says whether m serves feature ("transcribe", "translate",
// "rewrite").
func (m Model) For(feature string) bool {
	return len(m.Features) == 0 || slices.Contains(m.Features, feature)
}

// ForFeature is the models of list that serve feature, in order.
func ForFeature(list []Model, feature string) []Model {
	var out []Model
	for _, m := range list {
		if m.For(feature) {
			out = append(out, m)
		}
	}
	return out
}

// Chat is a language model as a session uses it (session.LLM).
type Chat interface {
	Chat(ctx context.Context, req llm.Request) (llm.Response, error)
}

// Services are the local models. Implementations are goroutine-safe.
type Services interface {
	// Installed lists kind's installed models, the recommended one
	// first. Memory only.
	Installed(kind Kind) []Model
	// ASR opens speech model id ("": the first installed): an
	// asr.FileTranscriber, an asr.Streamer, or both. It fails with an
	// error wrapping ErrNotInstalled when the model is not installed.
	ASR(ctx context.Context, id string) (asr.Provider, Model, error)
	// LLM opens language model id ("": the first installed), as ASR.
	LLM(ctx context.Context, id string) (Chat, Model, error)
}

// VoiceActivity is what a speech model's asr.Provider implements when it
// comes with a voice activity model (Silero VAD on the Omni runtime,
// beside Qwen3-ASR). VAD is nil when that model is not installed; a
// session then finds speech by energy (internal/vad).
type VoiceActivity interface {
	VAD() vad.Model
}

// ErrNotInstalled: the model asked for (or any model of its kind) is not
// installed.
var ErrNotInstalled = errors.New("local: model not installed")

// NotInstalledError is the error of an open whose model is not
// installed; it matches ErrNotInstalled.
func NotInstalledError(kind Kind, id string) error {
	if id == "" {
		return fmt.Errorf("%w: no %s model", ErrNotInstalled, kind)
	}
	return fmt.Errorf("%w: %s model %q", ErrNotInstalled, kind, id)
}

// Choose is the model a feature uses among the installed list: want
// when it is there, else the first; false when the list is empty.
func Choose(list []Model, want string) (Model, bool) {
	want = strings.TrimSpace(want)
	for _, m := range list {
		if m.ID == want {
			return m, true
		}
	}
	if len(list) > 0 {
		return list[0], true
	}
	return Model{}, false
}

// NotInstalled is the Services of a VoxtNew without models (tests):
// nothing is installed, and every open fails with ErrNotInstalled.
type NotInstalled struct{}

func (NotInstalled) Installed(Kind) []Model { return nil }

func (NotInstalled) ASR(_ context.Context, id string) (asr.Provider, Model, error) {
	return nil, Model{}, NotInstalledError(ASR, id)
}

func (NotInstalled) LLM(_ context.Context, id string) (Chat, Model, error) {
	return nil, Model{}, NotInstalledError(LLM, id)
}
