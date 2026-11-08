package providers

import (
	"context"
	"slices"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Services is base (the local models) with the remote language models and
// speech services that are ready listed after the local ones of their
// kind: a feature uses one when the user chose it, or when no local model
// of its kind is installed. Goroutine-safe; the app sets the ready lists
// on the main thread (SetReady).
type Services struct {
	base local.Services
	// name is what the UI calls a setup of a kind; openLLM and openASR
	// build its client (they read the key, off the main thread).
	name    func(local.Kind, settings.Provider) string
	openLLM func(ctx context.Context, p settings.Provider) (local.Chat, error)
	openASR func(ctx context.Context, p settings.Provider) (asr.Provider, error)

	mu    sync.Mutex
	ready map[local.Kind][]settings.Provider
}

// NewServices is base with the remote models SetReady lists.
func NewServices(base local.Services, name func(local.Kind, settings.Provider) string,
	openLLM func(context.Context, settings.Provider) (local.Chat, error),
	openASR func(context.Context, settings.Provider) (asr.Provider, error)) *Services {
	return &Services{base: base, name: name, openLLM: openLLM, openASR: openASR, ready: map[local.Kind][]settings.Provider{}}
}

// SetReady sets the setups of kind that can be used (Ready, SpeechReady).
func (s *Services) SetReady(kind local.Kind, list []settings.Provider) {
	s.mu.Lock()
	s.ready[kind] = list
	s.mu.Unlock()
}

// remote is the ready setup of kind's model id.
func (s *Services) remote(kind local.Kind, id string) (settings.Provider, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := slices.IndexFunc(s.ready[kind], func(p settings.Provider) bool { return p.ID == id })
	if i < 0 {
		return settings.Provider{}, false
	}
	return s.ready[kind][i], true
}

func (s *Services) Installed(kind local.Kind) []local.Model {
	out := slices.Clip(s.base.Installed(kind)) // appending never writes into base's array
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.ready[kind] {
		speech, _ := LookupSpeech(p.ID)
		out = append(out, local.Model{Kind: kind, ID: p.ID, Name: s.name(kind, p), Stream: kind == local.ASR && speech.Streams(SpeechModel(p))})
	}
	return out
}

func (s *Services) ASR(ctx context.Context, id string) (asr.Provider, local.Model, error) {
	m, ok := local.Choose(s.Installed(local.ASR), id)
	if p, remote := s.remote(local.ASR, m.ID); ok && remote {
		c, err := s.openASR(ctx, p)
		return c, m, err
	}
	return s.base.ASR(ctx, id)
}

func (s *Services) LLM(ctx context.Context, id string) (local.Chat, local.Model, error) {
	m, ok := local.Choose(s.Installed(local.LLM), id)
	if p, remote := s.remote(local.LLM, m.ID); ok && remote {
		c, err := s.openLLM(ctx, p)
		return c, m, err
	}
	return s.base.LLM(ctx, id)
}
