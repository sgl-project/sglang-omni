package providers

import (
	"context"
	"slices"
	"sync"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// Services is base (the local models) with the remote language models
// that are ready listed after the local ones: a feature uses one when the
// user chose it, or when no local language model is installed. Goroutine-
// safe; the app sets the ready list on the main thread (SetReady).
type Services struct {
	base local.Services
	// name is what the UI calls a setup; open builds its client (it reads
	// the key, off the main thread).
	name func(settings.Provider) string
	open func(ctx context.Context, p settings.Provider) (local.Chat, error)

	mu    sync.Mutex
	ready []settings.Provider
}

// NewServices is base with the remote models SetReady lists.
func NewServices(base local.Services, name func(settings.Provider) string, open func(context.Context, settings.Provider) (local.Chat, error)) *Services {
	return &Services{base: base, name: name, open: open}
}

// SetReady sets the setups that can be used (Ready).
func (s *Services) SetReady(list []settings.Provider) {
	s.mu.Lock()
	s.ready = list
	s.mu.Unlock()
}

// remote is the ready setup of model id.
func (s *Services) remote(id string) (settings.Provider, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.ready {
		if p.ID == id {
			return p, true
		}
	}
	return settings.Provider{}, false
}

func (s *Services) Installed(kind local.Kind) []local.Model {
	out := slices.Clip(s.base.Installed(kind)) // appending never writes into base's array
	if kind != local.LLM {
		return out
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.ready {
		out = append(out, local.Model{Kind: local.LLM, ID: p.ID, Name: s.name(p)})
	}
	return out
}

func (s *Services) ASR(ctx context.Context, id string) (asr.Provider, local.Model, error) {
	return s.base.ASR(ctx, id)
}

func (s *Services) LLM(ctx context.Context, id string) (local.Chat, local.Model, error) {
	m, ok := local.Choose(s.Installed(local.LLM), id)
	if p, remote := s.remote(m.ID); ok && remote {
		c, err := s.open(ctx, p)
		return c, m, err
	}
	return s.base.LLM(ctx, id)
}
