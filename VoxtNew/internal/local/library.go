package local

import (
	"context"
	"errors"
	"fmt"
	"log"

	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
)

// ASROpener opens speech model id (a models catalogue id) from dir, its
// installed and verified folder. It is called off the main thread for
// each session that needs the model: it may start the runtime and load
// the model, and should be quick once they are.
type ASROpener func(ctx context.Context, id, dir string) (asr.Provider, error)

// LLMOpener opens language model id from dir, as ASROpener.
type LLMOpener func(ctx context.Context, id, dir string) (Chat, error)

// Openers are the runtimes behind the models; nil: that runtime is not
// in this build (ErrNoRuntime).
type Openers struct {
	ASR ASROpener
	LLM LLMOpener
	// Unload stops what a runtime has loaded of model id, before its
	// files are deleted (Library.Delete); nil: nothing to stop.
	Unload func(id string)
}

// ErrNoRuntime: the model is installed, but this build has no runtime
// to run it.
var ErrNoRuntime = errors.New("local: runtime not available")

// Library is the local models of the app: those models.Manager installed,
// opened by the Openers. A model a session opens is claimed for VoxtNew
// first when it is not yet (another app installed it): the family then
// keeps it while VoxtNew uses it. Goroutine-safe.
type Library struct {
	models *models.Manager
	open   Openers
	name   func(id string) string
}

// NewLibrary is the models of m, opened by open; name says a model's
// Name (the UI's words, uitext.ModelName), nil: its id.
func NewLibrary(m *models.Manager, open Openers, name func(id string) string) *Library {
	if name == nil {
		name = func(id string) string { return id }
	}
	return &Library{models: m, open: open, name: name}
}

// Installed lists kind's installed models in the catalogue's order, the
// recommended first. Memory only.
func (l *Library) Installed(kind Kind) []Model {
	var out []Model
	for _, m := range models.Catalog() {
		if string(m.Kind) == string(kind) && l.models.Installed(m.ID) {
			out = append(out, Model{Kind: kind, ID: m.ID, Name: l.name(m.ID), Features: m.Features})
		}
	}
	return out
}

// Delete unloads model id from its runtime (Openers.Unload), then
// removes it from VoxtNew (models.Manager.Delete: its files go unless
// another ActNow app uses them, or whoever does with scope AllApps).
func (l *Library) Delete(ctx context.Context, id string, scope fammodels.Scope) ([]fammodels.Result, error) {
	if l.open.Unload != nil {
		l.open.Unload(id)
	}
	return l.models.Delete(ctx, id, scope)
}

// Dir is the folder of an installed model ("" when it is not), for the
// openers' wiring (the speech runtime's VAD: models.VADSilero).
func (l *Library) Dir(id string) string { return l.models.Dir(id) }

func (l *Library) ASR(ctx context.Context, id string) (asr.Provider, Model, error) {
	m, ok := Choose(l.Installed(ASR), id)
	if !ok {
		return nil, Model{}, NotInstalledError(ASR, id)
	}
	if l.open.ASR == nil {
		return nil, Model{}, fmt.Errorf("%w: speech recognition (%s)", ErrNoRuntime, m.ID)
	}
	l.claim(ctx, m.ID)
	p, err := l.open.ASR(ctx, m.ID, l.models.Dir(m.ID))
	if err != nil {
		return nil, Model{}, err
	}
	return p, m, nil
}

func (l *Library) LLM(ctx context.Context, id string) (Chat, Model, error) {
	m, ok := Choose(l.Installed(LLM), id)
	if !ok {
		return nil, Model{}, NotInstalledError(LLM, id)
	}
	if l.open.LLM == nil {
		return nil, Model{}, fmt.Errorf("%w: language model (%s)", ErrNoRuntime, m.ID)
	}
	l.claim(ctx, m.ID)
	c, err := l.open.LLM(ctx, m.ID, l.models.Dir(m.ID))
	if err != nil {
		return nil, Model{}, err
	}
	return c, m, nil
}

// claim makes sure VoxtNew claims model id (and the VAD a speech model
// uses) before it opens; a failure is only logged: the session goes on.
func (l *Library) claim(ctx context.Context, id string) {
	ids := []string{id}
	if m, ok := models.Lookup(id); ok {
		ids = append(ids, m.Needs...)
	}
	for _, x := range ids {
		if err := l.models.EnsureClaimed(ctx, x); err != nil {
			log.Printf("local: claim %s: %v", x, err)
		}
	}
}
