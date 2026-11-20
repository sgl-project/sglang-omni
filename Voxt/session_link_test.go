package main

import (
	"context"
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// llmShelf is installed language models, no speech model; opening
// broken fails.
type llmShelf struct {
	local.NotInstalled
	models     []local.Model
	broken     string
	substitute bool
}

func (f llmShelf) Installed(kind local.Kind) []local.Model {
	if kind == local.LLM {
		return f.models
	}
	return nil
}

func (f llmShelf) LLM(_ context.Context, id string) (local.Chat, local.Model, error) {
	m, _ := local.Choose(f.models, id)
	if f.substitute && len(f.models) > 0 {
		m = f.models[0]
	}
	if m.ID == f.broken {
		return nil, m, errors.New("no key in the Keychain")
	}
	return &llm.Client{Backend: &fakeChat{}}, m, nil
}

// TestResolverFallback: Translate's fallback model is opened beside its
// language model, and stands in for it when that one does not open;
// never for another feature, the same model, or a model not installed.
func TestResolverFallback(t *testing.T) {
	models := []local.Model{{Kind: local.LLM, ID: "deepseek", Name: "DeepSeek"}, {Kind: local.LLM, ID: "ollama", Name: "Ollama"}}
	for _, tc := range []struct {
		name               string
		kind               store.Kind
		main, fallback     string
		broken             string
		wantMain, wantBack string
	}{
		{"translate", store.KindTranslate, "deepseek", "ollama", "", "deepseek", "ollama"},
		{"automatic", store.KindTranslate, "", "ollama", "", "deepseek", "ollama"},
		{"none", store.KindTranslate, "deepseek", "", "", "deepseek", ""},
		{"the same model", store.KindTranslate, "", "deepseek", "", "deepseek", ""},
		{"not installed", store.KindTranslate, "deepseek", "gone", "", "deepseek", ""},
		{"rewrite has none", store.KindRewrite, "deepseek", "ollama", "", "deepseek", ""},
		{"the model does not open", store.KindTranslate, "deepseek", "ollama", "deepseek", "ollama", ""},
		{"the fallback does not open", store.KindTranslate, "deepseek", "ollama", "ollama", "deepseek", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := settings.Defaults()
			s.Features.Translate.LLM, s.Features.Translate.Fallback = tc.main, tc.fallback
			s.Features.Rewrite.LLM = tc.main
			svc, err := resolver(llmShelf{models: models, broken: tc.broken})(context.Background(), tc.kind, s, true)
			if err != nil {
				t.Fatal(err)
			}
			if svc.LLMInfo.ID != tc.wantMain || svc.FallbackInfo.ID != tc.wantBack || (svc.LLM != nil) != (tc.wantMain != "") || (svc.Fallback != nil) != (tc.wantBack != "") {
				t.Errorf("model %q, fallback %q; want %q, %q", svc.LLMInfo.ID, svc.FallbackInfo.ID, tc.wantMain, tc.wantBack)
			}
		})
	}
}

// TestResolverKeepsAnUnavailableChoiceClosed: a feature whose chosen model
// is not ready gets no other model, even if the opener substitutes one.
func TestResolverKeepsAnUnavailableChoiceClosed(t *testing.T) {
	models := []local.Model{{Kind: local.LLM, ID: "deepseek"}, {Kind: local.LLM, ID: "ollama"}}
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		t.Run(string(kind), func(t *testing.T) {
			s := settings.Defaults()
			s.Features.Transcribe.Enhance = true
			s.Features.Transcribe.LLM = "gone"
			s.Features.Translate.LLM = "gone"
			s.Features.Rewrite.LLM = "gone"
			svc, err := resolver(llmShelf{models: models})(context.Background(), kind, s, false)
			if err != nil {
				t.Fatal(err)
			}
			if svc.LLM != nil || svc.LLMInfo.ID != "" || svc.Fallback != nil {
				t.Errorf("unavailable choice opened %q with fallback %q", svc.LLMInfo.ID, svc.FallbackInfo.ID)
			}
		})
	}

	t.Run("translate uses only its explicit fallback", func(t *testing.T) {
		s := settings.Defaults()
		s.Features.Translate.LLM = "gone"
		s.Features.Translate.Fallback = "ollama"
		svc, err := resolver(llmShelf{models: models})(context.Background(), store.KindTranslate, s, false)
		if err != nil {
			t.Fatal(err)
		}
		if svc.LLMInfo.ID != "ollama" || svc.Fallback != nil {
			t.Errorf("model %q, fallback %q; want only the explicit fallback", svc.LLMInfo.ID, svc.FallbackInfo.ID)
		}
	})

	t.Run("an opener cannot substitute another model", func(t *testing.T) {
		s := settings.Defaults()
		s.Features.Rewrite.LLM = "ollama"
		svc, err := resolver(llmShelf{models: models, substitute: true})(context.Background(), store.KindRewrite, s, false)
		if err != nil {
			t.Fatal(err)
		}
		if svc.LLM != nil || svc.LLMInfo.ID != "" {
			t.Errorf("opener substituted %q for the chosen model", svc.LLMInfo.ID)
		}
	})

	t.Run("a fallback opener cannot substitute another model", func(t *testing.T) {
		s := settings.Defaults()
		s.Features.Translate.LLM = "deepseek"
		s.Features.Translate.Fallback = "ollama"
		svc, err := resolver(llmShelf{models: models, substitute: true})(context.Background(), store.KindTranslate, s, false)
		if err != nil {
			t.Fatal(err)
		}
		if svc.LLMInfo.ID != "deepseek" || svc.Fallback != nil || svc.FallbackInfo.ID != "" {
			t.Errorf("model %q, fallback %q; want no substituted fallback", svc.LLMInfo.ID, svc.FallbackInfo.ID)
		}
	})
}

// speechShelf is llmShelf with a speech model that counts its openings.
type speechShelf struct {
	llmShelf
	opened *int
}

func (f speechShelf) ASR(context.Context, string) (asr.Provider, local.Model, error) {
	*f.opened++
	return nil, local.Model{Kind: local.ASR, ID: "sensevoice"}, nil
}

// TestResolverWithoutSpeech: a session that hears nothing (a selection
// to translate) does not open the speech model, which may have to load
// again (#3838), yet gets its language model and fallback.
func TestResolverWithoutSpeech(t *testing.T) {
	models := []local.Model{{Kind: local.LLM, ID: "deepseek"}, {Kind: local.LLM, ID: "ollama"}}
	s := settings.Defaults()
	s.Features.Translate.LLM, s.Features.Translate.Fallback = "deepseek", "ollama"
	for _, speech := range []bool{false, true} {
		opened := 0
		svc, err := resolver(speechShelf{llmShelf{models: models}, &opened})(context.Background(), store.KindTranslate, s, speech)
		if err != nil {
			t.Fatal(err)
		}
		if want := map[bool]int{false: 0, true: 1}[speech]; opened != want {
			t.Errorf("speech %v: the speech model opened %d times; want %d", speech, opened, want)
		}
		if svc.LLMInfo.ID != "deepseek" || svc.FallbackInfo.ID != "ollama" {
			t.Errorf("speech %v: model %q, fallback %q", speech, svc.LLMInfo.ID, svc.FallbackInfo.ID)
		}
	}
}
