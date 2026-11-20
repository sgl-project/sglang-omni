package main

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
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

// TestResolverSwitch: with Models.LLMSwitch every feature falls back to
// the next model that opens, after Translate's own fallback model.
func TestResolverSwitch(t *testing.T) {
	// deepseek and openai are external APIs; custom (a loopback setup),
	// ollama and lmstudio are servers on this Mac.
	models := []local.Model{{Kind: local.LLM, ID: "deepseek"}, {Kind: local.LLM, ID: "openai"}, {Kind: local.LLM, ID: "custom"}, {Kind: local.LLM, ID: "ollama"}, {Kind: local.LLM, ID: "lmstudio"}}
	for _, tc := range []struct {
		name               string
		kind               store.Kind
		on                 bool
		fallback, broken   string
		wantMain, wantBack string
	}{
		{"off", store.KindRewrite, false, "", "", "deepseek", ""},
		{"rewrite skips the external API", store.KindRewrite, true, "", "", "deepseek", "custom"},
		{"transcribe skips the external API", store.KindTranscribe, true, "", "", "deepseek", "custom"},
		{"the next one does not open", store.KindRewrite, true, "", "custom", "deepseek", "ollama"},
		{"the model does not open", store.KindRewrite, true, "", "deepseek", "custom", ""},
		{"translate's own external fallback first", store.KindTranslate, true, "openai", "", "deepseek", "openai"},
		{"translate's fallback does not open", store.KindTranslate, true, "openai", "openai", "deepseek", "custom"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := settings.Defaults()
			s.Providers.LLM = []settings.Provider{{ID: "custom", Endpoint: "http://127.0.0.1:8080/v1", Model: "m"}}
			s.Models.LLMSwitch = tc.on
			s.Features.Transcribe.Enhance = true
			s.Features.Transcribe.LLM, s.Features.Translate.LLM, s.Features.Rewrite.LLM = "deepseek", "deepseek", "deepseek"
			s.Features.Translate.Fallback = tc.fallback
			svc, err := resolver(llmShelf{models: models, broken: tc.broken})(context.Background(), tc.kind, s, false)
			if err != nil {
				t.Fatal(err)
			}
			if svc.LLMInfo.ID != tc.wantMain || svc.FallbackInfo.ID != tc.wantBack || (svc.Fallback != nil) != (tc.wantBack != "") {
				t.Errorf("model %q, fallback %q; want %q, %q", svc.LLMInfo.ID, svc.FallbackInfo.ID, tc.wantMain, tc.wantBack)
			}
		})
	}
}

// TestResolverSwitchKeepsTextOnTheMac: with only external APIs besides
// the feature's own model, the switch finds no fallback, so a failure
// never sends the text to a provider the feature did not choose.
func TestResolverSwitchKeepsTextOnTheMac(t *testing.T) {
	models := []local.Model{{Kind: local.LLM, ID: "ollama"}, {Kind: local.LLM, ID: "deepseek"}, {Kind: local.LLM, ID: "openai"}}
	s := settings.Defaults()
	s.Models.LLMSwitch = true
	s.Features.Transcribe.Enhance = true
	s.Features.Transcribe.LLM, s.Features.Rewrite.LLM = "ollama", "deepseek"
	svc, err := resolver(llmShelf{models: models})(context.Background(), store.KindTranscribe, s, false)
	if err != nil {
		t.Fatal(err)
	}
	if svc.LLMInfo.ID != "ollama" || svc.Fallback != nil {
		t.Errorf("model %q, fallback %q; want ollama and none: Transcribe's text would go to an external API it did not choose", svc.LLMInfo.ID, svc.FallbackInfo.ID)
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

// TestReturnFinishQueuedStateChange: a Return the helper still reports as
// Finish from the old configuration (its new one is on its way) stops
// nothing once the setting is off or the recording is not a plain
// Transcribe one.
func TestReturnFinishQueuedStateChange(t *testing.T) {
	a := &app{sess: session.Snapshot{State: session.Recording, Kind: store.KindTranscribe}}
	a.prefs = settings.Defaults()
	a.prefs.Features.Transcribe.ReturnFinishes = true
	cfg := a.hotkeyPlan().Configure(true)
	if !slices.ContainsFunc(cfg.Bindings, func(b proto.HotkeyBinding) bool { return b.ID == "finish" }) {
		t.Fatal("finish was not configured")
	}
	e := proto.HotkeyEventParams{Binding: "finish", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap}
	a.prefs.Features.Transcribe.ReturnFinishes = false
	if d := hotkey.Route(e, a.hotkeyPhase(), a.sess.HoldBinding, a.strictFeature(), a.returnFinishes()); d.Action != hotkey.None {
		t.Errorf("opt-out event stopped the session: %+v", d)
	}
	a.sess.Kind = store.KindTranslate
	a.prefs.Features.Transcribe.ReturnFinishes = true
	if d := hotkey.Route(e, a.hotkeyPhase(), a.sess.HoldBinding, a.strictFeature(), a.returnFinishes()); d.Action != hotkey.None {
		t.Errorf("event from the old Transcribe configuration stopped Translate: %+v", d)
	}
	a.sess.Kind = store.KindTranscribe
	a.sess.Note = true
	if d := hotkey.Route(e, a.hotkeyPhase(), a.sess.HoldBinding, a.strictFeature(), a.returnFinishes()); d.Action != hotkey.None {
		t.Errorf("event from the old Transcribe configuration stopped a note dictation: %+v", d)
	}
}

// TestReturnFinishPlanScope: the helper gets Return only for a Transcribe
// recording with the setting on, never for another feature or idle work.
// With Paste last on Return, its tap routes to Stop in the same cases and
// nowhere else: opted out, or another feature's session, Return waits.
func TestReturnFinishPlanScope(t *testing.T) {
	for _, tc := range []struct {
		name  string
		state session.State
		kind  store.Kind
		note  bool
		on    bool
		want  bool
	}{
		{name: "Transcribe recording", state: session.Recording, kind: store.KindTranscribe, on: true, want: true},
		{name: "setting off", state: session.Recording, kind: store.KindTranscribe, want: false},
		{name: "idle", state: session.Idle, kind: store.KindTranscribe, on: true, want: false},
		{name: "recognizing", state: session.Recognizing, kind: store.KindTranscribe, on: true, want: false},
		{name: "Translate", state: session.Recording, kind: store.KindTranslate, on: true, want: false},
		{name: "Rewrite", state: session.Recording, kind: store.KindRewrite, on: true, want: false},
		{name: "note", state: session.Recording, kind: store.KindTranscribe, note: true, on: true, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := &app{sess: session.Snapshot{State: tc.state, Kind: tc.kind, Note: tc.note}}
			a.prefs = settings.Defaults()
			a.prefs.Features.Transcribe.ReturnFinishes = tc.on
			cfg := a.hotkeyPlan().Configure(tc.state.Live())
			got := slices.ContainsFunc(cfg.Bindings, func(b proto.HotkeyBinding) bool { return b.ID == "finish" })
			if got != tc.want {
				t.Errorf("Return bound %v, want %v: %+v", got, tc.want, cfg.Bindings)
			}
			// With Paste last on Return, Build drops Finish for it; its tap
			// must route to Stop in the same cases and nowhere else.
			a.prefs.Features.PasteLast = settings.Hotkey{Keys: "Return"}
			e := proto.HotkeyEventParams{Binding: hotkey.PasteLast, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap}
			d := hotkey.Route(e, a.hotkeyPhase(), a.sess.HoldBinding, a.strictFeature(), a.returnFinishes())
			if stopped := d.Action == hotkey.Stop; stopped != tc.want {
				t.Errorf("paste last on Return stops %v, want %v: %+v", stopped, tc.want, d)
			}
		})
	}
}
