package main

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type cleanupShelf struct {
	local.Services
	opens atomic.Int32
}

func (s *cleanupShelf) LLM(ctx context.Context, id string) (local.Chat, local.Model, error) {
	s.opens.Add(1)
	return s.Services.LLM(ctx, id)
}

// The session resolves its app override before calling this real opener.
// Off must avoid opening even an available model, not just discard its answer.
func TestResolverTranscribeCleanup(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kind    store.Kind
		cleanup bool
		opens   int32
	}{
		{"Transcribe off", store.KindTranscribe, false, 0},
		{"Transcribe on", store.KindTranscribe, true, 1},
		{"Translate unchanged", store.KindTranslate, false, 1},
		{"Rewrite unchanged", store.KindRewrite, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shelf := &cleanupShelf{Services: llmShelf{models: []local.Model{{Kind: local.LLM, ID: "ready", Name: "Ready"}}}}
			s := settings.Defaults()
			s.Features.Transcribe.Enhance = tc.cleanup
			svc, err := resolver(shelf)(context.Background(), tc.kind, s)
			if err != nil || shelf.opens.Load() != tc.opens || (svc.LLM != nil) != (tc.opens != 0) {
				t.Errorf("opens %d, LLM %v, error %v", shelf.opens.Load(), svc.LLM != nil, err)
			}
		})
	}
}

// Full app sessions use the actual resolver and fake services. The override
// controls both opening the model and its requests, without changing defaults.
func TestSessionAppCleanupModelOpening(t *testing.T) {
	for _, on := range []bool{false, true} {
		t.Run(map[bool]string{false: "Off with global on", true: "On with global off"}[on], func(t *testing.T) {
			chat := &fakeChat{text: "Cleaned text."}
			set, models, _ := fileSetup(t, "raw words", chat)
			shelf := &cleanupShelf{Services: models}
			front := "com.tinyspeck.slackmacgap"
			a := newSessionApp(t, script(t, hotkeyIn(300, front), hotkeyIn(800, front))+",front="+front, func(s *settings.Settings) {
				set(s)
				s.Features.Transcribe.Enhance = !on
			}, shelf)
			rules := e2eRules()
			rules.Groups[1].TranscribeCleanup = appctx.CleanupOff
			if on {
				rules.Groups[1].TranscribeCleanup = appctx.CleanupOn
			}
			a.setRules(rules)
			a.pump("cleanup override session", func() bool { return a.sess.ID != "" && a.sess.State.Ended() })
			wantText, wantOpens := "raw words", int32(0)
			if on {
				wantText, wantOpens = "Cleaned text.", 1
			}
			if a.sess.State != session.Done || !a.sess.Inserted || a.sess.Result != wantText || shelf.opens.Load() != wantOpens || len(chat.requests()) != int(wantOpens) {
				t.Errorf("end %+v, opens %d, chats %d", a.sess, shelf.opens.Load(), len(chat.requests()))
			}
			if a.sess.Scene == nil || a.sess.Scene.GroupID() != "g-chat" || a.settings.Get().Features.Transcribe.Enhance != !on {
				t.Errorf("scene %+v or global cleanup changed", a.sess.Scene)
			}
			if pastes := a.pasted(); len(pastes) != 1 || pastes[0].Text != wantText || len(pastes[0].Press) != 1 {
				t.Errorf("paste/auto-key %+v", pastes)
			}
		})
	}
}
