package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestPillThinkingLevel (#3940, #3957): while the language model works
// on Transcribe's text, or Translate's or Rewrite's answer, on a service
// that takes thinking levels, the pill ends its row with a labelled
// Thinking capsule, inside the face, in English and Chinese, in one line
// or in lines; it opens the level menu by the pill, above it at the
// bottom of the screen and under it at the top. A service without levels
// shows none.
func TestPillThinkingLevel(t *testing.T) {
	t.Cleanup(func() { i18n.SetLocale("en") })
	long := &llm.Thinking{Mode: llm.ThinkBudget, Budget: 16384} // the longest label
	for _, c := range []struct {
		lang       string
		lines, top bool
	}{{"en", false, false}, {"zh-Hans", false, true}, {"en", true, true}, {"zh-Hans", true, false}} {
		a, _, _ := newTestApp(t)
		a.measureHook = nil // the view measures its texts, as on the Mac: a guess may be narrower than Linux's fonts
		a.updateSettings(func(s *settings.Settings) {
			s.Overlay.LiveLines, s.General.Language = c.lines, c.lang
			s.Providers.LLM = []settings.Provider{{ID: "zhipu"}}
			if c.top {
				s.Overlay.Position = "top"
			}
		})
		label := askAgainText(long, settings.Generation{})
		snap(a, session.Snapshot{Kind: store.KindTranscribe, State: session.Composing, CanSkip: true, LLM: session.Model{ID: "zhipu"}, Thinking: long, Partial: "so the release moves to thursday"})
		tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		a.overlaySync() // the labels measured
		tt.SetSize(a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		r, ok := tt.Find(label)
		if !ok {
			t.Fatalf("%s: no %q in %q", c.lang, label, firstTexts(tt))
		}
		if right := float32(layout.Pad + a.ov.w); r.X+r.W > right {
			t.Errorf("%s lines %v: the capsule ends at %v, past the face's %v", c.lang, c.lines, r.X+r.W, right)
		}
		click(t, tt, label)
		if !a.answer.langOpen || !a.answer.pillMenu || !a.ov.lang.shown {
			t.Fatalf("%s: menu open %v pill %v shown %v", c.lang, a.answer.langOpen, a.answer.pillMenu, a.ov.lang.shown)
		}
		m, f := a.ov.lang.frame, a.ov.frame
		if m.X+m.Width != f.X+f.Width {
			t.Errorf("%s: the menu ends at %v, the pill at %v", c.lang, m.X+m.Width, f.X+f.Width)
		}
		if c.top && m.Y+layout.Pad <= f.Y+f.Height-layout.Pad || !c.top && m.Y+m.Height-layout.Pad >= f.Y+layout.Pad {
			t.Errorf("%s top %v: menu %+v over the pill %+v", c.lang, c.top, m, f)
		}
		mt := hudTester(a.langMenuView, m.Width, m.Height)
		frames(mt)
		for _, s := range []*llm.Thinking{nil, {Mode: llm.ThinkOff}, {Mode: llm.ThinkOn}, long} {
			if !has(mt, thinkLevelTitle(s, settings.Generation{})) {
				t.Errorf("%s: menu has no %q: %q", c.lang, thinkLevelTitle(s, settings.Generation{}), firstTexts(mt))
			}
		}
		click(t, tt, label) // again: it closes
		if a.answer.langOpen || a.ov.lang.shown {
			t.Errorf("%s: the capsule again left the menu open", c.lang)
		}

		// Translate's speech and Rewrite's answer: the same capsule; a
		// service without levels: none.
		for _, s := range []session.Snapshot{
			{ID: "s2", Kind: store.KindTranslate, State: session.Composing, CanSkip: true, LLM: session.Model{ID: "zhipu"}, Partial: "hello"},
			{ID: "s3", Kind: store.KindRewrite, State: session.Composing, LLM: session.Model{ID: "zhipu"}, Raw: "make it shorter"},
			{ID: "s4", Kind: store.KindTranscribe, State: session.Composing, CanSkip: true, LLM: session.Model{ID: "llm.balanced"}, Partial: "hello"},
		} {
			snap(a, s)
			frames(tt)
			if want := s.LLM.ID == "zhipu"; has(tt, askAgainText(nil, settings.Generation{})) != want {
				t.Errorf("%s: %s on %s: capsule %v, want %v: %q", c.lang, s.Kind, s.LLM.ID, !want, want, firstTexts(tt))
			} else if want && a.ov.passThrough {
				t.Errorf("%s: %s's pill lets the click on its capsule through", c.lang, s.Kind)
			}
		}
	}
}

// TestPillThinkingLevelEndToEnd (#3940): a level chosen in the pill's
// menu while the language model cleans up the dictation ends that
// request and asks again for the same text with the level; only that
// answer is pasted and kept, the snapshot says the level, and the
// service's setting stays.
func TestPillThinkingLevelEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "Hello, VoxtNew.", delay: time.Minute}
	set, keys, _ := fileSetup(t, "hello aura link", chat)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, set, keys)
	a.pump("the model's step", func() bool {
		return a.sess.State == session.Composing && a.sess.CanSkip && len(chat.requests()) == 1
	})
	chat.mu.Lock()
	chat.delay = 0
	chat.mu.Unlock()
	// The pill's menu (the local model here takes no levels, so no
	// capsule). The cancel shortcut closes it first: the step goes on.
	a.answer.langOpen, a.answer.thinkMenu, a.answer.pillMenu = true, true, true
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	time.Sleep(50 * time.Millisecond)
	if a.answer.langOpen || a.answer.pillMenu || a.sess.State != session.Composing || len(chat.requests()) != 1 {
		t.Fatalf("Escape on the menu: open %v, %s, %d requests", a.answer.langOpen, a.sess.State, len(chat.requests()))
	}
	a.answer.langOpen, a.answer.thinkMenu, a.answer.pillMenu = true, true, true
	off := llm.Thinking{Mode: llm.ThinkOff}
	a.chooseThinking(&off)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if s := a.sess; s.State != session.Done || s.Problem != nil || s.Result != "Hello, VoxtNew." || s.Thinking == nil || *s.Thinking != off {
		t.Fatalf("end %s %+v %q %+v", s.State, s.Problem, s.Result, s.Thinking)
	}
	if a.answer.langOpen || a.answer.pillMenu {
		t.Error("the menu stayed open")
	}
	reqs := chat.requests()
	if len(reqs) != 2 || reqs[0]["thinking"] != nil || reqs[1]["thinking"] != off {
		t.Errorf("requests %d: %v", len(reqs), reqs)
	}
	a.pump("the paste", func() bool { return len(a.pasted()) > 0 })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "Hello, VoxtNew." {
		t.Errorf("pasted %+v", p)
	}
	if es := a.history(); len(es) != 1 || es[0].Status != store.StatusDone || es[0].Result != "Hello, VoxtNew." {
		t.Errorf("history %+v", es)
	}
}

// TestThinkingDefaultNamed (#3957): the menu's first row, the service's
// own setting, names that level ("On (default)"), never "As in
// settings"; the capsule says it too. A change on the Language models
// page shows in the menu at once: both read the one setting.
func TestThinkingDefaultNamed(t *testing.T) {
	t.Cleanup(func() { i18n.SetLocale("en") })
	for _, lang := range []string{"en", "zh-Hans"} {
		a, _, _ := newTestApp(t)
		a.updateSettings(func(s *settings.Settings) {
			s.General.Language = lang
			s.Providers.LLM = []settings.Provider{{ID: "deepseek", Generation: settings.Generation{Thinking: llm.ThinkOn}}}
		})
		snap(a, session.Snapshot{Kind: store.KindRewrite, State: session.Composing, LLM: session.Model{ID: "deepseek"}, Raw: "make it shorter"})
		on, off := i18n.T("%s (default)", i18n.T("On")), i18n.T("Ask again: Thinking (%s, default)", i18n.T("On"))
		tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		a.overlaySync()
		tt.SetSize(a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		if !has(tt, off) {
			t.Fatalf("%s: no capsule %q in %q", lang, off, firstTexts(tt))
		}
		click(t, tt, off)
		m := a.ov.lang.frame
		mt := hudTester(a.langMenuView, m.Width, m.Height)
		frames(mt)
		if !has(mt, on) || has(mt, "As in settings") || has(mt, "按设置") {
			t.Errorf("%s: menu %q, want %q first", lang, firstTexts(mt), on)
		}
		// The Language models page sets High: the menu says so.
		a.updateSettings(func(s *settings.Settings) {
			s.Providers.LLM[0].Generation = settings.Generation{Thinking: llm.ThinkEffort, ThinkingEffort: "high"}
		})
		frames(mt)
		if high := i18n.T("%s (default)", i18n.T("High")); !has(mt, high) {
			t.Errorf("%s: after the setting changed, menu %q, want %q", lang, firstTexts(mt), high)
		}
	}
}
