package main

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestLLMWaitRow (#3821): the Language models page's select shows the
// saved wait and saves another, in English and in Chinese; a choice made
// elsewhere shows too.
func TestLLMWaitRow(t *testing.T) {
	t.Cleanup(func() { i18n.SetLocale("en") })
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	if !hasExact(tt, "Wait for the language model") || !hasExact(tt, "Automatic") {
		t.Fatalf("no row: %q", firstTexts(tt))
	}
	inView := func(what, text string) {
		t.Helper()
		for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
			tt.SetSize(size[0], size[1])
			frames(tt)
			if r, ok := tt.Find(text); !ok || r.Y+r.H > float32(size[1]) {
				t.Errorf("%s, %dx%d: the wait needs a scroll: %v %v", what, size[0], size[1], r, ok)
			}
		}
		tt.SetSize(windowWidth, windowHeight)
		frames(tt)
	}
	inView("English", "Automatic")
	click(t, tt, "Automatic")
	click(t, tt, "5 minutes")
	if got := a.settings.Get().Models.LLMWait; got != settings.Wait5Min {
		t.Errorf("saved %q", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.Models.LLMWait = settings.Wait10Min })
	if frames(tt); !hasExact(tt, "10 minutes") {
		t.Errorf("a choice made elsewhere: %q", firstTexts(tt))
	}
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	if frames(tt); !hasExact(tt, "等待语言模型") || !hasExact(tt, "10 分钟") {
		t.Errorf("in Chinese: %q", firstTexts(tt))
	}
	inView("zh-Hans", "10 分钟")
}

// TestSkipLLMPill (#3821): while the language model works on speech that
// can go in as said, the working pill ends its row with Use transcript,
// inside the face, and takes the pointer, with the live text in one line
// or in lines (the default); Rewrite's pill has none and lets the pointer
// through, as before.
func TestSkipLLMPill(t *testing.T) {
	t.Cleanup(func() { i18n.SetLocale("en") })
	for _, c := range []struct {
		lang  string
		lines bool
	}{{"en", false}, {"zh-Hans", false}, {"en", true}, {"zh-Hans", true}} { // lines: a fresh install's
		lang := c.lang
		a, _, _ := newTestApp(t)
		a.updateSettings(func(s *settings.Settings) { s.Overlay.LiveLines, s.General.Language = c.lines, lang })
		label := i18n.T("Use transcript")
		snap(a, session.Snapshot{Kind: store.KindTranscribe, State: session.Composing, CanSkip: true, Partial: "so the release moves to thursday"})
		tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		a.overlaySync() // the label measured
		tt.SetSize(a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tt)
		r, ok := tt.Find(label)
		if a.ov.mode != ovWorking || !ok || a.ov.passThrough {
			t.Fatalf("%s: mode %v, %q found %v, pass %v; texts %q", lang, a.ov.mode, label, ok, a.ov.passThrough, firstTexts(tt))
		}
		if right := float32(layout.Pad + a.ov.w); r.X+r.W > right {
			t.Errorf("%s: the button ends at %v, past the face's %v", lang, r.X+r.W, right)
		}

		snap(a, session.Snapshot{ID: "s2", Kind: store.KindRewrite, State: session.Composing, Partial: "make it shorter", Selection: "x"})
		frames(tt)
		if has(tt, label) || !a.ov.passThrough && !c.lines { // lines take the pointer anyway
			t.Errorf("%s: Rewrite's pill: %q, pass %v", lang, firstTexts(tt), a.ov.passThrough)
		}
	}
}

// TestSkipLLMEndToEnd (#3821): with the model slow, Use transcript in the
// pill, or the cancel shortcut, ends the language model step and the
// recognised text is pasted and kept; the model's answer never comes.
func TestSkipLLMEndToEnd(t *testing.T) {
	for _, how := range []string{"button", "shortcut"} {
		t.Run(how, func(t *testing.T) {
			chat := &fakeChat{text: "So the release moves to Thursday.", delay: time.Minute}
			set, keys, _ := fileSetup(t, "so the release moves to thursday", chat)
			ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
			a := newSessionApp(t, ev, func(s *settings.Settings) { set(s); s.Overlay.LiveLines = false }, keys)
			a.pump("the model's step", func() bool { return a.sess.State == session.Composing && a.sess.CanSkip })
			if how == "button" {
				tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
				click(t, tt, "Use transcript")
			} else {
				a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
			}
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != session.Done || a.sess.Problem != nil {
				t.Fatalf("end %s %+v", a.sess.State, a.sess.Problem)
			}
			a.pump("the paste", func() bool { return len(a.pasted()) > 0 })
			if p := a.pasted(); len(p) != 1 || !strings.Contains(p[0].Text, "so the release moves to thursday") {
				t.Errorf("pasted %+v", p)
			}
			if es := a.history(); len(es) != 1 || es[0].Status != store.StatusDone {
				t.Errorf("history %+v", es)
			}
		})
	}
}
