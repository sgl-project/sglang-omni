package main

// Whole sessions of Translate on Linux (the fakes are in session_e2e_test.go).

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// TestTranslateSelectionEndToEnd: with text selected, the Translate
// shortcut records nothing: the selection is translated (streamed) and
// pasted over it, and kept in the history with its target.
func TestTranslateSelectionEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "早上好，团队。"}
	set, keys, tr := fileSetup(t, "unused", chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "Good morning, team.")+","+ev, func(s *settings.Settings) {
		set(s)
		s.Features.Translate.Target = "zh-Hans"
	}, keys)
	var states []session.State
	a.pump("the end", func() bool {
		if n := len(states); n == 0 || states[n-1] != a.sess.State {
			states = append(states, a.sess.State)
		}
		return a.sess.State.Ended()
	})
	end := a.sess
	if end.State != session.Done || end.Result != "早上好，团队。" || !end.Inserted || end.Selection != "Good morning, team." || end.SelectionVia != proto.ViaAX {
		t.Fatalf("end %+v", end)
	}
	if got := fmt.Sprint(states); !strings.Contains(got, "reading composing") || strings.Contains(got, "recording") {
		t.Errorf("states %s", got)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != "早上好，团队。" || p[0].TargetPID != 812 {
		t.Errorf("pasted %+v", p)
	}
	if tr.count() != 0 {
		t.Errorf("%d uploads", tr.count())
	}
	reqs := chat.requests()
	if len(reqs) != 1 || reqs[0]["stream"] != true {
		t.Fatalf("chats %v", reqs)
	}
	msgs := chatMessages(reqs[0])
	if !strings.Contains(msgs[0][1], "into Simplified Chinese") || msgs[1][1] != "<input>\nGood morning, team.\n</input>" {
		t.Errorf("messages %q", msgs)
	}
	es := a.history()
	if len(es) != 1 || es[0].Kind != store.KindTranslate || es[0].SourceText != "Good morning, team." || es[0].TargetLang != "zh-Hans" ||
		es[0].Delivery != "insert" || es[0].AudioMS != 0 {
		t.Errorf("history %+v", es)
	}
}

// TestTranslateCopySelection: Accessibility shows no selection, ⌘C is
// allowed in Slack: the selection comes through the clipboard.
func TestTranslateCopySelection(t *testing.T) {
	chat := &fakeChat{text: "晚安"}
	set, keys, _ := fileSetup(t, "unused", chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("copy-selection", "Good night")+","+ev, set, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.Result != "晚安" || a.sess.SelectionVia != proto.ViaCopy {
		t.Errorf("end %+v", a.sess)
	}
}

// TestTranslateSpeechCard: nothing selected, the speech is translated
// into the card (the setting); the card's language select translates the
// same text again, into Japanese, without recording.
func TestTranslateSpeechCard(t *testing.T) {
	chat := &fakeChat{texts: []string{"See you on Thursday.", "木曜日に会いましょう。"}}
	set, keys, tr := fileSetup(t, "周四见", chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger), hotkeyLine(500, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Translate.ShowCard = true
		s.Features.Translate.Cleanup = true
		s.General.MainLanguage = "zh" // the target is English
	}, keys)
	// The draft shows in the pill while the model writes; the card comes
	// with the result, not before.
	streamed, early := false, false
	a.pump("the end", func() bool {
		streamed = streamed || a.sess.State == session.Composing && a.ov.mode == ovWorking && liveLineText(a.sess) == strings.TrimSpace(a.sess.Draft) && a.sess.Draft != ""
		early = early || a.sess.State.Live() && a.ov.mode == ovAnswer
		return a.sess.State.Ended()
	})
	if !streamed || early {
		t.Errorf("the draft in the pill %v, a card before the end %v", streamed, early)
	}
	if a.sess.Result != "See you on Thursday." || a.sess.Raw != "周四见" || a.sess.Inserted || !a.answer.open || a.answer.lang != "en" || a.ov.mode != ovAnswer {
		t.Fatalf("end %+v, card %+v", a.sess, a.answer)
	}
	if len(a.pasted()) != 0 || tr.count() != 1 {
		t.Errorf("pasted %+v, %d uploads", a.pasted(), tr.count())
	}
	if msgs := chatMessages(chat.requests()[0]); !strings.Contains(msgs[0][1], "基础清理") || len(msgs) != 4 || msgs[3][1] != "<input>\n周四见\n</input>" {
		t.Errorf("first request %q", msgs)
	}
	// Another language: into the card whatever the setting (Voxt's menu
	// replaces the card's answer); the card keeps its text meanwhile,
	// with a spinner and the new language in its menu.
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.ShowCard = false })
	first := a.sess.ID
	chat.mu.Lock()
	chat.delay = 150 * time.Millisecond
	chat.mu.Unlock()
	a.retryAnswer("ja")
	a.pump("the second session", func() bool { return a.sess.ID != first && a.sess.State == session.Composing })
	if v := a.cardNow(); a.ov.mode != ovAnswer || !v.busy || v.text != "See you on Thursday." || v.lang != "ja" || v.canInsert || v.canLang || !v.canCopy {
		t.Errorf("on its way: %s %+v", a.ov.mode, v)
	}
	a.pump("the second end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if a.sess.Result != "木曜日に会いましょう。" || a.sess.TargetLanguage != "ja" || !a.answer.open || a.answer.lang != "ja" || tr.count() != 1 {
		t.Errorf("again %+v, card %+v", a.sess, a.answer)
	}
	if a.answer.text != "木曜日に会いましょう。" || a.answer.id != a.sess.ID || a.answer.redo != "" || a.sess.Inserted || len(a.pasted()) != 0 || a.ov.mode != ovAnswer {
		t.Errorf("replaced: %+v %+v", a.answer, a.pasted())
	}
	if msgs := chatMessages(chat.requests()[1]); !strings.Contains(msgs[0][1], "翻译成日语") || msgs[1][1] != "<input>\n周四见\n</input>" {
		t.Errorf("second request %q", msgs)
	}
	if es := a.history(); len(es) != 2 || es[0].TargetLang != "ja" || es[0].Delivery != "card" {
		t.Errorf("history %+v", es)
	}
	// A language that fails: the card stays as it was, and says so.
	chat.set("", 500)
	second := a.sess.ID
	a.retryAnswer("ko")
	a.pump("the third end", func() bool { return a.sess.ID != second && a.sess.State.Ended() })
	if a.answer.text != "木曜日に会いましょう。" || a.answer.lang != "ja" || a.answer.redo != "" || !strings.HasPrefix(a.answer.msg, "Not translated into Korean: ") || a.notice != nil {
		t.Errorf("failed: %+v notice %+v", a.answer, a.notice)
	}
	// Escape while another language is on its way closes the card.
	chat.set("x", 0)
	chat.mu.Lock()
	chat.delay = time.Second
	chat.mu.Unlock()
	third := a.sess.ID
	a.retryAnswer("fr")
	a.pump("the fourth session", func() bool { return a.sess.ID != third && a.sess.State == session.Composing })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyEscape, Phase: proto.PhaseTrigger})
	a.pump("the cancel", func() bool { return a.sess.State.Ended() })
	if a.answer.open || a.sess.State != session.Cancelled || a.ov.shown {
		t.Errorf("escape: %+v %s", a.answer, a.sess.State)
	}
}

// TestTranslateFailureRetry: the model fails: the card shows why, with
// Try again, which translates the selection again and pastes it.
func TestTranslateFailureRetry(t *testing.T) {
	chat := &fakeChat{status: 500}
	set, keys, _ := fileSetup(t, "unused", chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "Hello")+","+ev, set, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Failed || !a.answer.open || a.answer.problem == nil || a.answer.problem.Code != session.ProblemLLM || a.notice != nil {
		t.Fatalf("end %+v card %+v notice %+v", a.sess, a.answer, a.notice)
	}
	if es := a.history(); len(es) != 1 || es[0].Status != store.StatusFailed || es[0].SourceText != "Hello" {
		t.Errorf("history %+v", es)
	}
	chat.set("你好", 0)
	first := a.sess.ID
	a.retryAnswer("")
	a.pump("the retry", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != "你好" || !a.sess.Inserted || a.answer.open {
		t.Errorf("retry %+v card %+v", a.sess, a.answer)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != "你好" {
		t.Errorf("pasted %+v", p)
	}
}
