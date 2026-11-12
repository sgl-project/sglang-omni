package main

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestCardThinkingLevel (#3879): the card offers the thinking levels of
// the service that answered; one asks the same input again with it, into
// the card, and keeps it for another language; the service's setting
// stays the default; a level that fails leaves the card as it was;
// Escape stops one on its way.
func TestCardThinkingLevel(t *testing.T) {
	chat := &fakeChat{texts: []string{"See you on Thursday.", "See you Thursday!", "", "木曜日に会いましょう。"}}
	set, keys, _ := fileSetup(t, "周四见", chat)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger), hotkeyLine(500, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Translate.ShowCard = true
		s.General.MainLanguage = "zh"
	}, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if !a.answer.open || a.answer.text != "See you on Thursday." {
		t.Fatalf("card %+v", a.answer)
	}
	if body := chat.requests()[0]; body["thinking"] != nil {
		t.Errorf("first request: thinking %v, want the service's", body["thinking"])
	}
	// A model that is not a service set up here (or takes no thinking
	// setting): no button.
	if v := a.cardNow(); v.thinks {
		t.Errorf("thinking offered for %q", a.answer.llmID)
	}
	tt := cardTester(t, a.app)
	frames(tt)
	if has(tt, "Thinking") {
		t.Errorf("button shows: %q", firstTexts(tt))
	}

	// DeepSeek: off, on, and its efforts.
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{{ID: "deepseek"}} })
	a.answer.llmID = "deepseek"
	a.overlaySync()
	frames(tt)
	if v := a.cardNow(); !v.thinks || !v.canThink || v.think != nil || !has(tt, "Thinking") {
		t.Fatalf("no button: %+v %q", v, firstTexts(tt))
	}
	click(t, tt, "Thinking")
	if !a.answer.langOpen || !a.answer.thinkMenu || !a.ov.lang.shown {
		t.Fatalf("menu: open %v think %v shown %v", a.answer.langOpen, a.answer.thinkMenu, a.ov.lang.shown)
	}
	if h := a.ov.lang.frame.Height; h != thinkMenuFrame(a.ov.lang.frame, true, 7).Height {
		t.Errorf("menu %d high for 7 levels", h)
	}
	mt := hudTester(a.langMenuView, a.ov.lang.frame.Width, a.ov.lang.frame.Height)
	frames(mt)
	for _, s := range []string{"As in settings", "Off", "On", "None", "Low", "High", "Max"} {
		if !has(mt, s) {
			t.Errorf("menu: no %q in %q", s, firstTexts(mt))
		}
	}
	// The same level: nothing is asked.
	if err := mt.Click("As in settings"); err != nil {
		t.Fatal(err)
	}
	if a.answer.langOpen || a.sess.State.Live() || len(chat.requests()) != 1 {
		t.Errorf("the same level asked again: %d requests", len(chat.requests()))
	}
	// Low: asked again, the card keeps its text meanwhile.
	chat.mu.Lock()
	chat.delay = 150 * time.Millisecond
	chat.mu.Unlock()
	first := a.sess.ID
	click(t, tt, "Thinking")
	frames(mt)
	if err := mt.Click("Low"); err != nil {
		t.Fatal(err)
	}
	a.pump("the second session", func() bool { return a.sess.ID != first && a.sess.State == session.Composing })
	low := llm.Thinking{Mode: llm.ThinkEffort, Effort: "low"}
	if v := a.cardNow(); !v.busy || v.text != "See you on Thursday." || v.canThink || v.think == nil || *v.think != low {
		t.Errorf("on its way: %+v", v)
	}
	a.pump("the second end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if a.answer.text != "See you Thursday!" || a.answer.think == nil || *a.answer.think != low || a.answer.lang != "en" || len(a.pasted()) != 0 {
		t.Errorf("again: %+v, pasted %v", a.answer, a.pasted())
	}
	if body := chat.requests()[1]; body["thinking"] != low || chatMessages(body)[len(chatMessages(body))-1][1] != "<input>\n周四见\n</input>" {
		t.Errorf("second request %v", body)
	}
	if tip := thinkText(a.cardNow().think); tip != "Thinking: Low" {
		t.Errorf("tip %q", tip)
	}
	if a.prefs.Providers.LLM[0].Generation.Thinking != "" {
		t.Errorf("the setting changed: %+v", a.prefs.Providers.LLM[0].Generation)
	}
	// Off fails: the card stays as it was and says so.
	second := a.sess.ID
	a.chooseThinking(&llm.Thinking{Mode: llm.ThinkOff})
	a.pump("the third end", func() bool { return a.sess.ID != second && a.sess.State.Ended() })
	if a.answer.text != "See you Thursday!" || *a.answer.think != low || !strings.HasPrefix(a.answer.msg, "Not answered again (Thinking: Off): ") {
		t.Errorf("failed: %+v", a.answer)
	}
	// Another language keeps the level.
	third := a.sess.ID
	a.retryAnswer("ja")
	a.pump("the fourth end", func() bool { return a.sess.ID != third && a.sess.State.Ended() })
	if body := chat.requests()[3]; body["thinking"] != low || a.answer.text != "木曜日に会いましょう。" || *a.answer.think != low {
		t.Errorf("another language: %v, card %+v", body["thinking"], a.answer)
	}
	// Escape while a level is on its way closes the card and stops it.
	chat.set("x", 0)
	chat.mu.Lock()
	chat.delay = time.Second
	chat.mu.Unlock()
	fourth := a.sess.ID
	a.chooseThinking(&llm.Thinking{Mode: llm.ThinkOn})
	a.pump("the fifth session", func() bool { return a.sess.ID != fourth && a.sess.State == session.Composing })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the cancel", func() bool { return a.sess.State.Ended() })
	if a.answer.open || a.sess.State != session.Cancelled || len(a.pasted()) != 0 {
		t.Errorf("escape: %+v %s", a.answer, a.sess.State)
	}
}

// TestRewriteCardThinkingLevel: a Rewrite answer asked again with another
// level becomes the conversation's first turn, so Continue follows up on
// the answer the card shows.
func TestRewriteCardThinkingLevel(t *testing.T) {
	chat := &fakeChat{texts: []string{"Docs ready, tests running.", "Docs are ready; the tests still run.", "Shorter."}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{{ID: "deepseek"}} })
	a.answer.llmID = "deepseek"
	if v := a.cardNow(); !v.thinks || !v.canThink {
		t.Fatalf("no button: %+v", v)
	}
	first := a.sess.ID
	high := llm.Thinking{Mode: llm.ThinkEffort, Effort: "high"}
	a.chooseThinking(&high)
	a.pump("the redo", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if body := chat.requests()[1]; body["thinking"] != high || !strings.Contains(lastUser(chat, 1), "summarize") {
		t.Errorf("request %v", body)
	}
	if a.answer.text != "Docs are ready; the tests still run." || len(a.answer.turns) != 1 || a.answer.turns[0].Answer.Content != a.answer.text || a.answer.conversation || len(a.pasted()) != 0 {
		t.Fatalf("card %+v, pasted %v", a.answer, a.pasted())
	}
	// Continue: the follow-up holds the new answer.
	a.recordFollowUp(t, tr, "shorter")
	msgs := chatMessages(chat.requests()[2])
	if !hasAnswer(msgs, "Docs are ready; the tests still run.") || hasAnswer(msgs, "Docs ready, tests running.") {
		t.Errorf("follow-up messages %q", msgs)
	}
}

// hasAnswer says whether msgs hold an assistant turn with text.
func hasAnswer(msgs [][2]string, text string) bool {
	for _, m := range msgs {
		if m[0] == "assistant" && strings.Contains(m[1], text) {
			return true
		}
	}
	return false
}
