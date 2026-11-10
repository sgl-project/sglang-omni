package main

import (
	"errors"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// pageTester is the test app on a feature page, in a tall window.
func pageTester(t *testing.T, path string) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate(path)
	frames(tt)
	return a, tt
}

func TestTranslatePage(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	for _, s := range []string{
		"Translate", "Shortcuts", "Right ⌥ ⇧", "Models", "No speech model is installed, so only selected text can be translated.",
		"No language model is set up. Set one up to translate.", "Translation", "Translate into", "Automatic (now: Simplified Chinese)",
		"Basic clean-up", "Show the result in a card", "Your rules", "Try it",
	} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	if has(tt, "Paste the last result") {
		t.Error("paste-last belongs to Transcribe")
	}
	clickSwitch(t, tt, "Show the result in a card")
	if !a.lib.settings.Get().Features.Translate.ShowCard {
		t.Error("show card not saved")
	}
	clickSwitch(t, tt, "Basic clean-up")
	if a.lib.settings.Get().Features.Translate.Cleanup {
		t.Error("clean-up not saved")
	}
	click(t, tt, "Target language")
	click(t, tt, "Japanese")
	if got := a.lib.settings.Get().Features.Translate.Target; got != "ja" {
		t.Errorf("target %q", got)
	}
	click(t, tt, "Target language")
	click(t, tt, "Automatic (now: Simplified Chinese)")
	if got := a.lib.settings.Get().Features.Translate.Target; got != "" {
		t.Errorf("target %q", got)
	}
	// Its own shortcuts and rules; the one added skips the keys Notes uses (Right ⌘).
	click(t, tt, "Add a shortcut")
	if hs := a.lib.settings.Get().Features.Translate.Hotkeys; len(hs) != 2 || hs[1].Keys != "RightShift" {
		t.Errorf("shortcuts %+v", hs)
	}
	click(t, tt, "Your rules")
	click(t, tt, "System prompt for Translate")
	a.fui(hotkey.Translate).prompt = "Use British spelling."
	a.fui(hotkey.Translate).promptDirty = true
	a.savePrompt(hotkey.Translate)
	click(t, tt, "Example user message")
	if p := a.lib.settings.Get().Features.Translate.Prompt; p != "Use British spelling." || a.lib.settings.Get().Features.Transcribe.Prompt != "" {
		t.Errorf("rules %q", p)
	}
	if !has(tt, "Below is the prompt for spoken text. Selected text is translated without the clean-up and keeps its form.") {
		t.Errorf("preview %q", firstTexts(tt))
	}
	// Ready services: no notices.
	readyModels(a)
	frames(tt)
	if has(tt, "No language model is set up. Set one up to translate.") {
		t.Error("notice with a ready model")
	}
}

func TestRewritePage(t *testing.T) {
	a, tt := pageTester(t, "/rewrite")
	for _, s := range []string{
		"Rewrite", "⌃ Right ⌥", "No speech model is installed. Install one to rewrite.", "No language model is set up. Set one up to rewrite.",
		"Rewriting", "Clean up the instruction",
		"While the answer card shows an answer, the shortcut records a follow-up that changes it (\"shorter\", \"in English\").",
	} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	clickSwitch(t, tt, "Clean up the instruction")
	if a.lib.settings.Get().Features.Rewrite.Cleanup {
		t.Error("clean-up not saved")
	}
	if has(tt, "Long answers in the card") {
		t.Error("0.1.0's card length setting is back")
	}
	click(t, tt, "Your rules")
	click(t, tt, "Example user message")
	if !has(tt, "Below is the prompt with a selection. With nothing selected, the model writes from your instruction and answers with a title and the text.") {
		t.Errorf("preview %q", firstTexts(tt))
	}
	// A try that is blocked says why, here.
	click(t, tt, "Start a test")
	if a.sess.State.Live() || !has(tt, "No speech model is installed.") {
		t.Errorf("try %+v", a.sess)
	}
}

// TestOverlayWriting: the overlay through Translate and Rewrite: reading
// shows the working pill, a draft shows under it, the card streams, a failure
// waits in the card with Try again, a follow-up records over the card.
func TestOverlayWriting(t *testing.T) {
	a, _, _ := newTestApp(t)
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Reading})
	if a.ov.mode != ovWorking || a.hotkeyPhase() != hotkey.Busy {
		t.Errorf("reading %s", a.ov.mode)
	}
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Composing, Selection: "Hi", Draft: "你"})
	if a.ov.mode != ovWorking || a.ov.h != layout.LiveH || liveLineText(a.sess) != "Hi" {
		t.Errorf("selection %s %d text %q", a.ov.mode, a.ov.w, liveLineText(a.sess))
	}
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Composing, Selection: "Hi", Draft: "你好", Card: true, TargetLanguage: "zh-Hans"})
	// The card streams the draft; the pill keeps the selection, not the draft.
	if a.ov.mode != ovWorking || a.answer.open || liveLineText(a.sess) != "Hi" {
		t.Errorf("the draft of a card's answer: %s %+v text %q", a.ov.mode, a.answer, liveLineText(a.sess))
	}
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Done, Selection: "Hi", Result: "你好", Card: true, Delivery: deliver.ReasonNotEditable, TargetLanguage: "zh-Hans"})
	if !a.answer.open || a.answer.input.Selection != "Hi" || a.answer.lang != "zh-Hans" {
		t.Fatalf("card %+v", a.answer)
	}
	if got := a.answer.why; got != "The selection in Slack cannot be edited: copy the result, or click where it should go and insert." {
		t.Errorf("why %q", got)
	}
	// Another language: nothing is set up, so the card says why.
	a.retryAnswer("ja")
	if !a.answer.open || a.answer.msg != "No language model is set up." || a.sess.ID != "t1" {
		t.Errorf("retry blocked %+v", a.answer)
	}
	a.closeAnswer()

	// A failure: the card, no notice.
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recording})
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Failed, Card: true, Raw: "shorter", Selection: "a long text",
		Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("503")}})
	if a.ov.mode != ovAnswer || a.answer.problem == nil || a.notice != nil || a.answer.input.Raw != "shorter" {
		t.Fatalf("failure %s %+v %+v", a.ov.mode, a.answer, a.notice)
	}
	tt := cardTester(t, a)
	frames(tt)
	for _, s := range []string{"AI answer", "Try again", "shorter", "a long text"} {
		if !has(tt, s) {
			t.Errorf("failure card: no %q in %q", s, firstTexts(tt))
		}
	}
	if has(tt, "Insert") {
		t.Error("a failed card offers Insert")
	}
	a.closeAnswer()

	// A Rewrite answer: its title, and the shortcut follows up.
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Recording})
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "write a note",
		Title: "A note", Result: "Long note."})
	if len(a.answer.turns) != 1 || a.answer.turns[0].Answer != (compose.Answer{Title: "A note", Content: "Long note."}) {
		t.Fatalf("turns %+v", a.answer.turns)
	}
	frames(tt)
	for _, s := range []string{"A note", "Long note.", "Continue", "Insert", "Copy", "Close"} {
		if !has(tt, s) {
			t.Errorf("answer card: no %q in %q", s, firstTexts(tt))
		}
	}
	// The follow-up of a card whose app is set up runs as a Rewrite with the turns.
	// It records in the card, a conversation (Voxt's).
	a.followUp = "r2"
	snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Recording})
	if v := a.cardNow(); a.ov.mode != ovAnswer || !a.answer.open || !v.conversation || !v.recording || !v.canStop {
		t.Errorf("follow-up recording %s %+v", a.ov.mode, v)
	}
	snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "shorter", Title: "Note", Result: "Note."})
	if a.answer.id != "r3" || len(a.answer.turns) != 2 || a.answer.turns[1].Instruction != "shorter" || a.followUp != "" {
		t.Errorf("after the follow-up %+v", a.answer)
	}
	// Without a model the shortcut on the card says so, and the card stays.
	a.hotkeyFired(proto.HotkeyEventParams{Binding: hotkey.ID(hotkey.Rewrite, 0), Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap})
	if !a.answer.open || a.notice == nil || a.followUp != "" {
		t.Errorf("blocked follow-up: card %v notice %+v", a.answer.open, a.notice)
	}
}

func TestSessionBlocker(t *testing.T) {
	a, _, _ := newTestApp(t)
	for _, k := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		if n := a.sessionBlocker(k); n == nil {
			t.Errorf("%s: no blocker with nothing set up", k)
		}
	}
	a.local = withChat(&fakeLocal{}, &fakeChat{})
	if n := a.sessionBlocker(store.KindTranslate); n != nil {
		t.Errorf("translate needs no speech model: %+v", n)
	}
	if n := a.sessionBlocker(store.KindRewrite); n == nil || n.text != "No speech model is installed." {
		t.Errorf("rewrite: %+v", n)
	}
	readyModels(a)
	for _, k := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		if n := a.sessionBlocker(k); n != nil {
			t.Errorf("%s: %+v", k, n)
		}
	}
}

// TestCardSwitchPerFeature: Transcribe, Translate and Rewrite each have
// their own answer-card switch (11-09: one switch for the three turned
// Transcribe's card back on from Rewrite's page); Rewrite's is on by
// default and its page says what its card does more.
func TestCardSwitchPerFeature(t *testing.T) {
	a, tt := pageTester(t, "/translate")
	rewriteCard := "In Rewrite's card you can keep going: click Continue or press the Rewrite shortcut again and say \"shorter\" or \"in English\". The card changes the answer and keeps the whole conversation."
	cards := func() [3]bool {
		f := a.lib.settings.Get().Features
		return [3]bool{f.Transcribe.ShowCard, f.Translate.ShowCard, f.Rewrite.ShowCard}
	}
	if got := cards(); got != [3]bool{false, false, true} {
		t.Fatalf("defaults %v, want Rewrite's card only", got)
	}
	want := cards()
	for i, page := range []string{"/transcribe", "/translate", "/rewrite"} {
		a.navigate(page)
		frames(tt)
		n := 0
		for _, s := range tt.AccessLabels() {
			if s == "Show the result in a card" {
				n++
			}
		}
		if n != 2 { // one row: its title and its switch
			t.Fatalf("%s: %d elements named for the card switch", page, n)
		}
		clickSwitch(t, tt, "Show the result in a card")
		want[i] = !want[i]
		if got := cards(); got != want {
			t.Errorf("%s: cards %v after a click, want %v", page, got, want)
		}
		if shown := has(tt, rewriteCard); shown != (page == "/rewrite") {
			t.Errorf("%s: the note on Rewrite's card shows: %v", page, shown)
		}
	}
	a.navigate("/transcribe") // still its own value
	frames(tt)
	if got := cards(); got != want {
		t.Errorf("back on /transcribe: cards %v, want %v", got, want)
	}
}
