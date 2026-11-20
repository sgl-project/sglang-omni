package main

import (
	"bytes"
	"errors"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/ink"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestCardRedoKeepsScrollKeyWhenWritingStarts(t *testing.T) {
	a, _, _ := newTestApp(t)
	local := sessionOverlayLocal{Overlay: &sessionOverlayLocalUI{}, Card: &answerCardUI{}}
	card := &sessionCardViewState{ID: "old-card", Blocks: []sessionCardBlock{{Kind: sessionCardBody, Text: "old answer"}}}
	tester := ui.NewTester(func(context *ui.Context) { cardBody(context, "redo-session", card, 120, layout.AnswerW, a, local, nil) }, 400, 300)
	frames(tester)
	card.Writing = true
	card.Live[0] = sessionCardWritingBlock("new answer", store.KindRewrite)
	card.LiveCount = 1
	frames(tester)
	if local.Overlay.cardScrollKey != "old-card/redo-session" {
		t.Fatalf("redo changed scroll key to %q", local.Overlay.cardScrollKey)
	}
}

// cardLong is a translation longer than the card's 220 of text.
var cardLong = strings.Repeat("goroutine 是由 Go 运行时管理的轻量级线程，启动成本很低，一个程序里同时运行成千上万个也很常见。", 12)

// cardQuote and cardQuoteLong are text selected for a follow-up: a
// short one, and one of 3,000 characters.
var (
	cardQuote     = "Anna: I can start the meeting."
	cardQuoteLong = strings.Repeat("Ben: I will send the notes after the meeting. ", 66)[:3000]
)

// cardStep is one state of the answer card, set on an app.
type cardStep struct {
	name  string
	apply func(a *app)
	// What it should be: the text's height (0: one line, -1: all of
	// it, layout.CardBodyMax: scrolled), a line under it, the header's offer.
	body                                                   int
	msg                                                    bool
	insert, copy, followUp, stop, retry, lang, busy, convo bool
	think                                                  bool // the thinking button
}

// cardSteps walk the answer card through its states, as one Translate
// card and one Rewrite conversation would (a first answer as the model
// writes it: overlay_stream_test.go).
func cardSteps() []cardStep {
	arc := proto.App{PID: 77, BundleID: "company.thebrowser.Browser", Name: "Arc"}
	sel := "A goroutine is a lightweight thread managed by the Go runtime."
	tr := func(st session.State, result, draft string) session.Snapshot {
		return session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: st, Card: true, Delivery: deliver.ReasonNotEditable, Target: arc,
			Selection: sel, TargetLanguage: "zh-Hans", Result: result, Draft: draft}
	}
	return []cardStep{
		{name: "short", apply: func(a *app) {
			snap(a, tr(session.Done, "goroutine 是由 Go 运行时管理的轻量级线程。", "goroutine 是由 Go 运行时管理的轻量级线程。"))
		},
			insert: true, copy: true, followUp: true, lang: true},
		// Another language, before its first words (then they stream in
		// over the card's: TestTranslateSpeechCard).
		{name: "redo", apply: func(a *app) {
			a.answer.redo, a.answer.redoLang = "t2", "ja"
			snap(a, session.Snapshot{ID: "t2", Kind: store.KindTranslate, State: session.Composing, Card: true, Target: arc, Selection: sel, TargetLanguage: "ja"})
		}, copy: true, busy: true},
		{name: "redone-long", apply: func(a *app) {
			snap(a, session.Snapshot{ID: "t2", Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Target: arc,
				Selection: sel, TargetLanguage: "ja", Result: cardLong})
		}, body: layout.CardBodyMax, insert: true, copy: true, followUp: true, lang: true},
		{name: "copied", apply: func(a *app) { a.answer.copied = true }, body: layout.CardBodyMax, insert: true, copy: true, followUp: true, lang: true},
		{name: "insert-failed", apply: func(a *app) {
			a.answer.copied, a.answer.text = false, "goroutine 是由 Go 运行时管理的轻量级线程。"
			a.answer.msg = uitext.DeliveryText(deliver.Result{Reason: deliver.ReasonNotFront}, arc)
		}, msg: true, insert: true, copy: true, followUp: true, lang: true},
		{name: "failed", apply: func(a *app) {
			snap(a, session.Snapshot{ID: "t3", Kind: store.KindTranslate, State: session.Recording})
			snap(a, session.Snapshot{ID: "t3", Kind: store.KindTranslate, State: session.Failed, Card: true, Target: arc, Selection: sel,
				Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("llm.balanced: the language model process stopped (signal 9)")}})
		}, body: -1, retry: true},
		{name: "rewrite", apply: func(a *app) {
			a.closeAnswer()
			snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "write that I am late",
				Title: "Running late", Result: "Sorry, I am running about ten minutes late: please start without me."})
		}, body: -1, insert: true, copy: true, followUp: true},
		{name: "follow-up-recording", apply: func(a *app) {
			a.followUp, a.answer.conversation = "r1", true
			now := time.Now()
			snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Recording, StartedAt: now, HeardAt: now, Levels: wave(), Partial: "shorter"})
		}, body: -1, copy: true, stop: true, busy: true, convo: true},
		{name: "follow-up-writing", apply: func(a *app) {
			snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Composing, Card: true, Raw: "shorter", Title: "Late", Draft: "Ten minutes late"})
		}, body: -1, copy: true, busy: true, convo: true},
		{name: "conversation", apply: func(a *app) {
			snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "shorter",
				Title: "Late", Result: "Ten minutes late: start without me."})
		}, body: -1, insert: true, copy: true, followUp: true, convo: true},
		// Continue with more text selected (#3816): it shows under the
		// follow-up, a long one cut until Show all.
		{name: "follow-up-selected", apply: func(a *app) {
			a.followUp = "r2"
			snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Recording, Selection: cardQuote})
			snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "add these",
				Selection: cardQuote, Result: "Ten minutes late, Anna starts."})
		}, body: -1, insert: true, copy: true, followUp: true, convo: true},
		{name: "follow-up-selected-long", apply: func(a *app) {
			a.followUp = "r3"
			snap(a, session.Snapshot{ID: "r4", Kind: store.KindRewrite, State: session.Recording, Selection: cardQuoteLong})
			snap(a, session.Snapshot{ID: "r4", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "and these",
				Selection: cardQuoteLong, Result: "Late; Anna starts; notes later."})
		}, body: -1, insert: true, copy: true, followUp: true, convo: true},
		{name: "follow-up-selected-open", apply: func(a *app) { a.toggleQuote(3) },
			body: -1, insert: true, copy: true, followUp: true, convo: true},
		// Cut off at the model's length limit: in the card, which says so.
		{name: "cut-off", apply: func(a *app) {
			a.closeAnswer()
			cut := tr(session.Done, "goroutine 是由 Go 运行时管理的", "")
			cut.ID, cut.Delivery = "t4", deliver.ReasonCard
			cut.Problem = &session.Problem{Code: session.ProblemTruncated, Degraded: true}
			snap(a, session.Snapshot{ID: "t4", Kind: store.KindTranslate, State: session.Composing})
			snap(a, cut)
		}, msg: true, insert: true, copy: true, followUp: true, lang: true},
		// A service with thinking levels (#3879): its button, by the
		// language capsule; after Continue in Rewrite's.
		{name: "thinking", apply: func(a *app) {
			a.closeAnswer()
			a.prefs.Providers.LLM = []settings.Provider{{ID: "deepseek"}}
			th := tr(session.Done, "goroutine 是由 Go 运行时管理的轻量级线程。", "")
			th.ID, th.LLM = "t5", session.Model{ID: "deepseek"}
			snap(a, session.Snapshot{ID: "t5", Kind: store.KindTranslate, State: session.Composing})
			snap(a, th)
			a.answer.think = &llm.Thinking{Mode: llm.ThinkEffort, Effort: "high"}
		}, insert: true, copy: true, followUp: true, lang: true, think: true},
		{name: "thinking-rewrite", apply: func(a *app) {
			a.closeAnswer()
			snap(a, session.Snapshot{ID: "r5", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "r5", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "write that I am late",
				Title: "Running late", Result: "Sorry, I am running about ten minutes late: please start without me.", LLM: session.Model{ID: "deepseek"}})
			a.answer.think = &llm.Thinking{Mode: llm.ThinkOff}
		}, body: -1, insert: true, copy: true, followUp: true, think: true},
		{name: "transcribe-dictionary", apply: func(a *app) {
			a.closeAnswer()
			const result = "Voxt uses SGLang."
			snap(a, session.Snapshot{ID: "d1", Kind: store.KindTranscribe, State: session.Recording})
			snap(a, session.Snapshot{ID: "d1", Kind: store.KindTranscribe, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Result: result, Highlights: testDictionaryHighlights(result)})
		}, insert: true, copy: true},
	}
}

// TestAnswerCardStates: the answer card through its states, the overlay
// at the bottom and at the top: 480 wide, as high as its text (Voxt's
// chrome and up to 220 of text, no empty space; a line under it when
// there is one), about the anchor; what its header offers.
func TestAnswerCardStates(t *testing.T) {
	for _, pos := range []string{"bottom", "top"} {
		t.Run(pos, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			shortScreen(t, a)
			// Saved, so that a later save (Setup.Tried) keeps it.
			a.updateSettings(func(s *settings.Settings) { s.Overlay.Position = pos })
			for _, st := range cardSteps() {
				st.apply(a)
				a.overlaySync()
				v := a.cachedCard()
				if a.ov.mode != ovAnswer || !a.ov.shown || a.ov.passThrough || a.ov.w != layout.AnswerW {
					t.Fatalf("%s: %s shown %v pass %v %d wide", st.name, a.ov.mode, a.ov.shown, a.ov.passThrough, a.ov.w)
				}
				body, _ := a.cardBodyHeight(&v)
				_, line := guessText(ovSized{"x", layout.CardTextSize, layout.CardTextWeight, layout.CardBodyW})
				switch st.body {
				case 0:
					if body != int(line) {
						t.Errorf("%s: the text is %d high, not a line (%v)", st.name, body, line)
					}
				case layout.CardBodyMax:
					if body != layout.CardBodyMax {
						t.Errorf("%s: the text is %d high, not %d", st.name, body, layout.CardBodyMax)
					}
				default:
					if body <= int(line) || body > layout.CardBodyMax {
						t.Errorf("%s: the text is %d high", st.name, body)
					}
				}
				want := layout.CardChrome + body + sessionCardAttachmentHeight(&v)
				if st.msg {
					want += layout.CardMsgGap + layout.CardMsgH
				}
				if a.ov.h != want || (v.Message != "") != st.msg {
					t.Errorf("%s: %d high, want %d (line %q)", st.name, a.ov.h, want, v.Message)
				}
				anchored(t, st.name, a.prefs.Overlay, a.ov.frame)
				got := []bool{v.CanInsert, v.CanCopy, v.CanFollowUp, v.CanStop, v.CanRetry, v.CanLanguage, v.Busy, v.Conversation, v.ShowThinking}
				exp := []bool{st.insert, st.copy, st.followUp, st.stop, st.retry, st.lang, st.busy, st.convo, st.think}
				for i := range got {
					if got[i] != exp[i] {
						t.Errorf("%s: insert, copy, follow up, stop, retry, language, busy, conversation = %v, want %v", st.name, got, exp)
						break
					}
				}
			}
		})
	}
}

// TestAnswerCardMarkdown (#4119): a Rewrite answer's Markdown shows in
// the card as read, not as its source: no **, ##, backticks or $$, the
// words and the math's TeX there, while it streams and in a
// conversation's answer bubbles (the user's own bubble stays raw). A
// Translate card keeps its text plain, as it goes in verbatim.
func TestAnswerCardMarkdown(t *testing.T) {
	a, _, _ := newTestApp(t)
	rw := func(st session.State, result, draft string) session.Snapshot {
		return session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: st, Card: true, Delivery: deliver.ReasonCard,
			Raw: "sum up the plan", Title: "The plan", Result: result, Draft: draft}
	}
	snap(a, rw(session.Recording, "", ""))
	snap(a, rw(session.Done, "## Release\n\n**First** we ship, then we rest.\n\n- one\n- two\n\n$$\nx^2+y\n$$\n\nRun `go test`.", ""))
	a.overlaySync()
	tt := cardTester(t, a)
	frames(tt)
	for _, s := range tt.Texts() {
		if strings.Contains(s, "**") || strings.Contains(s, "##") || strings.Contains(s, "`") || strings.Contains(s, "$$") {
			t.Errorf("the card shows Markdown's source: %q", s)
		}
	}
	for _, want := range []string{"Release", "First", "one", "two", "x^2+y", "go test"} {
		if !tt.HasText(want) {
			t.Errorf("no %q in the card (%q)", want, tt.Texts())
		}
	}

	// The answer streaming in heals its half-written Markdown.
	a.closeAnswer()
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Composing, Card: true,
		Draft: "**Bold** and **half"})
	a.overlaySync()
	frames(tt)
	for _, s := range tt.Texts() {
		if strings.Contains(s, "**") {
			t.Errorf("the streaming card shows Markdown's source: %q", s)
		}
	}
	if !tt.HasText("Bold") || !tt.HasText("half") {
		t.Errorf("the streaming answer lost words (%q)", tt.Texts())
	}

	// A conversation: the answers render, what the user said stays raw.
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
		Raw: "**Bold** and **half", Title: "Late", Result: "**Bold** and half"})
	a.followUp, a.answer.conversation = "r2", true
	snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Recording})
	snap(a, session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
		Raw: "make it **shorter**", Title: "Shorter", Result: "## Short\n\nBold half"})
	a.overlaySync()
	frames(tt)
	for _, s := range tt.Texts() {
		if strings.Contains(s, "##") {
			t.Errorf("a conversation's answer shows Markdown's source: %q", s)
		}
	}
	if !tt.HasText("Short") {
		t.Errorf("the follow-up's answer did not render (%q)", tt.Texts())
	}
	if !tt.HasText("**shorter**") {
		t.Errorf("the user's own words lost their text (%q)", tt.Texts())
	}

	// A Translate card's text stays plain: it goes in verbatim.
	a.closeAnswer()
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Recording})
	snap(a, session.Snapshot{ID: "t1", Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
		TargetLanguage: "zh-Hans", Result: "**not bold**"})
	a.overlaySync()
	frames(tt)
	if !tt.HasText("**not bold**") {
		t.Errorf("the translation's text changed (%q)", tt.Texts())
	}

	// A long Markdown answer scrolls, as a long plain one does.
	a.closeAnswer()
	snap(a, rw(session.Recording, "", ""))
	long := rw(session.Done, strings.Repeat("## 段落\n\n**加粗** 一段改写的回答，带列表和标题。\n\n- 一项\n", 20), "")
	long.ID = "r4"
	snap(a, long)
	a.overlaySync()
	frames(tt)
	if body, _ := a.cardBodyHeight(a.cachedCardRef()); body != layout.CardBodyMax {
		t.Errorf("a long Markdown answer is %d high, not %d", body, layout.CardBodyMax)
	}
}

// TestAnswerCardInk: each state as the view draws it, its texts measured
// (as on the Mac), checked from its pixels as the Mac's overlay check
// checks its captures: the face's size, its ink from the left padding
// down to just above the bottom padding (no empty space, nothing cut).
func TestAnswerCardInk(t *testing.T) {
	a, _, _ := newTestApp(t)
	shortScreen(t, a)
	defer func() { a.measureHook = guessText }()
	for _, st := range cardSteps() {
		st.apply(a)
		a.overlaySync()
		img := renderOverlay(t, a)
		v := a.cachedCard()
		k, bad := ink.Check(img, 2, cardInkWant(ink.Want{W: a.ov.w, H: a.ov.h}, &v))
		for _, b := range bad {
			t.Errorf("%s: %s", st.name, b)
		}
		t.Logf("%s: %v", st.name, k)
		if st.body == 0 && !st.msg && a.ov.h > layout.CardChrome+24 {
			t.Errorf("%s: one line of text, the card %d high", st.name, a.ov.h)
		}
		if scrolls := a.ov.ui.cardScroll.MaxY > 0; scrolls != (st.body == layout.CardBodyMax) && !v.Conversation {
			t.Errorf("%s: scrolls %v (%v)", st.name, scrolls, a.ov.ui.cardScroll.MaxY)
		}
		if dir := os.Getenv("VOXTNEW_INK"); dir != "" {
			var b bytes.Buffer
			png.Encode(&b, img)
			os.WriteFile(filepath.Join(dir, "card-"+st.name+".png"), b.Bytes(), 0o644)
		}
	}
}

// TestAnswerCardInteraction: the header's buttons, the language menu,
// Continue and Done, the bubbles' Copy under the pointer, and Escape:
// it closes the card, and only cancels a follow-up (Voxt's).
func TestAnswerCardInteraction(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	steps := cardSteps()
	steps[0].apply(a) // the short answer
	tt := cardTester(t, a)
	frames(tt)
	for _, s := range []string{"goroutine 是由 Go 运行时管理的轻量级线程。", "Simplified Chinese", "Continue", "Insert", "Copy", "Close"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	// No line of orange: why it is in the card is Insert's tip.
	if has(tt, a.answer.why) || !strings.HasPrefix(a.answer.why, "The selection in Arc cannot be edited") {
		t.Errorf("why %q shows", a.answer.why)
	}
	// The language menu: ours, in a window of its own; the language
	// checked; another asks again (here nothing is set up: the line says
	// so); the card does not move.
	card := a.ov.frame
	click(t, tt, "Translate into")
	if !a.answer.ui.langOpen || !a.ov.lang.shown || a.ov.frame != card {
		t.Fatalf("menu: open %v shown %v, card %+v (was %+v)", a.answer.ui.langOpen, a.ov.lang.shown, a.ov.frame, card)
	}
	mt := hudTester(a.langMenuView, a.ov.lang.frame.Width, a.ov.lang.frame.Height)
	frames(mt)
	for _, s := range []string{"English", "Japanese", "Simplified Chinese"} {
		if !has(mt, s) {
			t.Errorf("menu: no %q in %q", s, firstTexts(mt))
		}
	}
	if err := mt.Click("Japanese"); err != nil {
		t.Fatal(err)
	}
	frames(mt)
	if a.answer.ui.langOpen || a.ov.lang.shown || a.answer.msg != "No language model is set up." || a.answer.lang != "zh-Hans" || !a.answer.open || a.ov.frame.Y+a.ov.frame.Height != card.Y+card.Height {
		t.Errorf("another language: %+v, menu %v", a.answer, a.ov.lang.shown)
	}
	// Escape closes the menu, not the card.
	a.answer.msg = ""
	a.overlaySync()
	click(t, tt, "Translate into")
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.answer.ui.langOpen || a.ov.lang.shown || !a.answer.open {
		t.Errorf("Escape on the menu: open %v, card %v", a.answer.ui.langOpen, a.answer.open)
	}
	// Copy: the mark, then Copy again.
	click(t, tt, "Copy")
	if !has(tt, "Copied") || tt.Clipboard() != a.answer.text || !a.answer.open {
		t.Errorf("copied: %q", tt.Clipboard())
	}
	ft.advance(copyMark)
	if !has(tt, "Copy") || a.answer.copied {
		t.Error("the mark stays")
	}
	// Escape closes it.
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.answer.open || a.ov.shown {
		t.Fatalf("escape: %+v", a.answer)
	}

	// A Rewrite answer: Continue follows up (here the model is missing:
	// a notice, the card stays as it was).
	steps[6].apply(a) // a Rewrite answer
	frames(tt)
	click(t, tt, "Continue")
	if !a.answer.open || a.answer.conversation || a.notice == nil || a.notice.text != "No speech model is installed." {
		t.Errorf("continue: %+v %+v", a.answer, a.notice)
	}
	// A follow-up records in the card: Done stops it; Escape cancels it,
	// the card staying with its answer; a second Escape closes it.
	steps[7].apply(a) // its follow-up, recording
	frames(tt)
	if !has(tt, "Done") || !has(tt, "shorter") || !has(tt, "You") || has(tt, "Continue") {
		t.Errorf("recording: %q", firstTexts(tt))
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Cancelled}) // as the manager would
	if !a.answer.open || a.ov.mode != ovAnswer || a.followUp != "" || a.answer.text == "" {
		t.Errorf("escape on a follow-up: %s %+v", a.ov.mode, a.answer)
	}
	if v := a.cachedCard(); !v.Conversation || !v.CanFollowUp || len(cardBubblesForTest(v)) != 2 {
		t.Errorf("after the cancel %+v", v)
	}
	// An answer's bubble shows Copy under the pointer.
	frames(tt)
	r, ok := tt.Find("Sorry, I am running about ten minutes late: please start without me.")
	if !ok {
		t.Fatalf("no answer bubble in %q", firstTexts(tt))
	}
	tt.Move(r.X+r.W/2, r.Y+r.H/2)
	frames(tt)
	title, ok := tt.Find("Running late") // the bubble's heading, 12 inside it
	if !ok {
		t.Fatalf("no heading in %q", firstTexts(tt))
	}
	// Its Copy: 8 from the bubble's top right corner, 22 high.
	tt.ClickAt(title.X-layout.BubblePadX+layout.BubbleW-8-10, title.Y-layout.BubblePadY+8+11)
	frames(tt)
	if tt.Clipboard() != "Running late\n\nSorry, I am running about ten minutes late: please start without me." || a.ov.ui.bubbleCopied == "" {
		t.Errorf("bubble copied %q", tt.Clipboard())
	}
	ft.advance(bubbleMark)
	if a.ov.ui.bubbleCopied != "" {
		t.Error("the bubble's mark stays")
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.answer.open {
		t.Error("the second Escape left the card")
	}
}

// TestRewriteInputBubbleFitsText: the input bubble grows with one word and
// then one line instead of leaving the rest of its maximum width empty. A
// long paragraph takes that width, and the card still scrolls it.
func TestRewriteInputBubbleFitsText(t *testing.T) {
	inputs := []struct {
		name, text string
	}{
		{name: "word", text: "Short"},
		{name: "line", text: "Make this paragraph sound more professional"},
		{name: "long", text: strings.Repeat("Make this paragraph sound more professional and keep every detail. ", 12)},
	}
	widths := map[string]int{}
	for _, tc := range inputs {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			card := &sessionCardViewState{Conversation: true, Blocks: []sessionCardBlock{{Kind: sessionCardBubble, User: true, Title: "You", Text: tc.text}}}
			local := sessionOverlayLocal{Overlay: &sessionOverlayLocalUI{}, Card: &answerCardUI{}}
			tt := hudTester(func(c *ui.Context) {
				ui.Column(c).Fill().Background(ui.RGB(30, 31, 36)).Children(func() { cardBlockView(c, "", card, card.Blocks[0], 0, layout.AnswerW, a, local, nil) })
			}, layout.AnswerW, 500)
			frames(tt)
			title, ok := tt.Find("You")
			if !ok {
				t.Fatalf("no input heading in %q", firstTexts(tt))
			}
			img, y, left, right := tt.Image(), int(title.Y)/2, -1, -1
			background := img.RGBAAt(0, y)
			for x := range img.Bounds().Dx() {
				if img.RGBAAt(x, y) == background {
					continue
				}
				if left < 0 {
					left = x
				}
				right = x
			}
			if left < 0 {
				t.Fatal("no bubble pixels above its heading")
			}
			widths[tc.name] = right - left + 1
			if tc.name != "long" {
				return
			}
			shortScreen(t, a)
			a.answer = answerCard{open: true, id: "rewrite-input", kind: store.KindRewrite, conversation: true,
				turns: []compose.Turn{{Instruction: tc.text, Answer: compose.Answer{Content: "Done."}}}}
			a.overlaySync()
			renderOverlay(t, a)
			body, _ := a.cardBodyHeight(a.cachedCardRef())
			if body != layout.CardBodyMax || a.ov.ui.cardScroll.MaxY <= 0 {
				t.Errorf("long input body %d high, scroll limit %v", body, a.ov.ui.cardScroll.MaxY)
			}
		})
	}
	if widths["word"] >= widths["line"] || widths["line"] >= widths["long"] {
		t.Errorf("bubble widths for a word, a line and a paragraph are %v; want them to grow", widths)
	}
	if widths["long"] < layout.BubbleW-4 || widths["long"] > layout.BubbleW {
		t.Errorf("long input bubble %d wide, want the usual %d", widths["long"], layout.BubbleW)
	}
	t.Logf("word, line and paragraph bubble widths: %v", widths)
}

// TestOverlayShotsRewriteInput renders the input's fit in English and
// Chinese: one word, one line, and a paragraph long enough to scroll.
func TestOverlayShotsRewriteInput(t *testing.T) {
	runSnapshots(t, "VOXTNEW_OVERLAY_SHOTS", func(t *testing.T, out string) {
		for _, tc := range []struct {
			name, lang, text string
		}{
			{name: "rewrite-input-word-en", text: "Short"},
			{name: "rewrite-input-word-zh", lang: "zh-Hans", text: "短"},
			{name: "rewrite-input-line-en", text: "Make this paragraph sound more professional"},
			{name: "rewrite-input-long-zh", lang: "zh-Hans", text: strings.Repeat("请把这段文字改得更简洁、更正式，同时保留所有事实、数字和负责的人，不要遗漏任何重要信息。", 12)},
		} {
			a, _, _ := newTestApp(t)
			shortScreen(t, a)
			answer := "Done."
			if tc.lang != "" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = tc.lang })
				answer = "已完成。"
			}
			a.answer = answerCard{open: true, id: "rewrite-input", kind: store.KindRewrite, conversation: true,
				turns: []compose.Turn{{Instruction: tc.text, Answer: compose.Answer{Content: answer}}}}
			a.overlaySync()
			img := renderOverlay(t, a)
			f, err := os.Create(filepath.Join(out, tc.name+".png"))
			if err != nil {
				t.Fatal(err)
			}
			if err := png.Encode(f, img); err != nil {
				f.Close()
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
		}
	})
}

// TestLanguageMenuPlace: the language menu opens over the card when the
// overlay is at the bottom of the screen, under the capsule when it is at
// the top, its right edge under the capsule's; it goes when the card
// goes, or another app comes to the front.
func TestLanguageMenuPlace(t *testing.T) {
	for _, pos := range []string{"bottom", "top"} {
		t.Run(pos, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			shortScreen(t, a)
			// Saved, so that a later save (Setup.Tried) keeps it.
			a.updateSettings(func(s *settings.Settings) { s.Overlay.Position = pos })
			cardSteps()[0].apply(a) // a translation in the card
			a.overlaySync()
			card := a.ov.frame
			a.toggleLangMenu()
			m := a.ov.lang.frame
			if !a.ov.lang.shown || a.ov.frame != card {
				t.Fatalf("shown %v, card %+v (was %+v)", a.ov.lang.shown, a.ov.frame, card)
			}
			faceX, faceY := card.X+layout.Pad, card.Y+layout.Pad
			capRight := faceX + layout.AnswerW - layout.CardPadX - 3*layout.CardBtn - 3*layout.CardHeadGap
			if m.X+m.Width-layout.Pad != capRight || m.Width != ovLangW+2*layout.Pad || m.Height != langMenuHeight()+2*layout.Pad {
				t.Errorf("menu %+v, the capsule's right edge %d", m, capRight)
			}
			switch {
			case pos == "bottom" && m.Y+m.Height-layout.Pad != faceY-ovLangGap:
				t.Errorf("bottom: the menu ends at %d, the card's top is %d", m.Y+m.Height-layout.Pad, faceY)
			case pos == "top" && m.Y+layout.Pad != faceY+layout.CardPadTop+layout.CardHeadH+ovLangGap:
				t.Errorf("top: the menu starts at %d", m.Y+layout.Pad)
			}
			if len(compose.TargetLanguages)*ovLangRowH+2*ovLangPad > ovLangMaxH && langMenuHeight() != ovLangMaxH {
				t.Errorf("height %d", langMenuHeight())
			}
			// Another app in front: it goes; the card stays.
			a.helperNotified(appActivated(proto.App{PID: 9, BundleID: "com.apple.Notes", Name: "Notes"}))
			if a.ov.lang.shown || a.answer.ui.langOpen || !a.answer.open {
				t.Errorf("another app: menu %v", a.ov.lang.shown)
			}
			// The card going takes it.
			a.toggleLangMenu()
			a.closeAnswer()
			if a.ov.lang.shown || a.answer.ui.langOpen {
				t.Error("the menu stays without the card")
			}
		})
	}
}

// TestCardHeaderFits: at 480 the header holds its actions whatever the
// title: the language capsule at its widest (128 of name) and three
// buttons; Continue and three; Done and three over a conversation. A long
// title ends in "…" before them; nothing overlaps or leaves the card.
func TestCardHeaderFits(t *testing.T) {
	long := "A very long answer title that the model wrote, longer than any header could hold in one line"
	for _, c := range []struct {
		name    string
		apply   func(a *app)
		actions []string // left to right
	}{
		{"translate", func(a *app) {
			snap(a, session.Snapshot{ID: "h1", Kind: store.KindTranslate, State: session.Recording})
			snap(a, session.Snapshot{ID: "h1", Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Selection: "x", TargetLanguage: "zh-Hant", Result: "你好。"}) // "Traditional Chinese": the widest name
		}, []string{"Translate into", "Continue", "Insert", "Copy", "Close"}},
		{"rewrite", func(a *app) {
			snap(a, session.Snapshot{ID: "h2", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "h2", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Raw: "write", Title: long, Result: "Text."})
		}, []string{"Continue", "Insert", "Copy", "Close"}},
		{"conversation", func(a *app) {
			snap(a, session.Snapshot{ID: "h3", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "h3", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Raw: "write", Title: long, Result: "Text."})
			a.followUp, a.answer.conversation = "h3", true
			snap(a, session.Snapshot{ID: "h4", Kind: store.KindRewrite, State: session.Recording, StartedAt: time.Now(), HeardAt: time.Now(), Levels: wave()})
		}, []string{"Done", "Insert", "Copy", "Close"}},
	} {
		for _, lang := range []string{"en", "zh-Hans"} {
			t.Run(c.name+"/"+lang, func(t *testing.T) {
				a, _, _ := newTestApp(t)
				if lang != "en" {
					a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
				}
				c.apply(a)
				renderOverlay(t, a)
				tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
				frames(tt)
				right := float32(layout.Pad + layout.AnswerW - layout.CardPadX)
				prev := float32(-1)
				for _, label := range c.actions {
					r, ok := tt.Find(i18n.T(label))
					if !ok {
						t.Fatalf("no %q in %q", i18n.T(label), firstTexts(tt))
					}
					if r.X < prev || r.X+r.W > right+0.5 || r.H > layout.CardBtn+0.5 {
						t.Errorf("%q at %+v (after %v, the card's padding at %v)", label, r, prev, right)
					}
					prev = r.X + r.W
				}
				first, _ := tt.Find(i18n.T(c.actions[0]))
				// The name at its widest (112) and the capsule's padding and
				// chevron; Chinese names are short.
				if want := map[string]float32{"en": 140, "zh-Hans": 70}[lang]; c.name == "translate" && first.W < want {
					t.Errorf("the capsule is %v wide: the language's name is cut", first.W)
				}
				if c.name == "translate" {
					title, ok := tt.Find(i18n.T("Translation"))
					// The capsule names the card: no title to cut to "…" beside
					// Continue (#3897). (The Chinese icon's label is the title.)
					if ok && title.W > layout.CardIcon+0.5 {
						t.Errorf("a title %+v beside the capsule", title)
					}
				}
				if c.name == "rewrite" {
					title, ok := tt.Find(long)
					if !ok {
						t.Fatalf("no title in %q", firstTexts(tt))
					}
					if title.X+title.W > first.X-layout.CardHeadGap+0.5 {
						t.Errorf("the title %+v runs into %q at %v", title, c.actions[0], first.X)
					}
				}
			})
		}
	}
}

// TestCardDragged: a card the user dragged (overlayDragged keeps the
// offset) keeps its place as its answer grows, stays on the screen, and
// the next card shows at the anchor again.
func TestCardDragged(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: "Short."}
	a.overlaySync()
	anchored := a.ov.frame
	a.ov.dragged.x, a.ov.dragged.y = -300, -200
	a.answer.text = strings.Repeat("A longer answer, a few lines more. ", 12)
	a.overlaySync()
	if f := a.ov.frame; f.Height <= anchored.Height || f.X != anchored.X-300 || f.Y+f.Height != anchored.Y+anchored.Height-200 {
		t.Errorf("dragged, then grown: %+v; anchored %+v", f, anchored)
	}
	a.ov.dragged.y = -5000
	a.overlaySync()
	if a.ov.frame.Y != a.ov.area.Y-layout.Pad {
		t.Errorf("dragged past the top: %+v", a.ov.frame)
	}
	a.answer = answerCard{}
	a.overlaySync()
	a.answer = answerCard{open: true, kind: store.KindRewrite, text: "Short."}
	a.overlaySync()
	if a.ov.frame != anchored {
		t.Errorf("the next card: %+v, want %+v", a.ov.frame, anchored)
	}
}
