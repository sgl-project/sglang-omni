package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/plugins/glass"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/ink"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

var slackApp = proto.App{PID: 812, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}

// snap gives the app a session snapshot, as the manager would.
func snap(a *app, s session.Snapshot) {
	s.Rev = a.sess.Rev + 1
	if s.ID == "" {
		s.ID = "s1"
	}
	if s.Target.PID == 0 {
		s.Target = slackApp
	}
	a.sessionChanged(s)
}

func wave() []float64 {
	var out []float64
	for i := range session.Bars {
		out = append(out, 0.35+0.3*math.Sin(float64(i)/2))
	}
	return out
}

// TestOverlayModes: what the overlay shows through a session's states.
func TestOverlayModes(t *testing.T) {
	a, _, _ := newTestApp(t)
	if a.ov.mode != ovHidden || a.ov.shown {
		t.Fatalf("starts %s", a.ov.mode)
	}
	now := time.Now()
	snap(a, session.Snapshot{State: session.Recording, StartedAt: now})
	if a.ov.mode != ovRecording || a.ov.w != layout.PillW || a.ov.h != layout.PillH {
		t.Errorf("recording %s %d×%d", a.ov.mode, a.ov.w, a.ov.h)
	}
	snap(a, session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now, Partial: "hello"})
	if a.ov.w != layout.RowW+2*layout.LivePadX || a.ov.h != layout.LiveH {
		t.Errorf("live text %d×%d", a.ov.w, a.ov.h)
	}
	snap(a, session.Snapshot{State: session.Composing, Partial: "hello"})
	if a.ov.mode != ovWorking {
		t.Errorf("composing %s", a.ov.mode)
	}
	// Inserted: nothing more to show.
	snap(a, session.Snapshot{State: session.Done, Result: "Hello.", Inserted: true})
	if a.ov.mode != ovHidden || a.ov.shown {
		t.Errorf("done %s", a.ov.mode)
	}
	// A try shows on the page, not in the overlay.
	snap(a, session.Snapshot{ID: "s2", State: session.Recording, Try: true})
	if a.ov.mode != ovHidden {
		t.Errorf("try %s", a.ov.mode)
	}
	// Not inserted: the answer card, until closed (Escape closes it).
	snap(a, session.Snapshot{ID: "s3", State: session.Recording})
	snap(a, session.Snapshot{ID: "s3", State: session.Done, Result: "Card text", Card: true, Delivery: deliver.ReasonNotEditable})
	if a.ov.mode != ovAnswer || !a.answer.open || a.answer.text != "Card text" || a.ov.w != layout.AnswerW {
		t.Fatalf("card %s %+v", a.ov.mode, a.answer)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyEscape, Phase: proto.PhaseTrigger})
	if a.answer.open || a.ov.mode != ovHidden {
		t.Errorf("escape: %s %+v", a.ov.mode, a.answer)
	}
	// A failure: a notice with a way forward.
	snap(a, session.Snapshot{ID: "s4", State: session.Recording})
	snap(a, session.Snapshot{ID: "s4", State: session.Failed, Problem: &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindServer}}})
	if a.ov.mode != ovNotice || a.notice == nil || a.notice.action != actionModels || a.notice.text != "Speech recognition failed: The speech model stopped or did not answer. Try again in a moment." {
		t.Fatalf("notice %s %+v", a.ov.mode, a.notice)
	}
	a.noticeAction(a.notice.action)
	if a.router.Path() != modelsPath || a.ov.mode != ovHidden {
		t.Errorf("action: %s %s", a.router.Path(), a.ov.mode)
	}
	// Cancelled and too short: nothing.
	snap(a, session.Snapshot{ID: "s5", State: session.Recording})
	snap(a, session.Snapshot{ID: "s5", State: session.Cancelled, Problem: &session.Problem{Code: session.ProblemTooShort}})
	if a.ov.mode != ovHidden || a.notice != nil {
		t.Errorf("cancelled: %s %+v", a.ov.mode, a.notice)
	}
	// An older snapshot changes nothing.
	old := a.sess
	a.sessionChanged(session.Snapshot{Rev: old.Rev - 1, ID: "s4", State: session.Recording})
	if a.sess.Rev != old.Rev || a.ov.mode != ovHidden {
		t.Errorf("an older snapshot won: %+v", a.sess)
	}
}

// TestAnswerCardButtons: Copy copies and the card stays, its mark showing
// for 2 s (Voxt's); Insert hides the card, pastes through the helper into
// the session's app or a text field the user focused in another app, and
// brings the card back with why it could not.
func TestAnswerCardButtons(t *testing.T) {
	a, fs, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	var pasted []proto.PasteParams
	focus := proto.FocusResult{PID: 812, Role: "AXStaticText"}
	fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return focus, nil
	}
	fs.handlers[proto.MethodTextPaste] = func(raw json.RawMessage) (any, error) {
		var p proto.PasteParams
		json.Unmarshal(raw, &p)
		pasted = append(pasted, p)
		if a.ov.shown {
			t.Error("the card shows while it pastes")
		}
		return proto.PasteResult{Posted: true}, nil
	}
	card := func(id string) {
		snap(a, session.Snapshot{ID: id, State: session.Recording})
		snap(a, session.Snapshot{ID: id, State: session.Done, Result: "Answer " + id, Card: true, Delivery: deliver.ReasonNotEditable})
	}
	card("c1")
	tt := cardTester(a)
	frames(tt)
	if !tt.HasText("Answer c1") || tt.HasText("No text field is focused: click where it should go, then insert, or copy it.") {
		t.Errorf("card texts %q", firstTexts(tt))
	}
	if a.answer.why != "No text field is focused: click where it should go, then insert, or copy it." {
		t.Errorf("why %q", a.answer.why)
	}
	if err := tt.Click("Copy"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !a.answer.open || !a.answer.copied || tt.Clipboard() != "Answer c1" || !has(tt, "Copied") || len(a.toasts) != 0 {
		t.Errorf("copy: %+v %q %v", a.answer, tt.Clipboard(), a.toasts)
	}
	ft.advance(copyMark)
	if a.answer.copied || !a.answer.open || !has(tt, "Copy") {
		t.Errorf("the mark stays: %+v", a.answer)
	}
	card("c2")
	frames(tt)
	if err := tt.Click("Insert"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if a.answer.open || len(pasted) != 1 || pasted[0].Text != "Answer c2" || pasted[0].TargetPID != 812 {
		t.Errorf("insert: %+v %+v", a.answer, pasted)
	}
	focus = proto.FocusResult{PID: 530, Error: "no focused element"} // another app came to the front, nothing focused
	card("c3")
	a.insertAnswer()
	if !a.answer.open || a.answer.inserting || a.ov.mode != ovAnswer || a.answer.msg != "No text field is focused: click where the text should go, then insert." || len(pasted) != 1 {
		t.Errorf("elsewhere: %s %+v", a.ov.mode, a.answer)
	}
	h := a.ov.h
	if h != layout.CardChrome+a.mustBody(t)+layout.CardMsgGap+layout.CardMsgH {
		t.Errorf("the line under the text: %d high", h)
	}
	focus = proto.FocusResult{PID: 530, Role: "AXTextField", Subrole: "AXSecureTextField", Editable: true}
	a.insertAnswer()
	if !a.answer.open || a.answer.msg != "A password field is focused: VoxtNew never types into one." || len(pasted) != 1 {
		t.Errorf("a password field: %+v", a.answer)
	}
	textEdit := helpertest.Apps["com.apple.TextEdit"]
	focus = proto.FocusResult{PID: 530, App: &textEdit, Role: "AXTextArea", Editable: true} // a text field there now
	a.insertAnswer()
	if a.answer.open || len(pasted) != 2 || pasted[1].Text != "Answer c3" || pasted[1].TargetPID != 530 {
		t.Errorf("into the app in front: %+v %+v", a.answer, pasted)
	}
	card("c4")
	fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return nil, &proto.Error{Code: proto.CodePermissionDenied}
	}
	a.insertAnswer()
	if a.answer.msg != "Allow Accessibility to insert results where you type." {
		t.Errorf("no access: %+v", a.answer)
	}
	frames(tt)
	if !has(tt, "Allow Accessibility to insert results where you type.") {
		t.Errorf("the line: %q", firstTexts(tt))
	}
	if err := tt.Click("Close"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if a.answer.open || a.ov.mode != ovHidden {
		t.Errorf("close: %+v", a.answer)
	}
}

// mustBody is the card's text's height now.
func (a *app) mustBody(t *testing.T) int {
	t.Helper()
	h, _ := a.cardBodyHeight(a.cardNow())
	return h
}

func TestSessionProblemTexts(t *testing.T) {
	for _, tc := range []struct {
		p      session.Problem
		text   string
		action string
	}{
		{session.Problem{Code: session.ProblemNoASR}, "No speech model is installed.", actionModels},
		{session.Problem{Code: session.ProblemMicrophone, Err: errors.New("x")}, "VoxtNew may not use the microphone.", actionPermissions},
		{session.Problem{Code: session.ProblemHelper}, "The helper stopped: try again in a moment.", actionHelper},
		{session.Problem{Code: session.ProblemNoSpeech}, "Nothing was heard.", actionNone},
		{session.Problem{Code: session.ProblemTooShort}, "", actionNone},
		{session.Problem{Code: session.ProblemInternal, Err: errors.New("boom")}, "Something went wrong: boom", actionNone},
	} {
		text, _, action := sessionProblemText(&tc.p)
		if text != tc.text || action != tc.action {
			t.Errorf("%s: %q %q", tc.p.Code, text, action)
		}
	}
}

func TestClockAndTail(t *testing.T) {
	for d, want := range map[time.Duration]string{0: "0:00", 9 * time.Second: "0:09", 75 * time.Second: "1:15", 10 * time.Minute: "10:00"} {
		if got := clock(d); got != want {
			t.Errorf("clock(%v) = %q", d, got)
		}
	}
	// The face's width with live text: the row's or the text's, the text
	// at most layout.LiveTextW, and Voxt's padding.
	for _, c := range []struct {
		row  int
		text float32
		want int
	}{
		{layout.RowW, 30, layout.RowW + 2*layout.LivePadX},
		{layout.RowW, 156.2, 157 + 2*layout.LivePadX},
		{layout.RowW, 1200, layout.LiveMaxW},
		{layout.RowW + layout.Gap + 60, 100, layout.RowW + layout.Gap + 60 + 2*layout.LivePadX},
	} {
		if got := layout.LiveWidth(c.row, c.text); got != c.want {
			t.Errorf("liveWidth(%d, %v) = %d, want %d", c.row, c.text, got, c.want)
		}
	}
	if layout.PillW != 148 || layout.PillH != 48 || layout.LiveH != 76 || layout.LiveMaxW != 300 || math.Abs(layout.WaveW-88.7) > 1e-3 {
		t.Errorf("Voxt's sizes: pill %d×%d, live %d high, at most %d wide, the bars %v wide", layout.PillW, layout.PillH, layout.LiveH, layout.LiveMaxW, layout.WaveW)
	}
	if layout.RowWidth(true, 0) != layout.RowW+layout.Gap+layout.TimeW || layout.RowWidth(false, 200) != layout.RowW+layout.Gap+layout.SceneMaxW || layout.RowWidth(false, 40.2) != layout.RowW+layout.Gap+41 {
		t.Error("rowWidth")
	}
}

// TestFadeSpans: scrolled, the line shows from the letter at the left
// edge, the first layout.FadeW of it fading in, the rest whole.
func TestFadeSpans(t *testing.T) {
	measure := func(s string) float32 { return float32(len([]rune(s))) * 10 } // 10 a letter
	shown, at, spans := fadeSpans("abcdefghij", 25, measure)
	if shown != "cdefghij" || at != 20 {
		t.Fatalf("shown %q at %v", shown, at)
	}
	var b strings.Builder
	var prev uint8
	for i, sp := range spans {
		b.WriteString(sp.Text)
		if sp.Color.A < prev {
			t.Errorf("span %d %q: alpha %v after %v", i, sp.Text, sp.Color.A, prev)
		}
		prev = sp.Color.A
	}
	if b.String() != shown || spans[0].Color.A > 80 || spans[len(spans)-1].Color != ovLiveColor || spans[len(spans)-1].Text != "fghij" {
		t.Errorf("spans %+v", spans)
	}
	if shown, at, _ := fadeSpans("abc", 0, measure); shown != "abc" || at != 0 {
		t.Errorf("not scrolled: %q %v", shown, at)
	}
}

// overlayShots renders every state of the overlay (snapshot_test.go)
// over a desktop, light or dark, that shows through the face blurred, as
// the window's material blurs it on the Mac.
func overlayShots(t *testing.T, a *app, out, suffix string, dark bool) {
	t.Helper()
	now := time.Now()
	a.prefs.Overlay.ShowTime = false
	a.measureHook = nil // the overlay's view measures its texts, as on the Mac
	defer func() { a.measureHook = guessText }()
	longTr := "Go 的并发模型建立在 goroutine 和 channel 之上。goroutine 是由 Go 运行时管理的轻量级线程，启动成本很低，一个程序里同时运行成千上万个也很常见。channel 则是 goroutine 之间传递数据的管道：发送和接收默认都会阻塞，直到另一端准备好，这让同步变得自然。\n\n实践中，我们通常用 context 控制取消和超时，用 errgroup 等待一组任务结束并收集第一个错误，用带缓冲的 channel 限制并发数。需要共享状态时，sync.Mutex 依然是最直接的选择；但如果能把状态的所有权交给一个 goroutine，并通过 channel 和它通信，代码往往更容易推理。\n\n最后，别忘了用 -race 运行测试：数据竞争很少在本地复现，却常常在生产环境里出现。"
	turns := func() {
		a.answer.turns = []compose.Turn{{Instruction: "写一句话告诉同事我会晚到十分钟", Answer: compose.Answer{Title: "晚到通知", Content: "大家好，我这边路上有点堵，大概会晚到十分钟，会议请先开始，不用等我。"}}}
	}
	states := []struct {
		name  string
		s     session.Snapshot
		setup func()
		after func(id string) // once the state is in (the answer card's own states)
	}{
		{"overlay-starting", session.Snapshot{State: session.Recording, StartedAt: now}, nil, nil},
		// The microphone gives no audio yet, then the speech model starts
		// (10-20: the first session after launch).
		{"overlay-waiting-mic", session.Snapshot{State: session.Recording, StartedAt: now, Waiting: session.WaitMic}, nil, nil},
		{"overlay-waiting-model", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-2 * time.Second), Levels: wave(),
			Waiting: session.WaitModel}, nil, nil},
		{"overlay-waiting-model-recognizing", session.Snapshot{State: session.Recognizing, Waiting: session.WaitModel}, nil, nil},
		{"overlay-card-unconfirmed", session.Snapshot{State: session.Done, Card: true, Delivery: deliver.ReasonUnconfirmed, Target: proto.App{PID: 77, Name: "Arc"},
			Result: "详情中的左右滑动不够流畅，会卡住。"}, nil, nil},
		{"overlay-recording", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-12 * time.Second), Levels: wave()}, nil, nil},
		{"overlay-recording-time", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-12 * time.Second), Levels: wave()},
			func() { a.prefs.Overlay.ShowTime = true }, nil},
		{"overlay-live-short", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-3 * time.Second), Levels: wave(),
			Partial: "Meet at three tomorrow"}, nil, nil},
		{"overlay-live-short-zh", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-3 * time.Second), Levels: wave(),
			Partial: "明天三点开会"}, nil, nil},
		{"overlay-live-text", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-47 * time.Second), Levels: wave(),
			Partial: "So the plan for the release is to finish the onboarding first, then the history page, and after that we look at the translation card"}, nil, nil},
		{"overlay-live-text-zh", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-20 * time.Second), Levels: wave(),
			Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划，然后把引导页和历史页先做完"}, nil, nil},
		{"overlay-recording-scene", session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-5 * time.Second), Levels: wave(),
			Target: proto.App{Name: "Slack"}, Scene: sceneMatch("Team chat", appctx.SourceApp, "")}, nil, nil},
		{"overlay-working-scene", session.Snapshot{State: session.Composing, Target: proto.App{Name: "Google Chrome"},
			Scene: sceneMatch("Code review", appctx.SourceURL, "github.com")}, nil, nil},
		{"overlay-recognizing", session.Snapshot{State: session.Recognizing}, nil, nil},
		{"overlay-composing", session.Snapshot{State: session.Composing, Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划"}, nil, nil},
		{"overlay-error", session.Snapshot{State: session.Failed, Problem: &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindAuth}}}, nil, nil},
		{"overlay-no-speech", session.Snapshot{State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}}, nil, nil},
		{"overlay-warning", session.Snapshot{State: session.Done, Result: "x", Inserted: true, Problem: &session.Problem{Code: session.ProblemLLM, Degraded: true,
			Err: errors.New("timeout")}}, nil, nil},
		{"overlay-answer", session.Snapshot{State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Result: "Hello team, the release moves to Thursday.\n\n1. Finish the onboarding\n2. Polish the history page\n3. Ship the translation card\n\nThanks!"}, nil, nil},
		// Translate and Rewrite.
		{"overlay-translate-recording", session.Snapshot{Kind: store.KindTranslate, State: session.Recording, StartedAt: now, HeardAt: now.Add(-5 * time.Second), Levels: wave(), TargetLanguage: "en"}, nil, nil},
		{"overlay-translate-selection", session.Snapshot{Kind: store.KindTranslate, State: session.Composing, Selection: "A goroutine is a lightweight thread.", TargetLanguage: "zh-Hans",
			Draft: "goroutine 是一种轻量级"}, nil, nil},
		{"overlay-rewrite-writing", session.Snapshot{Kind: store.KindRewrite, State: session.Composing, Raw: "write a short note saying I will be late",
			Draft: "Sorry, I'm running about ten minutes late"}, nil, nil},
		{"overlay-translate-card-streaming", session.Snapshot{Kind: store.KindTranslate, State: session.Composing, Card: true, Delivery: deliver.ReasonNotEditable,
			Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Draft: "goroutine 是由 Go 运行时管理的"}, nil, nil},
		{"overlay-translate-card", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans",
			Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, nil, nil},
		{"overlay-translate-card-en", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "我们把发布推迟到周四，先把引导页和历史页做完。", TargetLanguage: "en",
			Result: "We are moving the release to Thursday, to finish the onboarding and the history page first."}, nil, nil},
		{"overlay-translate-card-long", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "Go's concurrency…", TargetLanguage: "zh-Hans", Result: longTr}, nil, nil},
		{"overlay-translate-card-redo", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans",
			Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, nil, func(id string) {
			a.answer.redo, a.answer.redoLang = id+"-ja", "ja"
			snap(a, session.Snapshot{ID: id + "-ja", Kind: store.KindTranslate, State: session.Composing, Card: true, TargetLanguage: "ja"})
		}},
		{"overlay-translate-card-menu", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans",
			Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, nil, nil},
		{"overlay-translate-card-copied", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans",
			Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, nil, func(string) { a.answer.copied = true }},
		{"overlay-translate-card-insert-failed", session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable,
			Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans",
			Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, nil, func(string) {
			a.answer.msg = uitext.DeliveryText(deliver.Result{Reason: deliver.ReasonNotFront}, proto.App{Name: "Arc"})
		}},
		{"overlay-rewrite-card", session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
			Raw: "写一段两百字的自我介绍，用于技术分享会的开场", Title: "技术分享开场自我介绍",
			Result: "大家好，我是关伟，一名专注于桌面应用和语音交互的工程师。过去几年，我主要在做跨平台的原生界面框架，也参与了几个语音输入工具的设计和开发。\n\n今天想和大家聊聊：怎样让语音输入真正融入日常的写作流程——从按下快捷键的那一刻，到文字出现在光标处，中间发生了什么，又有哪些地方值得打磨。"}, nil, nil},
		{"overlay-rewrite-conversation-recording", session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
			Raw: "写一句话告诉同事我会晚到十分钟", Title: "晚到通知", Result: "大家好，我这边路上有点堵，大概会晚到十分钟，会议请先开始，不用等我。"}, nil, func(id string) {
			a.followUp, a.answer.conversation = id, true
			snap(a, session.Snapshot{ID: id + "-f", Kind: store.KindRewrite, State: session.Recording, StartedAt: now, HeardAt: now, Levels: wave(), Partial: "再正式一点"})
		}},
		{"overlay-rewrite-conversation", session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
			Raw: "再正式一点", Title: "晚到说明", Result: "各位好：因交通拥堵，我预计将晚到约十分钟。会议请按时开始，无需等候，抱歉给大家带来不便。"}, nil, nil},
		{"overlay-rewrite-failed", session.Snapshot{Kind: store.KindRewrite, State: session.Failed, Card: true,
			Raw: "make it more formal", Selection: "hey, the release slips to thursday, sorry!",
			Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("llm.balanced: the language model process stopped (signal 9)")}}, nil, nil},
		// The Mac short of memory (memory_notice.go), after an insert.
		{"overlay-memory", session.Snapshot{State: session.Done, Result: "x", Inserted: true}, nil, func(string) { n := memoryNotice(); a.notice = &n }},
	}
	desk, line := ui.RGB(246, 246, 248), ui.RGB(150, 156, 166)
	if dark {
		desk, line = ui.RGB(30, 31, 36), ui.RGB(120, 128, 140)
	}
	for i, st := range states {
		id := "shot" + itoa(i)
		a.answer, a.notice, a.followUp = answerCard{}, nil, ""
		a.syncLangMenu()
		if st.name == "overlay-rewrite-conversation" {
			// The follow-up of a card that held the first answer.
			a.answer = answerCard{open: true, id: "first", kind: store.KindRewrite, conversation: true}
			turns()
			a.followUp = "first"
		}
		a.prefs.Overlay.ShowTime = false
		if st.setup != nil {
			st.setup()
		}
		final := st.s
		final.ID = id
		live := final
		live.State = session.Recording
		live.Problem, live.Result, live.Inserted, live.Card = nil, "", false, false
		snap(a, live)
		snap(a, final)
		if st.after != nil {
			st.after(id)
			a.overlaySync()
		}
		a.toasts = nil
		if a.ov.mode == ovHidden {
			t.Fatalf("%s: nothing shows", st.name)
		}
		renderOverlay(t, a) // its texts measured, the face fitted to them
		w, h := a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad
		radius := faceRadius(a.prefs.Overlay, a.ov.mode, a.ov.h)
		// With the language menu open: its window over the card's, where
		// it shows on the screen (the picture is as tall as both).
		var menuDY int
		if st.name == "overlay-translate-card-menu" {
			a.toggleLangMenu()
			menuDY = a.ov.frame.Y - a.ov.lang.frame.Y
			h += menuDY
		}
		tt := hudTester(func(c *ui.Context) {
			// The desktop: a page of text lines.
			ui.Column(c).Absolute().Fill().Background(desk).Padding(10, 14).Gap(8).Children(func() {
				for j := range 12 {
					ui.Box(c.Key(itoa(j))).Height(6).Width(float32(w) * (0.45 + 0.4*float32((j*37)%10)/10)).Radius(3).Background(line)
				}
			})
			// The window's material: what is under the face, blurred.
			ui.Box(c).Absolute().Left(layout.Pad).Top(float32(layout.Pad + menuDY)).Right(layout.Pad).Bottom(layout.Pad).Radius(radius).Material(glass.Blur{Radius: 8})
			// Positioned too, to paint over the blur.
			ui.Box(c).Absolute().Left(0).Top(float32(menuDY)).Right(0).Bottom(0).Children(func() { a.overlayView(c) })
			if menuDY > 0 {
				m := a.ov.lang.frame
				ui.Box(c).Absolute().Left(float32(m.X - a.ov.frame.X)).Top(0).Width(float32(m.Width)).Height(float32(m.Height)).Children(func() { a.langMenuView(c) })
			}
		}, w, h)
		tt.SetScale(2)
		frames(tt)
		f, err := os.Create(filepath.Join(out, st.name+"-"+suffix+".png"))
		if err != nil {
			t.Fatal(err)
		}
		png.Encode(f, tt.Image())
		f.Close()
	}
	a.answer, a.notice = answerCard{}, nil
	a.prefs.Overlay.ShowTime = false
	snap(a, session.Snapshot{ID: "shot-end", State: session.Cancelled})
}

// sceneMatch is a matched group for a snapshot.
func sceneMatch(name string, src appctx.Source, host string) *appctx.Match {
	return &appctx.Match{Source: src, Group: &appctx.Group{ID: "g", Name: name}, Host: host, Pattern: host}
}

func TestOverlaySceneLabel(t *testing.T) {
	for _, tc := range []struct {
		s    session.Snapshot
		want string
	}{
		{session.Snapshot{}, ""},
		{session.Snapshot{Target: proto.App{Name: "Slack"}, Scene: &appctx.Match{Source: appctx.SourceGlobal}}, ""},
		{session.Snapshot{Target: proto.App{Name: "Slack"}, Scene: sceneMatch("Team chat", appctx.SourceApp, "")}, "Slack · Team chat"},
		{session.Snapshot{Target: proto.App{Name: "Google Chrome"}, Scene: sceneMatch("Code review", appctx.SourceURL, "github.com")}, "github.com · Code review"},
		{session.Snapshot{Target: proto.App{Name: "Google Chrome"}, Scene: sceneMatch("Work", appctx.SourceBrowserApp, "news.ycombinator.com")}, "Google Chrome · Work"},
		{session.Snapshot{Scene: sceneMatch("Work", appctx.SourceApp, "")}, "Work"},
	} {
		if got := uitext.SceneLabel(tc.s); got != tc.want {
			t.Errorf("uitext.SceneLabel = %q, want %q", got, tc.want)
		}
	}
	a, _, _ := newTestApp(t)
	snap(a, session.Snapshot{ID: "s1", State: session.Recording, Target: proto.App{Name: "Slack"}, Scene: sceneMatch("Team chat", appctx.SourceApp, "")})
	if want := layout.PillW + layout.Gap + int(math.Ceil(float64(guessWidth(sceneText("Team chat"))))); a.ov.w != want || a.ov.h != layout.PillH {
		t.Errorf("size %dx%d", a.ov.w, a.ov.h)
	}
	if uitext.SceneShort(a.sess) != "Team chat" {
		t.Errorf("short %q", uitext.SceneShort(a.sess))
	}
}

// ovGone is how soon the overlay must be off the screen once nothing is
// left to show (the fade's two steps, and room).
const ovGone = 150 * time.Millisecond

// TestOverlayGoesAway: whatever ends what the overlay shows (an insert,
// Escape, a notice's time, the answer card's Close), it leaves the screen
// within ovGone and sets no timer again (10-15: after an insert, 043324a
// re-armed its "Inserted" timer from the timer itself, for ever).
func TestOverlayGoesAway(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	gone := func(what string) {
		t.Helper()
		if a.ov.shown {
			t.Errorf("%s: still shown (%s)", what, a.ov.mode)
		}
		ft.advance(ovGone)
		if a.ov.visible {
			t.Errorf("%s: still on screen after %v (%s)", what, ovGone, a.ov.mode)
		}
		before := ft.added
		ft.advance(time.Minute)
		if a.ov.visible || ft.added != before || len(ft.timers) != 0 {
			t.Errorf("%s: timers set again: %d more, %d pending", what, ft.added-before, len(ft.timers))
		}
	}
	// An insert.
	snap(a, session.Snapshot{ID: "s1", State: session.Recording})
	if !a.ov.visible {
		t.Fatal("recording: not on screen")
	}
	snap(a, session.Snapshot{ID: "s1", State: session.Delivering})
	snap(a, session.Snapshot{ID: "s1", State: session.Done, Result: "Hello.", Inserted: true})
	gone("inserted")
	// Escape.
	snap(a, session.Snapshot{ID: "s2", State: session.Recording})
	snap(a, session.Snapshot{ID: "s2", State: session.Cancelled})
	gone("cancelled")
	// A notice, after its time.
	snap(a, session.Snapshot{ID: "s3", State: session.Recording})
	snap(a, session.Snapshot{ID: "s3", State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}})
	if a.ov.mode != ovNotice || a.notice == nil || a.notice.text != "Nothing was heard." || !a.notice.warn {
		t.Fatalf("no speech: %s %+v", a.ov.mode, a.notice)
	}
	ft.advance(noticeTime - time.Millisecond)
	if a.ov.mode != ovNotice || !a.ov.visible {
		t.Fatalf("the notice went early: %s", a.ov.mode)
	}
	ft.advance(time.Millisecond)
	if a.notice != nil {
		t.Fatal("the notice stays")
	}
	gone("notice")
	// The answer card, closed.
	snap(a, session.Snapshot{ID: "s4", State: session.Recording})
	snap(a, session.Snapshot{ID: "s4", State: session.Done, Result: "Card", Card: true, Delivery: deliver.ReasonNotEditable})
	ft.advance(time.Minute)
	if a.ov.mode != ovAnswer || !a.ov.visible {
		t.Fatalf("the card went by itself: %s", a.ov.mode)
	}
	a.closeAnswer()
	gone("card closed")
}

// TestOverlayQuickSessions: what the last session left on its way out
// (the fade, a notice's timer) never hides the next one.
func TestOverlayQuickSessions(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	snap(a, session.Snapshot{ID: "s1", State: session.Recording})
	snap(a, session.Snapshot{ID: "s1", State: session.Done, Result: "One.", Inserted: true})
	ft.advance(20 * time.Millisecond) // mid-fade
	snap(a, session.Snapshot{ID: "s2", State: session.Recording})
	ft.advance(time.Second)
	if a.ov.mode != ovRecording || !a.ov.shown || !a.ov.visible {
		t.Fatalf("the second recording hidden: %s shown %v visible %v", a.ov.mode, a.ov.shown, a.ov.visible)
	}
	// A notice of the second, then a third session failing: the second's
	// timer leaves the third's notice alone.
	snap(a, session.Snapshot{ID: "s2", State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}})
	ft.advance(3 * time.Second)
	snap(a, session.Snapshot{ID: "s3", State: session.Recording})
	snap(a, session.Snapshot{ID: "s3", State: session.Failed, Problem: &session.Problem{Code: session.ProblemNoASR}})
	ft.advance(3 * time.Second) // the second's timer is due
	if a.notice == nil || a.notice.action != actionModels || !a.ov.visible {
		t.Fatalf("the third's notice went with the second's timer: %+v", a.notice)
	}
	ft.advance(noticeTime)
	if a.notice != nil || a.ov.visible {
		t.Errorf("the third's notice stays: %+v", a.notice)
	}
}

// renderOverlay draws the overlay's window as it shows now, at scale 2,
// its texts measured by its view (as on the Mac): a first build measures
// them and fits the face (overlayMeasure), the picture is the next.
func renderOverlay(t *testing.T, a *app) *image.RGBA {
	t.Helper()
	a.measureHook = nil
	for range 2 {
		w, h := a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad
		tt := hudTester(a.overlayView, w, h)
		tt.SetScale(2)
		frames(tt)
		if a.ov.w+2*layout.Pad == w && a.ov.h+2*layout.Pad == h {
			return tt.Image()
		}
	}
	t.Fatalf("the face does not settle: %d×%d", a.ov.w, a.ov.h)
	return nil
}

// TestOverlayInk: the face as it draws, measured from its pixels as the
// Mac's overlay check measures its captures (internal/overlay/ink): Voxt's
// pill, 148×48, its row's ink as far from both edges, the icon's and the
// waveform's middles one; with live text the face as wide as the text
// (centred under the row) or as the row, at most 300 with the text
// scrolled to its end; the anchor where the pill's is.
func TestOverlayInk(t *testing.T) {
	now := time.Now()
	short, shortZh := "Meet at three tomorrow", "明天三点开会"
	long := "So the plan for the release is to finish the onboarding first, then the history page, and after that the translation card"
	longZh := "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划，然后把引导页和历史页先做完"
	rec := func(k store.Kind, partial string) session.Snapshot {
		return session.Snapshot{Kind: k, State: session.Recording, StartedAt: now, HeardAt: now, Levels: previewSession().Levels, Partial: partial}
	}
	for _, c := range []struct {
		name  string
		s     session.Snapshot
		time  bool
		want  ink.Want
		width int // the face's, when known
	}{
		{"pill", rec(store.KindTranscribe, ""), false, ink.Want{W: 148, H: 48, Row: true, Even: true}, 148},
		// Before the first audio the bars are dim: not ink.
		{"pill-quiet", session.Snapshot{State: session.Recording, StartedAt: now}, false, ink.Want{W: 148, H: 48}, 148},
		{"pill-translate", rec(store.KindTranslate, ""), false, ink.Want{W: 148, H: 48, Row: true, Even: true}, 148},
		{"pill-rewrite", rec(store.KindRewrite, ""), false, ink.Want{W: 148, H: 48, Row: true, Even: true}, 148},
		{"pill-time", rec(store.KindTranscribe, ""), true, ink.Want{W: 188, H: 48, Row: true}, 188},
		{"live-short", rec(store.KindTranscribe, short), false, ink.Want{H: 76, Row: true, Even: true, Text: true}, 0},
		{"live-short-zh", rec(store.KindTranscribe, shortZh), false, ink.Want{W: 160, H: 76, Row: true, Even: true, Text: true}, 160},
		{"live-long", rec(store.KindTranscribe, long), false, ink.Want{W: 300, H: 76, Row: true, Even: true, Text: true, Scrolled: true}, 300},
		{"live-long-zh", rec(store.KindTranscribe, longZh), false, ink.Want{W: 300, H: 76, Row: true, Even: true, Text: true, Scrolled: true}, 300},
		{"working", session.Snapshot{State: session.Recognizing}, false, ink.Want{W: 148, H: 48, Row: true}, 148},
		// A notice's icon is coloured, not ink (as in the Mac's check,
		// overlayInkWant): its row's first stroke would be the text, whose
		// ink's middle is its glyphs' (Core Text's caps sit on the row's
		// middle, its descenders 1.5 under the chevron's), so its size
		// only.
		{"notice", session.Snapshot{State: session.Failed, Problem: &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindAuth}}}, false, ink.Want{H: 48}, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			if strings.HasSuffix(c.name, "-zh") {
				requireCJKFont(t)
			}
			a, _, _ := newTestApp(t)
			a.measureHook = nil // the view measures
			a.prefs.Overlay.ShowTime = c.time
			s := c.s
			s.ID = "ink"
			if s.State != session.Recording {
				snap(a, session.Snapshot{ID: s.ID, State: session.Recording, StartedAt: now})
			}
			snap(a, s)
			img := renderOverlay(t, a)
			if c.width != 0 && a.ov.w != c.width {
				t.Errorf("the face is %d wide, want %d", a.ov.w, c.width)
			}
			want := c.want
			want.W = a.ov.w
			anchored(t, c.name, a.prefs.Overlay, a.ov.frame)
			k, bad := ink.Check(img, 2, want)
			for _, b := range bad {
				t.Error(b)
			}
			t.Logf("%s: %v", c.name, k)
			if dir := os.Getenv("VOXTNEW_INK"); dir != "" {
				var b bytes.Buffer
				png.Encode(&b, img)
				os.WriteFile(filepath.Join(dir, c.name+".png"), b.Bytes(), 0o644)
			}
			if c.name == "live-short" {
				// The text's measured width and the padding: no guess.
				tw := a.ov.textW[liveTextKey(short)].w
				if tw <= layout.RowW || a.ov.w != int(math.Ceil(float64(tw)))+2*layout.LivePadX {
					t.Errorf("the face is %d wide for a text %v wide", a.ov.w, tw)
				}
			}
		})
	}
}

// TestOverlayShots renders the overlay's states alone (overlayShots),
// English over a light desktop and Chinese over a dark one:
//
//	VOXTNEW_OVERLAY_SHOTS=/tmp/ov go test -run TestOverlayShots .
func TestOverlayShots(t *testing.T) { runSnapshots(t, "VOXTNEW_OVERLAY_SHOTS", snapshotsOverlay) }

func snapshotsOverlay(t *testing.T, out string) {
	for _, run := range []struct {
		lang string
		dark bool
	}{{"en", false}, {"zh-Hans", true}} {
		a, _, _ := newTestApp(t)
		if run.lang != "en" {
			a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
		}
		suffix := "light"
		if run.dark {
			suffix = "dark"
		}
		if run.lang != "en" {
			suffix += "-" + run.lang
		}
		overlayShots(t, a, out, suffix, run.dark)
	}
}
