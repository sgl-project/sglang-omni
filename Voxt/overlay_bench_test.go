package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type overlayBenchmarkState struct {
	name     string
	snapshot session.Snapshot
	setup    func()
	after    func(string)
}

func sessionOverlayBenchmarkStates(a *app, now time.Time) []overlayBenchmarkState {
	longTranslation := "Go 的并发模型建立在 goroutine 和 channel 之上。goroutine 是由 Go 运行时管理的轻量级线程，启动成本很低，一个程序里同时运行成千上万个也很常见。channel 则是 goroutine 之间传递数据的管道：发送和接收默认都会阻塞，直到另一端准备好，这让同步变得自然。\n\n实践中，我们通常用 context 控制取消和超时，用 errgroup 等待一组任务结束并收集第一个错误，用带缓冲的 channel 限制并发数。需要共享状态时，sync.Mutex 依然是最直接的选择；但如果能把状态的所有权交给一个 goroutine，并通过 channel 和它通信，代码往往更容易推理。\n\n最后，别忘了用 -race 运行测试：数据竞争很少在本地复现，却常常在生产环境里出现。"
	return []overlayBenchmarkState{
		{name: "overlay-starting", snapshot: session.Snapshot{State: session.Recording, StartedAt: now}},
		{name: "overlay-waiting-mic", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, Waiting: session.WaitMic}},
		{name: "overlay-waiting-model", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-2 * time.Second), Levels: wave(), Waiting: session.WaitModel}},
		{name: "overlay-waiting-model-recognizing", snapshot: session.Snapshot{State: session.Recognizing, Waiting: session.WaitModel}},
		{name: "overlay-card-unconfirmed", snapshot: session.Snapshot{State: session.Done, Card: true, Delivery: deliver.ReasonUnconfirmed, Target: proto.App{PID: 77, Name: "Arc"}, Result: "详情中的左右滑动不够流畅，会卡住。"}},
		{name: "overlay-recording", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-12 * time.Second), Levels: wave()}},
		{name: "overlay-recording-time", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-12 * time.Second), Levels: wave()}, setup: func() { a.prefs.Overlay.ShowTime = true }},
		{name: "overlay-length-limit", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-9*time.Minute - 18*time.Second), Levels: wave(), MaxLen: 10 * time.Minute}, setup: func() { a.prefs.Overlay.ShowTime = true }},
		{name: "overlay-upload-limit", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-13*time.Minute - 20*time.Second), Levels: wave(), UploadLen: 13*time.Minute + 39*time.Second}},
		{name: "overlay-live-short", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-3 * time.Second), Levels: wave(), Partial: "Meet at three tomorrow"}},
		{name: "overlay-live-short-zh", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-3 * time.Second), Levels: wave(), Partial: "明天三点开会"}},
		{name: "overlay-live-text", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-47 * time.Second), Levels: wave(), Partial: "So the plan for the release is to finish the onboarding first, then the history page, and after that we look at the translation card"}},
		{name: "overlay-live-text-zh", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-20 * time.Second), Levels: wave(), Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划，然后把引导页和历史页先做完"}},
		{name: "overlay-lines-start", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-1 * time.Second), Levels: wave()}, setup: func() { a.prefs.Overlay.LiveLines = true }},
		{name: "overlay-live-lines", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-47 * time.Second), Levels: wave(), Partial: "So the plan for the release is to finish the onboarding first, then the history page, and after that we look at the translation card"}, setup: func() { a.prefs.Overlay.LiveLines = true }},
		{name: "overlay-live-lines-long-zh", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-90 * time.Second), Levels: wave(), Partial: strings.Repeat("那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划，然后把引导页和历史页先做完。", 8)}, setup: func() { a.prefs.Overlay.LiveLines = true }},
		{name: "overlay-recording-scene", snapshot: session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now.Add(-5 * time.Second), Levels: wave(), Target: proto.App{Name: "Slack"}, Scene: sceneMatch("Team chat", appctx.SourceApp, "")}},
		{name: "overlay-working-scene", snapshot: session.Snapshot{State: session.Composing, Target: proto.App{Name: "Google Chrome"}, Scene: sceneMatch("Code review", appctx.SourceURL, "github.com")}},
		{name: "overlay-recognizing", snapshot: session.Snapshot{State: session.Recognizing}},
		{name: "overlay-composing", snapshot: session.Snapshot{State: session.Composing, Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划"}},
		{name: "overlay-composing-skip", snapshot: session.Snapshot{Kind: store.KindTranscribe, State: session.Composing, CanSkip: true, Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划"}},
		{name: "overlay-composing-think", snapshot: session.Snapshot{Kind: store.KindTranscribe, State: session.Composing, CanSkip: true, LLM: session.Model{ID: llm.Custom}, Thinking: &llm.Thinking{Mode: llm.ThinkOff}, Partial: "那个我们明天，不对，后天下午三点开会，讨论一下 VoxtNew 的发布计划"}, setup: func() {
			a.prefs.Providers.LLM = []settings.Provider{{ID: llm.Custom}}
		}},
		{name: "overlay-composing-think-rewrite", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Composing, LLM: session.Model{ID: llm.Custom}, Raw: "make it shorter"}, setup: func() {
			a.prefs.Providers.LLM = []settings.Provider{{ID: llm.Custom, Generation: settings.Generation{Thinking: llm.ThinkEffort, ThinkingEffort: "high"}}}
		}},
		{name: "overlay-error", snapshot: session.Snapshot{State: session.Failed, Problem: &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindAuth}}}},
		{name: "overlay-no-speech", snapshot: session.Snapshot{State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}}},
		{name: "overlay-warning", snapshot: session.Snapshot{State: session.Done, Result: "x", Inserted: true, Problem: &session.Problem{Code: session.ProblemLLM, Degraded: true, Err: errors.New("timeout")}}},
		{name: "overlay-answer", snapshot: session.Snapshot{State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Result: "Hello team, the release moves to Thursday.\n\n1. Finish the onboarding\n2. Polish the history page\n3. Ship the translation card\n\nThanks!"}},
		{name: "overlay-translate-recording", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Recording, StartedAt: now, HeardAt: now.Add(-5 * time.Second), Levels: wave(), TargetLanguage: "en"}},
		{name: "overlay-translate-selection", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Composing, Selection: "A goroutine is a lightweight thread.", TargetLanguage: "zh-Hans", Draft: "goroutine 是一种轻量级"}},
		{name: "overlay-rewrite-writing", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Composing, Card: true, Raw: "write a short note saying I will be late", Draft: "Sorry, I'm running about ten minutes late"}},
		{name: "overlay-translate-card-streaming", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Composing, Card: true, Delivery: deliver.ReasonNotEditable, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Draft: "goroutine 是由 Go 运行时管理的"}},
		{name: "overlay-translate-card", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}},
		{name: "overlay-translate-card-cut-off", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的", Problem: &session.Problem{Code: session.ProblemTruncated, Degraded: true}}},
		{name: "overlay-translate-card-en", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "我们把发布推迟到周四，先把引导页和历史页做完。", TargetLanguage: "en", Result: "We are moving the release to Thursday, to finish the onboarding and the history page first."}},
		{name: "overlay-translate-card-long", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "Go's concurrency…", TargetLanguage: "zh-Hans", Result: longTranslation}},
		{name: "overlay-translate-card-redo", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, after: func(id string) {
			a.answer.redo, a.answer.redoLang = id+"-ja", "ja"
			snap(a, session.Snapshot{ID: id + "-ja", Kind: store.KindTranslate, State: session.Composing, Card: true, TargetLanguage: "ja"})
		}},
		{name: "overlay-translate-card-menu", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}},
		{name: "overlay-translate-card-copied", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, after: func(string) { a.answer.copied = true }},
		{name: "overlay-translate-card-insert-failed", snapshot: session.Snapshot{Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Target: proto.App{PID: 77, Name: "Arc"}, Selection: "A goroutine is a lightweight thread managed by the Go runtime.", TargetLanguage: "zh-Hans", Result: "goroutine 是由 Go 运行时管理的轻量级线程。"}, after: func(string) {
			a.answer.msg = uitext.DeliveryText(deliver.Result{Reason: deliver.ReasonNotFront}, proto.App{Name: "Arc"})
		}},
		{name: "overlay-rewrite-card", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "写一段两百字的自我介绍，用于技术分享会的开场", Title: "技术分享开场自我介绍", Result: "大家好，我是关伟，一名专注于桌面应用和语音交互的工程师。过去几年，我主要在做跨平台的原生界面框架，也参与了几个语音输入工具的设计和开发。\n\n今天想和大家聊聊：怎样让语音输入真正融入日常的写作流程——从按下快捷键的那一刻，到文字出现在光标处，中间发生了什么，又有哪些地方值得打磨。"}},
		{name: "overlay-rewrite-conversation-recording", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "写一句话告诉同事我会晚到十分钟", Title: "晚到通知", Result: "大家好，我这边路上有点堵，大概会晚到十分钟，会议请先开始，不用等我。"}, after: func(id string) {
			a.followUp, a.answer.conversation = id, true
			snap(a, session.Snapshot{ID: id + "-f", Kind: store.KindRewrite, State: session.Recording, StartedAt: now, HeardAt: now, Levels: wave(), Partial: "再正式一点"})
		}},
		{name: "overlay-rewrite-conversation", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "再正式一点", Title: "晚到说明", Result: "各位好：因交通拥堵，我预计将晚到约十分钟。会议请按时开始，无需等候，抱歉给大家带来不便。"}},
		{name: "overlay-rewrite-failed", snapshot: session.Snapshot{Kind: store.KindRewrite, State: session.Failed, Card: true, Raw: "make it more formal", Selection: "hey, the release slips to thursday, sorry!", Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("language model stopped")}}},
	}
}

func settledOverlayBenchmarkTester(tb testing.TB, a *app) *ui.Tester {
	tb.Helper()
	for range 2 {
		width, height := a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad
		tt := hudTester(a.overlayView, width, height)
		tt.SetScale(2)
		frames(tt)
		if a.ov.w+2*layout.Pad == width && a.ov.h+2*layout.Pad == height {
			return tt
		}
	}
	tb.Fatalf("the overlay face does not settle: %d×%d", a.ov.w, a.ov.h)
	return nil
}

func overlayBenchmarkSentinel(a *app) string {
	if a.ov.mode == ovAnswer {
		view := a.cardNow()
		for _, candidate := range []string{view.text, view.msg, view.title} {
			if candidate != "" {
				return candidate
			}
		}
	}
	if a.notice != nil && a.notice.text != "" {
		return a.notice.text
	}
	if a.sess.Problem != nil {
		text, _, _ := sessionProblemText(a.sess.Problem)
		return text
	}
	text, _ := pillLine(a.sess)
	if text != "" {
		return text
	}
	if a.sess.State == session.Recording {
		if a.sess.Waiting != session.WaitNone {
			return uitext.WaitText(a.sess)
		}
		return uitext.KindTitle(a.sess.Kind)
	}
	return uitext.WorkingText(a.sess)
}

func benchmarkOverlayContent(b *testing.B, tt *ui.Tester, sentinel string) {
	b.Helper()
	benchmarkRenderFrames(b, tt, sentinel)
}

func BenchmarkW0OverlayStates(b *testing.B) {
	a, _, _ := newRenderBenchmarkApp(b)
	a.prefs.Overlay.ShowTime = false
	a.measureHook = nil
	for index, state := range sessionOverlayBenchmarkStates(a, time.Now()) {
		b.Run(state.name, func(b *testing.B) {
			a.answer, a.notice, a.followUp = answerCard{}, nil, ""
			a.syncLangMenu()
			if state.name == "overlay-rewrite-conversation" {
				a.answer = answerCard{open: true, id: "first", kind: store.KindRewrite, conversation: true}
				a.answer.turns = []compose.Turn{{Instruction: "写一句话告诉同事我会晚到十分钟", Answer: compose.Answer{Title: "晚到通知", Content: "大家好，我这边路上有点堵，大概会晚到十分钟，会议请先开始，不用等我。"}}}
				a.followUp = "first"
			}
			a.prefs.Overlay.ShowTime, a.prefs.Overlay.LiveLines = false, false
			if state.setup != nil {
				state.setup()
			}
			id := "shot" + itoa(index)
			final := state.snapshot
			final.ID = id
			live := final
			live.State = session.Recording
			live.Problem, live.Result, live.Inserted, live.Card = nil, "", false, false
			snap(a, live)
			snap(a, final)
			if state.after != nil {
				state.after(id)
				a.overlaySync()
			}
			a.toasts = nil
			if a.ov.mode == ovHidden || !a.ov.shown {
				b.Fatalf("%s: overlay is not visible", state.name)
			}
			tt := settledOverlayBenchmarkTester(b, a)
			if state.name == "overlay-translate-card-menu" {
				a.toggleLangMenu()
				if !a.ov.lang.shown {
					b.Fatal("language menu is not visible")
				}
				menuDY := a.ov.frame.Y - a.ov.lang.frame.Y
				menuFrame := a.ov.lang.frame
				view := func(c *ui.Context) {
					ui.Box(c).Absolute().Left(0).Top(float32(menuDY)).Right(0).Bottom(0).Children(func() { a.overlayView(c) })
					if menuDY > 0 {
						ui.Box(c).Absolute().Left(float32(menuFrame.X - a.ov.frame.X)).Top(0).Width(float32(menuFrame.Width)).Height(float32(menuFrame.Height)).Children(func() { a.langMenuView(c) })
					}
				}
				tt = hudTester(view, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad+menuDY)
				tt.SetScale(2)
			}
			benchmarkOverlayContent(b, tt, overlayBenchmarkSentinel(a))
		})
	}
}

func BenchmarkW0AnswerCards(b *testing.B) {
	steps := cardSteps()
	for index, state := range steps {
		b.Run(state.name, func(b *testing.B) {
			a, _, _ := newRenderBenchmarkApp(b)
			savedArea := ovTestArea
			b.Cleanup(func() { ovTestArea = savedArea })
			ovTestArea.Height = layout.CardBodyMax + int(math.Round(a.prefs.Overlay.Margin)) + layout.Pad + layout.CardChrome
			a.ov.area = ovTestArea
			for prior := 0; prior <= index; prior++ {
				steps[prior].apply(a)
				a.overlaySync()
			}
			a.measureHook = nil
			if a.ov.mode != ovAnswer || !a.ov.shown {
				b.Fatalf("%s: answer card is not visible", state.name)
			}
			benchmarkOverlayContent(b, settledOverlayBenchmarkTester(b, a), overlayBenchmarkSentinel(a))
		})
	}
}

func BenchmarkW0MeetingOverlayStates(b *testing.B) {
	a, _, _ := newRenderBenchmarkApp(b)
	now := time.Now()
	segments := []meeting.Segment{
		{ID: "s1", Source: meeting.SourceSystem, Speaker: "s1", Text: "Mail and Voice Memos group by day and show two lines per row.", Final: true},
		{ID: "s2", Source: meeting.SourceMic, Speaker: "me", Text: "Yes. I'll ship the list first and the detail panel after.", Final: true},
	}
	levels := map[meeting.Source][]float64{}
	for index := range session.Bars {
		levels[meeting.SourceMic] = append(levels[meeting.SourceMic], session.Level(-40+float64(index%5)*7))
		levels[meeting.SourceSystem] = append(levels[meeting.SourceSystem], session.Level(-50+float64(index%3)*6))
	}
	recording := meeting.Snapshot{ID: "ov", Mode: meeting.ModeMeeting, State: meeting.StateRecording, Started: now.Add(-754 * time.Second), Elapsed: 754 * time.Second,
		Levels: levels, Live: map[meeting.Source]string{meeting.SourceSystem: "Sounds good, let's review again"}, Segments: segments}
	paused, problem, asking, finishing := recording, recording, recording, recording
	paused.State = meeting.StatePaused
	problem.Problem = &meeting.Problem{Key: meeting.ProblemKeyDiskLow}
	asking.State = meeting.StateAsking
	finishing.State, finishing.Stage, finishing.Progress = meeting.StateFinishing, meeting.StageTranscribing, 0.4
	problemText := meetingOverlayProblemText(meeting.ProblemKeyDiskLow)
	states := []struct {
		name      string
		snapshot  meeting.Snapshot
		collapsed bool
		menu      meetingMenu
		sentinel  string
	}{
		{name: "meeting-overlay-starting", snapshot: meeting.Snapshot{ID: "ov", Mode: meeting.ModeMeeting, State: meeting.StateStarting, Started: now}, sentinel: "End meeting"},
		{name: "meeting-overlay-recording", snapshot: recording, sentinel: segments[0].Text},
		{name: "meeting-overlay-paused", snapshot: paused, sentinel: "Paused"},
		{name: "meeting-overlay-problem", snapshot: problem, sentinel: problemText},
		{name: "meeting-overlay-pill", snapshot: recording, collapsed: true, sentinel: "Expand"},
		{name: "meeting-overlay-pill-problem", snapshot: problem, collapsed: true, sentinel: problemText},
		{name: "meeting-overlay-asking", snapshot: asking, sentinel: "Save this meeting?"},
		{name: "meeting-overlay-finishing", snapshot: finishing, sentinel: "Finalising transcript… 40%"},
		{name: "meeting-overlay-menu-mode", snapshot: recording, menu: meetingMenuMode, sentinel: "System audio only"},
		{name: "meeting-overlay-menu-more", snapshot: paused, menu: meetingMenuMore, sentinel: "Copy transcript"},
	}
	for _, state := range states {
		b.Run(state.name, func(b *testing.B) {
			a.meetSnap, a.meetOvUI.collapsed, a.meetOvUI.menu = state.snapshot, state.collapsed, state.menu
			model := overlayModel(state.snapshot, state.collapsed, 0, state.menu)
			tt := hudTester(a.meetingOverlayView, model.Layout.Width, model.Layout.Height)
			tt.SetScale(2)
			benchmarkRenderFrames(b, tt, state.sentinel)
		})
	}
}

const (
	waveformLevelsPerSecond   = 20
	answerStreamChunks        = 2000
	answerStreamChunk         = "x"
	meetingCommitInterval     = 8 * time.Second
	meetingUpdatesPerHour     = int(2 * time.Hour / meetingCommitInterval)
	meetingSnapshotSegmentCap = 200
)

func BenchmarkW0WaveformUpdate(b *testing.B) {
	a, _, _ := newRenderBenchmarkApp(b)
	now := time.Now()
	partial := "Benchmark stable partial"
	snapshot := session.Snapshot{ID: "benchmark-waveform", State: session.Recording, StartedAt: now, HeardAt: now, Partial: partial, Levels: wave()}
	snap(a, snapshot)
	tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	frames(tt)
	// The helper's capture.level input defaults to 20 Hz.
	levels := make([]float64, waveformLevelsPerSecond)
	for index := range levels {
		levels[index] = session.Level(-45 + float64(index%5)*6)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; b.Loop(); index++ {
		snapshot.Levels = append(snapshot.Levels, levels[index%len(levels)])
		if len(snapshot.Levels) > session.Bars {
			snapshot.Levels = snapshot.Levels[len(snapshot.Levels)-session.Bars:]
		}
		snap(a, snapshot)
		tt.Frame()
	}
	b.StopTimer()
	if a.sess.Partial != partial || !tt.HasText(partial) {
		b.Fatalf("stable partial is missing from the frame: %q", tt.Texts())
	}
}

func runW0AnswerStreamPath(tb testing.TB, recordPath string) []int {
	tb.Helper()
	fullDraft := strings.Repeat(answerStreamChunk, 6*answerStreamChunks)
	chat := &benchmarkChat{fakeChat: fakeChat{text: fullDraft}}
	events := benchmarkScript(tb, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newBenchmarkSessionApp(tb, helpertest.Opt("selection", "Benchmark selection")+","+events, func(s *settings.Settings) {
		s.Features.Translate.Target, s.Features.Translate.ShowCard = "en", true
	}, &fakeLocal{Chat: &llm.Client{Backend: chat}})
	tt := hudTester(a.overlayView, layout.AnswerW+2*layout.Pad, layout.CardMaxH+2*layout.Pad)
	var lengths []int
	var received strings.Builder
	chat.onDelta = func(text string) { received.WriteString(text) }
	lastDraft := ""
	updates := make(chan struct{}, answerStreamChunks)
	chat.afterDelta = func() { updates <- struct{}{} }
	deadline := time.After(time.Minute)
	for !a.sess.State.Ended() || len(updates) > 0 {
		select {
		case fn := <-a.queue:
			fn()
		case <-updates:
			draining := true
			for draining {
				select {
				case fn := <-a.queue:
					fn()
				default:
					draining = false
				}
			}
			if a.sess.Draft != lastDraft {
				lastDraft = a.sess.Draft
				lengths = append(lengths, len(lastDraft))
				tt.Frame()
			}
		case <-deadline:
			tb.Fatalf("timed out waiting for the streamed answer (session %s %+v)", a.sess.State, a.sess.Problem)
		}
	}
	tt.Frame()
	if a.sess.Result != fullDraft || !tt.HasText(fullDraft) {
		tb.Fatalf("completed draft is missing from the frame: result %q texts %q", a.sess.Result, tt.Texts())
	}
	if len(lengths) == 0 {
		tb.Fatal("no draft updates were published")
	}
	if received.String() != fullDraft {
		tb.Fatalf("received %d bytes in deltas, want %d", received.Len(), len(fullDraft))
	}
	if lengths[len(lengths)-1] != len(fullDraft) {
		// The completed result is a separate frame after the published drafts.
		lengths = append(lengths, len(fullDraft))
	}
	if recordPath != "" {
		data, err := json.Marshal(lengths)
		if err != nil {
			tb.Fatal(err)
		}
		if err := os.WriteFile(recordPath, data, 0o644); err != nil {
			tb.Fatal(err)
		}
	}
	return lengths
}

func TestW0AnswerStreamingPath(t *testing.T) {
	if os.Getenv("VOXTNEW_W0_STREAM_PATH") != "1" {
		t.Skip("set VOXTNEW_W0_STREAM_PATH=1 to verify the real stream path")
	}
	runW0AnswerStreamPath(t, os.Getenv("VOXTNEW_W0_STREAM_LENGTHS"))
}

func BenchmarkW0AnswerStreaming(b *testing.B) {
	data, err := os.ReadFile("testdata/w0-stream-lengths.json")
	if err != nil {
		b.Fatal(err)
	}
	var lengths []int
	if err := json.Unmarshal(data, &lengths); err != nil {
		b.Fatal(err)
	}
	fullDraft := strings.Repeat(answerStreamChunk, 6*answerStreamChunks)
	a, _, _ := newRenderBenchmarkApp(b)
	snapshot := session.Snapshot{ID: "benchmark-stream", Kind: store.KindTranslate, State: session.Composing, Card: true, Delivery: deliver.ReasonCard, TargetLanguage: "en"}
	tt := hudTester(a.overlayView, layout.AnswerW+2*layout.Pad, layout.CardMaxH+2*layout.Pad)
	frames(tt)
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; b.Loop(); index++ {
		position := index % len(lengths)
		snapshot.Draft = fullDraft[:lengths[position]]
		snap(a, snapshot)
		tt.Frame()
	}
	b.StopTimer()
	snapshot.State, snapshot.Result, snapshot.Draft = session.Done, fullDraft, fullDraft
	snap(a, snapshot)
	tt.Frame()
	if a.sess.Result != fullDraft || !tt.HasText(fullDraft) {
		b.Fatalf("completed draft is missing from the frame: %q", tt.Texts())
	}
	b.ReportMetric(float64(len(lengths)), "recorded_frames")
}

func BenchmarkW0MeetingOverlayHour(b *testing.B) {
	a, _, _ := newRenderBenchmarkApp(b)
	snapshot := meeting.Snapshot{ID: "benchmark-hour", Mode: meeting.ModeMeeting, State: meeting.StateRecording}
	a.meetSnap = snapshot
	model := overlayModel(snapshot, false, 0, meetingMenuNone)
	tt := hudTester(a.meetingOverlayView, model.Layout.Width, meetingOvMaxFaceH+2*layout.Pad)
	frames(tt)
	// Two tracks at the committer's 8 s cap produce 450 segments each in one hour.
	segments := make([]meeting.Segment, meetingUpdatesPerHour)
	for index := range segments {
		trackIndex := index / 2
		source, speaker := meeting.SourceMic, "me"
		if index%2 != 0 {
			source, speaker = meeting.SourceSystem, "s1"
		}
		start := time.Duration(trackIndex) * meetingCommitInterval
		segments[index] = meeting.Segment{ID: "segment-" + strconv.Itoa(index), Source: source, Speaker: speaker,
			Start: start, End: start + meetingCommitInterval, Text: "Benchmark segment " + strconv.Itoa(index) + ".", Final: true}
	}
	b.ReportAllocs()
	b.ResetTimer()
	updates := 0
	for b.Loop() {
		index := updates % meetingUpdatesPerHour
		if index == 0 {
			snapshot.Segments = snapshot.Segments[:0]
			snapshot.Elapsed = 0
		}
		snapshot.Segments = append(snapshot.Segments, segments[index])
		if len(snapshot.Segments) > meetingSnapshotSegmentCap {
			snapshot.Segments = snapshot.Segments[len(snapshot.Segments)-meetingSnapshotSegmentCap:]
		}
		snapshot.Elapsed = segments[index].End
		a.meetingChanged(snapshot)
		tt.Frame()
		updates++
	}
	b.StopTimer()
	newest := segments[(updates-1)%meetingUpdatesPerHour]
	if !tt.HasText(newest.Text) {
		b.Fatalf("newest segment %q is missing from the frame: %q", newest.Text, tt.Texts())
	}
	retained := min((updates-1)%meetingUpdatesPerHour+1, meetingSnapshotSegmentCap)
	if len(a.meetSnap.Segments) != retained {
		b.Fatalf("snapshot keeps %d segments, want %d", len(a.meetSnap.Segments), retained)
	}
	if updates%meetingUpdatesPerHour == 0 && a.meetSnap.Elapsed != time.Hour {
		b.Fatalf("simulated meeting elapsed time is %v, want %v", a.meetSnap.Elapsed, time.Hour)
	}
}
