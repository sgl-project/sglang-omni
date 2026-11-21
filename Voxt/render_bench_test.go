package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func newRenderBenchmarkApp(tb testing.TB) (*app, *fakeSupervisor, *ui.Tester) {
	tb.Helper()
	i18n.SetLocale("en")
	tb.Cleanup(func() { i18n.SetLocale("en") })
	lib, err := openLibrary(tb.TempDir())
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(lib.close)
	fs := newFakeSupervisor(lib.bus)
	a := newApp(lib, &helperHost{sup: fs})
	a.permSys = &permission.Fake{Mic: permission.Granted, Grant: true}
	useTestSounds(tb, a)
	a.playSound = func(sounds.Target) {}
	a.measureHook = guessText
	a.connectHelper()
	a.upd = newFakeUpdates()
	a.connectUpdates()
	a.startHelper()
	tb.Cleanup(a.quit)
	return a, fs, ui.NewTester(a.view, windowWidth, windowHeight)
}

func seedRenderBenchmarkFixtures(tb testing.TB, a *app) {
	tb.Helper()
	ctx := context.Background()
	if err := a.lib.store.History().Put(ctx, store.Entry{
		ID: "benchmark-history", Kind: store.KindTranscribe, CreatedAt: time.Now(),
		Status: store.StatusDone, Result: "Benchmark history result",
	}); err != nil {
		tb.Fatal(err)
	}
	if err := a.lib.store.Dictionary().Put(ctx, store.Term{ID: "benchmark-term", Term: "BenchmarkTerm", Enabled: true}); err != nil {
		tb.Fatal(err)
	}
	if _, err := a.lib.store.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "benchmark-meeting", CreatedAt: time.Now(), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Title: "Benchmark meeting title",
	}); err != nil {
		tb.Fatal(err)
	}
	a.setRules(sampleRules())
	a.logs.Append(core.LogEntry{Level: core.SeverityInfo, Source: helper.LogSource, Message: "benchmark log line"})
}

func benchmarkRenderFrames(b *testing.B, tt *ui.Tester, sentinel string) {
	b.Helper()
	if sentinel == "" {
		b.Fatal("empty frame sentinel")
	}
	b.ReportAllocs()
	frames(tt)
	b.ResetTimer()
	for b.Loop() {
		tt.Frame()
	}
	b.StopTimer()
	if !tt.HasText(sentinel) {
		b.Fatalf("frame does not contain %q: %q", sentinel, tt.Texts())
	}
}

func BenchmarkW0Pages(b *testing.B) {
	sentinels := map[string]string{
		"/home": "Today", "/history": "Benchmark history result",
		"/transcribe": "Models", "/translate": "Models", "/rewrite": "Models",
		"/app-enhancement": "Team chat", modelsPath: "Speech models", remotePath: "On this Mac",
		"/dictionary": "BenchmarkTerm", "/notes": "Keys to record a note", "/meetings": "Benchmark meeting title", "/logs": "benchmark log line",
	}
	for _, page := range pages {
		b.Run(strings.TrimPrefix(page.path, "/"), func(b *testing.B) {
			a, _, tt := newRenderBenchmarkApp(b)
			seedRenderBenchmarkFixtures(b, a)
			a.navigate(page.path)
			benchmarkRenderFrames(b, tt, sentinels[page.path])
		})
	}
	for _, section := range settingsSections() {
		b.Run("settings-"+section.id, func(b *testing.B) {
			a, _, tt := newRenderBenchmarkApp(b)
			a.navigate("/settings/" + section.id)
			sentinel := map[string]string{
				"general": "Languages", "permissions": "Microphone", "audio": "Recognition",
				"overlay": "Preview", "shortcuts": "Transcribe", "appearance": "Sidebar",
				"privacy": "What leaves this Mac", "storage": "Data folder", "about": "Self-check",
			}[section.id]
			benchmarkRenderFrames(b, tt, sentinel)
		})
	}
	b.Run("llm-debug", func(b *testing.B) {
		a, _, tt := newRenderBenchmarkApp(b)
		a.navigate(llmDebugPath)
		benchmarkRenderFrames(b, tt, "Try a prompt on your own text")
	})
	b.Run("onboarding", func(b *testing.B) {
		a, _, tt := newRenderBenchmarkApp(b)
		a.maybeOnboard()
		benchmarkRenderFrames(b, tt, "Set up Voxt")
	})
}

func benchmarkFirstFrame(b *testing.B, a *app, sentinel string) {
	b.Helper()
	fresh := func() *app {
		fresh := &app{lib: a.lib, shellState: newShellState(), helperLink: helperLink{host: a.host}}
		fresh.router = ui.NewRouter(a.shownPath())
		fresh.settings, fresh.prefs = a.settings, a.prefs
		fresh.notes, fresh.notices, fresh.logs = a.notes, a.notices, a.logs
		fresh.permSys, fresh.local, fresh.remote = a.permSys, a.local, a.remote
		fresh.models, fresh.modelStatus, fresh.upd = a.models, a.modelStatus, a.upd
		fresh.version, fresh.measureHook = a.version, a.measureHook
		fresh.hotkeys = a.hotkeys
		fresh.book, fresh.suggestions = a.book, a.suggestions
		fresh.hist, fresh.dv, fresh.mv, fresh.nv = a.hist, a.dv, a.mv, a.nv
		fresh.histUI, fresh.homeDir, fresh.defaultAudioDir = a.histUI, a.homeDir, a.defaultAudioDir
		fresh.dictUI, fresh.nvUI = a.dictUI, a.nvUI
		fresh.histUI.listUI = ui.ListState{}
		fresh.hist.view, fresh.hist.viewKey = historyViewState{}, historyViewCacheKey{}
		fresh.dv.view, fresh.dv.viewKey = dictionaryViewState{}, dictionaryViewCacheKey{}
		fresh.dictUI.listUI = ui.ListState{}
		fresh.mvUI.listUI = ui.ListState{}
		fresh.mv.listItems, fresh.mv.listKey, fresh.mv.detailView = nil, meetingListCacheKey{}, nil
		fresh.nvUI.listUI = ui.ListState{}
		return fresh
	}
	b.ReportAllocs()
	var pixels []uint8
	b.ResetTimer()
	for b.Loop() {
		pixels = ui.Render(fresh().view, windowWidth, windowHeight, 1).Pix
	}
	b.StopTimer()
	if len(pixels) == 0 {
		b.Fatal("empty first frame")
	}
	tt := ui.NewTester(fresh().view, windowWidth, windowHeight)
	if !tt.HasText(sentinel) {
		b.Fatalf("first frame does not contain %q: %q", sentinel, tt.Texts())
	}
}

func checkScrollChanges(tb testing.TB, tt *ui.Tester, x, y, delta float32) {
	tb.Helper()
	before := strings.Join(tt.Texts(), "\x00")
	tt.Scroll(x, y, 0, delta)
	after := strings.Join(tt.Texts(), "\x00")
	if before == after {
		tb.Fatalf("scroll by %v did not change the visible rows", delta)
	}
}

func BenchmarkW0LongLists(b *testing.B) {
	for _, count := range []int{2500, 5000} {
		b.Run(fmt.Sprintf("History%d_cap500", count), func(b *testing.B) {
			a, _, tt := newRenderBenchmarkApp(b)
			now := time.Now()
			for i := range count {
				if err := a.lib.store.History().Put(context.Background(), store.Entry{
					ID: fmt.Sprintf("history-%05d", i), Kind: store.KindTranscribe,
					CreatedAt: now.Add(time.Duration(i) * time.Second), Status: store.StatusDone,
					Result: fmt.Sprintf("History result %05d", i),
				}); err != nil {
					b.Fatal(err)
				}
			}
			a.navigate("/history")
			frames(tt)
			sentinel := fmt.Sprintf("History result %05d", count-1)
			b.Run("FirstFrame", func(b *testing.B) { benchmarkFirstFrame(b, a, sentinel) })
			b.Run("Scroll", func(b *testing.B) {
				checkScrollChanges(b, tt, float32(windowWidth)*0.75, float32(windowHeight)*0.6, 300)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; b.Loop(); i++ {
					dy := float32(300)
					if i%2 != 0 {
						dy = -dy
					}
					tt.Scroll(float32(windowWidth)*0.75, float32(windowHeight)*0.6, 0, dy)
				}
				b.StopTimer()
				tt.Scroll(float32(windowWidth)*0.75, float32(windowHeight)*0.6, 0, 1e6)
				oldest := fmt.Sprintf("History result %05d", count-500)
				if !tt.HasText(oldest) {
					b.Fatalf("history list did not scroll to %q: %q", oldest, tt.Texts())
				}
			})
		})
	}
	for _, count := range []int{250, 500} {
		b.Run(fmt.Sprintf("Meetings%d_page20", count), func(b *testing.B) {
			a, _, tt := newRenderBenchmarkApp(b)
			now := time.Now()
			for i := range count {
				if _, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{
					ID: fmt.Sprintf("meeting-%05d", i), CreatedAt: now.Add(time.Duration(i) * time.Second),
					Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Title: fmt.Sprintf("Meeting %05d", i),
				}); err != nil {
					b.Fatal(err)
				}
			}
			a.navigate("/meetings")
			frames(tt)
			sentinel := fmt.Sprintf("Meeting %05d", count-1)
			b.Run("FirstFrame", func(b *testing.B) { benchmarkFirstFrame(b, a, sentinel) })
			b.Run("Scroll", func(b *testing.B) {
				checkScrollChanges(b, tt, float32(windowWidth)*0.5, float32(windowHeight)*0.6, 300)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; b.Loop(); i++ {
					dy := float32(300)
					if i%2 != 0 {
						dy = -dy
					}
					tt.Scroll(float32(windowWidth)*0.5, float32(windowHeight)*0.6, 0, dy)
				}
				b.StopTimer()
				page := a.mvUI.page + 1
				if err := tt.ClickControl("Next page"); err != nil {
					b.Fatal(err)
				}
				frames(tt)
				next := fmt.Sprintf("Meeting %05d", count-20*page-1)
				if a.mvUI.page != page || len(a.mv.list) == 0 || a.mv.list[0].Title != next || !tt.HasText(fmt.Sprintf("Page %d", page+1)) {
					b.Fatalf("meetings did not reach page %d at %q: %q", page+1, next, tt.Texts())
				}
			})
		})
	}
	for _, count := range []int{2500, 5000} {
		b.Run(fmt.Sprintf("Dictionary%d", count), func(b *testing.B) {
			a, _, tt := newRenderBenchmarkApp(b)
			for i := range count {
				if err := a.lib.store.Dictionary().Put(context.Background(), store.Term{
					ID: fmt.Sprintf("term-%05d", i), Term: fmt.Sprintf("Term %05d", i), Enabled: true,
				}); err != nil {
					b.Fatal(err)
				}
			}
			a.navigate("/dictionary")
			frames(tt)
			b.Run("FirstFrame", func(b *testing.B) { benchmarkFirstFrame(b, a, "Term 00000") })
			b.Run("Scroll", func(b *testing.B) {
				checkScrollChanges(b, tt, float32(windowWidth)*0.5, float32(windowHeight)*0.75, 300)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; b.Loop(); i++ {
					dy := float32(300)
					if i%2 != 0 {
						dy = -dy
					}
					tt.Scroll(float32(windowWidth)*0.5, float32(windowHeight)*0.75, 0, dy)
				}
				b.StopTimer()
				tt.Scroll(float32(windowWidth)*0.5, float32(windowHeight)*0.75, 0, 1e6)
				last := fmt.Sprintf("Term %05d", count-1)
				if !tt.HasText(last) {
					b.Fatalf("dictionary list did not scroll to %q: %q", last, tt.Texts())
				}
			})
		})
	}
}
