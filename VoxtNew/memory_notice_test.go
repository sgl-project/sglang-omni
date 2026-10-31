package main

import (
	"bytes"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/ink"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

var (
	paceSlow    = llmrt.ChatPace{Model: llmrt.Balanced, Skill: llm.SkillTranscribeEnhance, FirstToken: 4 * time.Second, TokensPerSecond: 9.8, Tokens: 140}
	paceFast    = llmrt.ChatPace{Model: llmrt.Balanced, Skill: llm.SkillTranscribeEnhance, FirstToken: 700 * time.Millisecond, TokensPerSecond: 24, Tokens: 140}
	paceStopped = llmrt.ChatPace{Model: llmrt.Balanced, Skill: llm.SkillTranscribeEnhance, FirstToken: 2 * time.Second, Tokens: 90, Stopped: true}
)

// swapUsed is the Mac's memory with the swap files percent full.
func swapUsed(percent int64) macx.SystemMemory {
	return macx.SystemMemory{Pressure: "normal", AvailablePercent: 55, SwapUsedBytes: 18 << 30 * percent / 100, SwapTotalBytes: 18 << 30}
}

// memoryLines are the log's lines of the memory notice.
func memoryLines(a *app) []string {
	var out []string
	for _, e := range a.logs.Tail(1000, sessionSource) {
		if strings.Contains(e.Message, "short of memory") {
			out = append(out, e.Message)
		}
	}
	return out
}

// TestMemoryNotice: the overlay says the Mac is short of memory after a
// session whose chat was slow (or stopped) with the swap over 90% full,
// once a run: after the result went in, after the session's own notice;
// not for a quick chat, a swap half full, a try, a cancel or under the
// answer card; the log says when it shows.
func TestMemoryNotice(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	paced := a.onPace() // llmrt.Config.OnPace; a.post runs at once in tests
	text := i18n.T("Low memory slows the model: quit unused apps.")
	session1 := func(id string, try bool, pace llmrt.ChatPace, memory macx.SystemMemory, end session.Snapshot) {
		t.Helper()
		snap(a, session.Snapshot{ID: id, State: session.Recording, Try: try})
		snap(a, session.Snapshot{ID: id, State: session.Composing, Try: try})
		paced(pace, memory)
		end.ID, end.Try = id, try
		snap(a, end)
	}
	inserted := session.Snapshot{State: session.Done, Result: "Hello.", Inserted: true}
	none := func(what string) {
		t.Helper()
		if a.notice != nil || a.shortMemory.shown || len(memoryLines(a)) != 0 {
			t.Fatalf("%s: %+v, shown %v, %q", what, a.notice, a.shortMemory.shown, memoryLines(a))
		}
		ft.advance(time.Minute)
	}
	session1("quick", false, paceFast, swapUsed(95), inserted)
	none("a quick chat, swap 95%")
	session1("half", false, paceSlow, swapUsed(50), inserted)
	none("a slow chat, swap 50%")
	session1("try", true, paceSlow, swapUsed(95), inserted)
	none("a try")
	session1("cancel", false, paceSlow, swapUsed(95), session.Snapshot{State: session.Cancelled})
	none("a cancel")
	session1("card", false, paceSlow, swapUsed(95), session.Snapshot{State: session.Done, Result: "Hello.", Card: true, Delivery: deliver.ReasonNotEditable})
	if a.ov.mode != ovAnswer {
		t.Fatalf("the card: %s", a.ov.mode)
	}
	none("the answer card")
	a.closeAnswer()

	// A deadline stopped the chat: the session's own notice first; the
	// next session, starting meanwhile, drops the memory notice.
	timedOut := session.Snapshot{State: session.Done, Result: "hello", Inserted: true,
		Problem: &session.Problem{Code: session.ProblemLLM, Degraded: true, Err: &llm.Error{Code: llm.CodeTimeout}}}
	session1("stopped", false, paceStopped, swapUsed(95), timedOut)
	ownText := "Inserted as recognised: the language model did not answer in time."
	if a.notice == nil || a.notice.text != ownText {
		t.Fatalf("the session's notice: %+v", a.notice)
	}
	snap(a, session.Snapshot{ID: "next", State: session.Recording})
	ft.advance(noticeTime)
	snap(a, session.Snapshot{ID: "next", State: session.Cancelled})
	none("the next session started")

	// Again: its own notice for its time, then the memory notice in its
	// place, then nothing.
	session1("stopped-2", false, paceStopped, swapUsed(95), timedOut)
	ft.advance(noticeTime - time.Millisecond)
	if a.notice == nil || a.notice.text != ownText {
		t.Fatalf("the session's notice went early: %+v", a.notice)
	}
	ft.advance(time.Millisecond)
	if a.notice == nil || a.notice.text != text || !a.notice.warn || a.notice.action != actionNone || a.ov.mode != ovNotice || !a.ov.visible {
		t.Fatalf("the memory notice: %s %+v", a.ov.mode, a.notice)
	}
	// Through the view: its one line.
	renderOverlay(t, a)
	tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	frames(tt)
	if _, ok := tt.Find(text); !ok {
		t.Errorf("the overlay does not show %q: %q", text, firstTexts(tt))
	}
	lines := memoryLines(a)
	if len(lines) != 1 || !strings.Contains(lines[0], "llm.balanced transcribe_enhance stopped by a deadline, 90 pieces streamed; memory pressure normal, 55% available, swap 17.1 of 18.0 GB used") {
		t.Errorf("the log: %q", lines)
	}
	ft.advance(noticeTime)
	if a.notice != nil || a.ov.shown {
		t.Fatalf("the memory notice stays: %+v", a.notice)
	}

	// Once a run.
	session1("again", false, paceSlow, swapUsed(99), inserted)
	if a.notice != nil || len(memoryLines(a)) != 1 {
		t.Errorf("shown twice: %+v, %q", a.notice, memoryLines(a))
	}
}

// TestMemoryNoticeSlow: a slow chat (no deadline) shows it as soon as the
// text went in; VOXTNEW_DEV_SHORT_OF_MEMORY makes any chat call for it.
func TestMemoryNoticeSlow(t *testing.T) {
	for _, dev := range []bool{false, true} {
		if dev {
			t.Setenv(devShortOfMemoryEnv, "1")
		}
		a, _, _ := newTestApp(t)
		new(fakeTimers).install(a)
		pace, memory := paceSlow, swapUsed(95)
		if dev {
			pace, memory = paceFast, macx.SystemMemory{}
		}
		snap(a, session.Snapshot{ID: "s1", Kind: store.KindTranslate, State: session.Composing})
		a.onPace()(pace, memory)
		snap(a, session.Snapshot{ID: "s1", Kind: store.KindTranslate, State: session.Done, Result: "Bonjour.", Inserted: true})
		if a.notice == nil || a.notice.text != memoryNotice().text || a.ov.mode != ovNotice {
			t.Fatalf("dev %v: %s %+v", dev, a.ov.mode, a.notice)
		}
		lines := memoryLines(a)
		want := "llm.balanced transcribe_enhance first token 4.0 s, 9.8 tokens/s; memory pressure normal, 55% available, swap 17.1 of 18.0 GB used"
		if dev {
			want = "llm.balanced (VOXTNEW_DEV_SHORT_OF_MEMORY) transcribe_enhance first token 4.0 s, 24.0 tokens/s;"
		}
		if len(lines) != 1 || !strings.Contains(lines[0], want) {
			t.Errorf("dev %v: the log %q", dev, lines)
		}
	}
}

// TestOverlayInkMemoryNotice: the memory notice as it draws, English and
// Chinese: the notice's height, its line whole (measured, it fits the
// notice's widest, and keeps to one line at its own width).
func TestOverlayInkMemoryNotice(t *testing.T) {
	for _, lang := range []string{"en", "zh-Hans"} {
		t.Run(lang, func(t *testing.T) {
			if lang != "en" {
				requireCJKFont(t)
			}
			a, _, _ := newTestApp(t)
			if lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
			}
			snap(a, session.Snapshot{ID: "ink", State: session.Recording})
			snap(a, session.Snapshot{ID: "ink", State: session.Done, Result: "x", Inserted: true})
			a.showNotice(memoryNotice())
			img := renderOverlay(t, a)
			text := a.notice.text
			tw, exact := a.ovTextWidth(noticeText(text))
			need := 2*layout.PillPadX + layout.IconSlot + layout.Gap + int(math.Ceil(float64(tw)))
			if !exact || need > layout.NoticeW2 || a.ov.w != layout.NoticeWidth(tw, false) {
				t.Errorf("%q is %v wide (measured %v): the notice needs %d, is %d, at most %d", text, tw, exact, need, a.ov.w, layout.NoticeW2)
			}
			for s, where := range misfits(t, []string{text}, lang) {
				t.Errorf("%q breaks at its own width (%s)", s, where)
			}
			anchored(t, lang, a.prefs.Overlay, a.ov.frame)
			k, bad := ink.Check(img, 2, ink.Want{W: a.ov.w, H: layout.PillH})
			for _, b := range bad {
				t.Error(b)
			}
			t.Logf("%s: %v", lang, k)
			if dir := os.Getenv("VOXTNEW_INK"); dir != "" {
				var b bytes.Buffer
				png.Encode(&b, img)
				name := "notice-memory"
				if lang != "en" {
					name += "-zh"
				}
				os.WriteFile(filepath.Join(dir, name+".png"), b.Bytes(), 0o644)
			}
		})
	}
}
