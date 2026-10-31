package llmrt

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
)

// TestProgressDeadlines: llm.Client's progress deadlines over the real
// process: a late first token and a stall end the chat (cancelled in the
// process, which answers the next chat at once), with a log line that
// says how far it got and how the Mac's memory stood; a slow but steady
// chat passes, its line marked slow. Config.OnPace hears each, with the
// memory: the three would show the overlay's memory notice, a quick
// chat would not.
func TestProgressDeadlines(t *testing.T) {
	readSystemMemory = func() (macx.SystemMemory, error) {
		return macx.SystemMemory{Pressure: "warning", AvailablePercent: 21, SwapUsedBytes: 17 << 30, SwapTotalBytes: 18 << 30}, nil
	}
	t.Cleanup(func() { readSystemMemory = macx.ReadSystemMemory })
	words := strings.Repeat("word ", 20) // 25 pieces
	reply := llmrttest.Opt("reply", words)
	limits := llm.Request{FirstTokenTimeout: 400 * time.Millisecond, StallTimeout: 300 * time.Millisecond, Timeout: 10 * time.Second}
	for _, c := range []struct {
		name, opts string
		wait, log  string // the timeout's wait and a line of the log; wait "" for an answer
		short      bool   // ShortOfMemory, at OnPace
	}{
		{"first token late", "first-ms=5000," + reply, llm.WaitFirstToken, "stopped after 0.4 s without an answer: nothing streamed; the longest wait between two pieces 0.0 s; memory pressure warning, 21% available, swap 17.0 of 18.0 GB used", true},
		{"stall", "stall-at=3,stall-ms=5000," + reply, llm.WaitStall, "3 pieces streamed, the first after", true},
		{"slow but steady", "piece-ms=80," + reply, "", "; slow: the longest wait between two pieces 0.", true},
		{"quick", reply, "", "tokens/s", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			type paced struct {
				pace   ChatPace
				memory macx.SystemMemory
			}
			paces := make(chan paced, 4)
			rt, logs := newRuntime(t, c.opts, func(cfg *Config) {
				cfg.OnPace = func(p ChatPace, m macx.SystemMemory) { paces <- paced{p, m} }
			})
			chat, err := Open(context.Background(), rt, Balanced, mlxDir(t))
			if err != nil {
				t.Fatal(err)
			}
			var streamed strings.Builder
			req := limits
			req.Messages = []llm.Message{{Role: llm.User, Content: "hi"}}
			req.OnDelta = func(d string) { streamed.WriteString(d) }
			t0 := time.Now()
			resp, err := chat.Chat(context.Background(), req)
			took := time.Since(t0)
			var e *llm.Error
			switch {
			case c.wait == "":
				if err != nil || resp.Text != words || c.short && took < 1500*time.Millisecond {
					t.Fatalf("%q, %v after %v", resp.Text, err, took)
				}
			case !errors.As(err, &e) || e.Code != llm.CodeTimeout || e.Params["wait"] != c.wait || resp.Attempts != 1:
				t.Fatalf("%+v, %v", resp, err)
			case took > 2*time.Second:
				t.Errorf("took %v", took)
			}
			waitFor(t, "the line "+c.log, func() bool { return logs.has(c.log) })
			if p := <-paces; p.pace.Model != Balanced || p.pace.Stopped != (c.wait != "") || ShortOfMemory(p.pace, p.memory, false) != c.short {
				t.Errorf("pace %+v, %v: short of memory %v", p.pace, p.memory, !c.short)
			}
			if c.wait == "" {
				return
			}
			// The process stopped the chat: the next one is not behind it.
			t0 = time.Now()
			if _, err := chat.Chat(context.Background(), llm.Request{Messages: req.Messages, MaxTokens: 1}); err != nil || time.Since(t0) > 2*time.Second {
				t.Errorf("next chat: %v after %v", err, time.Since(t0))
			}
		})
	}
}

// TestShortOfMemory: the overlay's memory notice wants a slow (or
// stopped) chat and the swap over 90% full, once.
func TestShortOfMemory(t *testing.T) {
	swap := func(percent int64) macx.SystemMemory {
		return macx.SystemMemory{Pressure: "normal", AvailablePercent: 60, SwapUsedBytes: percent << 30, SwapTotalBytes: 100 << 30}
	}
	slowFirst := ChatPace{FirstToken: 3500 * time.Millisecond, TokensPerSecond: 20, Tokens: 120}
	slowRate := ChatPace{FirstToken: 900 * time.Millisecond, TokensPerSecond: 10.5, Tokens: 200}
	fast := ChatPace{FirstToken: 800 * time.Millisecond, TokensPerSecond: 24, Tokens: 200}
	few := ChatPace{FirstToken: 800 * time.Millisecond, TokensPerSecond: 8, Tokens: 12} // too few tokens for a rate
	stopped := ChatPace{FirstToken: 16 * time.Second, Stopped: true}
	for _, c := range []struct {
		name   string
		pace   ChatPace
		memory macx.SystemMemory
		shown  bool
		want   bool
	}{
		{"slow first token, swap 95%", slowFirst, swap(95), false, true},
		{"slow rate, swap 95%", slowRate, swap(95), false, true},
		{"stopped, swap 95%", stopped, swap(95), false, true},
		{"fast, swap 95%", fast, swap(95), false, false},
		{"few tokens, swap 95%", few, swap(95), false, false},
		{"slow, swap 50%", slowFirst, swap(50), false, false},
		{"slow, swap 90%", slowFirst, swap(90), false, false},
		{"stopped, swap 50%", stopped, swap(50), false, false},
		{"slow, no swap", slowFirst, macx.SystemMemory{}, false, false},
		{"slow, swap 95%, shown before", slowFirst, swap(95), true, false},
		{"stopped, swap 95%, shown before", stopped, swap(95), true, false},
	} {
		if got := ShortOfMemory(c.pace, c.memory, c.shown); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}
