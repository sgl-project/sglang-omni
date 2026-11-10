package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const (
	learnSpoken = "open Go Host now"
	// learnBefore is what the fake helper's field holds before the paste:
	// the paste's check reports the selection at 40.
	learnBefore = "aaaaaaaaaa bbbbbbbbbb cccccccccc ddddddd "
	learnEdited = "open Ghostty now"
)

func learnField(text string) helpertest.Field {
	return helpertest.Field{Value: learnBefore + text + " tail", Element: 1}
}

// learnApp starts a Transcribe session over the fake helper's field: the
// first read is the baseline, the next ones the script; set changes the
// settings on top of the learning switch.
func learnApp(t *testing.T, on bool, set func(*settings.Settings), fields ...helpertest.Field) (*sessionApp, string) {
	t.Helper()
	dir := t.TempDir()
	readFile, readLog := filepath.Join(dir, "read.jsonl"), filepath.Join(dir, "read-log.jsonl")
	var lines []string
	for _, f := range fields {
		line, _ := json.Marshal(f)
		lines = append(lines, string(line))
	}
	if err := os.WriteFile(readFile, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatal(err)
	}
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
	opts := strings.Join([]string{helpertest.Opt("read", readFile), helpertest.Opt("read-log", readLog), ev}, ",")
	a := newSessionApp(t, opts, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.Dictionary.Enabled = true
		s.Dictionary.Learn = on
		if set != nil {
			set(s)
		}
	}, &fakeLocal{ASRP: fakeStreamer{final: learnSpoken}})
	return a, readLog
}

// reads is what the fake helper was asked to read.
func readsOf(t *testing.T, log string) []proto.ReadParams {
	t.Helper()
	data, err := os.ReadFile(log)
	if err != nil {
		return nil
	}
	var out []proto.ReadParams
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var p proto.ReadParams
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			t.Fatalf("read log: %v", err)
		}
		out = append(out, p)
	}
	return out
}

// runWatch is the fake clock of a whole watch: no time passes.
func runWatch(ctx context.Context, _ time.Duration) error { return ctx.Err() }

func sessionDone(a *sessionApp) {
	a.pump("the end of the session", func() bool { return a.sess.State.Ended() })
}

func TestLearnFromCorrection(t *testing.T) {
	a, log := learnApp(t, true, nil, learnField(learnSpoken), learnField(learnEdited))
	a.learner.Sleep = runWatch
	sessionDone(a)
	a.pump("the suggestion", func() bool { return len(a.suggestions) == 1 })
	if a.navBadge("/dictionary") != "1" {
		t.Errorf("sidebar badge %q", a.navBadge("/dictionary"))
	}
	if s := a.suggestions[0]; s.Term != "Ghostty" || !slices.Equal(s.Heard, []string{"Go Host"}) || s.Source != learn.FromEdit {
		t.Errorf("suggestion %+v", s)
	}
	reads := readsOf(t, log)
	if len(reads) != 31 || reads[0].Range.Location != 0 || reads[0].PID != a.pasteTarget() {
		t.Errorf("%d reads, first %+v", len(reads), reads[0])
	}
	tt := ui.NewTester(a.view, windowWidth, 1200)
	a.navigate("/dictionary")
	a.pump("the dictionary page", func() bool { return a.dv.read })
	frames(tt)
	if !hasExact(tt, "1") || !hasExact(tt, "Ghostty") || !slices.Contains(tt.AccessLabels(), "Add Ghostty to the dictionary") {
		t.Errorf("the page does not show it: %q", tt.AccessLabels())
	}
}

// pasteTarget is the pid the session pasted into.
func (s *sessionApp) pasteTarget() int { return s.pasted()[0].TargetPID }

// awaitPumping runs posted functions until ch closes: a.pump only looks
// at its condition after a posted function ran.
func awaitPumping(a *sessionApp, ch <-chan struct{}, what string) {
	a.t.Helper()
	deadline := time.After(10 * time.Second)
	for {
		select {
		case <-ch:
			return
		case fn := <-a.queue:
			fn()
		case <-deadline:
			a.t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func TestLearnOffNeverReads(t *testing.T) {
	for _, tc := range []struct {
		name string
		on   bool
		set  func(*settings.Settings)
	}{
		{"the switch off (default)", false, nil},
		{"the dictionary off", true, func(s *settings.Settings) { s.Dictionary.Enabled = false }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, log := learnApp(t, tc.on, tc.set, learnField(learnSpoken), learnField(learnEdited))
			a.learner.Sleep = runWatch
			sessionDone(a)
			a.pump("the history", func() bool { return len(a.history()) == 1 })
			if reads := readsOf(t, log); len(reads) != 0 {
				t.Fatalf("%d textio.read with learning off: %+v", len(reads), reads)
			}
			if a.navBadge("/dictionary") != "" {
				t.Errorf("sidebar badge %q", a.navBadge("/dictionary"))
			}
		})
	}
}

// countingReader is a Learner.Read that notes the reads.
type countingReader struct{ reads atomic.Int32 }

func (r *countingReader) Read(context.Context, proto.ReadParams) (proto.ReadResult, error) {
	r.reads.Add(1)
	return proto.ReadResult{Error: proto.ReadNotFront}, nil
}

// TestLearnOnlyWatchesTranscribe: a started watch would read; synctest
// makes "no watch started" certain, not a matter of time.
func TestLearnOnlyWatchesTranscribe(t *testing.T) {
	landing := &deliver.Landing{At: 40}
	for _, tc := range []struct {
		name string
		s    session.Snapshot
	}{
		{"translate", session.Snapshot{Kind: store.KindTranslate, Landing: landing, Result: "x"}},
		{"a try", session.Snapshot{Kind: store.KindTranscribe, Try: true, Landing: landing, Result: "x"}},
		{"not inserted", session.Snapshot{Kind: store.KindTranscribe, Result: "x"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _ := newTestAppIn(t, testDir(t))
			reader := &countingReader{}
			a.learner.Read, a.learner.Sleep = reader, runWatch
			a.prefs.Dictionary.Enabled, a.prefs.Dictionary.Learn = true, true
			a.hstatus = helper.Status{State: helper.Ready, Hello: &proto.HelloResult{Methods: []string{proto.MethodTextRead}}}
			synctest.Test(t, func(t *testing.T) {
				a.learnFromSession(tc.s)
				synctest.Wait()
				if n := reader.reads.Load(); n != 0 {
					t.Fatalf("%d reads", n)
				}
			})
		})
	}
}

func TestLearnHelperWithoutRead(t *testing.T) {
	a, log := learnApp(t, true, nil, learnField(learnSpoken))
	a.learner.Sleep = runWatch
	hello := *a.hstatus.Hello
	hello.Methods = slices.DeleteFunc(slices.Clone(hello.Methods), func(m string) bool { return m == proto.MethodTextRead })
	a.hstatus.Hello = &hello
	sessionDone(a)
	a.pump("the history", func() bool { return len(a.history()) == 1 })
	if reads := readsOf(t, log); len(reads) != 0 {
		t.Fatalf("%d reads on a helper without the method", len(reads))
	}
}

// watchHeld is a Learner.Sleep that passes n waits (the settling and the
// first polls) and then waits for the watch's end, saying when it comes.
func watchHeld(n int) (sleep func(context.Context, time.Duration) error, reached, ended chan struct{}) {
	reached, ended = make(chan struct{}), make(chan struct{})
	waits := 0
	sleep = func(ctx context.Context, _ time.Duration) error {
		waits++
		if waits < n {
			return ctx.Err()
		}
		close(reached)
		<-ctx.Done()
		close(ended)
		return ctx.Err()
	}
	return sleep, reached, ended
}

func TestLearnTurnedOffMidWatch(t *testing.T) {
	for _, tc := range []struct {
		name string
		off  func(*settings.Settings)
	}{
		{"the learning switch", func(s *settings.Settings) { s.Dictionary.Learn = false }},
		{"the dictionary switch", func(s *settings.Settings) { s.Dictionary.Enabled = false }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, log := learnApp(t, true, nil, learnField(learnSpoken), learnField(learnEdited))
			sleep, reached, ended := watchHeld(3) // settling, poll 1, then held in poll 2's wait
			a.learner.Sleep = sleep
			sessionDone(a)
			awaitPumping(a, reached, "the watch at its second poll")
			before := len(readsOf(t, log))
			if before != 2 {
				t.Fatalf("%d reads before the switch, want the baseline and one poll", before)
			}
			a.updateSettings(tc.off)
			awaitPumping(a, ended, "the watch's end")
			if got := len(readsOf(t, log)); got != before {
				t.Fatalf("%d reads after the switch, want %d", got, before)
			}
			got, err := a.book.Pending(context.Background())
			if err != nil || len(got) != 0 || a.navBadge("/dictionary") != "" {
				t.Fatalf("proposed %v (%v), badge %q", got, err, a.navBadge("/dictionary"))
			}
		})
	}
}

func TestLearnNeverLogsTheEdit(t *testing.T) {
	a, _ := learnApp(t, true, func(s *settings.Settings) { s.Privacy.LogText = true }, learnField(learnSpoken), learnField(learnEdited))
	a.learner.Sleep = runWatch
	sessionDone(a)
	a.pump("the suggestion", func() bool { return len(a.suggestions) == 1 })
	logs := a.logText()
	if !strings.Contains(logs, "watch") {
		t.Fatalf("no line of the watch in the log:\n%s", logs)
	}
	for _, text := range []string{learnEdited, "Ghostty", "tail", "bbbbbbbbbb"} {
		if strings.Contains(logs, text) {
			t.Errorf("the log has %q with LogText on:\n%s", text, logs)
		}
	}
}

// TestLearnSwitch: the page's switch is off at first, saves the setting,
// and its tip says how long the field is read, from the watch's constant.
func TestLearnSwitch(t *testing.T) {
	a, _, tt := newTestAppIn(t, testDir(t))
	a.navigate("/dictionary")
	frames(tt)
	if a.prefs.Dictionary.Learn {
		t.Fatal("learning is on by default")
	}
	const name = "Learn from my corrections"
	r, ok := tt.Find(name)
	if !ok {
		t.Fatalf("no switch: %q", tt.AccessLabels())
	}
	tt.Move(r.X+r.W/2, r.Y+r.H/2)
	frames(tt)
	tt.Advance(time.Second)
	frames(tt)
	if want := i18n.T("For %d seconds after inserting, Voxt reads the text around it. If you correct a word, it suggests your spelling. Nothing it reads is saved or sent.", learn.WatchSeconds); !hasExact(tt, want) || !strings.Contains(want, fmt.Sprintf("For %d seconds", learn.WatchSeconds)) {
		t.Fatalf("no tip %q in %q", want, firstTexts(tt))
	}
	clickControl(t, tt, name)
	if !a.prefs.Dictionary.Learn || !a.settings.Get().Dictionary.Learn {
		t.Fatalf("the switch saved nothing: %+v", a.settings.Get().Dictionary)
	}
}

func TestLearnQuitStopsTheWatch(t *testing.T) {
	a, log := learnApp(t, true, nil, learnField(learnSpoken), learnField(learnEdited))
	sleep, reached, ended := watchHeld(3)
	a.learner.Sleep = sleep
	sessionDone(a)
	awaitPumping(a, reached, "the watch at its second poll")
	a.quit() // the first of two: the test's clean-up quits too
	awaitPumping(a, ended, "the watch's end")
	if got := len(readsOf(t, log)); got != 2 {
		t.Fatalf("%d reads after quit, want 2", got)
	}
	if a.navBadge("/dictionary") != "" {
		t.Fatalf("badge %q", a.navBadge("/dictionary"))
	}
}

func TestLearnFinishOnNextSession(t *testing.T) {
	a, _ := learnApp(t, true, nil, learnField(learnSpoken), learnField(learnEdited))
	sleep, reached, _ := watchHeld(3)
	a.learner.Sleep = sleep
	sessionDone(a)
	awaitPumping(a, reached, "the watch at its second poll")
	next := a.sess
	next.Rev, next.ID, next.State = next.Rev+1, "next", session.Recording
	a.sessionChanged(next)
	a.pump("the judged edit", func() bool { return len(a.suggestions) == 1 })
	if a.suggestions[0].Term != "Ghostty" {
		t.Fatalf("suggestion %+v", a.suggestions[0])
	}
}
