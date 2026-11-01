package main

// The output's mute in whole sessions on Linux: the app over the real
// supervisor and the fake helper process, whose pretend output devices
// live in a file (out=), so that what each helper process did can be read
// from the test, and they outlive a helper's crash.

import (
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestSessionMutesOutput: each feature records with the output muted,
// and it comes back as the recording stops, while the local models work on
// it: by the language model's request, or within releaseBound of it (the
// release does not wait for the session: it goes from the stopped
// snapshot through the main thread, the mute's queue and the helper,
// while the session goes on to recognition; on in-process fake models under
// load the request may come first). A cancel gives it back too, and the
// follow-up of a Rewrite answer mutes again.
func TestSessionMutesOutput(t *testing.T) {
	// releaseBound is how long after the request the output may still be
	// muted: the release's few calls, on a loaded machine. A product that
	// held the mute through the model's work would hold it until the
	// answer, which this request withholds meanwhile.
	const releaseBound = 2 * time.Second
	for _, tc := range []struct {
		name, opts string
		setup      func(t *testing.T) (func(*settings.Settings), *fakeLocal)
		events     []string
		end        session.State
	}{
		{name: "transcribe, whole recording", events: []string{hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(500, "transcribe/0", proto.PhaseTrigger)}, end: session.Done},
		{name: "transcribe, stream", setup: func(t *testing.T) (func(*settings.Settings), *fakeLocal) {
			return func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
			}, &fakeLocal{ASRP: fakeStreamer{final: "你好，VoxtNew"}}
		}, events: []string{hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(600, "transcribe/0", proto.PhaseTrigger)}, end: session.Done},
		{name: "translate by voice", events: []string{hotkeyLine(0, "translate/0", proto.PhaseTrigger), hotkeyLine(500, "translate/0", proto.PhaseTrigger)}, end: session.Done},
		{name: "rewrite", opts: "readonly", events: []string{hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger)}, end: session.Done},
		{name: "cancelled", events: []string{hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, proto.HotkeyEscape, proto.PhaseTrigger)}, end: session.Cancelled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := filepath.Join(t.TempDir(), "output.json")
			muted := func() bool { return helpertest.LoadOutputSystem(state).Muted(testSpeakers) }
			var mu sync.Mutex
			var backAfter []time.Duration // after each request's start; -1: not within releaseBound
			chat := &fakeChat{text: "Hello there.", onRequest: func() {
				t0 := time.Now()
				d := time.Duration(-1)
				for time.Since(t0) < releaseBound {
					if !muted() {
						d = time.Since(t0)
						break
					}
					time.Sleep(5 * time.Millisecond)
				}
				mu.Lock()
				backAfter = append(backAfter, d)
				mu.Unlock()
			}}
			set, svc, _ := fileSetup(t, "hello there", chat)
			if tc.setup != nil {
				set, svc = tc.setup(t)
			}
			opts := helpertest.Opt("out", state) + "," + script(t, tc.events...)
			if tc.opts != "" {
				opts = tc.opts + "," + opts
			}
			a := newSessionApp(t, opts, set, svc)
			seen := false
			a.pump("the output muted while recording", func() bool {
				seen = seen || a.sess.State == session.Recording && muted()
				return seen || a.sess.State.Ended()
			})
			if !seen {
				t.Fatalf("not muted while recording (%s)", a.sess.State)
			}
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != tc.end {
				t.Fatalf("end %s %+v", a.sess.State, a.sess.Problem)
			}
			a.mute.Wait()
			if muted() {
				t.Fatal("still muted after the session")
			}
			mu.Lock()
			back := slices.Clone(backAfter)
			mu.Unlock()
			if slices.Contains(back, -1) || (tc.end == session.Done && tc.name != "transcribe, stream" && len(back) == 0) {
				t.Errorf("still muted %v after the language model's request (or no request): %v", releaseBound, back)
			}
			t.Logf("output back %v after the language model's request", back)
			if w := helpertest.LoadOutputSystem(state).WriteLog(); !slices.Equal(w, []string{testSpeakers + ":true", testSpeakers + ":false"}) {
				t.Errorf("writes %v", w)
			}
			if tc.name != "rewrite" {
				return
			}
			// The answer card's follow-up records with the output muted too.
			front := helpertest.Apps["com.tinyspeck.slackmacgap"]
			a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
			a.pump("the follow-up muted", func() bool { return a.sess.State == session.Recording && muted() })
			time.Sleep(300 * time.Millisecond)
			a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
			a.pump("the follow-up's end", func() bool { return a.sess.State.Ended() })
			a.mute.Wait()
			if muted() {
				t.Fatal("still muted after the follow-up")
			}
		})
	}
}

// TestSessionMuteHelperCrash: the helper dies right after muting; the
// session fails, and the next helper unmutes what the first one left.
func TestSessionMuteHelperCrash(t *testing.T) {
	state := filepath.Join(t.TempDir(), "output.json")
	muted := func() bool { return helpertest.LoadOutputSystem(state).Muted(testSpeakers) }
	set, svc, _ := fileSetup(t, "hello", &fakeChat{text: "x"})
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, "crash-mute,"+helpertest.Opt("out", state)+","+ev, set, svc)
	a.pump("the session's end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Failed {
		t.Fatalf("end %s", a.sess.State)
	}
	deadline := time.Now().Add(10 * time.Second)
	for muted() && time.Now().Before(deadline) {
		select {
		case fn := <-a.queue:
			fn()
		case <-time.After(20 * time.Millisecond):
		}
	}
	if muted() {
		t.Fatal("the next helper did not unmute")
	}
}
