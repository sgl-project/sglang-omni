package main

import (
	"errors"
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestCardFirstOverlay covers the whole live card lifecycle, including
// hidden live text, without changing Rewrite's separate input flow.
func TestCardFirstOverlay(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranslate, store.KindTranscribe} {
		t.Run(string(kind), func(t *testing.T) {
			a, _, _ := newTestApp(t)
			for _, state := range []session.State{session.Recording, session.Recognizing, session.Composing} {
				snap(a, session.Snapshot{ID: "card-first", Kind: kind, State: state, Card: true,
					Selection: "We plan to ship on Friday.", Partial: "this Friday", Levels: wave(), HeardAt: time.Now()})
				if a.ov.mode != ovAnswer || a.ov.passThrough || !a.ov.shown || a.pill.shown {
					t.Fatalf("%s: mode %s, pass-through %v, card shown %v, pill shown %v", state, a.ov.mode, a.ov.passThrough, a.ov.shown, a.pill.shown)
				}
				if state == session.Recording {
					a.prefs.Features.SessionsOn = true
					a.overlaySync()
					if len(a.cachedCard().Sessions) != 1 {
						t.Fatal("the live card lost its session row")
					}
					tt := cardTester(t, a)
					frames(tt)
					if !has(tt, a.sess.Selection) || !has(tt, "Input level") || !has(tt, "Done") {
						t.Fatalf("recording card: %v", firstTexts(tt))
					}
				}
				a.prefs.Overlay.HideLiveText = true
				if a.overlayModeNow() != ovAnswer || a.cardNow().text != "" {
					t.Fatal("hiding live text must keep the card and hide partial text")
				}
				a.prefs.Overlay.HideLiveText = false
			}
			for _, state := range []session.State{session.Done, session.Cancelled} {
				snap(a, session.Snapshot{ID: "card-first", Kind: kind, State: state})
				if a.ov.mode != ovHidden {
					t.Fatalf("%s: the live card stayed open", state)
				}
			}
		})
	}
}

// TestCardFirstOverlayWithBackgroundAnswerInAnotherSession keeps a background
// answer in its own slot from hiding the current session's speech card.
func TestCardFirstOverlayWithBackgroundAnswerInAnotherSession(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.SessionsOn = true
	a.bgID = "rewrite-in-session-1"
	a.bgSlot = a.slot
	a.bgSess = session.Snapshot{ID: a.bgID, Kind: store.KindRewrite, State: session.Composing, Card: true}
	a.newSession()
	if a.slot != 1 || a.bgVisible() {
		t.Fatalf("background answer setup: slot %d, visible %v", a.slot, a.bgVisible())
	}
	snap(a, session.Snapshot{ID: "card-first", Kind: store.KindTranscribe, State: session.Recording, Card: true, Levels: wave(), HeardAt: time.Now()})
	if !a.liveCard() || a.ov.mode != ovAnswer || a.pill.shown || a.cachedCard().Empty || !a.cachedCard().Speech {
		t.Fatalf("live %v, mode %s, pill %v, card %+v", a.liveCard(), a.ov.mode, a.pill.shown, a.cachedCard())
	}
}

// TestCardFirstFrontOwnership keeps the live card above the target app
// without activating Voxt, then restores the answer card's front behavior.
func TestCardFirstFrontOwnership(t *testing.T) {
	t.Run("live card floats until the answer opens", func(t *testing.T) {
		a, _, _ := newTestApp(t)
		previous := frontmostPID
		t.Cleanup(func() { frontmostPID = previous })
		frontmostPID = func() int { return slackApp.PID }
		win := &frontRecorder{onTop: true}
		a.sess = session.Snapshot{ID: "card-first", Kind: store.KindTranscribe, State: session.Recording, Card: true, Target: slackApp}
		a.syncOverlayFront(win, ovAnswer, true)
		if !win.onTop || win.focused != 0 {
			t.Fatalf("recording: on top %v, focused %d", win.onTop, win.focused)
		}
		a.answer = answerCard{open: true, id: "answer", target: slackApp}
		a.syncOverlayFront(win, ovAnswer, false)
		if win.onTop || win.focused != 1 {
			t.Fatalf("real answer: on top %v, focused %d", win.onTop, win.focused)
		}
	})

	t.Run("close returns to the session target", func(t *testing.T) {
		a, _, _ := newTestApp(t)
		previousFront, previousActivate := frontmostPID, activateTarget
		t.Cleanup(func() { frontmostPID, activateTarget = previousFront, previousActivate })
		activated := make(chan int, 1)
		activateTarget = func(pid int) (bool, time.Duration) {
			activated <- pid
			return true, 0
		}
		a.sess = session.Snapshot{ID: "card-first", Kind: store.KindTranscribe, State: session.Recording, Card: true, Target: slackApp}
		a.ov.native = true
		frontmostPID = func() int { return os.Getpid() }
		a.giveAnswerFrontBack(a.liveCard())
		select {
		case pid := <-activated:
			if pid != slackApp.PID {
				t.Fatalf("activated %d, want %d", pid, slackApp.PID)
			}
		case <-time.After(time.Second):
			t.Fatal("Close did not return the front to the session target")
		}
	})
}

// TestCardFirstFailureOffersRetry: a card-first Transcribe whose speech
// model fails ends with Card off (session.run.endFailed): no card opens,
// and the overlay's notice retries the kept recording.
func TestCardFirstFailureOffersRetry(t *testing.T) {
	a, _, _ := newTestApp(t)
	snap(a, session.Snapshot{ID: "card-first", Kind: store.KindTranscribe, State: session.Recording, Card: true, Levels: wave(), HeardAt: time.Now()})
	snap(a, session.Snapshot{ID: "card-first", Kind: store.KindTranscribe, State: session.Failed, Recording: true,
		Problem: &session.Problem{Code: session.ProblemASR, Err: errors.New("500")}})
	if a.answer.open {
		t.Fatal("a card opened for a failure with nothing to show")
	}
	if a.notice == nil || a.notice.action != actionRetry || a.notice.entry != "card-first" {
		t.Fatalf("notice %+v", a.notice)
	}
}
