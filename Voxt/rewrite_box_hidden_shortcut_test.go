package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// stubNativeOverlay lets the native card paths run without a display
// server: the headless tester has no mygo main loop to ask for one.
func stubNativeOverlay(t *testing.T, a *app) {
	t.Helper()
	savedFrame, savedPointer := overlayFocusedWindowFrame, overlayPointerDisplay
	overlayFocusedWindowFrame = func() (macx.WindowFrame, bool) { return macx.WindowFrame{}, false }
	overlayPointerDisplay = func() mygo.Display { return mygo.Display{ID: 1, Label: "test", WorkArea: ovTestArea} }
	t.Cleanup(func() { overlayFocusedWindowFrame, overlayPointerDisplay = savedFrame, savedPointer })
	a.ov.native = true
}

// answeredRewriteWindow puts an answered Rewrite card's panel on screen
// and returns its window and the shortcut the card shows.
func answeredRewriteWindow(t *testing.T) (*app, *fakeBoxWindow, string) {
	t.Helper()
	a, _, _ := newTestApp(t)
	stubNativeOverlay(t, a)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Rewrite.Hotkeys = []settings.Hotkey{{Keys: "Control+RightOption", Trigger: settings.TriggerTap}}
		s.Appearance.ReduceMotion = true // fade at once: no timer after the test
	})
	a.answer = answerCard{open: true, id: "v1", conv: "v1", kind: store.KindRewrite, text: "Shorter.", conversation: true,
		turns: []compose.Turn{{Instruction: "make it shorter", Answer: compose.Answer{Content: "Shorter."}}}}
	w := &fakeBoxWindow{}
	a.box = rewriteBox{open: true, shown: true, win: w, winShown: true}
	a.boxSync() // the card is on screen with its hint
	view := a.boxViewState()
	if view.Card == nil || !view.Card.CanFollowUp {
		t.Fatalf("the answered card takes no follow-up to begin with: %+v", view)
	}
	key := a.sessionCardProjectionInput().FollowKeys
	if key == "" || !strings.Contains(view.Card.FollowTooltip, key) || view.SpeakKeys == "" {
		t.Fatalf("the open card shows no Rewrite shortcut to begin with: %q, %q", view.Card.FollowTooltip, view.SpeakKeys)
	}
	return a, w, key
}

// visibleCard is the card the on-screen window draws: the cached
// projection, read without refreshing it.
func visibleCard(a *app) *sessionCardViewState {
	return a.ov.projection.view.Card
}

func visibleCardTooltip(a *app) string {
	card := visibleCard(a)
	if card == nil {
		return ""
	}
	return card.FollowTooltip
}

// TestHiddenRewriteLeavesVoiceHints: turning Show Rewrite off while an
// answered Rewrite card is on screen refreshes that same window and drops
// its shortcut from the follow-up tooltip and the microphone, at once.
// Showing Rewrite again restores both in the same open card.
func TestHiddenRewriteLeavesVoiceHints(t *testing.T) {
	a, w, key := answeredRewriteWindow(t)
	before := w.invalidations

	a.setFeatureShown(string(store.KindRewrite), false)

	if card := visibleCard(a); card == nil || strings.Contains(card.FollowTooltip, key) {
		t.Errorf("the visible card still draws the hidden shortcut: %q", visibleCardTooltip(a))
	}
	if w.invalidations <= before {
		t.Errorf("the visible card was not redrawn after Rewrite was hidden (%d invalidations, was %d)", w.invalidations, before)
	}
	if got := a.sessionCardProjectionInput().FollowKeys; got != "" {
		t.Errorf("a hidden Rewrite feature still advertises %q in the follow-up tooltip", got)
	}
	if view := a.boxViewState(); view.SpeakKeys != "" {
		t.Errorf("the microphone still advertises the hidden shortcut %q", view.SpeakKeys)
	}

	before = w.invalidations
	a.setFeatureShown(string(store.KindRewrite), true)

	if card := visibleCard(a); card == nil || !strings.Contains(card.FollowTooltip, key) {
		t.Errorf("showing Rewrite does not restore the shortcut in the open card: %q", visibleCardTooltip(a))
	}
	if view := a.boxViewState(); view.SpeakKeys == "" {
		t.Error("showing Rewrite does not restore the microphone's shortcut")
	}
	if w.invalidations <= before {
		t.Error("the visible card was not redrawn after Rewrite was shown again")
	}
}

// TestHiddenRewriteWithoutShortcut: a card with no Rewrite shortcut has
// nothing to advertise, and hiding or showing Rewrite never invents one.
func TestHiddenRewriteWithoutShortcut(t *testing.T) {
	a, _, _ := newTestApp(t)
	stubNativeOverlay(t, a)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Rewrite.Hotkeys = nil
		s.Appearance.ReduceMotion = true // fade at once: no timer after the test
	})
	a.answer = answerCard{open: true, id: "v1", conv: "v1", kind: store.KindRewrite, text: "Shorter.", conversation: true,
		turns: []compose.Turn{{Instruction: "make it shorter", Answer: compose.Answer{Content: "Shorter."}}}}
	w := &fakeBoxWindow{}
	a.box = rewriteBox{open: true, shown: true, win: w, winShown: true}
	a.boxSync()

	a.setFeatureShown(string(store.KindRewrite), false)
	a.setFeatureShown(string(store.KindRewrite), true)

	if view := a.boxViewState(); view.SpeakKeys != "" || a.sessionCardProjectionInput().FollowKeys != "" {
		t.Errorf("a card without a configured shortcut invented one: %q, %q", view.SpeakKeys, a.sessionCardProjectionInput().FollowKeys)
	}
	if w.invalidations == 0 {
		t.Error("the visible card was never redrawn")
	}
}

// TestRewriteHideShowToggles: rapid hide and show toggles keep the visible
// card in step with the setting, redrawing it each time.
func TestRewriteHideShowToggles(t *testing.T) {
	a, w, key := answeredRewriteWindow(t)
	before := w.invalidations
	for _, shown := range []bool{false, true, false, true} {
		a.setFeatureShown(string(store.KindRewrite), shown)
		card := visibleCard(a)
		if card == nil {
			t.Fatalf("the visible card went away after setting Rewrite shown=%v", shown)
		}
		if got := strings.Contains(card.FollowTooltip, key); got != shown {
			t.Errorf("after shown=%v the visible card shows the shortcut %v", shown, got)
		}
		if got := a.boxViewState().SpeakKeys != ""; got != shown {
			t.Errorf("after shown=%v the microphone shows the shortcut %v", shown, got)
		}
	}
	if got := w.invalidations - before; got < 4 {
		t.Errorf("four hide/show toggles redrew the visible card %d times", got)
	}
}
