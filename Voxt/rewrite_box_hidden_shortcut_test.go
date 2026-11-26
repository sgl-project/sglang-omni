package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestHiddenRewriteLeavesVoiceHints: turning Show Rewrite off while an
// answered Rewrite card is on screen drops its shortcut from the card's
// follow-up tooltip and from the microphone's, at once.
func TestHiddenRewriteLeavesVoiceHints(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Rewrite.Hotkeys = []settings.Hotkey{{Keys: "Control+RightOption", Trigger: settings.TriggerTap}}
	})
	a.box = rewriteBox{open: true, shown: true}
	a.answer = answerCard{open: true, id: "v1", conv: "v1", kind: store.KindRewrite, text: "Shorter.", conversation: true,
		turns: []compose.Turn{{Instruction: "make it shorter", Answer: compose.Answer{Content: "Shorter."}}}}

	view := a.boxViewState()
	if view.Card == nil || !view.Card.CanFollowUp {
		t.Fatalf("the answered card takes no follow-up to begin with: %+v", view)
	}
	cardKey, speakKey := a.sessionCardProjectionInput().FollowKeys, view.SpeakKeys
	if cardKey == "" || !strings.Contains(view.Card.FollowTooltip, cardKey) {
		t.Fatalf("the follow-up tooltip does not show the Rewrite shortcut to begin with: %q", view.Card.FollowTooltip)
	}
	if speakKey == "" {
		t.Fatal("the microphone does not show the Rewrite shortcut to begin with")
	}

	a.setFeatureShown(string(store.KindRewrite), false)

	view = a.boxViewState()
	if view.Card == nil {
		t.Fatal("hiding Rewrite took away the visible answer")
	}
	if got := a.sessionCardProjectionInput().FollowKeys; got != "" {
		t.Errorf("a hidden Rewrite feature still advertises %q in the follow-up tooltip", got)
	}
	if strings.Contains(view.Card.FollowTooltip, cardKey) {
		t.Errorf("the follow-up tooltip still advertises the hidden shortcut: %q", view.Card.FollowTooltip)
	}
	if view.SpeakKeys != "" {
		t.Errorf("the microphone still advertises the hidden shortcut %q", view.SpeakKeys)
	}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if has(tt, speakKey) {
		t.Errorf("the rendered card still shows the hidden shortcut %q: %q", speakKey, firstTexts(tt))
	}
}
