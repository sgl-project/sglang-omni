package main

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRevealRunes: the shown text moves on at a steady pace, faster the
// more is left but never past revealMax, and at once to a shorter draft.
func TestRevealRunes(t *testing.T) {
	for _, c := range []struct {
		shown, total int
		dt           time.Duration
		want         int
	}{
		{0, 0, time.Second, 0},
		{0, 5, 33 * time.Millisecond, 1},       // a few left: revealMin, at least one
		{0, 10, time.Second, 10},               // never past the draft
		{0, 100, time.Second, 100},             // what is left, over revealCatchUp
		{0, 4000, time.Second, int(revealMax)}, // a burst: no faster than revealMax
		{100, 4000, 100 * time.Millisecond, 116},
		{50, 20, 33 * time.Millisecond, 20}, // the draft got shorter
	} {
		if got := revealRunes(c.shown, c.total, c.dt); got != c.want {
			t.Errorf("revealRunes(%d, %d, %v) = %d, want %d", c.shown, c.total, c.dt, got, c.want)
		}
	}
}

// TestStreamedAnswerCalm (#3836): a fast model streams a long Rewrite
// answer, 8 runes every 20 ms as the fake model does. The card shows it
// at a steady pace, the text a part of the draft, and its height changes
// at most twice in any second; when the model ends, the whole answer
// shows at once.
func TestStreamedAnswerCalm(t *testing.T) {
	a, _, _ := newTestApp(t)
	clock := (&fakeTimers{}).install(a)
	long := strings.Repeat("The meeting moves to Thursday at three, and the room is the big one upstairs. ", 20)
	s := session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Partial: "make it longer", Selection: "x"}
	var changes []time.Duration
	lastH := -1
	for n := 8; n < len(long); n += 8 {
		s.Draft = long[:n]
		snap(a, s)
		clock.advance(20 * time.Millisecond)
		v := a.cardNow()
		if !v.writing || !strings.HasPrefix(s.Draft, v.text) {
			t.Fatalf("at %d runes the card shows %q (writing %v)", n, v.text, v.writing)
		}
		if _, h, _ := a.cardSize(); h != lastH {
			changes, lastH = append(changes, clock.now), h
		}
	}
	if v := a.cardNow(); len(v.text) >= len(long)-8 {
		t.Errorf("the card kept up with the model (%d of %d): no pace", len(v.text), len(long))
	}
	for i := range changes {
		if i >= 2 && changes[i]-changes[i-2] < time.Second {
			t.Fatalf("the card changed height 3 times within a second: %v", changes)
		}
	}
	if len(changes) < 3 {
		t.Errorf("the card grew %d times over %v: it should grow with the answer", len(changes), clock.now)
	}
	s.State, s.Result, s.Delivery = session.Done, long, deliver.ReasonCard
	snap(a, s)
	if v := a.cardNow(); v.text != long {
		t.Errorf("the model ended: the card shows %d of %d runes", len(v.text), len(long))
	}
}
