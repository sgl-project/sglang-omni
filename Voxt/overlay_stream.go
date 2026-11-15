package main

// The pace of an answer streaming into the card (#3836): a fast model
// sends many words a second, and the card jumped on every chunk. The card
// shows the draft at a steady pace instead, a little behind the model,
// and grows in steps of a few lines (cardBodyHeight). The whole answer
// shows the moment the model ends: the card's text is then the result.

import (
	"math"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

const (
	// revealFrame is how often the shown text moves on.
	revealFrame = 33 * time.Millisecond
	// The pace, in runes a second: what is left to show over
	// revealCatchUp, between revealMin and revealMax.
	revealMin, revealMax = 30.0, 160.0
	revealCatchUp        = time.Second
)

// streamReveal is how much of a session's draft the card shows.
type streamReveal struct {
	id      string
	shown   int     // runes
	ticking bool    // a step is on its way
	h       float32 // the card's last measured body height (cardBodyHeight)
}

// revealRunes is how many runes of a total-rune draft show dt after shown
// did: at least one more while some are left, all of a draft that got
// shorter.
func revealRunes(shown, total int, dt time.Duration) int {
	left := total - shown
	if left <= 0 {
		return total
	}
	rate := min(max(float64(left)/revealCatchUp.Seconds(), revealMin), revealMax)
	return min(total, shown+max(1, int(math.Round(rate*dt.Seconds()))))
}

// shownDraft is the part of s's draft the card shows now; all of it
// where no timer can move it on (tests without timers).
func (a *app) shownDraft(s session.Snapshot) string {
	r := a.ov.stream
	if r.id != s.ID || !a.timersRun() {
		return s.Draft
	}
	i := 0
	for n := 0; n < r.shown && i < len(s.Draft); n++ {
		_, size := utf8.DecodeRuneInString(s.Draft[i:])
		i += size
	}
	return s.Draft[:i]
}

// paceDraft moves the shown draft on, a step every revealFrame, until it
// caught up with the session's. overlaySync calls it.
func (a *app) paceDraft() {
	r := &a.ov.stream
	if r.id != a.sess.ID {
		*r = streamReveal{id: a.sess.ID}
	}
	if r.ticking || r.shown >= utf8.RuneCountInString(a.sess.Draft) || !a.sess.State.Live() || !a.timersRun() || a.overlayModeNow() != ovAnswer {
		return
	}
	r.ticking = true
	id := r.id
	a.later(revealFrame, func() {
		if r.id != id {
			return // another session's: its own steps run
		}
		r.ticking = false
		r.shown = revealRunes(r.shown, utf8.RuneCountInString(a.sess.Draft), revealFrame)
		a.overlaySync()
	})
}

// timersRun says whether later's timers run: with a window, or in tests
// that run them.
func (a *app) timersRun() bool { return a.laterHook != nil || a.overlayNative() }
