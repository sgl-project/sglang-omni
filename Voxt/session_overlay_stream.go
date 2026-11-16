package main

import (
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// streamReveal is how much of a session's draft the card shows.
type streamReveal struct {
	id      string
	shown   int     // runes
	ticking bool    // a step is on its way
	h       float32 // the card's last measured body height (cardBodyHeight)
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
