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
	return runePrefix(s.Draft, r.shown)
}

func runePrefix(text string, runes int) string {
	i := 0
	for n := 0; n < runes && i < len(text); n++ {
		_, size := utf8.DecodeRuneInString(text[i:])
		i += size
	}
	return text[:i]
}

// revealedAnswer is the part of the card's ended answer the card shows
// while the reveal catches up with it, as Claude's does: a fast model's
// answer streams on after it ended instead of its rest showing at once.
// ok is false once all of it shows.
func (a *app) revealedAnswer() (text string, ok bool) {
	r, s := a.ov.stream, a.cardSnapshot()
	if !a.answer.open || r.id != a.answer.id || r.id != s.ID || s.State != session.Done || !a.timersRun() || r.shown >= utf8.RuneCountInString(a.answer.text) {
		return "", false
	}
	return runePrefix(a.answer.text, r.shown), true
}

// paceDraft moves the shown draft on, a step every revealFrame, until it
// caught up with the card's session's (the backgrounded answer's too),
// and after the model ended with its answer. overlaySync calls it.
func (a *app) paceDraft() {
	r := &a.ov.stream
	s := a.cardSnapshot()
	if r.id != s.ID {
		*r = streamReveal{id: s.ID}
	}
	total, live := a.revealTotal(s)
	// The model asked again (Translate's strict retry, a fallback, another
	// thinking level): the new answer shows at the pace from its start.
	r.shown = min(r.shown, total)
	if r.ticking || r.shown >= total || !live && r.shown == 0 || !a.timersRun() || a.overlayModeNow() != ovAnswer {
		return
	}
	r.ticking = true
	id := r.id
	a.later(revealFrame, func() {
		if r.id != id {
			return // another session's: its own steps run
		}
		r.ticking = false
		total, live := a.revealTotal(a.cardSnapshot())
		r.shown = revealRunes(r.shown, total, revealFrame, live)
		a.overlaySync()
	})
}

// revealTotal is how many runes the reveal moves towards: the live draft,
// or the ended answer it streamed into; nothing for any other end.
func (a *app) revealTotal(s session.Snapshot) (int, bool) {
	switch {
	case s.State.Live():
		return utf8.RuneCountInString(s.Draft), true
	case s.State == session.Done && a.answer.open && a.answer.id == s.ID:
		return utf8.RuneCountInString(a.answer.text), false
	}
	return 0, false
}

// timersRun says whether later's timers run: with a window, or in tests
// that run them.
func (a *app) timersRun() bool { return a.laterHook != nil || a.overlayNative() }
