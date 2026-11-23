package main

// The session slots of the multi-session prototype (#4005): the session
// shortcuts (Option+1 … by default, Settings → Shortcuts) switch the
// slot Rewrite, Translate and Transcribe run in. Each slot keeps its own
// answer card (and the card waiting under it, #4013) with its
// conversation; switching back restores them, so a covered Rewrite
// result is never lost. With no session shortcut pressed, everything
// stays in slot 0 and behaves as before.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// slotCount is how many session slots the session shortcuts switch
// between (settings.Features.Sessions).
const slotCount = 3

// parkedSlot is a session slot's cards while another slot is current.
type parkedSlot struct {
	answer, under answerCard
	followUp      string
}

// switchSlot parks the current slot's cards and restores slot n's (a
// session shortcut's tap, session_hotkeys.go). While a session runs it
// does nothing; routing already keeps a slot switch to an idle app
// (internal/hotkey's Route). An open Rewrite box belongs to the slot
// being left: it closes, as it does for another feature's session; its
// answered card parks with the slot.
func (a *app) switchSlot(n int) {
	if n == a.slot || n < 0 || n >= slotCount || a.sess.State.Live() || a.bgID != "" && a.bgSess.State.Live() || a.answer.inserting {
		return // an insert pastes: its callback must find its card in place
	}
	if a.box.open {
		a.closeBox()
	}
	a.slots[a.slot] = parkedSlot{answer: a.answer, under: a.under, followUp: a.followUp}
	p := a.slots[n]
	a.slots[n] = parkedSlot{}
	a.slot, a.slotsUsed = n, true
	a.answer, a.under, a.followUp = p.answer, p.under, p.followUp
	a.notice = nil
	a.resurface()
	if !a.answer.open {
		a.showNotice(sessionNotice{text: i18n.T("Session %d", n+1), ok: true})
	}
	a.buildTrayMenu()
	a.syncHotkeys()
	a.overlaySync()
}
