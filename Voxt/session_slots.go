package main

// The session slots of the multi-session prototype (#4005): the session
// shortcuts (Control+Option+1 … by default, Settings → Shortcuts) switch
// the slot Rewrite, Translate and Transcribe run in. Each slot keeps its
// own answer card (and the card waiting under it, #4013) with its
// conversation; switching back restores them, so a covered Rewrite
// result is never lost. An answer still streaming when the shortcut
// fires finishes in the background in its own slot (wall 116): its card
// waits there until the slot comes back. With no session shortcut
// pressed, everything stays in slot 0 and behaves as before.

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
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
// session shortcut's tap, session_hotkeys.go). While a session records
// or works it does nothing; a streaming answer finishes in the
// background in its own slot. An open Rewrite box belongs to the slot
// being left: it closes, as it does for another feature's session; its
// answered card parks with the slot.
func (a *app) switchSlot(n int) {
	if n == a.slot || n < 0 || n >= slotCount || a.answer.inserting {
		return // an insert pastes: its callback must find its card in place
	}
	if a.sess.State.Live() {
		// Only a streaming answer can keep going meanwhile; anything
		// earlier needs the session's place (session_background.go).
		if !a.answerBackgroundable() || !a.backgroundAnswer() {
			return
		}
		// No new session starts here (startSessionID starts one right
		// after backgrounding, which covers this): the manager's
		// foreground snapshot stays the backgrounded session's until
		// then, so the app must not read it as live.
		a.sess = session.Snapshot{State: session.Idle, Rev: a.sess.Rev}
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
	if a.prefs.Setup.SessionSwitches < settings.SessionGuideTimes {
		// The first switches teach sessions, beside a card that came back too.
		a.updateSettings(func(s *settings.Settings) { s.Setup.SessionSwitches++ })
		a.showNotice(sessionNotice{text: sessionGuide(n, a.prefs.Features.Sessions), ok: true})
	} else if !a.answer.open && !a.bgVisible() {
		a.showNotice(sessionNotice{text: i18n.T("Session %d", n+1), ok: true})
	}
	a.buildTrayMenu()
	a.syncHotkeys()
	a.overlaySync()
}

// sessionGuide is a first switch's notice: the session's name, what a
// session keeps, and the shortcuts that switch.
func sessionGuide(n int, sessions [slotCount]settings.Hotkey) string {
	var keys []string
	for _, h := range sessions {
		if h.Keys != "" {
			keys = append(keys, uitext.HotkeyGlyphs(h.Keys))
		}
	}
	return i18n.T("Session %d. Each session keeps its own card and conversation; switch with %s.", n+1, strings.Join(keys, " "))
}

// parkAnswer gives an ended backgrounded answer its card in its own
// session slot while another slot is current (#4005): the card, with its
// conversation, waits there until the slot comes back. A redo of the
// slot's card replaces its answer there; another open card of the slot
// waits under the new one, as setAside would hold it.
func (a *app) parkAnswer(s session.Snapshot) {
	saved := parkedSlot{answer: a.answer, under: a.under, followUp: a.followUp}
	p := a.slots[a.bgSlot]
	a.answer, a.under, a.followUp = p.answer, p.under, p.followUp
	switch {
	case a.answer.open && a.answer.redo == s.ID:
		a.endRedo(s)
	default:
		if a.answer.open && a.followUp == "" {
			a.setAside(false)
		}
		a.openAnswer(s, a.bgSlot)
	}
	a.slots[a.bgSlot] = parkedSlot{answer: a.answer, under: a.under, followUp: a.followUp}
	a.answer, a.under, a.followUp = saved.answer, saved.under, saved.followUp
}
