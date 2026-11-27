package main

// A Rewrite or Translate answer that is still streaming when the
// transcribe shortcut fires (wall 116): the session's composing step
// backgrounds (session.Manager.Background) and keeps streaming into its
// card, while the new session records. The card is never set aside or
// recreated: it shows in the overlay from the backgrounded session's
// snapshots (bgSess), the recording pill in its own window
// (session_pill_window.go), and the answer's end opens the finished
// card as any session's would.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// answerBackgroundable says whether the live session can finish in the
// background: a Rewrite's or Translate's answer streaming into the card.
// The earlier states need the session's place (the microphone, the
// recognition the pill waits on), and Transcribe's clean-up drops the
// rest of what was said when its place goes.
func (a *app) answerBackgroundable() bool {
	s := a.sess
	return s.State == session.Composing && s.Card && !s.Try &&
		(s.Kind == store.KindRewrite || s.Kind == store.KindTranslate)
}

// backgroundAnswer moves the streaming answer's session to the
// background. The card is not touched: a fresh answer keeps streaming
// into it from the backgrounded snapshots (bgSess), exactly as it did
// from the live session's, and its end opens the finished card as any
// session's would. It reports whether the session backgrounded.
func (a *app) backgroundAnswer() bool {
	s := a.sess
	if !a.answerBackgroundable() || !a.mgr.Background(s.ID) {
		return false
	}
	a.bgID, a.bgSess, a.bgSlot = s.ID, s, a.slot
	return true
}

// bgVisible says whether the backgrounded answer belongs to the current
// session: only then does it drive the visible card or take the
// cancel shortcut (#4005).
func (a *app) bgVisible() bool { return a.bgID != "" && a.bgSlot == a.slot }

// backgroundChanged is sessionChanged for the backgrounded answer: its
// snapshots update only its card; its end is any session's end (the
// finished card, the history).
func (a *app) backgroundChanged(s session.Snapshot) {
	if s.Rev <= a.bgSess.Rev {
		return
	}
	was := a.bgSess
	a.bgSess = s
	if s.State.Ended() && !was.State.Ended() {
		// bgID stays set through sessionEnded: its openAnswer is the
		// answer's own card, not another session's.
		a.sessionEnded(s)
		a.bgID = ""
		a.syncHotkeys() // the answer no longer runs: the shortcuts settle
	}
	a.overlaySync()
}

// cardSnapshot is the session snapshot the answer card reads: the
// backgrounded answer's while it belongs to this session, else the current
// session's.
func (a *app) cardSnapshot() session.Snapshot {
	if a.bgVisible() {
		return a.bgSess
	}
	return a.sess
}

// cancelCardSession cancels the session the card belongs to: this
// session's backgrounded answer, or the live follow-up or redo. Another
// session's answer keeps streaming (#4005).
func (a *app) cancelCardSession() {
	if a.mgr == nil {
		return
	}
	if a.bgVisible() {
		a.mgr.Cancel(a.bgID)
		return
	}
	a.cancelSession()
}
