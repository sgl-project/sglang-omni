package main

// Starting and stopping sessions (session_link.go): what blocks one (the
// helper, the services a feature needs), starting it (or a try), stopping,
// cancelling, the menu bar's toggle, and pasting the last result again.

import (
	"context"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// helperReady reports whether the helper can record.
func (a *app) helperReady() bool {
	return a.hstatus.State == helper.Ready
}

// transcribeBlocker checks, without I/O, what a Transcribe session
// needs (sessionBlocker).
func (a *app) transcribeBlocker() *sessionNotice { return a.sessionBlocker(store.KindTranscribe) }

// sessionBlocker checks, without I/O, what a session of kind needs: a
// running helper, a speech model (Transcribe, Rewrite: Translate may
// translate a selection without one) and a language model (Translate,
// Rewrite). It returns the notice that says what is missing, or nil.
func (a *app) sessionBlocker(kind store.Kind) *sessionNotice {
	if !a.helperReady() {
		return &sessionNotice{text: i18n.T("The helper is not running."), action: actionHelper}
	}
	if kind != store.KindTranslate {
		if _, ok := a.featureModel(local.ASR, string(kind)); !ok {
			return &sessionNotice{text: i18n.T("No speech model is installed."), action: actionModels}
		}
	}
	if kind != store.KindTranscribe {
		if _, ok := a.featureModel(local.LLM, string(kind)); !ok {
			return &sessionNotice{text: i18n.T("No language model is installed."), action: actionModels}
		}
	}
	return nil
}

// startTranscribe starts a Transcribe session (startSession).
func (a *app) startTranscribe(target proto.App, hold string, try bool) *sessionNotice {
	return a.startSession(session.StartRequest{Kind: store.KindTranscribe, Target: target, HoldBinding: hold, Try: try})
}

// startSession starts a session for the app req.Target (the app in front
// when the shortcut fired); req.HoldBinding names the long-pressed
// binding; a try keeps the result on its page. When something it needs
// is missing it returns the notice saying so, which a shortcut's session
// shows in the overlay (a try shows it on the page); nil when it started
// (or a session runs already).
func (a *app) startSession(req session.StartRequest) *sessionNotice {
	_, n := a.startSessionID(req)
	return n
}

// startSessionID is startSession that also returns the session's id ("" when
// none started), so a Try area can tell its own session from any other.
func (a *app) startSessionID(req session.StartRequest) (string, *sessionNotice) {
	if a.mgr == nil || a.sess.State.Live() {
		return "", nil
	}
	if n := a.sessionBlocker(req.Kind); n != nil {
		if !req.Try {
			a.showNotice(*n)
		}
		return "", n
	}
	if len(req.Previous) == 0 {
		a.answer = answerCard{}
	}
	a.notice = nil
	id, err := a.mgr.Start(req)
	if err != nil {
		return "", nil // a session runs already
	}
	return id, nil
}

// startTry starts a try of kind (a Try area's Start).
func (a *app) startTry(kind store.Kind) (string, *sessionNotice) {
	return a.startSessionID(session.StartRequest{Kind: kind, Try: true})
}

// stopSession stops the recording (the shortcut, the menu bar).
func (a *app) stopSession() {
	if a.mgr != nil && a.sess.State == session.Recording {
		a.mgr.Stop(a.sess.ID)
	}
}

// cancelSession cancels the session (Escape, the answer card's Close).
func (a *app) cancelSession() {
	if a.mgr != nil && a.sess.State.Live() {
		a.mgr.Cancel(a.sess.ID)
	}
}

// toggleTranscribe is the menu bar's Start / Stop transcribing.
func (a *app) toggleTranscribe() {
	switch {
	case a.sess.State == session.Recording:
		a.stopSession()
	case !a.sess.State.Live():
		a.startTranscribe(a.front, "", false)
	}
}

// setDelivery records in the history how a result left the card, and
// the app it was inserted into when that is not the entry's (into: zero
// otherwise).
func (a *app) setDelivery(id, how string, into proto.App) {
	h := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		e, err := h.Get(ctx, id)
		if err != nil {
			return // not kept
		}
		e.Delivery = how
		e.InsertedBundle, e.InsertedName = into.BundleID, into.Name
		if err := h.Put(ctx, e); err == nil {
			a.post(a.historyChanged)
		}
	})
}

// pasteLast pastes the last result into target (⌃⌘V, the menu bar).
func (a *app) pasteLast(target proto.App) {
	if a.mgr == nil {
		return
	}
	text := a.mgr.Last()
	if text == "" {
		a.showNotice(sessionNotice{text: i18n.T("Nothing to paste yet.")})
		return
	}
	io := a.textIO() // family_runtime.go: the family's paste rule
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		res := deliver.Insert(ctx, io, deliver.Request{Text: text, Target: target, VerifyMS: deliver.DefaultVerifyMS})
		a.post(func() {
			if !res.Inserted {
				a.showNotice(sessionNotice{text: uitext.DeliveryText(res, target)})
			}
		})
	})
}

// loadLast reads the newest delivered result from the history, for
// "paste the last result" after a restart.
func (a *app) loadLast() {
	if a.lastLoaded || a.mgr == nil {
		return
	}
	a.lastLoaded = true
	h, mgr := a.lib.store.History(), a.mgr
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		es, err := h.List(ctx, store.HistoryQuery{Limit: 20})
		if err != nil {
			return
		}
		for _, e := range es {
			if e.Status == store.StatusDone && strings.TrimSpace(e.Result) != "" {
				mgr.SetLast(e.Result)
				return
			}
		}
	})
}
