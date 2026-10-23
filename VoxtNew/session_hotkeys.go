package main

// The shortcuts as sessions use them (session_link.go): routing a pressed
// shortcut to an action by the session's phase, and sending the bindings to
// the helper's event tap, or to MyGo's Carbon shortcuts without
// Accessibility.

import (
	"fmt"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// ---- shortcuts ----

// hotkeyPhase is the session's phase for routing.
func (a *app) hotkeyPhase() hotkey.Phase {
	switch a.sess.State {
	case session.Recording:
		return hotkey.Recording
	case session.Reading, session.Recognizing, session.Composing, session.Delivering:
		return hotkey.Busy
	}
	return hotkey.Idle
}

// hotkeyFired acts on a shortcut (the helper's hotkey.event, or a Carbon
// shortcut).
func (a *app) hotkeyFired(e proto.HotkeyEventParams) {
	target := a.front
	if e.Front != nil {
		target = *e.Front
	}
	if e.Binding == proto.HotkeyEscape && a.answer.langOpen {
		a.closeLangMenu() // Escape closes the menu first, then the card
		return
	}
	if e.Binding == proto.HotkeyEscape && a.answer.open {
		switch {
		case !a.sess.State.Live() || a.sess.ID == a.answer.redo:
			// Voxt's: Escape closes the card, and stops another language
			// on its way.
			a.closeAnswer()
			return
		case a.followUp != "" && a.followUp == a.answer.id:
			// A follow-up: Escape cancels it, the card stays (Voxt's).
			a.cancelSession()
			return
		}
	}
	d := hotkey.Route(e, a.hotkeyPhase(), a.sess.HoldBinding)
	switch d.Action {
	case hotkey.Start:
		hold := ""
		if d.Hold {
			hold = e.Binding
		}
		if d.Feature == hotkey.Rewrite && a.followUpAnswer(hold) {
			return
		}
		kind := store.KindTranscribe
		switch d.Feature {
		case hotkey.Translate:
			kind = store.KindTranslate
		case hotkey.Rewrite:
			kind = store.KindRewrite
		}
		req := session.StartRequest{Kind: kind, Target: target, HoldBinding: hold}
		// VoxtNew in front, on a page with a Try area for it: a try there.
		switch place := a.tryPlace(kind, target); place {
		case placeNone:
		case placeIgnore:
			a.tryIgnored(kind) // onboarding.go
			return
		default:
			req.Try = true
			id, n := a.startSessionID(req)
			a.tryStarted(place, kind, id, n)
			return
		}
		a.startSession(req)
	case hotkey.Stop:
		a.stopSession()
	case hotkey.Cancel:
		a.cancelSession()
	case hotkey.Paste:
		a.pasteLast(target)
	}
}

// hotkeyPlan is what the settings' shortcuts become.
func (a *app) hotkeyPlan() hotkey.Plan { return hotkey.Build(a.prefs.Features, liveFeatures) }

// syncHotkeys sends the shortcuts to the helper (and flips "active" while
// a session runs or the answer card shows), and keeps the Carbon
// fallback in step with the event tap.
func (a *app) syncHotkeys() {
	plan := a.hotkeyPlan()
	a.syncCarbon(plan)
	if !a.helperReady() {
		a.hotkeysSent = ""
		return
	}
	params := plan.Configure(a.sess.State.Live() || a.answer.open)
	key := fmt.Sprintf("%+v", params)
	if key == a.hotkeysSent {
		return
	}
	a.hotkeysSent = key
	helperCall(a, proto.MethodHotkeyConfigure, params, 5*time.Second, func(st proto.HotkeyStatus, err error) {
		if err != nil {
			a.hotkeysSent = ""
			a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityWarning, Source: sessionSource, Message: "hotkey.configure: " + err.Error()})
			return
		}
		was := a.hotkeys.Installed
		a.hotkeys = st
		if st.Installed != was {
			a.syncCarbon(a.hotkeyPlan())
		}
	})
}

// syncCarbon registers the plan's key shortcuts with MyGo while the
// helper's event tap is not installed (no Accessibility), and removes
// them once it is. Only the real app (it needs mygo.App).
func (a *app) syncCarbon(plan hotkey.Plan) {
	if !a.hasMenu {
		return
	}
	var want []string
	byAcc := map[string]string{}
	if !a.hotkeys.Installed {
		for _, c := range plan.Carbon {
			want = append(want, c.Accelerator)
			byAcc[c.Accelerator] = c.ID
		}
	}
	if strings.Join(want, "\n") == strings.Join(a.carbon, "\n") {
		return
	}
	for _, acc := range a.carbon {
		mygo.GlobalShortcut.Unregister(acc)
	}
	a.carbon = nil
	for _, acc := range want {
		id := byAcc[acc]
		err := mygo.GlobalShortcut.Register(acc, func() {
			a.post(func() {
				front := a.front
				a.hotkeyFired(proto.HotkeyEventParams{Binding: id, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
			})
		})
		if err != nil {
			a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityWarning, Source: sessionSource, Message: "shortcut " + acc + ": " + err.Error()})
			continue
		}
		a.carbon = append(a.carbon, acc)
	}
}
