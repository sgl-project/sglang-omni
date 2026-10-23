package main

import (
	"encoding/json"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// TestSessionSwitchedApp: the shortcut is pressed in Slack, then the user
// goes to another app. A text field focused there gets the result (the
// history keeps Slack and names that app apart); nothing focused (Finder's desktop) or a
// password field leaves it in the answer card, saying why.
func TestSessionSwitchedApp(t *testing.T) {
	for _, tc := range []struct {
		name, opts string
		into       string // the app it went into; "": the card
		why        string
		reason     deliver.Reason
	}{
		{name: "a text field", opts: "switch=com.apple.TextEdit", into: "TextEdit"},
		{name: "nothing focused", opts: "switch=com.apple.finder,focus=none", reason: deliver.ReasonNotEditable,
			why: "No text field is focused: click where it should go, then insert, or copy it."},
		{name: "a password field", opts: "switch=com.apple.TextEdit,focus=secure", reason: deliver.ReasonSecure,
			why: "A password field is focused, and VoxtNew never types into one: copy it, or click another field and insert."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
			a := newSessionApp(t, tc.opts+","+ev, func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
			}, &fakeLocal{ASRP: fakeStreamer{final: "你好"}})
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			end := a.sess
			if end.State != session.Done || end.Target.Name != "Slack" {
				t.Fatalf("end %+v", end)
			}
			if tc.into == "" {
				if end.Inserted || !end.Card || end.Delivery != tc.reason || len(a.pasted()) != 0 || !a.answer.open || a.answer.why != tc.why {
					t.Errorf("card: end %+v, pasted %+v, card %+v", end, a.pasted(), a.answer)
				}
				return
			}
			p := a.pasted()
			if !end.Inserted || end.InsertedInto.Name != tc.into || len(p) != 1 || p[0].TargetPID != 530 || p[0].Text != "你好" {
				t.Errorf("end %+v, pasted %+v", end, p)
			}
			a.pump("the history", func() bool { return len(a.history()) == 1 })
			e := a.history()[0]
			if e.AppName != "Slack" || e.AppBundle != "com.tinyspeck.slackmacgap" || e.Delivery != "insert" || e.InsertedBundle != "com.apple.TextEdit" || e.InsertedName != tc.into {
				t.Errorf("history %+v", e)
			}
			if got := uitext.DeliveryLabel(e); got != "Inserted into TextEdit" {
				t.Errorf("the details say %q", got)
			}
		})
	}
}

// activatedLine is a script's app.activated: the user brings bundle's app
// to the front (the fake helper's front follows).
func activatedLine(afterMS int, bundle string) string {
	app, ok := helpertest.Apps[bundle]
	if !ok {
		app = proto.App{PID: 640, BundleID: bundle, Name: "Finder"} // Finder (not in the fake's app.list)
	}
	b, _ := json.Marshal(helpertest.Script{AfterMS: afterMS, Method: proto.MethodAppActivated, Params: mustJSON(app)})
	return string(b)
}

// TestCardInsertSwitchedApp: the user moved to Finder, where nothing
// takes text, so the result waits in the card. Once the user is in
// TextEdit, the card's Insert pastes there; the history keeps Slack, the
// start app, and names TextEdit apart.
func TestCardInsertSwitchedApp(t *testing.T) {
	ev := script(t,
		hotkeyLine(0, "transcribe/0", proto.PhaseTrigger),
		activatedLine(150, "com.apple.finder"),
		hotkeyLine(250, "transcribe/0", proto.PhaseTrigger),
		activatedLine(1500, "com.apple.TextEdit"), // well after the end
	)
	a := newSessionApp(t, "readonly=com.apple.finder,"+ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
	}, &fakeLocal{ASRP: fakeStreamer{final: "你好"}})
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Inserted || !end.Card || end.Target.Name != "Slack" || !a.answer.open || len(a.pasted()) != 0 {
		t.Fatalf("end %+v, card %v, pasted %+v", end, a.answer.open, a.pasted())
	}
	a.pump("the card in the history", func() bool { es := a.history(); return len(es) == 1 && es[0].Delivery == "card" })
	a.pump("TextEdit in front", func() bool { return a.front.PID == 530 })
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].TargetPID != 530 || p[0].Text != "你好" {
		t.Fatalf("pasted %+v", p)
	}
	a.pump("the history", func() bool { return a.history()[0].Delivery == "insert" })
	e := a.history()[0]
	if e.AppBundle != "com.tinyspeck.slackmacgap" || e.AppName != "Slack" || e.InsertedBundle != "com.apple.TextEdit" || e.InsertedName != "TextEdit" {
		t.Errorf("history %+v", e)
	}
	if got := uitext.DeliveryLabel(e); got != "Inserted into TextEdit" {
		t.Errorf("the details say %q", got)
	}
}
