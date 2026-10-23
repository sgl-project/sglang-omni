package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestSessionPasteChecked: every paste is checked (verifyMs). The focused
// field not changing after ⌘V (10-20, Arc: "pasted, nothing there, no
// word") leaves the result in the answer card saying so, the history
// says card and the log says what the field read; with secure input on in
// another app, the card says that. A check that cannot tell (a web area)
// is an insert, as before. The card's Insert checks again and says why
// when it fails once more.
func TestSessionPasteChecked(t *testing.T) {
	for _, tc := range []struct {
		name, opts string
		inserted   bool
		reason     deliver.Reason
		why        string
		again      string // the card's Insert's message
	}{
		{name: "inserted", opts: "paste-check=inserted", inserted: true},
		{name: "unknown", opts: "paste-check=unknown", inserted: true},
		{name: "unchanged", opts: "paste-check=unchanged", reason: deliver.ReasonUnconfirmed,
			why:   "The text did not show up where you type. It is on the clipboard: press ⌘V, or insert it again.",
			again: "The text did not show up where you type: it is on the clipboard, press ⌘V."},
		{name: "secure input", opts: "paste-check=unchanged,secure-input=Terminal", reason: deliver.ReasonSecureInput,
			why:   "Secure input is on in another app, which can block the keys VoxtNew sends. The text is on the clipboard: press ⌘V, or insert it again.",
			again: "Secure input is on in Terminal, which can block the keys VoxtNew sends: the text is on the clipboard, press ⌘V."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(400, "transcribe/0", proto.PhaseTrigger))
			a := newSessionApp(t, tc.opts+","+ev, func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
			}, &fakeLocal{ASRP: fakeStreamer{final: "你好"}})
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			end := a.sess
			p := a.pasted()
			if end.State != session.Done || len(p) != 1 || p[0].VerifyMS != deliver.DefaultVerifyMS {
				t.Fatalf("end %+v, pasted %+v", end, p)
			}
			if !strings.Contains(a.logText(), "delivery: focus AXTextArea in Slack") {
				t.Errorf("the log has no delivery line:\n%s", a.logText())
			}
			if tc.inserted {
				if !end.Inserted || end.Card || a.answer.open {
					t.Errorf("end %+v, card %v", end, a.answer.open)
				}
				return
			}
			if end.Inserted || !end.Card || end.Delivery != tc.reason || !a.answer.open || a.answer.why != tc.why {
				t.Fatalf("end %+v, card %+v", end, a.answer)
			}
			a.pump("the card in the history", func() bool { es := a.history(); return len(es) == 1 && es[0].Delivery == "card" })
			a.insertAnswer()
			a.pump("the second paste", func() bool { return !a.answer.inserting && len(a.pasted()) == 2 })
			if !a.answer.open || a.answer.msg != tc.again {
				t.Errorf("after Insert: open %v, msg %q", a.answer.open, a.answer.msg)
			}
		})
	}
}

// logText is the app's log so far, one entry a line.
func (s *sessionApp) logText() string {
	var b strings.Builder
	for _, e := range s.logs.Tail(1000, "") {
		b.WriteString(e.Message + "\n")
	}
	return b.String()
}
