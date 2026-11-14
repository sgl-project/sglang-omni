package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestSessionPasteChecked: every paste is checked (verifyMs). The focused
// field not changing after ⌘V (10-20, Arc: "pasted, nothing there, no
// word") is no insert: with the card off, the notice says the clipboard
// has it, and the log says what the field read. A check that cannot tell (a web area) is an insert,
// as before.
func TestSessionPasteChecked(t *testing.T) {
	for _, tc := range []struct {
		name, opts string
		inserted   bool
		reason     deliver.Reason
	}{
		{name: "inserted", opts: "paste-check=inserted", inserted: true},
		{name: "unknown", opts: "paste-check=unknown", inserted: true},
		{name: "unchanged", opts: "paste-check=unchanged", reason: deliver.ReasonUnconfirmed},
		{name: "secure input", opts: "paste-check=unchanged,secure-input=Terminal", reason: deliver.ReasonSecureInput},
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
			// The card is off: no card, the notice says the clipboard has it.
			if end.Inserted || end.Card || end.Delivery != tc.reason || a.answer.open || a.notice == nil || a.notice.text != copiedNotice {
				t.Fatalf("end %+v, card %v, notice %+v", end, a.answer.open, a.notice)
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
