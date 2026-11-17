package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestSessionNote: the note shortcut, speak, the shortcut again: the
// cleaned-up text is saved as a note, not pasted and not kept in the
// history; the overlay says so and the Notes page has it.
func TestSessionNote(t *testing.T) {
	set, keys, _ := fileSetup(t, "buy milk", &fakeChat{text: "Buy milk."})
	ev := script(t,
		hotkeyLine(0, hotkey.Note, proto.PhaseTrigger),
		hotkeyLine(400, hotkey.Note, proto.PhaseTrigger),
	)
	a := newSessionApp(t, ev, set, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if end := a.sess; end.State != session.Done || end.Result != "Buy milk." || !end.Note || end.Inserted || end.Card || end.Target.Name != "Slack" {
		t.Fatalf("end %+v", end)
	}
	a.pump("the notes", func() bool { return len(a.nv.list) == 1 })
	if n := a.nv.list[0]; n.Text != "Buy milk." || n.ID != a.sess.ID {
		t.Errorf("note %+v", n)
	}
	if a.notice == nil || a.notice.text != "Saved to Notes" {
		t.Errorf("notice %+v", a.notice)
	}
	if p, es := a.pasted(), a.history(); len(p) != 0 || len(es) != 0 {
		t.Errorf("pasted %+v, history %+v", p, es)
	}
}
