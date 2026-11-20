package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestEverySessionLeavesOneHistoryEntry: each kind of session, and a
// cancelled and a failed one, leaves exactly one entry in History, of the
// session's kind and status, and no second one once the app has settled.
func TestEverySessionLeavesOneHistoryEntry(t *testing.T) {
	tap := func(binding string, stopMS int) []string {
		lines := []string{hotkeyLine(0, binding, proto.PhaseTrigger)}
		if stopMS > 0 {
			lines = append(lines, hotkeyLine(stopMS, binding, proto.PhaseTrigger))
		}
		return lines
	}
	selection := helpertest.Opt("selection", "Good morning, team.") + ","
	for _, c := range []struct {
		name   string
		opts   string
		events []string
		chat   *fakeChat
		kind   store.Kind
		status store.Status
	}{
		{"transcribe", "", tap("transcribe/0", 400), &fakeChat{text: "Hello."}, store.KindTranscribe, store.StatusDone},
		{"translate speech", "", tap("translate/0", 400), &fakeChat{text: "Hello."}, store.KindTranslate, store.StatusDone},
		{"translate selection", selection, tap("translate/0", 0), &fakeChat{text: "早上好，团队。"}, store.KindTranslate, store.StatusDone},
		{"rewrite", selection, tap("rewrite/0", 500), &fakeChat{text: "Good morning to the team."}, store.KindRewrite, store.StatusDone},
		{"cancelled", "", []string{hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(500, proto.HotkeyCancel, proto.PhaseTrigger)},
			&fakeChat{text: "Hello."}, store.KindTranscribe, store.StatusCancelled},
		{"failed", selection, tap("translate/0", 0), &fakeChat{status: 500}, store.KindTranslate, store.StatusFailed},
	} {
		t.Run(c.name, func(t *testing.T) {
			set, keys, _ := fileSetup(t, "hello there", c.chat)
			a := newSessionApp(t, c.opts+script(t, c.events...), func(s *settings.Settings) { set(s) }, keys)
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			a.pump("the entry", func() bool { return len(a.history()) > 0 })
			for settled := time.After(300 * time.Millisecond); ; { // what the app still does after the end
				select {
				case fn := <-a.queue:
					fn()
					continue
				case <-settled:
				}
				break
			}
			es := a.history()
			if len(es) != 1 || es[0].Kind != c.kind || es[0].Status != c.status {
				t.Fatalf("want one %s entry with status %s, got %+v", c.kind, c.status, es)
			}
		})
	}
}
