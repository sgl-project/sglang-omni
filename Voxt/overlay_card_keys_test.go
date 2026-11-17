package main

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestAnswerCardKeys: while the answer card shows, its single-letter
// keys are bound (#4127): c copies the answer (the card stays, its mark
// shows) and r inserts it where the user types (the card closes); with
// no card the letters are not bound and a late event does nothing, and
// a key does no more than its button (a failure's card has no copy).
func TestAnswerCardKeys(t *testing.T) {
	a, fs, _ := newTestApp(t)
	var copied []string
	a.copyText = func(s string) error { copied = append(copied, s); return nil }
	var pasted []proto.PasteParams
	fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}, nil
	}
	fs.handlers[proto.MethodTextPaste] = func(raw json.RawMessage) (any, error) {
		var p proto.PasteParams
		json.Unmarshal(raw, &p)
		pasted = append(pasted, p)
		return proto.PasteResult{Posted: true}, nil
	}
	cardBindings := func() (copy, insert bool) {
		fs.mu.Lock()
		defer fs.mu.Unlock()
		for i := len(fs.calls) - 1; i >= 0; i-- {
			if fs.calls[i].Method == proto.MethodHotkeyConfigure {
				var p proto.HotkeyConfigureParams
				json.Unmarshal(fs.calls[i].Params, &p)
				for _, b := range p.Bindings {
					copy = copy || b.ID == hotkey.CardCopy
					insert = insert || b.ID == hotkey.CardInsert
				}
				return
			}
		}
		return
	}
	fire := func(binding string) {
		a.hotkeyFired(proto.HotkeyEventParams{Binding: binding, Phase: proto.PhaseTrigger})
	}

	// A failure's card: no text, so c does nothing (Copy is disabled).
	snap(a, session.Snapshot{ID: "k0", State: session.Recording})
	snap(a, session.Snapshot{ID: "k0", State: session.Failed, Card: true, Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("boom")}})
	fire(hotkey.CardCopy)
	if len(copied) != 0 {
		t.Errorf("a failure's copy: %v", copied)
	}

	snap(a, session.Snapshot{ID: "k1", State: session.Recording})
	snap(a, session.Snapshot{ID: "k1", State: session.Done, Result: "Answer k1", Card: true, Delivery: deliver.ReasonNotEditable})
	if !a.answer.open || a.answer.text != "Answer k1" {
		t.Fatalf("card %+v", a.answer)
	}
	if copy, insert := cardBindings(); !copy || !insert {
		t.Errorf("bound while the card shows: copy %v, insert %v", copy, insert)
	}
	fire(hotkey.CardCopy)
	if len(copied) != 1 || copied[0] != "Answer k1" || !a.answer.open || !a.answer.copied {
		t.Errorf("c copies: %v %+v", copied, a.answer)
	}
	fire(hotkey.CardInsert)
	if a.answer.open || len(pasted) != 1 || pasted[0].Text != "Answer k1" {
		t.Errorf("r inserts: %+v %v", a.answer, pasted)
	}
	if copy, insert := cardBindings(); copy || insert {
		t.Errorf("bound after the card closed: copy %v, insert %v", copy, insert)
	}
	fire(hotkey.CardCopy) // a late event of a closed card does nothing
	fire(hotkey.CardInsert)
	if len(copied) != 1 || len(pasted) != 1 {
		t.Errorf("late events: %v %v", copied, pasted)
	}
}
