package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

func newAnswerKeyTest(t *testing.T) (*app, *fakeSupervisor, *[]string, *[]proto.PasteParams) {
	t.Helper()
	a, fs, _ := newTestApp(t)
	copied := []string{}
	a.copyText = func(s string) error { copied = append(copied, s); return nil }
	pasted := []proto.PasteParams{}
	fs.handlers[proto.MethodTextFocus] = func(json.RawMessage) (any, error) {
		return proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}, nil
	}
	fs.handlers[proto.MethodTextPaste] = func(raw json.RawMessage) (any, error) {
		var p proto.PasteParams
		json.Unmarshal(raw, &p)
		pasted = append(pasted, p)
		return proto.PasteResult{Posted: true}, nil
	}
	return a, fs, &copied, &pasted
}

func openAnswerKeyCard(a *app, id, text string) {
	snap(a, session.Snapshot{ID: id, State: session.Recording})
	snap(a, session.Snapshot{ID: id, State: session.Done, Result: text, Card: true, Delivery: deliver.ReasonNotEditable})
}

func configuredCardKeys(fs *fakeSupervisor) (copyID, insertID string) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	for i := len(fs.calls) - 1; i >= 0; i-- {
		if fs.calls[i].Method != proto.MethodHotkeyConfigure {
			continue
		}
		var p proto.HotkeyConfigureParams
		json.Unmarshal(fs.calls[i].Params, &p)
		for _, b := range p.Bindings {
			if strings.HasPrefix(b.ID, hotkey.CardCopy) {
				copyID = b.ID
			}
			if strings.HasPrefix(b.ID, hotkey.CardInsert) {
				insertID = b.ID
			}
		}
		return copyID, insertID
	}
	return "", ""
}

func fireAnswerKey(a *app, binding string) {
	a.hotkeyFired(proto.HotkeyEventParams{Binding: binding, Phase: proto.PhaseTrigger})
}

// TestAnswerCardKeys: while the answer card shows, its single-letter
// keys are bound (#4127): c copies the answer (the card stays, its mark
// shows) and r inserts it where the user types (the card closes); with
// no card the letters are not bound and a late event does nothing, and
// a key does no more than its button (a failure's card has no copy).
func TestAnswerCardKeys(t *testing.T) {
	a, fs, copied, pasted := newAnswerKeyTest(t)

	// A failure's card: no text, so c does nothing (Copy is disabled).
	snap(a, session.Snapshot{ID: "k0", State: session.Recording})
	snap(a, session.Snapshot{ID: "k0", State: session.Failed, Card: true, Problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("boom")}})
	copyID, _ := configuredCardKeys(fs)
	fireAnswerKey(a, copyID)
	if len(*copied) != 0 {
		t.Errorf("a failure's copy: %v", *copied)
	}

	openAnswerKeyCard(a, "k1", "Answer k1")
	if !a.answer.open || a.answer.text != "Answer k1" {
		t.Fatalf("card %+v", a.answer)
	}
	copyID, insertID := configuredCardKeys(fs)
	if copyID == "" || insertID == "" {
		t.Fatalf("bound while the card shows: copy %q, insert %q", copyID, insertID)
	}
	fireAnswerKey(a, copyID)
	if len(*copied) != 1 || (*copied)[0] != "Answer k1" || !a.answer.open || !a.answer.copied {
		t.Errorf("c copies: %v %+v", *copied, a.answer)
	}
	fireAnswerKey(a, insertID)
	if a.answer.open || len(*pasted) != 1 || (*pasted)[0].Text != "Answer k1" {
		t.Errorf("r inserts: %+v %v", a.answer, *pasted)
	}
	if copyID, insertID := configuredCardKeys(fs); copyID != "" || insertID != "" {
		t.Errorf("bound after the card closed: copy %q, insert %q", copyID, insertID)
	}
	fireAnswerKey(a, copyID)
	fireAnswerKey(a, insertID)
	if len(*copied) != 1 || len(*pasted) != 1 {
		t.Errorf("late events: %v %v", *copied, *pasted)
	}
}

func TestAnswerCardKeysStreamingFirstDraft(t *testing.T) {
	a, fs, copied, pasted := newAnswerKeyTest(t)
	snap(a, session.Snapshot{ID: "stream", State: session.Recording})
	snap(a, session.Snapshot{ID: "stream", State: session.Composing, Card: true, Draft: "first"})
	if a.overlayModeNow() != ovAnswer {
		t.Fatalf("first draft mode %s", a.overlayModeNow())
	}
	copyID, insertID := configuredCardKeys(fs)
	if copyID == "" || insertID == "" {
		t.Fatalf("first draft keys: copy %q, insert %q", copyID, insertID)
	}
	fireAnswerKey(a, copyID)
	fireAnswerKey(a, insertID)
	if len(*copied) != 0 || len(*pasted) != 0 {
		t.Errorf("partial answer actions: copied %v, pasted %v", *copied, *pasted)
	}
}

func TestAnswerCardCopyKeyWaitsForRedo(t *testing.T) {
	a, fs, copied, _ := newAnswerKeyTest(t)
	openAnswerKeyCard(a, "committed", "old answer")
	copyID, _ := configuredCardKeys(fs)
	a.answer.redo = "redo"
	snap(a, session.Snapshot{ID: "redo", State: session.Composing, Card: true, Draft: "new answer"})
	if a.cachedCard().CanCopy {
		t.Fatal("Copy is enabled while the redo streams")
	}
	fireAnswerKey(a, copyID)
	if len(*copied) != 0 {
		t.Errorf("c copied the old answer during redo: %v", *copied)
	}
}

func TestAnswerCardKeysOnlyWhenVisible(t *testing.T) {
	for _, test := range []struct {
		name   string
		notice bool
	}{
		{"notice covers the card", true},
		{"insert hides the card", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, fs, copied, pasted := newAnswerKeyTest(t)
			openAnswerKeyCard(a, "visible", "visible answer")
			copyID, insertID := configuredCardKeys(fs)
			if test.notice {
				a.showNotice(sessionNotice{text: "cover", over: true})
			} else {
				a.answer.inserting = true
				a.overlaySync()
			}
			if a.overlayModeNow() == ovAnswer {
				t.Fatal("the answer is still visible")
			}
			if copyID, insertID := configuredCardKeys(fs); copyID != "" || insertID != "" {
				t.Errorf("hidden card keys: copy %q, insert %q", copyID, insertID)
			}
			fireAnswerKey(a, copyID)
			fireAnswerKey(a, insertID)
			if len(*copied) != 0 || len(*pasted) != 0 {
				t.Errorf("hidden card actions: copied %v, pasted %v", *copied, *pasted)
			}
		})
	}
}

func TestAnswerCardKeyRejectsPreviousGeneration(t *testing.T) {
	a, fs, copied, pasted := newAnswerKeyTest(t)
	openAnswerKeyCard(a, "card-a", "Answer A")
	oldCopy, oldInsert := configuredCardKeys(fs)
	a.closeAnswer()
	openAnswerKeyCard(a, "card-b", "Answer B")
	newCopy, newInsert := configuredCardKeys(fs)
	if oldCopy == newCopy || oldInsert == newInsert {
		t.Errorf("card generations did not change: %q %q", oldCopy, oldInsert)
	}
	fireAnswerKey(a, oldCopy)
	fireAnswerKey(a, oldInsert)
	if len(*copied) != 0 || len(*pasted) != 0 {
		t.Errorf("previous card events acted on B: copied %v, pasted %v", *copied, *pasted)
	}
	fireAnswerKey(a, newInsert)
	if len(*pasted) != 1 || (*pasted)[0].Text != "Answer B" {
		t.Errorf("current card insert: %v", *pasted)
	}
}

func TestHotkeyConfigureHiddenCardUnbindsImmediately(t *testing.T) {
	a, fs, _ := newTestApp(t)
	sequences := []uint64{}
	fs.handlers[proto.MethodHotkeyConfigure] = func(raw json.RawMessage) (any, error) {
		var p proto.HotkeyConfigureParams
		json.Unmarshal(raw, &p)
		sequences = append(sequences, p.Sequence)
		if copyID, _ := configuredCardKeys(fs); copyID != "" {
			a.closeAnswer()
		}
		return proto.HotkeyStatus{Installed: true, Bindings: len(p.Bindings)}, nil
	}
	sequences = nil
	a.answer = answerCard{open: true, id: "A", text: "Answer A"}
	a.syncHotkeys()
	want := len(a.hotkeyPlan().Configure(false).Bindings)
	if a.hotkeys.Bindings != want {
		t.Errorf("helper kept %d bindings after the close completed, want %d", a.hotkeys.Bindings, want)
	}
	if len(sequences) < 2 || sequences[len(sequences)-2] == 0 || sequences[len(sequences)-1] <= sequences[len(sequences)-2] {
		t.Errorf("configure sequences %v, want an increasing open and close", sequences)
	}
	if copyID, insertID := configuredCardKeys(fs); copyID != "" || insertID != "" {
		t.Errorf("hidden card keys: copy %q, insert %q", copyID, insertID)
	}
}

func TestAnswerCardCancelInsertKeyKeepsPriority(t *testing.T) {
	a, fs, copied, _ := newAnswerKeyTest(t)
	openAnswerKeyCard(a, "cancel-r", "Answer")
	a.prefs.Features.Cancel.Keys = "R"
	a.syncHotkeys()
	copyID, insertID := configuredCardKeys(fs)
	if copyID == "" || insertID != "" {
		t.Errorf("cancel R conflicts with the card: copy %q, insert %q", copyID, insertID)
	}
	fireAnswerKey(a, proto.HotkeyCancel)
	if a.answer.open {
		t.Fatal("the user's cancel key did not close the card")
	}
	openAnswerKeyCard(a, "cancel-r-again", "Answer again")
	a.prefs.Features.Cancel.Keys = "R"
	a.syncHotkeys()
	copyID, _ = configuredCardKeys(fs)
	fireAnswerKey(a, copyID)
	if len(*copied) != 1 || (*copied)[0] != "Answer again" {
		t.Errorf("the nonconflicting copy key: %v", *copied)
	}
}

func TestAnswerCardCancelKeyKeepsPriority(t *testing.T) {
	a, fs, _, pasted := newAnswerKeyTest(t)
	openAnswerKeyCard(a, "cancel-c", "Answer")
	a.prefs.Features.Cancel.Keys = "C"
	a.syncHotkeys()
	copyID, insertID := configuredCardKeys(fs)
	if copyID != "" || insertID == "" {
		t.Errorf("cancel C conflicts with the card: copy %q, insert %q", copyID, insertID)
	}
	fireAnswerKey(a, proto.HotkeyCancel)
	if a.answer.open {
		t.Fatal("the user's cancel key did not close the card")
	}
	openAnswerKeyCard(a, "cancel-c-again", "Answer again")
	_, insertID = configuredCardKeys(fs)
	fireAnswerKey(a, insertID)
	if len(*pasted) != 1 || (*pasted)[0].Text != "Answer again" {
		t.Errorf("the nonconflicting insert key: %v", *pasted)
	}
}
