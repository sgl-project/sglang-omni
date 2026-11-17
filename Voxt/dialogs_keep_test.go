package main

// Form dialogs keep what was typed (ui_forms.go dialogFrame): a click
// beside the dialog, on the dimmed page, or Escape closes one only while
// it holds nothing to save; Cancel always closes it. A click back into
// Voxt's window after copying a key in another app used to land beside
// the remote service dialog and lose the key.

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestDialogsKeepInput(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	for _, d := range []struct {
		name, path string
		open       func()
		typed      func()
		isOpen     func() bool
	}{
		{"remote service", remotePath, func() { a.openProviderDialog(local.LLM, llm.Custom) },
			func() { a.provDlg.key = "sk-typed" }, func() bool { return a.provDlg.open }},
		{"term", "/dictionary", func() { a.openTerm(store.Term{}) },
			func() { a.dv.edit.term = "Voxt" }, func() bool { return a.dv.editOpen }},
		{"network", modelsPath, a.openNetwork,
			func() { a.netForm.proxyURL = "http://127.0.0.1:7897" }, func() bool { return a.netOpen }},
	} {
		a.navigate(d.path)
		beside := func() {
			tt.ClickAt(4, 4) // the dimmed page, beside the dialog
			frames(tt)
		}
		d.open()
		frames(tt)
		beside()
		if d.isOpen() {
			t.Errorf("%s: an untouched form stayed open after a click beside it", d.name)
		}
		d.open()
		frames(tt)
		d.typed()
		beside()
		if !d.isOpen() {
			t.Errorf("%s: a click beside it closed a form with input", d.name)
			continue
		}
		tt.Key(0, ui.KeyEscape)
		frames(tt)
		if !d.isOpen() {
			t.Errorf("%s: Escape closed a form with input", d.name)
			continue
		}
		click(t, tt, "Cancel")
		if d.isOpen() {
			t.Errorf("%s: Cancel did not close it", d.name)
		}
	}
	if a.provDlg.key != "sk-typed" {
		t.Errorf("the typed key is gone: %q", a.provDlg.key)
	}
}
