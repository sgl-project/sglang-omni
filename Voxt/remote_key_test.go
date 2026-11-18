package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestRemoteDialogDismissalUsesLiveInput(t *testing.T) {
	for _, test := range []struct {
		name     string
		before   func(*providerDialog)
		during   func(*providerDialog)
		wantOpen bool
	}{
		{name: "endpoint edit", during: func(dialog *providerDialog) { dialog.endpoint = "changed" }, wantOpen: true},
		{name: "endpoint revert", before: func(dialog *providerDialog) { dialog.endpoint = "changed" }, during: func(dialog *providerDialog) { dialog.endpoint = dialog.was.Endpoint }},
		{name: "key edit", during: func(dialog *providerDialog) { dialog.key = "fixture" }, wantOpen: true},
		{name: "key revert", before: func(dialog *providerDialog) { dialog.key = "fixture" }, during: func(dialog *providerDialog) { dialog.key = "" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := remoteFixtureDialogInput(local.LLM)
			a := &app{provDlg: *input.Dialog}
			input.Dialog = &a.provDlg
			if test.before != nil {
				test.before(&a.provDlg)
			}
			input.Unsaved = a.provDlg.unsaved()
			state := buildRemoteDialogViewState(i18n.NewTranslator("en"), *input)
			tt := ui.NewTester(func(c *ui.Context) {
				test.during(&a.provDlg)
				remoteProviderDialogView(c, &state, a, &a.provDlg.providerDialogLocalUI)
			}, 1000, 1000)
			tt.Key(0, ui.KeyEscape)
			frames(tt)
			if a.provDlg.open != test.wantOpen {
				t.Fatalf("dialog open %t, want %t", a.provDlg.open, test.wantOpen)
			}
		})
	}
}

// TestRemoteKeyWhitespace: a saved key with a space in it (copied where
// the text wrapped) is flagged on its card and in its dialog; a pasted
// key says how many spaces or line breaks it has, shows in clear on Show,
// and is saved without them.
func TestRemoteKeyWhitespace(t *testing.T) {
	keys := &memKeys{m: map[string]string{"llm.custom": "rxinf_9a97836 4624"}}
	useKeys(t, keys)
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.updateSettings(func(s *settings.Settings) {
		providers.Put(s, settings.Provider{ID: llm.Custom, Endpoint: "https://example.com/v1", Model: "m"})
	})
	a.navigate(remotePath)
	frames(tt)
	if !has(tt, "Key has a space") {
		t.Fatalf("no warning on the card: %q", tt.Texts())
	}
	a.openProviderDialog(local.LLM, llm.Custom)
	frames(tt)
	if !has(tt, "The saved key has a space or line break in it, so the service refuses it. Paste the key again.") {
		t.Fatalf("no warning in the dialog: %q", tt.Texts())
	}

	a.provDlg.key = "sk-abc def\tghi\n"
	frames(tt)
	if !has(tt, "The key has 2 spaces or line breaks in it, removed when you test or save.") {
		t.Fatalf("no hint for the pasted key: %q", tt.Texts())
	}
	click(t, tt, "Show")
	if !a.provDlg.showKey || !hasExact(tt, "Hide") {
		t.Fatal("Show does not show the key")
	}
	click(t, tt, "Save")
	frames(tt)
	if keys.m["llm.custom"] != "sk-abcdefghi" || a.keySpaced["llm.custom"] {
		t.Fatalf("saved %q, flagged %v", keys.m["llm.custom"], a.keySpaced["llm.custom"])
	}
	if has(tt, "Key has a space") {
		t.Error("the card still warns after a clean key was saved")
	}
}
