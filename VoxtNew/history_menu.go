package main

// The context menu of a history entry, on History's rows and Home's recent
// results: Copy (its result), Copy the recognised text, Show details, Show
// the recording in Finder, and Delete (confirmed by historyDeleteDialog).
// On History a focused row takes ⌘C and ⌫ too. Auralink has the same
// menu without the recognised text and the recording.

import (
	"path/filepath"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// entryCopyText is what Copy puts on the clipboard: the entry's result (as
// the details' "Copy the result" and Home's copy button), else, for an
// entry that failed after recognition, the recognised text; "" for none.
func entryCopyText(e store.Entry) string {
	if s := strings.TrimSpace(e.Result); s != "" {
		return s
	}
	return strings.TrimSpace(e.RawText)
}

// entryRawCopy is the recognised text the menu offers apart: "" when
// there is none or Copy copies it already.
func entryRawCopy(e store.Entry) string {
	if raw := strings.TrimSpace(e.RawText); raw != entryCopyText(e) {
		return raw
	}
	return ""
}

// entryMenu gives row an entry's context menu; details shows the entry's
// details, as a click on the row does. keys adds ⌘C and ⌫ while the row
// has the focus, and shows them by the items. The menu's function runs in
// build passes (as it opens, and in the frame after an item was chosen):
// it reaches the store only through the Delete dialog.
func (a *app) entryMenu(c *ui.Context, row ui.Element, e store.Entry, keys bool, details func()) {
	text, raw := entryCopyText(e), entryRawCopy(e)
	row.ContextMenu(func(m *ui.Menu) {
		cp := m.Item(i18n.T("Copy")).Disabled(text == "")
		if keys {
			cp.Shortcut(ui.Cmd, ui.KeyC)
		}
		if cp.Chosen() {
			a.copyEntry(c, text)
		}
		if raw != "" && m.Item(i18n.T("Copy the recognised text")).Chosen() {
			a.copyEntry(c, raw)
		}
		if m.Item(i18n.T("Show details")).Chosen() {
			details()
		}
		if e.Audio != "" && m.Item(i18n.T("Show the recording in Finder")).Chosen() {
			revealInFinder(filepath.Join(a.lib.store.History().AudioDir(), e.Audio))
		}
		if e.Audio != "" && m.Item(retryLabel(e)).Disabled(a.hist.retry.id != "").Chosen() {
			a.retryEntry(e) // history_retry.go
		}
		m.Separator()
		del := m.Item(i18n.T("Delete"))
		if keys {
			del.Shortcut(0, ui.KeyBackspace)
		}
		if del.Chosen() {
			a.askDeleteEntry(e.ID)
		}
	})
	if !keys {
		return
	}
	// On macOS the Edit menu's Copy takes ⌘C (shell_commands.go): it
	// comes as the "copy" command, not as the key.
	svc := c.Services()
	row.HandleInput(func(ev ui.InputEvent) bool {
		if ev.Kind != ui.InputCommand || ev.Text != "copy" || text == "" {
			return false
		}
		svc.WriteClipboard(text)
		a.toast(i18n.T("Copied"))
		return true
	})
	if row.Shortcut(ui.Cmd, ui.KeyC) && text != "" {
		a.copyEntry(c, text)
	}
	if row.Shortcut(0, ui.KeyBackspace) || row.Shortcut(0, ui.KeyDelete) {
		a.askDeleteEntry(e.ID)
	}
}

// copyEntry copies an entry's text, as the details' copy buttons do.
func (a *app) copyEntry(c *ui.Context, text string) {
	mygoClipboard(c, text)
	a.toast(i18n.T("Copied"))
}

// askDeleteEntry opens the Delete dialog (historyDeleteDialog, on History
// and Home) for an entry.
func (a *app) askDeleteEntry(id string) {
	a.hist.delAsk, a.hist.delID = true, id
}
