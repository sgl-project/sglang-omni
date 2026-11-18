package main

// The context menu of a history entry, on History's rows and Home's recent results.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func entryMenu(c *ui.Context, row ui.Element, state *historyRowViewState, actions historyActions, local *historyLocalUI, keys bool, details func()) {
	row.ContextMenu(func(menu *ui.Menu) {
		copyItem := menu.Item(i18n.T("Copy")).Disabled(state.CopyText == "")
		if keys {
			copyItem.Shortcut(ui.Cmd, ui.KeyC)
		}
		if copyItem.Chosen() {
			actions.copyHistoryText(state.CopyText)
		}
		if state.RawCopyText != "" && menu.Item(i18n.T("Copy recognised text")).Chosen() {
			actions.copyHistoryText(state.RawCopyText)
		}
		if menu.Item(i18n.T("Show details")).Chosen() {
			details()
		}
		if state.CanReveal && menu.Item(i18n.T("Show recording in Finder")).Chosen() {
			actions.showHistoryRecording(state.ID)
		}
		if state.Retry.Available && menu.Item(state.Retry.Label).Disabled(state.Retry.Disabled).Chosen() {
			actions.retryHistoryEntry(state.ID)
		}
		menu.Separator()
		deleteItem := menu.Item(i18n.T("Delete"))
		if keys {
			deleteItem.Shortcut(0, ui.KeyBackspace)
		}
		if deleteItem.Chosen() {
			local.delAsk, local.delID = true, state.ID
		}
	})
	if !keys {
		return
	}
	row.HandleInput(func(event ui.InputEvent) bool {
		if event.Kind != ui.InputCommand || event.Text != "copy" || state.CopyText == "" {
			return false
		}
		actions.copyHistoryText(state.CopyText)
		return true
	})
	if row.Shortcut(ui.Cmd, ui.KeyC) && state.CopyText != "" {
		actions.copyHistoryText(state.CopyText)
	}
	if row.Shortcut(0, ui.KeyBackspace) || row.Shortcut(0, ui.KeyDelete) {
		local.delAsk, local.delID = true, state.ID
	}
}
