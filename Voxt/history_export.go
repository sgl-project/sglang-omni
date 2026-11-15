package main

// The History settings dialog's Export text (docs/history-access.md):
// the same rows as `Voxt history export`, to a file the user picks,
// whatever the Privacy switch says, since the user does this by hand.

import (
	"context"
	"path/filepath"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// chooseHistoryFile asks where to save; tests replace it.
var chooseHistoryFile = func(a *app, name string, then func(string)) {
	go func() {
		ext := filepath.Ext(name)[1:]
		dst, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: i18n.T("Export text"), DefaultPath: name,
			Filters: []mygo.FileFilter{{Name: strings.ToUpper(ext), Extensions: []string{ext}}}})
		if err != nil || dst == "" {
			return
		}
		a.post(func() { then(dst) })
	}()
}

// histExportFormats are Export text's choices; proper names, untranslated.
var histExportFormats = []string{"Markdown", "JSONL"}

// histExportTextRow is the dialog's Export text row.
func (a *app) histExportTextRow(c *ui.Context) {
	t := c.Theme()
	hs := &a.histSet
	settingRow(c, "hist-set-text", i18n.T("Export text"), "", func() {
		scopes := []string{i18n.T("All entries")}
		if l := a.shownLabel(); l != "" {
			scopes = append(scopes, l)
		}
		if !slices.Contains(scopes, hs.textScope) {
			hs.textScope = scopes[0]
		}
		if !slices.Contains(histExportFormats, hs.textFormat) {
			hs.textFormat = histExportFormats[0]
		}
		ui.Select(c.Key("hist-set-text-scope"), &hs.textScope, scopes).Label(i18n.T("Which entries")).Width(px(t, 150)).Shrink(1).MinWidth(px(t, 100))
		ui.Select(c.Key("hist-set-text-format"), &hs.textFormat, histExportFormats).Label(i18n.T("Format")).Width(px(t, 110)).Shrink(0)
		if ui.Button(c.Key("hist-set-text-go"), i18n.T("Export text…")).Disabled(hs.textBusy).Clicked() {
			format, name := historyapi.FormatMarkdown, "voxt-history.md"
			if hs.textFormat == "JSONL" {
				format, name = historyapi.FormatJSONL, "voxt-history.jsonl"
			}
			shown := len(scopes) > 1 && hs.textScope == scopes[1]
			chooseHistoryFile(a, name, func(dst string) { a.exportHistoryText(dst, format, shown) })
		}
	})
}

// exportHistoryText writes the entries, or those the History page shows,
// to dst.
func (a *app) exportHistoryText(dst, format string, shown bool) {
	var q store.HistoryQuery
	if shown {
		q.Kind, q.Search = a.hist.kind(), a.histSearch()
	}
	h := a.lib.store.History()
	selection := a.prefs.Privacy.HistorySelection
	a.histSet.textBusy = true
	a.async(func() {
		n, err := historyapi.WriteFile(context.Background(), h, q, selection, dst, format)
		a.post(func() {
			a.histSet.textBusy = false
			if err != nil {
				a.toast(i18n.T("Could not export the text: %v", err))
				return
			}
			a.toast(i18n.N(n, "Exported %d entry to %s", "Exported %d entries to %s", n, filepath.Base(dst)))
		})
	})
}
