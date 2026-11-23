package main

// History text export asks for a destination and writes through historyapi.

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var chooseHistoryFile = func(a *app, name string, then func(string)) {
	go func() {
		extension := filepath.Ext(name)[1:]
		destination, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: i18n.T("Export text"), DefaultPath: name,
			Filters: []mygo.FileFilter{{Name: strings.ToUpper(extension), Extensions: []string{extension}}}})
		if err != nil || destination == "" {
			return
		}
		a.post(func() { then(destination) })
	}()
}

func (a *app) exportShownHistoryText(shown bool, format historyTextFormat) {
	apiFormat, name := historyapi.FormatMarkdown, "voxt-history.md"
	if format == historyTextJSONL {
		apiFormat, name = historyapi.FormatJSONL, "voxt-history.jsonl"
	}
	chooseHistoryFile(a, name, func(destination string) { a.exportHistoryText(destination, apiFormat, shown) })
}

func (a *app) exportHistoryText(destination, format string, shown bool) {
	var query store.HistoryQuery
	if shown {
		query.Kind, query.Search = a.histUI.kind(), a.histSearch()
	}
	history := a.lib.store.History()
	selection := a.prefs.Privacy.HistorySelection
	a.histSet.textBusy = true
	a.async(func() {
		count, err := historyapi.WriteFile(context.Background(), history, query, selection, destination, format)
		a.post(func() {
			a.histSet.textBusy = false
			if err != nil {
				a.toast(i18n.T("Could not export the text: %v", err))
				return
			}
			a.toast(i18n.N(count, "Exported %d entry to %s", "Exported %d entries to %s", count, filepath.Base(destination)))
		})
	})
}
