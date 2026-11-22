package main

// Importing snippets from a JSON file (snippet.Import): Wispr Flow's and
// Raycast's exports, and Voxt's own list.

import (
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

var chooseSnippetsFile = func(a *app, then func(string)) {
	go func() {
		files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import snippets"),
			Filters: []mygo.FileFilter{{Name: i18n.T("Snippets (JSON)"), Extensions: []string{"json"}}}})
		if err != nil || len(files) == 0 {
			return
		}
		a.post(func() { then(files[0]) })
	}()
}

// importSnippets asks for a file and adds its snippets: one whose
// trigger is already there changes that one's text.
func (a *app) importSnippets() {
	if !a.snip.loaded {
		return
	}
	chooseSnippetsFile(a, a.importSnippetsFile)
}

func (a *app) importSnippetsFile(path string) {
	a.async(func() {
		var data []byte
		info, err := os.Stat(path)
		if err == nil && info.Size() > snippet.MaxImportBytes {
			err = snippet.ErrNotSnippets
		}
		if err == nil {
			data, err = os.ReadFile(path)
		}
		var in []snippet.Snippet
		var skipped int
		if err == nil {
			in, skipped, err = snippet.Import(data)
		}
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not import %s: it is not a JSON list of snippets.", filepath.Base(path)))
				return
			}
			n := time.Now().UnixNano()
			list, added, updated := snippet.Merge(a.snip.list, in, func() string {
				n++
				return strconv.FormatInt(n, 36)
			})
			a.snip.list = list
			if added+updated > 0 {
				a.saveSnippets()
			}
			if skipped > 0 {
				a.toast(i18n.T("Imported snippets: %d new, %d updated, %d skipped.", added, updated, skipped))
				return
			}
			a.toast(i18n.T("Imported snippets: %d new, %d updated.", added, updated))
		})
	})
}
