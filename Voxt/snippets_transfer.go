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
		a.toast(i18n.T("Could not read the snippets, so nothing was imported."))
		return
	}
	chooseSnippetsFile(a, a.importSnippetsFile)
}

// importSnippetsFile adds the snippets in the file at path, from the
// import dialog's file chooser or a drop on it. A file that imports
// closes the dialog and says how many came in, in the middle of the
// page; the problem of one that does not shows in the dialog, or as a
// toast once it is closed.
func (a *app) importSnippetsFile(path string) {
	a.async(func() {
		in, skipped, problem := readSnippetsFile(path)
		a.post(func() {
			if problem == "" && !a.snip.loaded {
				problem = i18n.T("Could not read the snippets, so nothing was imported.")
			}
			if problem != "" {
				if a.snipUI.importOpen {
					a.snipUI.importErr, a.snipUI.importExample = problem, true
					return
				}
				a.toast(problem)
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
			a.snipUI.importOpen, a.snipUI.importErr = false, ""
			msg, d := i18n.T("Imported snippets: %d new, %d updated.", added, updated), snippetsImportNoticeFor
			if skipped > 0 {
				msg, d = i18n.T("Imported snippets: %d new, %d updated, %d skipped.", added, updated, skipped), snippetsImportNoticeSkippedFor
			}
			// Shown by snippetsImportNotice.
			a.snipUI.notice, a.snipUI.noticeAt, a.snipUI.noticeFor = msg, time.Time{}, d
		})
	})
}

// readSnippetsFile reads the snippets in path, or says why it could not.
func readSnippetsFile(path string) (in []snippet.Snippet, skipped int, problem string) {
	name := filepath.Base(path)
	info, err := os.Stat(path)
	if err == nil && info.Size() > snippet.MaxImportBytes {
		return nil, 0, i18n.T("Could not import %s: it is larger than %d MB.", name, snippet.MaxImportBytes>>20)
	}
	var data []byte
	if err == nil {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, 0, i18n.T("Could not open %s.", name)
	}
	if in, skipped, err = snippet.Import(data); err != nil {
		return nil, 0, i18n.T("Could not import %s: it is not a JSON list of snippets.", name)
	}
	if len(in) == 0 {
		return nil, 0, i18n.T("Found no snippets Voxt can use in %s.", name)
	}
	return in, skipped, ""
}
