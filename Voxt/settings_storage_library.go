package main

// Settings → Storage → Another Mac: one file with the settings, history,
// meetings, dictionary and snippets (internal/datadir/library.go).

import (
	"context"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

var chooseLibraryFile = func(a *app, then func(string)) {
	go func() {
		files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import library"),
			Filters: []mygo.FileFilter{{Name: i18n.T("Voxt library"), Extensions: []string{"voxtlibrary"}}}})
		if err != nil || len(files) == 0 {
			return
		}
		a.post(func() { then(files[0]) })
	}()
}

func (a *app) exportLibrary() {
	name := "Voxt-" + time.Now().Format("2006-01-02") + ".voxtlibrary"
	chooseExportFile(a, i18n.T("Export library"), name, func(dst string) {
		dir := a.dataDir()
		a.async(func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			err := datadir.ExportLibrary(ctx, dir, dst)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not export: %v", err))
					return
				}
				a.toast(i18n.T("Exported the library to %s", filepath.Base(dst)))
			})
		})
	})
}

func (a *app) importLibrary() {
	chooseLibraryFile(a, func(src string) {
		dir := a.dataDir()
		a.async(func() {
			err := datadir.StageImport(context.Background(), dir, src)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not import: %v", err))
					return
				}
				appRelaunch() // the start swaps the library in (datadir.Resolve)
			})
		})
	})
}
