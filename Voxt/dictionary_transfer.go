package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

var (
	chooseImportFile = func(a *app, then func(string)) {
		go func() {
			files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import terms"),
				Filters: []mygo.FileFilter{{Name: i18n.T("Dictionary (CSV, text or Voxt 1.x export)"), Extensions: []string{"csv", "txt", "json"}}}})
			if err != nil || len(files) == 0 {
				return
			}
			a.post(func() { then(files[0]) })
		}()
	}
	chooseExportFile = func(a *app, title, name string, then func(string)) {
		go func() {
			dst, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: title, DefaultPath: name,
				Filters: []mygo.FileFilter{{Name: strings.ToUpper(filepath.Ext(name)[1:]), Extensions: []string{filepath.Ext(name)[1:]}}}})
			if err != nil || dst == "" {
				return
			}
			a.post(func() { then(dst) })
		}()
	}
)

func (a *app) exportTerms(format string) {
	name := "voxtnew-dictionary." + format
	chooseExportFile(a, i18n.T("Export terms"), name, func(dst string) { a.exportFile(dst, format) })
}

func (a *app) exportFile(dst, format string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		terms, err := repo.List(ctx, "")
		if err == nil {
			err = os.WriteFile(dst, dict.Export(terms, format), 0o644)
		}
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not export: %v", err))
				return
			}
			a.toast(i18n.N(len(terms), "Exported %d term to %s", "Exported %d terms to %s", len(terms), filepath.Base(dst)))
		})
	})
}
