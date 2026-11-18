package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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

func (a *app) importTerms() {
	if a.dv.importing {
		return
	}
	chooseImportFile(a, func(path string) { a.importFile(path) })
}

func (a *app) importFile(path string) {
	a.dv.importing = true
	repo := a.lib.store.Dictionary()
	a.async(func() {
		msg := func() string {
			data, err := os.ReadFile(path)
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			var terms []store.Term
			var skipped int
			if strings.EqualFold(filepath.Ext(path), ".json") {
				terms, skipped, err = parseLegacyExport(data)
			} else {
				terms, skipped, err = dict.Import(data, dict.FormatOf(path))
			}
			if err != nil {
				return i18n.T("Could not import %s: it is not a CSV or text list of terms, or a Voxt 1.x dictionary export.", filepath.Base(path))
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			added, updated, err := mergeImported(ctx, repo, terms)
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			if skipped > 0 {
				return i18n.T("Imported: %d new, %d updated, %d rows skipped.", added, updated, skipped)
			}
			return i18n.T("Imported: %d new, %d updated.", added, updated)
		}()
		a.post(func() {
			a.dv.importing = false
			a.toast(msg)
			a.loadDict()
		})
	})
}

func mergeImported(ctx context.Context, repo store.Dictionary, terms []store.Term) (added, updated int, err error) {
	have, err := repo.List(ctx, "")
	if err != nil {
		return 0, 0, err
	}
	put, added, updated := dict.Merge(have, terms)
	now := time.Now()
	for index, term := range put {
		if term.ID == "" {
			term.ID, term.CreatedAt = core.NewID(now), now.Add(time.Duration(index))
		}
		term.UpdatedAt = now
		if err := repo.Put(ctx, term); err != nil {
			return 0, 0, err
		}
	}
	return added, updated, nil
}

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
