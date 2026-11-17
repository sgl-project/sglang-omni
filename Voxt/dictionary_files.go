package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// importTerms reads a CSV or text file and merges its terms in.
func (a *app) importTerms() {
	if a.dv.importing {
		return
	}
	chooseImportFile(a, func(path string) { a.importFile(path) })
}

// importFile merges the terms of the file at path into the dictionary.
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
				terms, skipped, err = parseLegacyExport(data) // legacy_import.go
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

// mergeImported merges imported terms into the dictionary: a file import
// and the previous Voxt's import share it.
func mergeImported(ctx context.Context, repo store.Dictionary, terms []store.Term) (added, updated int, err error) {
	have, err := repo.List(ctx, "")
	if err != nil {
		return 0, 0, err
	}
	put, added, updated := dict.Merge(have, terms)
	now := time.Now()
	for i, tm := range put {
		if tm.ID == "" {
			tm.ID, tm.CreatedAt = core.NewID(now), now.Add(time.Duration(i))
		}
		tm.UpdatedAt = now
		if err := repo.Put(ctx, tm); err != nil {
			return 0, 0, err
		}
	}
	return added, updated, nil
}

// exportTerms writes every term to a file the user picks.
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
