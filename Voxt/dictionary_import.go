package main

// Importing terms into the dictionary. Dictionary → Import… (importTerms)
// and `Voxt dictionary import` (dictionary_socket.go) share the parsing
// and the merge, so a file gives the same terms either way.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// withDictionaryLock serializes a read-modify-write with every other
// dictionary writer, so an import cannot overwrite a page edit or deletion.
func withDictionaryLock(repo store.Dictionary, write func() error) error {
	repo.Lock()
	defer repo.Unlock()
	return write()
}

// errImportFormat: the file is not one importTerms reads.
var errImportFormat = errors.New("not a CSV or text list of terms, or a Voxt 1.x dictionary export")

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
			file, err := os.Open(path)
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			data, readErr := readDictionaryData(file)
			closeErr := file.Close()
			if readErr != nil {
				err = readErr
			} else {
				err = closeErr
			}
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			if len(data) > dictapi.MaxBytes {
				return i18n.T("Could not import: %v", fmt.Errorf("the file is larger than %d MB", dictapi.MaxBytes>>20))
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			res, err := importData(ctx, repo, filepath.Base(path), data, false)
			if errors.Is(err, errImportFormat) {
				return i18n.T("Could not import %s: it is not a CSV or text list of terms, or a Voxt 1.x dictionary export.", filepath.Base(path))
			}
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			return importMessage(res)
		}()
		a.post(func() {
			a.dv.importing = false
			a.toast(msg)
			a.loadDict()
		})
	})
}

// importMessage is the toast of an import that saved.
func importMessage(res dictapi.Result) string {
	if res.Skipped > 0 {
		return i18n.T("Imported: %d new, %d updated, %d rows skipped.", res.Added, res.Updated, res.Skipped)
	}
	return i18n.T("Imported: %d new, %d updated.", res.Added, res.Updated)
}

// parseImport reads the terms of the file called name, by its extension:
// .json a Voxt 1.x export (legacy_import.go), .txt text, else CSV.
func parseImport(name string, data []byte) (terms []store.Term, skipped int, err error) {
	if strings.EqualFold(filepath.Ext(name), ".json") {
		terms, skipped, err = parseLegacyExport(data)
	} else {
		terms, skipped, err = dict.Import(data, dict.FormatOf(name))
	}
	if err != nil {
		return nil, 0, fmt.Errorf("%s is %w (%v)", name, errImportFormat, err)
	}
	return terms, skipped, nil
}

// importData merges the terms of the file called name into repo, or with
// dryRun only counts what the merge would add and update.
func importData(ctx context.Context, repo store.Dictionary, name string, data []byte, dryRun bool) (dictapi.Result, error) {
	terms, skipped, err := parseImport(name, data)
	if err != nil {
		return dictapi.Result{}, err
	}
	res := dictapi.Result{Skipped: skipped}
	if !dryRun {
		res.Added, res.Updated, err = mergeImported(ctx, repo, terms)
		return res, err
	}
	repo.Lock()
	defer repo.Unlock()
	have, err := repo.List(ctx, "")
	if err != nil {
		return dictapi.Result{}, err
	}
	_, res.Added, res.Updated = dict.Merge(have, terms)
	return res, nil
}

// mergeImported merges imported terms into the dictionary: a file import
// and the previous Voxt's import share it.
func mergeImported(ctx context.Context, repo store.Dictionary, terms []store.Term) (added, updated int, err error) {
	repo.Lock()
	defer repo.Unlock()
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
