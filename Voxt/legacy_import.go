package main

// Importing the previous Voxt (com.voxt.Voxt 1.x, a sandboxed Swift app).
// Its dictionary lives in its container's voxt.sqlite (GRDB, WAL journal).
// macOS keeps other apps out of a container until the user picks the
// folder in an open panel, so the import asks for one pick of the
// container's Data/Library, preset in the panel; it then reads the
// database there read-only, with its -wal and -shm, and merges the terms
// like a file import. Nothing in the old app's data is changed.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/sqlite"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const legacyBundleID = "com.voxt.Voxt"

// legacyLibrary is the previous Voxt's Data/Library, or "" when it never
// ran on this Mac. Seeing the container needs no permission.
func legacyLibrary() string {
	home := os.Getenv("VOXTNEW_LEGACY_HOME") // automation: tests that run the app with a fake HOME (as VOXTNEW_HELPER)
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	container := filepath.Join(home, "Library", "Containers", legacyBundleID)
	if st, err := os.Stat(container); err != nil || !st.IsDir() {
		return ""
	}
	return filepath.Join(container, "Data", "Library")
}

// legacyDatabase is voxt.sqlite under a picked Data/Library, or under the
// Application Support or Voxt folder below it if the user picked those.
func legacyDatabase(picked string) string {
	for _, rel := range []string{
		filepath.Join("Application Support", "Voxt", "voxt.sqlite"),
		filepath.Join("Voxt", "voxt.sqlite"),
		"voxt.sqlite",
	} {
		if p := filepath.Join(picked, rel); fileExists(p) {
			return p
		}
	}
	return ""
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// readLegacyTerms reads the previous Voxt's dictionary: each entry's term,
// its replacement terms as aliases, and whether it is active.
func readLegacyTerms(ctx context.Context, path string) (terms []store.Term, skipped int, err error) {
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{ReadOnly: true})
	if err != nil {
		return nil, 0, err
	}
	defer db.Close()
	rows, err := db.Query(ctx, `SELECT e.id, e.term, e.status, r.text
		FROM dictionary_entries e LEFT JOIN dictionary_replacement_terms r ON r.entryID = e.id
		ORDER BY e.createdAt, e.id, r.rowid`)
	if err != nil {
		return nil, 0, fmt.Errorf("not a Voxt dictionary: %w", err)
	}
	index := map[string]int{}
	var all []store.Term
	for _, v := range rows.Values {
		id, _ := v[0].(string)
		term, _ := v[1].(string)
		status, _ := v[2].(string)
		i, ok := index[id]
		if !ok {
			i = len(all)
			index[id] = i
			all = append(all, store.Term{Term: term, Enabled: status == "active"})
		}
		if alias, _ := v[3].(string); alias != "" {
			all[i].Aliases = append(all[i].Aliases, alias)
		}
	}
	for _, t := range all {
		t = dict.Clean(t)
		if dict.Check(t) != nil {
			skipped++
			continue
		}
		terms = append(terms, t)
	}
	return terms, skipped, nil
}

// legacyExport is the previous Voxt's dictionary export (Dictionary >
// Export, "Voxt-Dictionary.json", version 3): its terms and their
// replacement terms. Its groups and categories have no place here.
type legacyExport struct {
	Version int `json:"version"`
	Entries []struct {
		Term             string   `json:"term"`
		ReplacementTerms []string `json:"replacementTerms"`
	} `json:"entries"`
}

// parseLegacyExport reads the previous Voxt's dictionary export as terms,
// the replacement terms as aliases; terms the dictionary would refuse are
// counted in skipped.
func parseLegacyExport(data []byte) (terms []store.Term, skipped int, err error) {
	var x legacyExport
	if err := json.Unmarshal(data, &x); err != nil {
		return nil, 0, fmt.Errorf("%w: %v", dict.ErrFormat, err)
	}
	if x.Entries == nil {
		return nil, 0, fmt.Errorf("%w: no entries", dict.ErrFormat)
	}
	for _, e := range x.Entries {
		t := dict.Clean(store.Term{Term: e.Term, Aliases: e.ReplacementTerms, Enabled: true})
		if dict.Check(t) != nil {
			skipped++
			continue
		}
		terms = append(terms, t)
	}
	return terms, skipped, nil
}

// legacyKinds maps the previous Voxt's history kinds to this app's; its
// "transcript" entries (meetings and file transcription) have no place here.
var legacyKinds = map[string]store.Kind{"normal": store.KindTranscribe, "translation": store.KindTranslate, "rewrite": store.KindRewrite}

// readLegacyHistory reads the previous Voxt's history: the text of each
// dictation, translation and rewrite, when, in which app and with which
// models. IDs are stable ("legacy-" + the old id), so importing again
// replaces instead of duplicating. Recordings stay with the previous Voxt.
func readLegacyHistory(ctx context.Context, path string) (entries []store.Entry, skipped int, err error) {
	db, err := sqlite.Open(ctx, path, sqlite.OpenOptions{ReadOnly: true})
	if err != nil {
		return nil, 0, err
	}
	defer db.Close()
	rows, err := db.Query(ctx, `SELECT id, text, createdAt, kind, COALESCE(focusedAppName, ''), COALESCE(focusedAppBundleID, ''),
		COALESCE(transcriptionModel, ''), COALESCE(enhancementModel, ''), COALESCE(audioDurationSeconds, 0)
		FROM history_entries ORDER BY createdAt`)
	if err != nil {
		if strings.Contains(err.Error(), "no such table") {
			return nil, 0, nil // an old database without history
		}
		return nil, 0, err
	}
	for _, v := range rows.Values {
		id, _ := v[0].(string)
		text, _ := v[1].(string)
		kind, ok := legacyKinds[fmt.Sprint(v[3])]
		if !ok || strings.TrimSpace(text) == "" {
			skipped++
			continue
		}
		app, _ := v[4].(string)
		bundle, _ := v[5].(string)
		asr, _ := v[6].(string)
		llm, _ := v[7].(string)
		entries = append(entries, store.Entry{
			ID: "legacy-" + id, Kind: kind, CreatedAt: time.Unix(0, int64(number(v[2])*1e9)), Status: store.StatusDone,
			AppName: app, AppBundle: bundle, RawText: text, Result: text, ASRModel: asr, LLMModel: llm,
			AudioMS: int64(number(v[8]) * 1000),
		})
	}
	return entries, skipped, nil
}

// number is a SQLite REAL or INTEGER as a float64.
func number(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int64:
		return float64(n)
	}
	return 0
}

// reuseLegacyModels brings the models the previous Voxt downloaded into the
// library when its files are the catalogue's: every file of the pinned
// commit at its size, the large ones also at their SHA-256. The previous
// Voxt kept each model flat in model-storage/<owner>_<name>; the library
// wants $HUB/models--<owner>--<name>/snapshots/<commit>. Files are hard
// linked (copied across volumes), so nothing is downloaded and no space is
// used twice. It returns the models brought and their bytes.
func reuseLegacyModels(ctx context.Context, m *models.Manager, voxtDir string) (ids []string, bytes int64) {
	root := filepath.Join(voxtDir, "model-storage")
	if st, err := os.Stat(root); err != nil || !st.IsDir() {
		return nil, 0
	}
	for _, mod := range models.Catalog() {
		if mod.Borrowed || len(mod.Files) == 0 || m.Installed(mod.ID) || ctx.Err() != nil {
			continue
		}
		src := filepath.Join(root, strings.ReplaceAll(mod.Repo, "/", "_"))
		if !legacyModelMatches(src, mod) {
			continue
		}
		dst := models.SnapshotDir(m.Hub(), mod, mod.Revision)
		n, err := linkModelFiles(src, dst, mod)
		if err != nil {
			log.Printf("legacy import: model %s: %v", mod.ID, err)
			continue
		}
		ids, bytes = append(ids, mod.ID), bytes+n
	}
	if len(ids) > 0 {
		m.Refresh()
		if err := m.Use(ctx, ids...); err != nil {
			log.Printf("legacy import: claiming %v: %v", ids, err)
		}
	}
	return ids, bytes
}

// legacyModelMatches reports whether dir holds every file of mod's pinned
// commit, at its size and, for the large (LFS) ones, its SHA-256.
func legacyModelMatches(dir string, mod models.Model) bool {
	for _, f := range mod.Files {
		p := filepath.Join(dir, filepath.FromSlash(f.Path))
		st, err := os.Stat(p)
		if err != nil || st.IsDir() || st.Size() != f.Size {
			return false
		}
		if f.LFS && f.SHA256 != "" {
			sum, err := fileSHA256(p)
			if err != nil || sum != f.SHA256 {
				return false
			}
		}
	}
	return true
}

func fileSHA256(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// linkModelFiles puts mod's files from src into the snapshot folder dst:
// a hard link, or a copy when the two are on different volumes.
func linkModelFiles(src, dst string, mod models.Model) (bytes int64, err error) {
	for _, f := range mod.Files {
		from, to := filepath.Join(src, filepath.FromSlash(f.Path)), filepath.Join(dst, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
			return 0, err
		}
		os.Remove(to)
		if err := os.Link(from, to); err != nil {
			if err := copyFile(from, to); err != nil {
				return 0, err
			}
		}
		bytes += f.Size
	}
	return bytes, nil
}

func copyFile(from, to string) error {
	in, err := os.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(to)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// importLegacy asks for the previous Voxt's data folder (one click: the
// panel opens on it) and merges its dictionary in.
func (a *app) importLegacy() {
	lib := legacyLibrary()
	if lib == "" || a.dv.importing {
		return
	}
	go func() {
		files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import from the previous Voxt"),
			Message:     i18n.T("This is the previous Voxt's data. Click Import to bring its dictionary over; the previous Voxt keeps everything as it is."),
			ButtonLabel: i18n.T("Import"), DefaultPath: lib, Directory: true})
		if err != nil || len(files) == 0 {
			return
		}
		a.post(func() { a.importLegacyFrom(files[0]) })
	}()
}

func (a *app) importLegacyFrom(picked string) {
	a.dv.importing = true
	repo, hist, mods := a.lib.store.Dictionary(), a.lib.store.History(), a.models
	a.async(func() {
		msg := func() string {
			path := legacyDatabase(picked)
			if path == "" {
				return i18n.T("No previous Voxt dictionary in this folder. Choose the folder the import window opens on.")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute) // hashing the models' large files
			defer cancel()
			terms, skipped, err := readLegacyTerms(ctx, path)
			if err != nil {
				log.Printf("legacy import: %s: %v", path, err)
				if strings.Contains(err.Error(), "no such table") {
					return i18n.T("This folder is not the previous Voxt's. Nothing was imported.")
				}
				return i18n.T("The previous Voxt's dictionary could not be read; it may be damaged. Nothing was imported.")
			}
			added, updated, err := mergeImported(ctx, repo, terms)
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			msg := i18n.T("From the previous Voxt: %d new, %d updated.", added, updated)
			if skipped > 0 {
				msg = i18n.T("From the previous Voxt: %d new, %d updated; %d left out (too long or empty).", added, updated, skipped)
			}
			entries, _, err := readLegacyHistory(ctx, path)
			if err != nil {
				log.Printf("legacy import: history: %v", err)
				return msg
			}
			n := 0
			for _, e := range entries {
				if _, err := hist.Get(ctx, e.ID); errors.Is(err, core.ErrNotFound) {
					n++
				}
				if err := hist.Put(ctx, e); err != nil {
					log.Printf("legacy import: history %s: %v", e.ID, err)
				}
			}
			if n > 0 {
				msg += " " + i18n.T("History: %d entries added.", n)
			}
			if ids, bytes := reuseLegacyModels(ctx, mods, filepath.Dir(path)); len(ids) > 0 {
				msg += " " + i18n.N(len(ids), "%d model (%s) came over too: no download needed.", "%d models (%s) came over too: no download needed.", len(ids), uitext.HumanBytes(bytes))
			}
			return msg
		}()
		a.post(func() {
			a.dv.importing = false
			a.modelsChanged()
			a.ob.legacyMsg = msg
			a.toast(msg)
			a.loadDict()
		})
	})
}
