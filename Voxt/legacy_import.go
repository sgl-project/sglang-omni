package main

// Importing the previous Voxt (com.voxt.Voxt 1.x, a sandboxed Swift app).
// Its dictionary lives in its container's voxt.sqlite (GRDB, WAL journal).
// macOS keeps other apps out of a container until the user picks the
// folder in an open panel, so the import asks for one pick of the
// container's Data/Library, preset in the panel; it then reads the
// database there read-only, with its -wal and -shm, and merges the terms
// like a file import, then the history, the settings that have a place
// here (legacy_settings.go) and the speech models. Nothing in the old
// app's data is changed.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

// legacyReadable reports whether the database in lib opens and reads without
// asking: so it does for a build signed by the previous Voxt's team
// (RY2TQTYSKQ), as macOS lets one team's apps read each other's
// containers. Other builds get EPERM, ad hoc ones without a prompt, and ask
// with the open panel. It reads a byte, as the container may let a file be
// found and not opened.
func legacyReadable(lib string) bool {
	path := legacyDatabase(lib)
	if path == "" {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	_, err = f.Read(make([]byte, 1))
	return err == nil
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

// linkImportLegacy is the link of the card that offers the previous Voxt's
// data (offerLegacy): its action is the import.
const linkImportLegacy = "action:import-legacy"

// legacyOffered is that card's id, and the document that says it was
// posted: once, so a dismissed card stays away.
const legacyOffered = "legacy-import-offered"

// offerLegacy posts, once, a card offering the previous Voxt's data to who
// will not see the setup guide's: who set up a version without the import
// (2.0.0) and then installed this one.
func (a *app) offerLegacy() {
	if a.notes == nil || !a.prefs.Setup.Done || legacyLibrary() == "" {
		return
	}
	docs, ctx := a.lib.store.Documents(), context.Background()
	if _, posted, err := docs.Get(ctx, legacyOffered); err != nil || posted {
		return
	}
	if err := docs.Put(ctx, legacyOffered, []byte("true")); err != nil {
		log.Printf("legacy import: %v", err)
		return
	}
	a.notes.Post(core.Notification{ID: legacyOffered, Title: i18n.T("The previous Voxt is on this Mac"),
		Body: i18n.T("Bring its dictionary, history and settings over."), Link: linkImportLegacy})
}

// importLegacy merges the previous Voxt's data in: at once when it reads
// without asking (legacyReadable), else after the user picks its data folder
// (one click: the panel opens on it).
func (a *app) importLegacy() {
	lib := legacyLibrary()
	if lib == "" || a.dv.importing {
		return
	}
	if legacyReadable(lib) {
		a.importLegacyFrom(lib)
		return
	}
	go func() {
		files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import from the previous Voxt"),
			Message:     i18n.T("This is the previous Voxt's data. Click Import to bring its dictionary, history and settings over; the previous Voxt keeps everything as it is."),
			ButtonLabel: i18n.T("Import"), DefaultPath: lib, Directory: true})
		if err != nil || len(files) == 0 {
			return
		}
		a.post(func() { a.importLegacyFrom(files[0]) })
	}()
}

func (a *app) importLegacyFrom(picked string) {
	a.dv.importing = true
	repo, hist, mods, docs, prefs := a.lib.store.Dictionary(), a.lib.store.History(), a.models, a.lib.store.Documents(), a.settings
	a.async(func() {
		done := false // the dictionary came over: the card offers the import again, quietly
		report := func() legacyReport {
			path := legacyDatabase(picked)
			if path == "" {
				return legacyReport{failed: i18n.T("No previous Voxt dictionary in this folder. Choose the folder the import window opens on.")}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute) // hashing the models' large files
			defer cancel()
			terms, skipped, err := readLegacyTerms(ctx, path)
			if err != nil {
				log.Printf("legacy import: %s: %v", path, err)
				if strings.Contains(err.Error(), "no such table") {
					return legacyReport{failed: i18n.T("This folder is not the previous Voxt's. Nothing was imported.")}
				}
				return legacyReport{failed: i18n.T("The previous Voxt's dictionary could not be read; it may be damaged. Nothing was imported.")}
			}
			added, updated, err := mergeImported(ctx, repo, terms)
			if err != nil {
				return legacyReport{failed: i18n.T("Could not import: %v", err)}
			}
			done = true
			if err := docs.Put(ctx, legacyOffered, []byte("true")); err != nil { // imported: offerLegacy's card is not wanted any more
				log.Printf("legacy import: %v", err)
			}
			var r legacyReport
			r.imported = append(r.imported, i18n.T("Dictionary: %d new, %d updated", added, updated))
			if skipped > 0 {
				r.skipped = append(r.skipped, i18n.N(skipped, "%d dictionary term: too long or empty", "%d dictionary terms: too long or empty", skipped))
			}
			entries, histSkipped, err := readLegacyHistory(ctx, path)
			if err != nil {
				log.Printf("legacy import: history: %v", err)
				return r
			}
			if histSkipped > 0 {
				r.skipped = append(r.skipped, i18n.N(histSkipped, "%d history entry: a meeting or empty", "%d history entries: meetings or empty", histSkipped))
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
				r.imported = append(r.imported, i18n.N(n, "History: %d entry", "History: %d entries", n))
			}
			voxtDir := filepath.Dir(path)
			l := parseLegacySettings(legacyPreference(voxtDir, "remoteLLMProviderConfigurations"), legacyPreference(voxtDir, "remoteASRProviderConfigurations"),
				legacyPreference(voxtDir, "featureSettings"), legacyPreference(voxtDir, "mlxLocalASRTuningSettings"))
			// The models first: a feature keeps its local speech model when it came over.
			if ids, bytes := reuseLegacyModels(ctx, mods, voxtDir); len(ids) > 0 {
				n := 0 // the tokenizers are not counted
				for _, id := range ids {
					if m, _ := models.Lookup(id); m.Kind != models.Aux {
						n++
					}
				}
				r.imported = append(r.imported, i18n.N(n, "%d speech model (%s): no download needed", "%d speech models (%s): no download needed", n, uitext.HumanBytes(bytes)))
			}
			for i, id := range l.featureLocal {
				if id != "" && !mods.Installed(id) {
					l.featureLocal[i] = ""
				}
			}
			var llmServices, speechServices []settings.Provider
			prompts := 0
			if err := prefs.Update(ctx, func(s *settings.Settings) {
				llmServices, speechServices, prompts = l.apply(s)
				legacyApplyGeneral(voxtDir, s)
			}); err != nil {
				log.Printf("legacy import: settings: %v", err)
				llmServices, speechServices, prompts = nil, nil, 0 // not saved: nothing came over
			}
			if n := len(llmServices); n > 0 {
				r.imported = append(r.imported, i18n.N(n, "%d language model service", "%d language model services", n))
			}
			if n := len(speechServices); n > 0 {
				r.imported = append(r.imported, i18n.N(n, "%d speech service", "%d speech services", n))
			}
			if prompts > 0 {
				r.imported = append(r.imported, i18n.N(prompts, "%d prompt", "%d prompts", prompts))
			}
			// The keys stay in the previous Voxt's Keychain: each service
			// that needs one is to complete.
			for _, p := range llmServices {
				if missing := providers.Missing(p, false); missing != "" {
					r.todo = append(r.todo, legacyTodo{text: setupNeedsText(local.LLM, p, missing), kind: local.LLM, id: p.ID})
				}
			}
			for _, p := range speechServices {
				if !providers.SpeechKeyOptional(p) {
					r.todo = append(r.todo, legacyTodo{text: setupNeedsText(local.ASR, p, providers.MissingKey), kind: local.ASR, id: p.ID})
				}
			}
			llmCheck, speechCheck := l.toCheck(llmServices, speechServices)
			for _, p := range llmCheck {
				r.todo = append(r.todo, legacyTodo{text: i18n.T("Check %s: its address or model may not work as it did in the previous Voxt.", uitext.PresetName(p.ID)), kind: local.LLM, id: p.ID})
			}
			for _, p := range speechCheck {
				r.todo = append(r.todo, legacyTodo{text: i18n.T("Check %s: its address or model may not work as it did in the previous Voxt.", uitext.SpeechName(p.ID)), kind: local.ASR, id: p.ID})
			}
			return r
		}()
		a.post(func() {
			a.dv.importing = false
			a.modelsChanged()
			a.ob.legacy, a.ob.legacyDone = report, done
			if done {
				a.dismissNotice(legacyOffered) // the sidebar card, if it shows
			}
			switch {
			case report.failed != "":
				a.toast(report.failed)
			case !a.ob.open: // the guide's card shows it
				a.ob.legacyOpen = true // legacy_report.go
			}
			log.Printf("legacy import: %s", report.text())
			a.loadDict()
		})
	})
}
