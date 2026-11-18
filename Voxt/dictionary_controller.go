package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type dictionaryState struct {
	dv              dictView
	dictUI          dictionaryLocalUI
	dictProfiles    dictionaryProfileState
	book            *learn.Book
	suggestions     []learn.Suggestion
	suggestionReads int
	scanner         *learn.Scanner
	selPress        *selectionPress
	// glossaryTerms is the whole dictionary for the feature pages' prompt
	// preview (feature_prompts.go), read again after each change;
	// glossaryReads numbers the reads: only the last one lands.
	glossaryTerms []store.Term
	glossaryReads int
}

type dictView struct {
	list            []store.Term
	listFor         string
	read            bool
	seq             int
	loading         bool
	err             string
	total           int
	importing       bool
	legacyAvailable bool
	legacyReads     int
	scan            scanState
	termRows        []dictionaryTermViewState
	termRowsKey     dictionaryTermRowsCacheKey
	view            dictionaryViewState
	viewKey         dictionaryViewCacheKey
}

type dictionaryTermRowsCacheKey struct {
	seq     int
	locale  string
	loading bool
}

// dictionaryViewCacheKey is every input of a dictionaryViewState: when it
// matches the last build's, dictionaryPageModel reuses the state instead
// of rebuilding it every frame.
type dictionaryViewCacheKey struct {
	seq           int
	read          bool
	loading       bool
	err           string
	listFor       string
	total         int
	legacy        bool
	query         string
	tab           int
	scanOpen      bool
	suggestions   string
	settings      uint64
	locale        string
	scanModelID   string
	scanModelName string
	canScan       bool
	scan          dictionaryScanCacheKey
	edit          dictionaryEditCacheKey
	notesText     string
	notesQuery    string
	notesFind     int
	profiles      string
	profileEdit   dictionaryProfileEdit
	profileDelete string
}

type dictionaryScanCacheKey struct {
	modelID, modelName         string
	counted, running, stopping bool
	ended, stopped             bool
	count, read, total, found  int
	failed, failedType         string
}

type dictionaryEditCacheKey struct {
	open                    bool
	id, term, aliases, note string
	enabled                 bool
	err                     string
	saving                  bool
	original, suggestion    string
	profile                 string
}

func dictionaryScanCacheKeyFor(state scanState) dictionaryScanCacheKey {
	key := dictionaryScanCacheKey{
		modelID: state.model.ID, modelName: state.model.Name,
		counted: state.counted, running: state.running, stopping: state.stopping,
		ended: state.ended, stopped: state.stopped,
		count: state.count, read: state.read, total: state.total, found: state.found,
	}
	if state.failed != nil {
		key.failed, key.failedType = state.failed.Error(), fmt.Sprintf("%T", state.failed)
	}
	return key
}

// dictionarySuggestionsFingerprint is the suggestions' content as one
// comparable string: loadSuggestions swaps the slice when a read lands,
// and tests seed a.suggestions directly, so a read counter alone can go
// stale.
func dictionarySuggestionsFingerprint(suggestions []learn.Suggestion) string {
	if len(suggestions) == 0 {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%d", len(suggestions))
	for _, suggestion := range suggestions {
		fmt.Fprintf(&b, "|%s\x00%s\x00%s", suggestion.Key(), strings.Join(suggestion.Heard, "\x00"), suggestion.Source)
	}
	return b.String()
}

func dictionaryEditCacheKeyFor(local dictionaryLocalUI) dictionaryEditCacheKey {
	edit := local.edit
	return dictionaryEditCacheKey{
		open: local.editOpen, id: edit.id, term: edit.term, aliases: edit.aliases, note: edit.note,
		enabled: edit.enabled, err: edit.err, saving: edit.saving, original: edit.original, suggestion: edit.suggestion,
		profile: edit.profile,
	}
}

func (a *app) dictShown() {
	a.dv.legacyReads++
	read := a.dv.legacyReads
	a.async(func() {
		available := legacyLibrary() != ""
		a.post(func() {
			if read == a.dv.legacyReads {
				a.dv.legacyAvailable = available
			}
		})
	})
	a.loadDict()
}

func (a *app) prepareDictionaryPage() {
	if !a.dv.read && !a.dv.loading {
		a.loadDict()
	}
	notes := &a.dictUI.notes
	if saved := a.prefs.Dictionary.Notes; !notes.dirty && saved != notes.seen {
		notes.text, notes.seen = saved, saved
	}
}

func (a *app) loadDict() {
	d := &a.dv
	d.seq++
	d.loading = true
	seq, query := d.seq, strings.TrimSpace(a.dictUI.query)
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.List(ctx, query)
		a.post(func() {
			d := &a.dv
			if seq != d.seq {
				return
			}
			d.loading, d.read, d.listFor = false, true, query
			if err != nil {
				d.err = err.Error()
				return
			}
			d.list, d.err = list, ""
			// The glossary cache (loadGlossaryTerms in feature_prompts.go)
			// follows every reload, so the prompt preview does too.
			if query == "" {
				d.total = len(list)
				a.glossaryReads++
				a.glossaryTerms = list
			} else {
				a.loadGlossaryTerms()
			}
		})
	})
}

func (a *app) searchDictionary(query string) {
	a.dictUI.query = query
	a.loadDict()
}

func (a *app) dictionaryPageModel() dictionaryViewState {
	translator := i18n.CurrentTranslator()
	key := dictionaryTermRowsCacheKey{seq: a.dv.seq, locale: translator.Locale(), loading: a.dv.loading}
	if key != a.dv.termRowsKey {
		a.dv.termRows = buildDictionaryTermViewStates(translator, a.dv.list)
		a.dv.termRowsKey = key
	}
	installed := a.installed(local.LLM)
	scanModel, canScan := dictionaryScanModel(a.prefs, installed)
	viewKey := dictionaryViewCacheKey{
		seq: a.dv.seq, read: a.dv.read, loading: a.dv.loading, err: a.dv.err, listFor: a.dv.listFor,
		total: a.dv.total, legacy: a.dv.legacyAvailable, query: a.dictUI.query, tab: a.dictUI.tab, scanOpen: a.dictUI.scanOpen,
		suggestions: dictionarySuggestionsFingerprint(a.suggestions), settings: a.settingsRevision, locale: translator.Locale(),
		scanModelID: scanModel.ID, scanModelName: scanModel.Name, canScan: canScan,
		scan: dictionaryScanCacheKeyFor(a.dv.scan), edit: dictionaryEditCacheKeyFor(a.dictUI),
		notesText: a.dictUI.notes.text, notesQuery: a.dictUI.notesQuery, notesFind: a.dictUI.notesFind.cur,
		profiles: dictionaryProfilesFingerprint(a.dictProfiles.list), profileEdit: a.dictUI.profileEdit,
	}
	if a.dictUI.profileDeleteOpen {
		viewKey.profileDelete = a.dictUI.profileDeleteID
	}
	if viewKey != a.dv.viewKey {
		input := dictionaryViewStateInput{
			TermsReady: true, TermRows: a.dv.termRows, Suggestions: a.suggestions, Settings: a.prefs,
			Installed: installed, Read: a.dv.read, Loading: a.dv.loading, Error: a.dv.err,
			Query: a.dictUI.query, ListFor: a.dv.listFor, Total: a.dv.total, LegacyImport: a.dv.legacyAvailable,
			Tab: a.dictUI.tab, ScanOpen: a.dictUI.scanOpen, Scan: dictionaryScanInput(a.dv.scan),
			Edit: dictionaryTermEditInput(a.dictUI), NotesText: a.dictUI.notes.text,
			NotesQuery: a.dictUI.notesQuery, NotesFind: a.dictUI.notesFind.cur,
			Profiles: a.dictProfiles.list, ProfileEdit: a.dictUI.profileEdit, ProfileDeleteOpen: a.dictUI.profileDeleteOpen,
			ProfileDeleteID: a.dictUI.profileDeleteID,
		}
		a.dv.view, a.dv.viewKey = buildDictionaryViewState(translator, input), viewKey
	}
	return a.dv.view
}

func dictionaryTermEditInput(local dictionaryLocalUI) dictionaryTermEditStateInput {
	edit := local.edit
	return dictionaryTermEditStateInput{
		Open: local.editOpen, ID: edit.id, Term: edit.term, Aliases: edit.aliases, Note: edit.note,
		Enabled: edit.enabled, Error: edit.err, Saving: edit.saving, Original: edit.original, Suggestion: edit.suggestion,
	}
}

func (a *app) setDictionaryEnabled(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Enabled = on })
}

func (a *app) setDictionaryASRHints(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.ASRHints = on })
}

func (a *app) setDictionaryGlossary(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Glossary = on })
}

func (a *app) setDictionaryLearn(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.Learn = on })
}

func (a *app) setDictionaryNotesOff(off bool) {
	a.updateSettings(func(s *settings.Settings) { s.Dictionary.NotesOff = off })
}

func (a *app) openDictionaryTranscribe() { a.navigate("/transcribe") }

func (a *app) openDictionaryTerm(id string) {
	if id == "" {
		a.openTerm(store.Term{Enabled: true})
		return
	}
	for _, term := range a.dv.list {
		if term.ID == id {
			a.openTerm(term)
			return
		}
	}
}

func (a *app) openTerm(term store.Term) {
	local := &a.dictUI
	local.editOpen = true
	local.edit = termEdit{id: term.ID, term: term.Term, aliases: strings.Join(term.Aliases, ", "), note: term.Note,
		enabled: term.Enabled, createdAt: term.CreatedAt, original: term.Term, profile: a.activeDictionaryProfile().ID}
	local.editWas = local.edit
}

func (a *app) submitTerm() {
	edit := &a.dictUI.edit
	term := dict.Clean(store.Term{ID: edit.id, Term: edit.term, Aliases: dict.SplitAliases(edit.aliases), Note: edit.note, Enabled: edit.enabled, CreatedAt: edit.createdAt, ProfileID: edit.profile})
	if err := dict.Check(term); err != nil {
		edit.err = uitext.TermErrorText(err)
		return
	}
	edit.saving = true
	if edit.suggestion != "" {
		a.acceptSuggestion(edit.suggestion, term, true)
		return
	}
	a.saveTerm(term, true)
}

func (a *app) saveTerm(term store.Term, fromDialog bool) {
	if term.ID == "" {
		term.ID = core.NewID(time.Now())
	}
	if term.CreatedAt.IsZero() {
		term.CreatedAt = time.Now()
	}
	term.UpdatedAt = time.Now()
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.Put(ctx, term)
		a.post(func() {
			if fromDialog {
				a.dictUI.edit.saving = false
				if err != nil {
					a.dictUI.edit.err = uitext.TermErrorText(err)
					return
				}
				a.dictUI.editOpen = false
			} else if err != nil {
				a.toast(uitext.TermErrorText(err))
			}
			a.loadDict()
		})
	})
}

func (a *app) setDictionaryTermEnabled(id string, enabled bool) {
	for _, term := range a.dv.list {
		if term.ID == id {
			term.Enabled = enabled
			a.saveTerm(term, false)
			return
		}
	}
}

// deleteDictionaryTerm deletes a term from profile ("": the active one).
func (a *app) deleteDictionaryTerm(id, original, profile string) {
	a.deleteTerm(store.Term{ID: id, Term: original, ProfileID: profile})
}

func (a *app) deleteTerm(term store.Term) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.Delete(ctx, term.ID, term.ProfileID)
		a.post(func() {
			if err != nil && !errors.Is(err, core.ErrNotFound) {
				a.toast(i18n.T("Could not delete the term: %v", err))
				return
			}
			a.toast(i18n.T("Deleted “%s”", term.Term))
			a.loadDict()
		})
	})
}
