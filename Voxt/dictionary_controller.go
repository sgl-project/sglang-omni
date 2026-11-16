package main

import (
	"context"
	"errors"
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
	book            *learn.Book
	suggestions     []learn.Suggestion
	suggestionReads int
	scanner         *learn.Scanner
	selPress        *selectionPress
}

type dictView struct {
	list            []store.Term
	listFor         string
	read            bool
	seq             int
	loading         bool
	err             string
	total           int
	all             []store.Term
	importing       bool
	legacyAvailable bool
	scan            scanState
	example         notesExampleCache
	termRows        []dictionaryTermViewState
	termRowsKey     dictionaryTermRowsCacheKey
}

type dictionaryTermRowsCacheKey struct {
	seq    int
	locale string
}

func (a *app) dictShown() {
	a.dv.legacyAvailable = legacyLibrary() != ""
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
			if query == "" {
				d.total, d.all = len(list), list
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
	key := dictionaryTermRowsCacheKey{seq: a.dv.seq, locale: translator.Locale()}
	if key != a.dv.termRowsKey {
		a.dv.termRows = buildDictionaryTermViewStates(translator, a.dv.list)
		a.dv.termRowsKey = key
	}
	input := dictionaryViewStateInput{
		TermsReady: true, TermRows: a.dv.termRows, Suggestions: a.suggestions, Settings: a.prefs,
		Installed: a.installed(local.LLM), Read: a.dv.read, Loading: a.dv.loading, Error: a.dv.err,
		Query: a.dictUI.query, ListFor: a.dv.listFor, Total: a.dv.total, LegacyImport: a.dv.legacyAvailable,
		Tab: a.dictUI.tab, ScanOpen: a.dictUI.scanOpen, Scan: dictionaryScanInput(a.dv.scan),
		Edit: dictionaryTermEditInput(a.dictUI), NotesText: a.dictUI.notes.text,
		NotesQuery: a.dictUI.notesQuery, NotesFind: a.dictUI.notesFind.cur,
	}
	state := buildDictionaryViewState(translator, input)
	if a.dictUI.tab == dictNotesTab {
		state.Notes = a.dictionaryNotesModel(a.dictionaryNotesInput(notesFeatures[a.dictUI.notesFeature]))
	}
	return state
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
		enabled: term.Enabled, createdAt: term.CreatedAt, original: term.Term}
	local.editWas = local.edit
}

func (a *app) submitTerm() {
	edit := &a.dictUI.edit
	term := dict.Clean(store.Term{ID: edit.id, Term: edit.term, Aliases: dict.SplitAliases(edit.aliases), Note: edit.note, Enabled: edit.enabled, CreatedAt: edit.createdAt})
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

func (a *app) deleteDictionaryTerm(id, original string) {
	a.deleteTerm(store.Term{ID: id, Term: original})
}

func (a *app) deleteTerm(term store.Term) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := repo.Delete(ctx, term.ID)
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
