package main

import (
	"context"
	"log"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func (a *app) openDictionary() {
	a.book = learn.NewBook(a.lib.store.Documents(), a.lib.store.Dictionary())
	a.scanner = &learn.Scanner{History: a.lib.store.History(), Book: a.book}
	a.loadSuggestions()
}

func (a *app) loadSuggestions() {
	a.suggestionReads++
	read, book := a.suggestionReads, a.book
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		list, err := book.Pending(ctx)
		a.post(func() {
			if read != a.suggestionReads {
				return
			}
			if err != nil {
				log.Printf("dictionary: reading the suggestions: %v", err)
				return
			}
			a.suggestions = list
			a.suggestionReads++
		})
	})
}

func (a *app) acceptDictionarySuggestion(key string) {
	for _, suggestion := range a.suggestions {
		if suggestion.Key() == key {
			a.acceptSuggestion(key, suggestion.AsTerm(), false)
			return
		}
	}
}

func (a *app) editDictionarySuggestion(key string) {
	for _, suggestion := range a.suggestions {
		if suggestion.Key() == key {
			a.openTerm(suggestion.AsTerm())
			a.dictUI.edit.suggestion = key
			a.dictUI.editWas = a.dictUI.edit
			return
		}
	}
}

func (a *app) acceptSuggestion(key string, term store.Term, fromDialog bool) {
	book := a.book
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := book.Accept(ctx, key, term)
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
				return
			}
			a.loadDict()
			a.loadSuggestions()
		})
	})
}

func (a *app) dismissSuggestion(key string) {
	book := a.book
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		err := book.Dismiss(ctx, key)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not ignore the suggestion: %v", err))
				return
			}
			a.loadSuggestions()
		})
	})
}
