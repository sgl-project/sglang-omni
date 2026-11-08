package main

// The Dictionary page's suggestions (internal/learn): terms waiting for
// the user, above the list. Add saves the term alone, Edit… opens the term
// dialog first, Ignore hides it for 90 days. The sidebar shows how many
// wait, from the start: openDictionary reads them once.

import (
	"context"
	"log"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/learn"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const (
	suggestionRowsShown  = 5
	suggestionRowsHeight = 240 // the list below keeps the rest of the page at the smallest window
)

func (a *app) openDictionary() {
	a.book = learn.NewBook(a.lib.store.Documents(), a.lib.store.Dictionary())
	a.loadSuggestions()
}

// loadSuggestions reads the suggestions in the background; only the last
// read lands.
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
		})
	})
}

// acceptSuggestion puts term in the dictionary for the suggestion key;
// fromDialog closes the term dialog when it worked, or shows why not.
func (a *app) acceptSuggestion(key string, term store.Term, fromDialog bool) {
	book := a.book
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := book.Accept(ctx, key, term)
		a.post(func() {
			if fromDialog {
				a.dv.edit.saving = false
				if err != nil {
					a.dv.edit.err = uitext.TermErrorText(err)
					return
				}
				a.dv.editOpen = false
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

// suggestionsSection is the suggestions above the list: the first five,
// or all of them after All.
func (a *app) suggestionsSection(c *ui.Context) {
	if len(a.suggestions) == 0 {
		return
	}
	t := c.Theme()
	d := &a.dv
	list := a.suggestions
	if !d.allSuggestions {
		list = list[:min(len(list), suggestionRowsShown)]
	}
	ui.Column(c.Key("dict-suggestions")).Shrink(0).Padding(0, 0, px(t, sp2)).Children(func() {
		ui.Row(c).Padding(0, px(t, sp6)).AlignItems(ui.Center).Children(func() {
			sectionTitleText(c, i18n.T("Suggestions")).Grow(1)
			if len(list) < len(a.suggestions) && ghostButton(c.Key("dict-suggest-all"), i18n.T("All (%d)", len(a.suggestions))).Clicked() {
				d.allSuggestions = true
			}
		})
		ui.Scroll(c.Key("dict-suggest-rows")).MaxHeight(px(t, suggestionRowsHeight)).Children(func() {
			for _, s := range list {
				a.suggestionRow(c, s)
			}
		})
	})
}

// suggestionRow is a suggestion: the term, what Voxt wrote instead, where
// it came from, and Add, Edit… and Ignore.
func (a *app) suggestionRow(c *ui.Context, s learn.Suggestion) {
	t := c.Theme()
	k := tok(t)
	key := s.Key()
	row := ui.Row(c.Key("suggest-"+key)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center)
	if row.Hovered() {
		row.Background(k.Hover)
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Text(c, s.Term).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				if len(s.Heard) > 0 {
					truncText(c, i18n.T("Heard: %s", strings.Join(s.Heard, i18n.T(", ")))).FontSize(px(t, fsCaption)).TextColor(k.Text2).Shrink(1)
				}
				source := i18n.T("From your correction")
				if s.Source == learn.FromHistory {
					source = i18n.T("From your history")
				}
				caption(c, source).SingleLine().Shrink(0)
			})
		})
		if ghostButton(c.Key("suggest-add-"+key), i18n.T("Add")).Label(i18n.T("Add %s to the dictionary", s.Term)).Clicked() {
			a.acceptSuggestion(key, s.AsTerm(), false)
		}
		if ghostButton(c.Key("suggest-edit-"+key), i18n.T("Edit…")).Label(i18n.T("Edit %s first, then add it", s.Term)).Clicked() {
			a.openTerm(s.AsTerm())
			a.dv.edit.suggestion = key
			a.dv.editWas = a.dv.edit
		}
		if ghostButton(c.Key("suggest-ignore-"+key), i18n.T("Ignore")).Label(i18n.T("Ignore %s", s.Term)).Clicked() {
			a.dismissSuggestion(key)
		}
	})
}
