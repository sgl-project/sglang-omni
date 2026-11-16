package main

// The Dictionary page (/dictionary): terms, personal Markdown, suggestions,
// and the history scan. Controllers own storage, settings, files, and models.

import (
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type dictionaryLocalUI struct {
	query          string
	find           finder
	listUI         ui.ListState
	guideOpen      bool
	editOpen       bool
	edit           termEdit
	editWas        termEdit
	allSuggestions bool
	tab            int
	notes          speechContextState
	notesFeature   int
	notesQuery     string
	notesFind      finder
	scanOpen       bool
}

type termEdit struct {
	id            string
	term, aliases string
	note          string
	enabled       bool
	err           string
	saving        bool
	createdAt     time.Time
	original      string
	suggestion    string
}

func dictionaryPage(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	t := c.Theme()
	widePageHeader(c, i18n.T("Dictionary"), func() {
		if state.CountLabel != "" {
			caption(c, state.CountLabel).SingleLine()
		}
		on := state.Enabled
		status := i18n.T("Off")
		if on {
			status = i18n.T("On")
		}
		caption(c, status).SingleLine()
		if ui.Switch(c.Key("dict-on"), &on).Label(i18n.T("Use the dictionary")).Tooltip(i18n.T("Use the dictionary")).Changed() {
			actions.setDictionaryEnabled(on)
		}
		if ui.Button(c.Key("dict-scan"), i18n.T("Scan history…")).Disabled(!state.CanScan).Clicked() {
			actions.openDictionaryScan()
		}
		moreMenu(c.Key("dict-more"), "", func(menu *ui.Menu) {
			if menu.Item(i18n.T("Import…")).Chosen() {
				actions.importTerms()
			}
			if state.LegacyImport && menu.Item(i18n.T("Import from Voxt 1.x…")).Chosen() {
				actions.importLegacy()
			}
			if menu.Item(i18n.T("Export as CSV…")).Chosen() {
				actions.exportTerms("csv")
			}
			if menu.Item(i18n.T("Export as text…")).Chosen() {
				actions.exportTerms("txt")
			}
		})
		if primaryIconButton(c.Key("dict-add"), icons.Add, i18n.T("Add term")).Clicked() {
			actions.openDictionaryTerm("")
		}
	})
	pageBody(c).Children(func() {
		ui.Row(c).Padding(px(t, sp1), px(t, sp6), 0).Shrink(0).Children(func() {
			segmentedFit(c.Key("dict-tab"), &local.tab, i18n.T("Terms"), i18n.T("Personal dictionary")).Label(i18n.T("Dictionary view"))
		})
		if local.tab == dictNotesTab {
			dictionaryNotesView(c, &state.Notes, actions, local)
			return
		}
		hits := findHits(state.ListFor, len(state.Terms), func(index int) []string { return termTexts(state.Terms[index]) })
		if local.find.advance(hits[len(state.Terms)]) {
			row, _ := rowOf(hits, local.find.cur)
			local.listUI.ScrollIntoView(row)
		}
		ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp2)).Shrink(0).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				if local.find.field(c, "dict-search", &local.query, i18n.T("Search terms"), px(t, 320)).Label(i18n.T("Search dictionary")).Changed() {
					actions.searchDictionary(local.query)
				}
				local.find.count(c, state.ListFor, hits[len(state.Terms)])
				dictSwitches(c, state, actions)
			})
			if !state.CanScan {
				ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
					caption(c, i18n.T("Choose a language model for Transcribe first")).SingleLine().MinWidth(0).Shrink(1)
					if ghostButton(c.Key("dict-scan-model"), i18n.T("Open Transcribe")).Clicked() {
						actions.openDictionaryTranscribe()
					}
				})
			}
			dictGuide(c, state, local)
			if !state.Enabled {
				notice(c, "dict-off", noticeInfo, i18n.T("The dictionary is off, so its terms are not used."), "", nil, nil)
			}
			if state.Error != "" {
				notice(c, "dict-err", noticeError, i18n.T("Could not read the dictionary."), state.Error, func() {
					if ghostButton(c.Key("dict-retry"), i18n.T("Try again")).Clicked() {
						actions.loadDict()
					}
				}, nil)
			}
		})
		suggestionsSection(c, state, actions, local)
		current, currentAt := rowOf(hits, local.find.cur)
		ui.Column(c).Grow(1).MinHeight(0).Children(func() {
			switch {
			case !state.Read:
			case len(state.Terms) == 0:
				dictEmpty(c, state, actions)
			default:
				ui.List(c.Key("dict-list"), &local.listUI, len(state.Terms)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
					ItemKey(func(index int) any { return state.Terms[index].ID }).Rows(func(row ui.ListRow) {
					at := -1
					if row.Index == current {
						at = currentAt
					}
					termRow(c, state.Terms[row.Index], state.ListFor, at, actions)
				})
			}
		})
	})
	termDialog(c, actions, local)
	scanDialog(c, &state.Scan, actions, local)
}

func dictSwitches(c *ui.Context, state *dictionaryViewState, actions dictionaryActions) {
	t := c.Theme()
	off := !state.Enabled
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Wrap().Children(func() {
		hints := state.ASRHints
		if ui.Checkbox(c.Key("dict-hints"), &hints, i18n.T("Speech hints")).Disabled(off).
			Tooltip(i18n.T("Terms go to the speech model as words to expect.")).Changed() {
			actions.setDictionaryASRHints(hints)
		}
		glossary := state.Glossary
		if ui.Checkbox(c.Key("dict-glossary"), &glossary, i18n.T("Glossary for the language model")).Disabled(off).
			Tooltip(i18n.T("The language model writes terms as spelled here.")).Changed() {
			actions.setDictionaryGlossary(glossary)
		}
		learnOn := state.Learn
		if ui.Checkbox(c.Key("dict-learn"), &learnOn, i18n.T("Learn from my corrections")).Disabled(off).
			Tooltip(i18n.T("For %d seconds after inserting, Voxt reads the text around it. If you correct a word, it suggests your spelling. Nothing it reads is saved or sent.", state.LearnWatchSeconds)).Changed() {
			actions.setDictionaryLearn(learnOn)
		}
	})
}

func dictGuide(c *ui.Context, state *dictionaryViewState, local *dictionaryLocalUI) {
	collapsible(c.Key("dict-guide"), i18n.T("How the dictionary works"), &local.guideOpen, nil, func() {
		caption(c, i18n.T("Each enabled term is used in three places:"))
		caption(c, i18n.T("Speech recognition: with Speech hints on, terms go to the speech model as words to expect. Aliases are not sent."))
		caption(c, i18n.T("Language model: with Glossary for the language model on, the model writes terms as spelled here. With more than %d terms, only those found in the text are sent, up to %d.", state.GlossaryAllUpTo, state.MaxGlossaryTerms))
		caption(c, i18n.T("Aliases: misspellings Voxt replaces with the term, even without a language model."))
		caption(c, i18n.T("Rules a term cannot hold, such as when a spelling applies, go in the Personal dictionary: Markdown notes for the language model only, with an example of the prompt."))
	})
}

func dictEmpty(c *ui.Context, state *dictionaryViewState, actions dictionaryActions) {
	if query := strings.TrimSpace(state.ListFor); query != "" {
		emptyState(c, icons.Magnifer, i18n.T("No term matches “%s”", query), "", func() {
			if ghostButton(c.Key("dict-clear-search"), i18n.T("Clear search")).Clicked() {
				actions.searchDictionary("")
			}
		})
		return
	}
	emptyState(c, dictionaryIcon, i18n.T("Names and terms you use"),
		i18n.T("Add the names and jargon you say, spelled as they should be written."), func() {
			if ghostIconButton(c.Key("dict-empty-add"), icons.Add, i18n.T("Add term")).Clicked() {
				actions.openDictionaryTerm("")
			}
			if ghostIconButton(c.Key("dict-empty-import"), icons.Import, i18n.T("Import…")).Label(i18n.T("Import terms")).Clicked() {
				actions.importTerms()
			}
		})
}

func termTexts(term dictionaryTermViewState) []string {
	return []string{term.Term, term.AliasesText, term.Note}
}

func termRow(c *ui.Context, term dictionaryTermViewState, query string, at int, actions dictionaryActions) {
	t := c.Theme()
	colors := tok(t)
	texts := termTexts(term)
	termMatches, aliasMatches := len(findFolded(texts[0], query)), len(findFolded(texts[1], query))
	row := ui.Row(c.Key("term-"+term.ID)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center)
	if row.Hovered() {
		row.Background(colors.Hover)
	}
	row.Children(func() {
		on := term.Enabled
		ui.Switch(c.Key("on"), &on).Label(i18n.T("Use %s", term.Term)).OnChange(func() {
			actions.setDictionaryTermEnabled(term.ID, on)
		})
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				text := markedText(c, term.Term, query, curIn(at, 0)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				if !term.Enabled {
					text.TextColor(colors.Text3)
				}
				if term.AliasesText != "" {
					heard := i18n.T("heard as %s", texts[1])
					before, after, _ := strings.Cut(i18n.T("heard as %s", "\x00"), "\x00")
					spans := append([]ui.Span{{Text: before}}, markSpans(colors.Danger, ui.Span{}, texts[1], query, curIn(at, termMatches))...)
					ui.RichText(c, append(spans, ui.Span{Text: after})...).Label(heard).Tooltip(heard).
						FontSize(px(t, fsCaption)).TextColor(colors.Text2).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				}
			})
			if term.Note != "" {
				markedText(c, term.Note, query, curIn(at, termMatches+aliasMatches)).Tooltip(term.Note).
					FontSize(px(t, fsCaption)).TextColor(colors.Text3).SingleLine().Ellipsis("…").MinWidth(0)
			}
		})
		if iconButton(c.Key("edit"), icons.Pen2, i18n.T("Edit %s", term.Term)).Clicked() {
			actions.openDictionaryTerm(term.ID)
		}
		if iconButton(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete %s", term.Term)).Clicked() {
			actions.deleteDictionaryTerm(term.ID, term.Term)
		}
	})
}

func termDialog(c *ui.Context, actions dictionaryActions, local *dictionaryLocalUI) {
	t := c.Theme()
	if !local.editOpen {
		return
	}
	edit := &local.edit
	title := i18n.T("Add term")
	if edit.id != "" {
		title = i18n.T("Edit term")
	}
	if dialogFrame(c, "term-dlg", &local.editOpen, title, "", 460, func() {
		fieldInfo(c, i18n.T("Term"), i18n.T("As it should be written: Voxt, Kubernetes, 豆包."), func() {
			input := ui.TextInput(c.Key("term-term"), &edit.term).Label(i18n.T("Term as written")).Placeholder("Voxt").AutoFocus()
			if input.Changed() {
				edit.err = ""
			}
			if input.Submitted() {
				actions.submitTerm()
			}
		})
		fieldInfo(c, i18n.T("Aliases"), i18n.T("Wrong spellings to replace, separated by commas: Vox T, 沃克斯特."), func() {
			ui.TextInput(c.Key("term-aliases"), &edit.aliases).Label(i18n.T("Aliases of the term")).Placeholder(i18n.T("Vox T, 沃克斯特"))
		})
		fieldInfo(c, i18n.T("Note"), i18n.T("Only for you, never sent."), func() {
			ui.TextArea(c.Key("term-note"), &edit.note).Label(i18n.T("Note on the term")).MinHeight(px(t, 56))
		})
		ui.Checkbox(c.Key("term-enabled"), &edit.enabled, i18n.T("Use this term"))
		if edit.err != "" {
			warnLine(c, edit.err)
		}
	}, func() {
		if edit.id != "" && ghostButton(c.Key("term-delete"), i18n.T("Delete")).Clicked() {
			local.editOpen = false
			actions.deleteDictionaryTerm(edit.id, edit.original)
		}
		ui.Spacer(c)
		if ui.Button(c.Key("term-cancel"), i18n.T("Cancel")).Clicked() {
			local.editOpen = false
		}
		if primaryButton(c.Key("term-save"), i18n.T("Save")).Disabled(edit.saving).Clicked() {
			actions.submitTerm()
		}
	}).Dismissed() && local.edit != local.editWas {
		local.editOpen = true
	}
}
