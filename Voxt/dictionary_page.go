package main

// The Dictionary page (/dictionary): one column, as Snippets. The header
// holds the active profile (with more than one), the settings, the more
// menu and Add term; under it a line says what the page is for, then the
// two views (terms, the personal dictionary) and their search, the
// suggestions waiting for a click, and the terms in a card. The switches
// that decide where terms go live in the settings dialog
// (dictionary_settings.go). Controllers own storage, settings, files and
// models.

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
	settingsOpen   bool
	editOpen       bool
	edit           termEdit
	editWas        termEdit
	allSuggestions bool
	tab            int
	notes          speechContextState
	notesQuery     string
	notesFind      finder
	scanOpen       bool
	profileEdit    dictionaryProfileEdit
	// profileDeleteOpen asks whether to delete profileDeleteID.
	profileDeleteOpen bool
	profileDeleteID   string
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
	profile       string // the profile the term was opened in: a switch must not retarget Save/Delete
}

// dictRowH is a term row's height: one height for rows of one line and of
// two, so the list is as tall as its rows and its card ends under the last.
const dictRowH = 48

func dictionaryPage(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	t := c.Theme()
	dictionaryHeader(c, state, actions, local)
	ui.Column(c).FillWidth().MaxWidth(px(t, pageMaxW)).AlignSelf(ui.Center).Grow(1).MinHeight(0).
		Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp3)).Children(func() {
		subtitle := i18n.T("Add the names and jargon you say, spelled as they should be written.")
		if local.tab == dictNotesTab {
			subtitle = i18n.T("Rules for the language model in Markdown, such as when a spelling applies.")
		}
		ui.Text(c, subtitle).TextColor(tok(t).Text2).Shrink(0)
		if local.tab == dictNotesTab {
			matches := findFolded(local.notes.text, local.notesQuery)
			dictToolbar(c, local, func() {
				local.notesFind.count(c, local.notesQuery, len(matches))
				local.notesFind.field(c, "dict-notes-find", &local.notesQuery, i18n.T("Find in the personal dictionary"), px(t, 260)).Label(i18n.T("Search personal dictionary"))
			})
			dictionaryNotesView(c, &state.Notes, actions, local, matches)
		} else {
			dictionaryTermsView(c, state, actions, local)
		}
	})
	termDialog(c, actions, local)
	scanDialog(c, &state.Scan, actions, local)
	dictSettingsDialog(c, state, actions, local)
	dictionaryProfileDialogs(c, state, actions, local)
}

// dictionaryHeader is the title, the profile picker beside it (only with
// more than one profile: one is just the dictionary), and on the right
// the settings, the more menu and the page's one primary action.
func dictionaryHeader(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	var byTitle func()
	if len(state.Profiles) > 1 {
		byTitle = func() { dictionaryProfilePicker(c, state, actions, local) }
	}
	centeredPageHeaderBy(c, pageMaxW, i18n.T("Dictionary"), byTitle, func() {
		if iconButton(c.Key("dict-settings"), icons.Settings, i18n.T("Dictionary settings")).Clicked() {
			local.settingsOpen = true
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
			if len(state.Profiles) <= 1 {
				menu.Separator()
				if menu.Item(i18n.T("New profile…")).Chosen() {
					actions.openDictionaryProfileEdit("")
				}
			}
		})
		if primaryIconButton(c.Key("dict-add"), icons.Add, i18n.T("Add term")).Clicked() {
			actions.openDictionaryTerm("")
		}
	})
}

// dictToolbar is the views on the left and, on the right, what trailing
// adds: the view's count and search.
func dictToolbar(c *ui.Context, local *dictionaryLocalUI, trailing func()) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Shrink(0).Children(func() {
		segmentedFit(c.Key("dict-tab"), &local.tab, i18n.T("Terms"), i18n.T("Personal dictionary")).Label(i18n.T("Dictionary view")).Shrink(0)
		ui.Spacer(c)
		trailing()
	})
}

func dictionaryTermsView(c *ui.Context, state *dictionaryViewState, actions dictionaryActions, local *dictionaryLocalUI) {
	t := c.Theme()
	k := tok(t)
	hits := findHits(state.ListFor, len(state.Terms), func(index int) []string { return termTexts(state.Terms[index]) })
	if local.find.advance(hits[len(state.Terms)]) {
		row, _ := rowOf(hits, local.find.cur)
		local.listUI.ScrollIntoView(row)
	}
	dictToolbar(c, local, func() {
		if strings.TrimSpace(state.ListFor) == "" && state.CountLabel != "" {
			caption(c, state.CountLabel).TextColor(k.Text3).SingleLine().Shrink(0)
		}
		local.find.count(c, state.ListFor, hits[len(state.Terms)])
		if local.find.field(c, "dict-search", &local.query, i18n.T("Search terms"), px(t, 260)).Label(i18n.T("Search dictionary")).Changed() {
			actions.searchDictionary(local.query)
		}
	})
	if !state.Enabled {
		notice(c, "dict-off", noticeInfo, i18n.T("The dictionary is off, so its terms are not used."), "", func() {
			if ghostButton(c.Key("dict-turn-on"), i18n.T("Turn on")).Label(i18n.T("Turn on the dictionary")).Clicked() {
				actions.setDictionaryEnabled(true)
			}
		}, nil)
	}
	if state.Error != "" {
		notice(c, "dict-err", noticeError, i18n.T("Could not read the dictionary."), state.Error, func() {
			if ghostButton(c.Key("dict-retry"), i18n.T("Try again")).Clicked() {
				actions.loadDict()
			}
		}, nil)
	}
	suggestionsSection(c, state, actions, local)
	switch {
	case !state.Read:
	case len(state.Terms) == 0:
		dictEmpty(c, state, actions)
	default:
		current, currentAt := rowOf(hits, local.find.cur)
		inset := px(t, sp1)
		// The card is as tall as its rows and shrinks with the window;
		// then the list scrolls inside it.
		card(c.Key("dict-card")).Padding(0).Gap(0).Shrink(1).MinHeight(0).Children(func() {
			ui.List(c.Key("dict-list"), &local.listUI, len(state.Terms)).Height(float32(len(state.Terms))*px(t, dictRowH) + 2*inset).
				Shrink(1).MinHeight(0).Padding(inset).
				ItemKey(func(index int) any { return state.Terms[index].ID }).Rows(func(row ui.ListRow) {
				at := -1
				if row.Index == current {
					at = currentAt
				}
				termRow(c, state.Terms[row.Index], state.ListFor, at, actions)
			})
		})
	}
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
	emptyState(c, dictionaryIcon, i18n.T("No terms yet"), "", func() {
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

// termRow is a term as written and, under it, what speech recognition
// may write instead and the note; a click edits it. Its switch is on the
// right, as a setting's; Delete shows under the pointer, and its place is
// kept when it hides, so the switch does not move.
func termRow(c *ui.Context, term dictionaryTermViewState, query string, at int, actions dictionaryActions) {
	t := c.Theme()
	k := tok(t)
	texts := termTexts(term)
	termMatches, aliasMatches := len(findFolded(texts[0], query)), len(findFolded(texts[1], query))
	r := ui.Row(c.Key("term-"+term.ID)).Height(px(t, dictRowH)).Padding(0, px(t, sp3)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center)
	hover := r.Hovered()
	if hover {
		r.Background(k.Hover)
	}
	r.Children(func() {
		open := ui.ButtonBase(c.Key("open")).Grow(1).MinWidth(0).AlignSelf(ui.Stretch).AlignItems(ui.Center).Label(i18n.T("Edit %s", term.Term))
		if open.Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
				text := markedText(c, term.Term, query, curIn(at, 0)).SingleLine().Ellipsis("…").MinWidth(0)
				if !term.Enabled {
					text.TextColor(k.Text3)
				}
				if line, label := termDetail(k, term, query, at, termMatches, aliasMatches); len(line) > 0 {
					ui.RichText(c, line...).Label(label).Tooltip(label).FontSize(px(t, fsCaption)).TextColor(k.Text2).
						SingleLine().Ellipsis("…").MinWidth(0)
				}
			})
		}).Clicked() {
			actions.openDictionaryTerm(term.ID)
		}
		ui.Row(c).Width(px(t, snippetIcon)).Shrink(0).AlignItems(ui.Center).Children(func() {
			if hover && iconButtonSized(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete"), snippetIcon, 14).Label(i18n.T("Delete %s", term.Term)).Clicked() {
				actions.deleteDictionaryTerm(term.ID, term.Term, "")
			}
		})
		on := term.Enabled
		ui.Switch(c.Key("on"), &on).Label(i18n.T("Use %s", term.Term)).OnChange(func() {
			actions.setDictionaryTermEnabled(term.ID, on)
		})
	})
}

// termDetail is a term's second line: "heard as …" in the secondary grey,
// then " · " and the note in the tertiary one, with the search's marks;
// empty for a term with neither.
func termDetail(k *tokens, term dictionaryTermViewState, query string, at, termMatches, aliasMatches int) ([]ui.Span, string) {
	var spans []ui.Span
	var label []string
	if term.AliasesText != "" {
		heard := i18n.T("heard as %s", term.AliasesText)
		before, after, _ := strings.Cut(i18n.T("heard as %s", "\x00"), "\x00")
		spans = append(spans, ui.Span{Text: before})
		spans = append(spans, markSpans(k.Danger, ui.Span{}, term.AliasesText, query, curIn(at, termMatches))...)
		spans = append(spans, ui.Span{Text: after})
		label = append(label, heard)
	}
	if term.Note != "" {
		faint := ui.Span{Color: k.Text3}
		if len(spans) > 0 {
			dot := faint
			dot.Text = "  ·  "
			spans = append(spans, dot)
		}
		spans = append(spans, markSpans(k.Danger, faint, term.Note, query, curIn(at, termMatches+aliasMatches))...)
		label = append(label, term.Note)
	}
	return spans, strings.Join(label, " · ")
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
		fieldInfo(c, i18n.T("Heard as"), i18n.T("Wrong spellings to replace, separated by commas: Vox T, 沃克斯特."), func() {
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
			actions.deleteDictionaryTerm(edit.id, edit.original, edit.profile)
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
