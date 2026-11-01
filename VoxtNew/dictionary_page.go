package main

// The Dictionary page (/dictionary): the user's terms,
// kept by hand in 0.1. Each term has the spelling it must have, aliases
// (what speech recognition tends to hear instead), a note and a switch.
// Terms go to the speech model as hints (dict.Hints) and to the
// language model as a glossary (compose.Glossary); the switches at the
// top say where. Search, add, edit, delete, and import or export CSV and
// plain text (internal/dict).
//
// The page keeps the terms of its search in memory, read in the
// background when it shows and after each change; sessions read the
// store themselves, so a change applies to the next one.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dict"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// dictionaryState is the Dictionary page's view state.
type dictionaryState struct {
	dv dictView
}

type dictView struct {
	query string
	// list holds the terms read for listFor (the search); "\x00" before
	// the first read. seq numbers the reads: only the last one lands.
	list    []store.Term
	listFor string
	read    bool
	seq     int
	loading bool
	err     string
	total   int // terms in the dictionary (the last read without a search)
	listUI  ui.ListState
	// The term dialog.
	editOpen bool
	edit     termEdit
	// importing: an import runs.
	importing bool
}

// termEdit is the term dialog's fields: a new term when id is "".
type termEdit struct {
	id              string
	term, aliases   string
	note            string
	enabled         bool
	err             string
	saving          bool
	createdAt       time.Time
	original, title string
}

// Test seams: the file dialogs of Import and Export.
var (
	chooseImportFile = func(a *app, then func(string)) {
		go func() {
			files, err := mygo.Dialog.Open(mygo.OpenDialogOptions{Parent: a.win, Title: i18n.T("Import terms"),
				Filters: []mygo.FileFilter{{Name: i18n.T("Dictionary (CSV or text)"), Extensions: []string{"csv", "txt"}}}})
			if err != nil || len(files) == 0 {
				return
			}
			a.post(func() { then(files[0]) })
		}()
	}
	chooseExportFile = func(a *app, name string, then func(string)) {
		go func() {
			dst, err := mygo.Dialog.Save(mygo.SaveDialogOptions{Parent: a.win, Title: i18n.T("Export terms"), DefaultPath: name,
				Filters: []mygo.FileFilter{{Name: strings.ToUpper(filepath.Ext(name)[1:]), Extensions: []string{filepath.Ext(name)[1:]}}}})
			if err != nil || dst == "" {
				return
			}
			a.post(func() { then(dst) })
		}()
	}
)

// dictShown runs when the router goes to the page (pageShown).
func (a *app) dictShown() { a.loadDict() }

// loadDict reads the terms of the search in the background.
func (a *app) loadDict() {
	d := &a.dv
	d.seq++
	d.loading = true
	seq, q := d.seq, strings.TrimSpace(d.query)
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.List(ctx, q)
		a.post(func() {
			d := &a.dv
			if seq != d.seq {
				return
			}
			d.loading, d.read, d.listFor = false, true, q
			if err != nil {
				d.err = err.Error()
				return
			}
			d.list, d.err = list, ""
			if q == "" {
				d.total = len(list)
			}
		})
	})
}

func (a *app) dictionaryPage(c *ui.Context) {
	t := c.Theme()
	d := &a.dv
	if !d.read && !d.loading {
		a.loadDict()
	}
	pageHeader(c, i18n.T("Dictionary"), func() {
		if d.total > 0 {
			caption(c, i18n.N(d.total, "%d term", "%d terms", d.total)).SingleLine()
		}
		on := &a.prefs.Dictionary.Enabled
		state := i18n.T("Off")
		if *on {
			state = i18n.T("On")
		}
		caption(c, state).SingleLine()
		if ui.Switch(c.Key("dict-on"), on).Label(i18n.T("Use the dictionary")).Tooltip(i18n.T("Use the dictionary")).Changed() {
			v := *on
			a.updateSettings(func(s *settings.Settings) { s.Dictionary.Enabled = v })
		}
		moreMenu(c.Key("dict-more"), "", func(m *ui.Menu) {
			if m.Item(i18n.T("Import…")).Chosen() {
				a.importTerms()
			}
			if m.Item(i18n.T("Export as CSV…")).Chosen() {
				a.exportTerms(dict.CSV)
			}
			if m.Item(i18n.T("Export as text…")).Chosen() {
				a.exportTerms(dict.Text)
			}
		})
		if primaryIconButton(c.Key("dict-add"), icons.Add, i18n.T("Add term")).Clicked() {
			a.openTerm(store.Term{Enabled: true})
		}
	})
	ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp2)).Gap(px(t, sp2)).Shrink(0).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
			if searchField(c, "dict-search", &d.query, i18n.T("Search terms"), px(t, 320)).Label(i18n.T("Search the dictionary")).Changed() {
				a.loadDict()
			}
			a.dictSwitches(c)
		})
		if !a.prefs.Dictionary.Enabled {
			notice(c, "dict-off", noticeInfo, i18n.T("The dictionary is off: its terms go nowhere."), "", nil, nil)
		}
		if d.err != "" {
			notice(c, "dict-err", noticeError, i18n.T("Could not read the dictionary."), d.err, func() {
				if ghostButton(c.Key("dict-retry"), i18n.T("Try again")).Clicked() {
					a.loadDict()
				}
			}, nil)
		}
	})
	list := d.list
	ui.Column(c).Grow(1).MinHeight(0).Children(func() {
		switch {
		case !d.read:
		case len(list) == 0:
			a.dictEmpty(c)
		default:
			ui.List(c.Key("dict-list"), &d.listUI, len(list)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
				ItemKey(func(i int) any { return list[i].ID }).Rows(func(r ui.ListRow) {
				a.termRow(c, list[r.Index])
			})
		}
	})
	a.termDialog(c)
}

// dictSwitches say where the terms go.
func (a *app) dictSwitches(c *ui.Context) {
	t := c.Theme()
	off := !a.prefs.Dictionary.Enabled
	ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Wrap().Children(func() {
		hints := &a.prefs.Dictionary.ASRHints
		if ui.Checkbox(c.Key("dict-hints"), hints, i18n.T("Speech hints")).Disabled(off).
			Tooltip(i18n.T("Sent to the speech model as words to expect. It keeps a limited number: the terms changed last go first.")).Changed() {
			v := *hints
			a.updateSettings(func(s *settings.Settings) { s.Dictionary.ASRHints = v })
		}
		gl := &a.prefs.Dictionary.Glossary
		if ui.Checkbox(c.Key("dict-glossary"), gl, i18n.T("Glossary for the language model")).Disabled(off).
			Tooltip(i18n.T("Clean-up writes the terms as spelled here, also when recognition heard them a little differently. A dictionary of up to 30 terms goes in whole; a larger one only with the terms found in what you said.")).Changed() {
			v := *gl
			a.updateSettings(func(s *settings.Settings) { s.Dictionary.Glossary = v })
		}
	})
}

// dictEmpty is the list's empty state.
func (a *app) dictEmpty(c *ui.Context) {
	d := &a.dv
	if q := strings.TrimSpace(d.listFor); q != "" {
		emptyState(c, icons.Magnifer, i18n.T("No term matches “%s”", q), i18n.T("The search looks in terms, aliases and notes."), func() {
			if ghostButton(c.Key("dict-clear-search"), i18n.T("Clear the search")).Clicked() {
				d.query = ""
				a.loadDict()
			}
		})
		return
	}
	emptyState(c, dictionaryIcon, i18n.T("Names and terms you use"),
		i18n.T("Add the names, products and jargon you say: speech recognition expects them, and clean-up writes them as you spell them here."), func() {
			if ghostIconButton(c.Key("dict-empty-add"), icons.Add, i18n.T("Add term")).Clicked() {
				a.openTerm(store.Term{Enabled: true})
			}
			if ghostIconButton(c.Key("dict-empty-import"), icons.Import, i18n.T("Import…")).Label(i18n.T("Import terms")).Clicked() {
				a.importTerms()
			}
		})
}

// termRow is a term in the list: its switch, the term, its aliases and
// note, Edit and Delete.
func (a *app) termRow(c *ui.Context, term store.Term) {
	t := c.Theme()
	k := tok(t)
	row := ui.Row(c.Key("term-"+term.ID)).Padding(px(t, 6), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center)
	if row.Hovered() {
		row.Background(k.Hover)
	}
	row.Children(func() {
		on := term.Enabled // derived: committed in OnChange
		ui.Switch(c.Key("on"), &on).Label(i18n.T("Use %s", term.Term)).OnChange(func() {
			tm := term
			tm.Enabled = on
			a.saveTerm(tm, false)
		})
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				tx := ui.Text(c, term.Term).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
				if !term.Enabled {
					tx.TextColor(k.Text3)
				}
				if len(term.Aliases) > 0 {
					truncText(c, i18n.T("heard as %s", strings.Join(term.Aliases, i18n.T(", ")))).FontSize(px(t, fsCaption)).TextColor(k.Text2).Grow(1)
				}
			})
			if term.Note != "" {
				truncText(c, term.Note).FontSize(px(t, fsCaption)).TextColor(k.Text3)
			}
		})
		if iconButton(c.Key("edit"), icons.Pen2, i18n.T("Edit %s", term.Term)).Clicked() {
			a.openTerm(term)
		}
		if iconButton(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete %s", term.Term)).Clicked() {
			a.deleteTerm(term)
		}
	})
}

// openTerm opens the term dialog on term (a new one without an id).
func (a *app) openTerm(term store.Term) {
	d := &a.dv
	d.editOpen = true
	d.edit = termEdit{id: term.ID, term: term.Term, aliases: strings.Join(term.Aliases, ", "), note: term.Note,
		enabled: term.Enabled, createdAt: term.CreatedAt, original: term.Term}
}

func (a *app) termDialog(c *ui.Context) {
	t := c.Theme()
	d := &a.dv
	if !d.editOpen {
		return
	}
	e := &d.edit
	title := i18n.T("Add term")
	if e.id != "" {
		title = i18n.T("Edit term")
	}
	dialogFrame(c, "term-dlg", &d.editOpen, title, "", 460, func() {
		fieldInfo(c, i18n.T("Term"), i18n.T("As it must be written: VoxtNew, Kubernetes, 豆包."), func() {
			in := ui.TextInput(c.Key("term-term"), &e.term).Label(i18n.T("Term as written")).Placeholder("VoxtNew").AutoFocus()
			if in.Changed() {
				e.err = ""
			}
			if in.Submitted() {
				a.submitTerm()
			}
		})
		fieldInfo(c, i18n.T("Aliases"), i18n.T("What speech recognition writes instead, separated by commas: Vox Tunew, 沃克斯牛. VoxtNew puts the term in their place, also without a language model."), func() {
			ui.TextInput(c.Key("term-aliases"), &e.aliases).Label(i18n.T("Aliases of the term")).Placeholder(i18n.T("Vox Tunew, 沃克斯牛"))
		})
		fieldInfo(c, i18n.T("Note"), i18n.T("For you only: it is sent nowhere."), func() {
			ui.TextArea(c.Key("term-note"), &e.note).Label(i18n.T("Note on the term")).MinHeight(px(t, 56))
		})
		ui.Checkbox(c.Key("term-enabled"), &e.enabled, i18n.T("Use this term"))
		if e.err != "" {
			warnLine(c, e.err)
		}
	}, func() {
		if e.id != "" && ghostButton(c.Key("term-delete"), i18n.T("Delete")).Clicked() {
			d.editOpen = false
			a.deleteTerm(store.Term{ID: e.id, Term: e.original})
		}
		ui.Spacer(c)
		if ui.Button(c.Key("term-cancel"), i18n.T("Cancel")).Clicked() {
			d.editOpen = false
		}
		if primaryButton(c.Key("term-save"), i18n.T("Save")).Disabled(e.saving).Clicked() {
			a.submitTerm()
		}
	})
}

// submitTerm checks and saves the dialog's term.
func (a *app) submitTerm() {
	e := &a.dv.edit
	term := dict.Clean(store.Term{ID: e.id, Term: e.term, Aliases: dict.SplitAliases(e.aliases), Note: e.note, Enabled: e.enabled, CreatedAt: e.createdAt})
	if err := dict.Check(term); err != nil {
		e.err = uitext.TermErrorText(err)
		return
	}
	e.saving = true
	a.saveTerm(term, true)
}

// saveTerm writes a term (a new id for a new one); fromDialog closes the
// dialog when it worked, or shows why not.
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
				a.dv.edit.saving = false
				if err != nil {
					a.dv.edit.err = uitext.TermErrorText(err)
					return
				}
				a.dv.editOpen = false
			} else if err != nil {
				a.toast(uitext.TermErrorText(err))
			}
			a.loadDict()
		})
	})
}

// deleteTerm deletes a term.
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
			terms, skipped, err := dict.Import(data, dict.FormatOf(path))
			if err != nil {
				return i18n.T("Could not import %s: it is not a CSV or text list of terms.", filepath.Base(path))
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			have, err := repo.List(ctx, "")
			if err != nil {
				return i18n.T("Could not import: %v", err)
			}
			put, added, updated := dict.Merge(have, terms)
			now := time.Now()
			for i, tm := range put {
				if tm.ID == "" {
					tm.ID, tm.CreatedAt = core.NewID(now), now.Add(time.Duration(i))
				}
				tm.UpdatedAt = now
				if err := repo.Put(ctx, tm); err != nil {
					return i18n.T("Could not import: %v", err)
				}
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

// exportTerms writes every term to a file the user picks.
func (a *app) exportTerms(format string) {
	name := "voxtnew-dictionary." + format
	chooseExportFile(a, name, func(dst string) { a.exportFile(dst, format) })
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
