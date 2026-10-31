package main

// The History page (/history): finished sessions by feature
// (Transcribe, Translate, Rewrite), newest first, searchable (the store's
// full-text index), and the details of one in a side panel beside the
// list, or over it when the window is narrow: its texts to copy, where it
// went, the models, the time of each stage, why it failed,
// its recording in Finder, and Delete. An entry's context menu has the
// same (history_menu.go).
//
// The History settings dialog (history_settings.go) holds what the
// history keeps: its recordings and the automatic clean-up.
//
// The page keeps the entries of its tab and search in memory, read in the
// background when it shows, when the tab or the search changes, and when a
// session keeps an entry (historyChanged).

import (
	"context"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// historyLimit is how many entries the list holds at most: the newest.
const historyLimit = 500

// Widths of the details panel and of the list beside it (DIPs at 100 %):
// under both, the panel shows over the list. Both fit beside the sidebar
// in the default window (800 wide: a 600-wide page) and beside the rail in
// the smallest (680).
const (
	historyPanelW   = 290
	historyListMinW = 300
)

// historyState is the History page's view state.
type historyState struct {
	hist    historyView
	histSet histSettingsState // history_settings.go
}

// historyView is the History page's state.
type historyView struct {
	tab   int    // the kind shown: an index in store.Kinds
	query string // the search field
	// list holds the entries read for listFor (the kind and the search),
	// "" before the first read; seq numbers the reads, so that only the
	// last one lands.
	list    []store.Entry
	listFor string
	seq     int
	loading bool
	stale   bool   // an entry changed while the page was not shown
	err     string // the last read's error
	sel     string // the entry in the details panel
	listUI  ui.ListState
	// The Delete dialog: open, and the entry it deletes.
	delAsk bool
	delID  string
	retry  entryRetry // the entry made again from its recording (history_retry.go)
}

// revealInFinder shows a file in Finder; tests replace it.
var revealInFinder = func(path string) { mygo.Shell.ShowItemInFolder(path) }

// historyKind is the kind of the tab shown.
func (h *historyView) kind() store.Kind {
	return store.Kinds[min(max(h.tab, 0), len(store.Kinds)-1)]
}

// wants says what the list should hold: the kind and the search.
func (h *historyView) wants() string {
	return string(h.kind()) + "\x00" + strings.TrimSpace(h.query)
}

// selected is the entry in the details panel, nil for none.
func (h *historyView) selected() *store.Entry {
	if h.sel == "" {
		return nil
	}
	for i := range h.list {
		if h.list[i].ID == h.sel {
			return &h.list[i]
		}
	}
	return nil
}

// loadHistory reads the entries of the tab and the search in the
// background.
func (a *app) loadHistory() {
	h := &a.hist
	h.seq++
	h.loading, h.stale = true, false
	seq, kind, search, want := h.seq, h.kind(), strings.TrimSpace(h.query), h.wants()
	hs := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := hs.List(ctx, store.HistoryQuery{Kind: kind, Search: search, Limit: historyLimit})
		a.post(func() {
			if seq != a.hist.seq {
				return // a newer read is on its way
			}
			h := &a.hist
			h.loading, h.listFor = false, want
			if err != nil {
				h.err = err.Error()
				return
			}
			h.list, h.err = list, ""
			if h.selected() == nil {
				h.sel = ""
			}
		})
	})
}

// historyChanged runs when the history changed (an entry kept, changed
// or deleted, a try ended): the observers of onHistory (events.go) hear
// of it.
func (a *app) historyChanged() {
	for _, fn := range a.onHistory {
		fn()
	}
}

// historyReload reads the history again now when the page shows, else
// when it shows next.
func (a *app) historyReload() {
	if a.currentPage() == "history" {
		a.loadHistory()
		return
	}
	a.hist.stale = true
}

// historyShown runs when the router goes to the History page (pageShown).
func (a *app) historyShown() { a.loadHistory() }

// historyPage builds the page.
func (a *app) historyPage(c *ui.Context) {
	t := c.Theme()
	h := &a.hist
	// A read that pageShown did not start: the first, or after a change.
	if !h.loading && (h.listFor == "" || h.stale) {
		a.loadHistory()
	}
	pageHeader(c, i18n.T("History"), func() {
		if n := len(h.list); n > 0 {
			label := i18n.N(n, "%d entry", "%d entries", n)
			if n >= historyLimit {
				label = i18n.T("Newest %d", n)
			}
			caption(c, label).SingleLine()
		}
	})
	ui.Row(c).Padding(px(t, sp1), px(t, sp6), px(t, sp3)).Gap(px(t, sp2)).AlignItems(ui.Center).Shrink(0).Children(func() {
		tabs := []string{i18n.T("Transcribe"), i18n.T("Translate"), i18n.T("Rewrite")}
		if ui.Segmented(c.Key("hist-tab"), &h.tab, tabs...).Label(i18n.T("Feature")).Shrink(0).Changed() {
			h.sel = ""
			a.loadHistory()
		}
		if searchField(c, "hist-search", &h.query, i18n.T("Search"), px(t, 360)).Label(i18n.T("Search the history")).Changed() {
			a.loadHistory()
		}
		a.historySettingsButton(c) // history_settings.go
	})
	a.historyFolderNotice(c)
	if !a.prefs.Privacy.KeepHistory {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
			notice(c, "hist-off", noticeInfo, i18n.T("History is off: new results are not kept."), "", func() {
				if ghostButton(c.Key("hist-privacy"), i18n.T("Privacy settings")).Clicked() {
					a.navigate("/settings/privacy")
				}
			}, nil)
		})
	}
	if h.err != "" {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(func() {
			notice(c, "hist-err", noticeError, i18n.T("Could not read the history."), h.err, func() {
				if ghostButton(c.Key("hist-retry"), i18n.T("Try again")).Clicked() {
					a.loadHistory()
				}
			}, nil)
		})
	}
	list, sel := h.list, h.selected()
	w := px(t, historyPanelW)
	inline := a.pageWidth(c)-w >= px(t, historyListMinW)
	ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Children(func() {
			switch {
			case h.listFor == "":
				// The first read is on its way: no empty state yet.
			case len(list) == 0:
				a.historyEmpty(c)
			default:
				c.After(time.Minute) // the times are relative
				ui.List(c.Key("hist-list"), &h.listUI, len(list)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp4)).
					ItemKey(func(i int) any { return list[i].ID }).Rows(func(r ui.ListRow) {
					a.historyRow(c, list[r.Index])
				})
			}
		})
		if sel != nil {
			closePanel := func() {
				if !h.delAsk { // a click in the Delete dialog is outside the drawer
					h.sel = ""
				}
			}
			sidePanel(c, "hist-detail", w, inline, i18n.T("Entry details"), closePanel, func() {
				a.historyDetail(c, *sel)
			})
		}
	})
	a.historyDeleteDialog(c)
	a.historySettingsDialog(c)
}

// historyEmpty is the list's empty state: nothing kept yet for the tab,
// or nothing matching the search.
func (a *app) historyEmpty(c *ui.Context) {
	h := &a.hist
	if q := strings.TrimSpace(h.query); q != "" {
		if r := []rune(q); len(r) > 40 {
			q = string(r[:40]) + "…"
		}
		emptyState(c, icons.Magnifer, i18n.T("Nothing matches “%s”", q),
			i18n.T("The search looks in what was said, selected and written."), func() {
				if ghostButton(c.Key("hist-clear-search"), i18n.T("Clear the search")).Clicked() {
					h.query = ""
					a.loadHistory()
				}
			})
		return
	}
	switch h.kind() {
	case store.KindTranslate:
		emptyState(c, translateIcon, i18n.T("No translations yet"),
			i18n.T("Select text, or speak, and press your Translate shortcut: each translation shows up here with its source."), func() {
				if ghostButton(c.Key("hist-go-translate"), i18n.T("Set up Translate")).Clicked() {
					a.navigate("/translate")
				}
			})
	case store.KindRewrite:
		emptyState(c, rewriteIcon, i18n.T("No rewrites yet"),
			i18n.T("Press your Rewrite shortcut and say what to write, or how to change the selected text: each answer shows up here with your instruction."), func() {
				if ghostButton(c.Key("hist-go-rewrite"), i18n.T("Set up Rewrite")).Clicked() {
					a.navigate("/rewrite")
				}
			})
	default:
		emptyState(c, icons.History, i18n.T("No transcriptions yet"),
			i18n.T("Press your Transcribe shortcut and speak: each result shows up here, to find and copy again."), func() {
				if ghostButton(c.Key("hist-go-transcribe"), i18n.T("Set up Transcribe")).Clicked() {
					a.navigate("/transcribe")
				}
			})
	}
}

// historyRow is an entry in the list: the first line of its result (why
// it failed, when it did), its marks, and when, in which app and how long
// it was spoken. A click shows its details, or hides them; its context
// menu copies, shows or deletes it.
func (a *app) historyRow(c *ui.Context, e store.Entry) {
	t := c.Theme()
	k := tok(t)
	h := &a.hist
	title, failed := uitext.EntryTitle(e)
	row := ui.ButtonBase(c.Key("hist-row-"+e.ID)).Padding(px(t, sp2), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Label(title).Gap(px(t, sp2)).AlignItems(ui.Start)
	switch {
	case h.sel == e.ID:
		row.Background(k.Pressed)
	case row.Hovered():
		row.Background(k.Hover)
	}
	if row.Clicked() {
		if h.sel == e.ID {
			h.sel = ""
		} else {
			h.sel = e.ID
		}
	}
	a.entryMenu(c, row, e, true, func() { h.sel = e.ID }) // history_menu.go
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 3)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				tx := ui.Text(c, title).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if failed {
					tx.TextColor(k.Text2)
				}
				a.entryStatus(c, e) // history_retry.go
			})
			ui.Text(c, uitext.EntryMeta(e)).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
		})
	})
}

// entryMarks are an entry's status marks: failed or cancelled, or left
// in the answer card, or copied from it.
func entryMarks(c *ui.Context, e store.Entry) {
	k := tok(c.Theme())
	switch e.Status {
	case store.StatusFailed:
		tagChip(c, i18n.T("Failed")).TextColor(k.Danger).Tooltip(i18n.T("No recording was kept, so it cannot be retried"))
		return
	case store.StatusCancelled:
		tagChip(c, i18n.T("Cancelled"))
		return
	}
	switch e.Delivery {
	case "card":
		tagChip(c, i18n.T("Not inserted")).Tooltip(i18n.T("Shown in the answer card, and not inserted"))
	case "copy":
		tagChip(c, i18n.T("Copied")).Tooltip(i18n.T("Copied from the answer card"))
	}
}
