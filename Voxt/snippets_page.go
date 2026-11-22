package main

// The Snippets page: each snippet as "trigger → text", found by a search
// and sorted, a dialog to add or change one, an import from a JSON file,
// and on an empty page a few triggers to start from.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type snippetsActions interface {
	saveSnippet(id, trigger, text string) string
	deleteSnippet(id string)
	importSnippets()
}

// snippetsLocalUI is the page's search and sort, and its dialog: open,
// the snippet it edits (id "" for a new one), its fields as they were
// opened (was) and the problem a save gave.
type snippetsLocalUI struct {
	query    string
	order    snippetOrder
	editOpen bool
	edit     snippetEdit
	was      snippetEdit
	err      string
}

type snippetEdit struct{ id, trigger, text string }

func (l *snippetsLocalUI) open(e snippetEdit) {
	l.editOpen, l.edit, l.was, l.err = true, e, e, ""
}

func snippetsPage(c *ui.Context, state *snippetsViewState, actions snippetsActions, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	columnPageHeader(c, i18n.T("Snippets"), func() {
		if ui.Button(c.Key("snippet-import"), i18n.T("Import…")).Label(i18n.T("Import snippets")).Clicked() {
			actions.importSnippets()
		}
		if primaryButton(c.Key("snippet-add"), i18n.T("Add snippet")).Clicked() {
			local.open(snippetEdit{})
		}
	})
	ui.Scroll(c.Key("snippets-scroll")).Grow(1).MinHeight(0).Children(func() {
		pageColumn(c).Gap(px(t, sp3)).Children(func() {
			ui.Text(c, i18n.T("Say a trigger and Voxt types the text you saved, as it is.")).TextColor(k.Text2)
			if state.Error != "" {
				warnLine(c, state.Error)
			}
			switch {
			case state.Loading:
			case state.Total == 0:
				snippetExamples(c, state.Examples, local)
			default:
				snippetsToolbar(c, local)
				if len(state.Rows) == 0 {
					emptyState(c, icons.Magnifer, i18n.T("No snippet matches “%s”", strings.TrimSpace(local.query)), "", func() {
						if ghostButton(c.Key("snippet-clear-search"), i18n.T("Clear search")).Clicked() {
							local.query = ""
						}
					})
					break
				}
				card(c.Key("snippets")).Padding(px(t, sp1)).Gap(0).Children(func() {
					for _, row := range state.Rows {
						snippetRow(c, row, actions, local)
					}
				})
			}
		})
	})
	snippetDialog(c, actions, local)
}

// snippetsToolbar is the search, and the sort menu at the end of its
// line.
func snippetsToolbar(c *ui.Context, local *snippetsLocalUI) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		searchField(c, "snippet-search", &local.query, i18n.T("Search snippets"), px(t, 320))
		ui.Box(c.Key("snippet-toolbar-gap")).Grow(1)
		iconButton(c.Key("snippet-sort"), icons.SortVertical, i18n.T("Sort")).Menu(func(m *ui.Menu) {
			for _, o := range []struct {
				order snippetOrder
				label string
			}{
				{snippetsNewest, i18n.T("Newest first")},
				{snippetsOldest, i18n.T("Oldest first")},
				{snippetsByTrigger, i18n.T("By trigger (A–Z)")},
			} {
				if m.Item(o.label).Checked(local.order == o.order).Chosen() {
					local.order = o.order
				}
			}
		})
	})
}

// snippetRow is "trigger → text"; a click edits it, and Edit and Delete
// show under the pointer, beside it (not inside its button, so that they
// are controls of their own).
func snippetRow(c *ui.Context, row snippetRowViewState, actions snippetsActions, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	r := ui.Row(c.Key("snippet-"+row.ID)).Padding(px(t, sp1), px(t, sp2), px(t, sp1), px(t, sp3)).Radius(px(t, radiusControl)).
		Gap(px(t, sp1)).AlignItems(ui.Center).MinHeight(px(t, 44))
	hover := r.Hovered()
	if hover {
		r.Background(k.Hover)
	}
	edit := func() { local.open(snippetEdit{id: row.ID, trigger: row.Trigger, text: row.Text}) }
	r.Children(func() {
		b := ui.ButtonBase(c.Key("open")).Grow(1).MinWidth(0).Gap(px(t, sp2)).AlignItems(ui.Center).Label(row.Trigger + " → " + row.Preview)
		if b.Children(func() {
			ui.Text(c, row.Trigger).SingleLine().Shrink(0).MaxWidth(px(t, 240)).Ellipsis("…")
			ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3).Shrink(0)
			ui.Text(c, row.Preview).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
		}).Clicked() {
			edit()
		}
		// The icons' place is kept when they hide, so the text does not
		// move under the pointer.
		ui.Row(c).Width(px(t, 2*snippetIcon+2)).Shrink(0).Gap(px(t, 2)).AlignItems(ui.Center).Children(func() {
			if !hover {
				return
			}
			if snippetIconButton(c.Key("edit"), icons.Pen2, i18n.T("Edit")).Label(i18n.T("Edit %s", row.Trigger)).Clicked() {
				edit()
			}
			if snippetIconButton(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete")).Label(i18n.T("Delete %s", row.Trigger)).Clicked() {
				actions.deleteSnippet(row.ID)
			}
		})
	})
}

// snippetIcon is the size of a row's icon buttons: smaller than
// iconButton's, a pair that sits close.
const snippetIcon = 22

func snippetIconButton(c *ui.Context, ic icons.Icon, tip string) ui.Element {
	t := c.Theme()
	b := ghostBase(c).Size(px(t, snippetIcon), px(t, snippetIcon)).Center().Label(tip).Tooltip(tip)
	b.Children(func() {
		col := tok(t).Text3
		if b.Hovered() {
			col = tok(t).Text
		}
		ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col)
	})
	return b
}

// snippetExamples are an empty page's triggers to start from: each opens
// the dialog with its trigger filled in.
func snippetExamples(c *ui.Context, examples []snippetExampleViewState, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	emptyState(c, icons.TextSquare, i18n.T("No snippets yet"), i18n.T("Save what you type again and again: an address, a link, a reply, a prompt."), func() {
		// One example a line, each as wide as its text, in English and
		// in Chinese alike.
		ui.Column(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			for index, ex := range examples {
				b := ui.ButtonBase(c.Key(index)).Padding(px(t, 6), px(t, sp3)).Radius(px(t, radiusControl)).Border(1, k.Line).
					Gap(px(t, 6)).AlignItems(ui.Center).Label(ex.Trigger)
				if b.Hovered() {
					b.Background(k.Hover)
				}
				if b.Children(func() {
					ui.Text(c, "“"+ex.Trigger+"”").SingleLine()
					ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 12), px(t, 12)).TextColor(k.Text3)
					ui.Text(c, ex.Hint).TextColor(k.Text3).SingleLine()
				}).Clicked() {
					local.open(snippetEdit{trigger: ex.Trigger})
				}
			}
		})
	})
}

// snippetDialog adds or changes a snippet.
func snippetDialog(c *ui.Context, actions snippetsActions, local *snippetsLocalUI) {
	t := c.Theme()
	if !local.editOpen {
		return
	}
	edit := &local.edit
	title := i18n.T("Add snippet")
	if edit.id != "" {
		title = i18n.T("Edit snippet")
	}
	submit := func() {
		if local.err = actions.saveSnippet(edit.id, edit.trigger, edit.text); local.err == "" {
			local.editOpen = false
		}
	}
	if dialogFrame(c, "snippet-dlg", &local.editOpen, title, "", 480, func() {
		fieldInfo(c, i18n.T("Trigger"), i18n.T("Words you won’t say by accident, like “insert my intro”: Voxt replaces them wherever they come up."), func() {
			input := ui.TextInput(c.Key("snippet-trigger"), &edit.trigger).Label(i18n.T("Trigger to say")).Placeholder(i18n.T("my email")).AutoFocus()
			if input.Changed() {
				local.err = ""
			}
		})
		fieldInfo(c, i18n.T("Snippet"), i18n.T("What Voxt types, as written here."), func() {
			area := ui.TextArea(c.Key("snippet-text"), &edit.text).Label(i18n.T("Text of the snippet")).MinHeight(px(t, 120))
			if area.Changed() {
				local.err = ""
			}
		})
		if local.err != "" {
			warnLine(c, local.err)
		}
	}, func() {
		if edit.id != "" && ghostButton(c.Key("snippet-delete"), i18n.T("Delete")).Clicked() {
			local.editOpen = false
			actions.deleteSnippet(edit.id)
		}
		ui.Spacer(c)
		if ui.Button(c.Key("snippet-cancel"), i18n.T("Cancel")).Clicked() {
			local.editOpen = false
		}
		if primaryButton(c.Key("snippet-save"), i18n.T("Save")).Clicked() {
			submit()
		}
	}).Dismissed() && local.edit != local.was {
		local.editOpen = true
	}
}
