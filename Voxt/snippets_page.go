package main

// The Snippets page: each snippet as "trigger → text", a dialog to add or
// change one, and on an empty page a few triggers to start from.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type snippetsActions interface {
	saveSnippet(id, trigger, text string) string
	deleteSnippet(id string)
}

// snippetsLocalUI is the page's dialog: open, the snippet it edits (id
// "" for a new one), its fields as they were opened (was) and the problem
// a save gave.
type snippetsLocalUI struct {
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
			case len(state.Rows) == 0:
				snippetExamples(c, state.Examples, local)
			default:
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

// snippetRow is "trigger → text"; a click edits it, and Delete shows
// under the pointer.
func snippetRow(c *ui.Context, row snippetRowViewState, actions snippetsActions, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	b := ui.ButtonBase(c.Key("snippet-"+row.ID)).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
		Gap(px(t, sp2)).AlignItems(ui.Center).Label(row.Trigger + " → " + row.Preview)
	hover := b.Hovered()
	if hover {
		b.Background(k.Hover)
	}
	if b.Children(func() {
		ui.Text(c, row.Trigger).FontWeight(600).SingleLine().Shrink(0).MaxWidth(px(t, 240)).Ellipsis("…")
		ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3).Shrink(0)
		ui.Text(c, row.Preview).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
		if hover && iconButton(c.Key("delete"), icons.TrashBinMinimalistic, i18n.T("Delete")).Label(i18n.T("Delete %s", row.Trigger)).Clicked() {
			actions.deleteSnippet(row.ID)
		}
	}).Clicked() {
		local.open(snippetEdit{id: row.ID, trigger: row.Trigger, text: row.Text})
	}
}

// snippetExamples are an empty page's triggers to start from: each opens
// the dialog with its trigger filled in.
func snippetExamples(c *ui.Context, examples []snippetExampleViewState, local *snippetsLocalUI) {
	t := c.Theme()
	k := tok(t)
	emptyState(c, icons.TextSquare, i18n.T("No snippets yet"), i18n.T("Save what you type again and again: an address, a link, a reply, a prompt."), func() {
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
		fieldInfo(c, i18n.T("Trigger"), i18n.T("What you say: my email, intro reply."), func() {
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
