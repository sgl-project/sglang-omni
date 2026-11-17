package main

// The shell's command views (shell_commands_controller.go keeps the
// registry): the ⌘K palette, the "?" sheet listing the shortcuts, and
// the window's own shortcuts.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// paletteLocalUI is the palette's interface-local state: open, the
// query and the selection.
type paletteLocalUI struct {
	open  *bool
	query *string
	sel   *int
}

// commandIcon is a command's icon in the palette: its page's for "Go
// to" commands, a settings section's, or nil.
func commandIcon(id string) *icons.Icon {
	if rest, ok := strings.CutPrefix(id, "go/settings"); ok {
		ic := icons.Settings
		if sec := strings.TrimPrefix(rest, "/"); sec != "" {
			ic = settingsSectionIcon(sec)
		}
		return &ic
	}
	if rest, ok := strings.CutPrefix(id, "go"); ok {
		for _, p := range pages {
			if p.path == rest {
				ic := p.icon
				return &ic
			}
		}
	}
	return nil
}

// kbd shows a shortcut as a key cap.
func kbd(c *ui.Context, keys string) ui.Element {
	t := c.Theme()
	k := tok(t)
	return ui.Text(c, keys).FontSize(px(t, fsMicro)).TextColor(k.Text2).Padding(px(t, 1), px(t, 5)).Radius(px(t, 4)).
		Background(k.Subtle).Border(1, k.Line).SingleLine().Shrink(0)
}

// paletteView is the ⌘K command palette.
func paletteView(c *ui.Context, state *paletteViewState, actions paletteActions, local *paletteLocalUI) {
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	ui.DialogBase(c, local.open, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.2)).Justify(ui.Start).Padding(min(px(t, 90), h*0.12), px(t, sp6), 0)
		pw := min(px(t, 560), w-2*px(t, sp6))
		panel.Width(pw).MaxWidth(pw).Radius(px(t, radiusCard)).Background(k.Bg).Border(1, k.Line).Clip().
			Shadow(0, 16, 40, 0, ui.RGBA(0, 0, 0, 0.22)).Label(i18n.T("Command palette"))
		items := state.Items
		*local.sel = max(0, min(*local.sel, len(items)-1))
		ui.Row(c).Padding(px(t, 10), px(t, 14)).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, searchIcon.Linear).Size(px(t, 18), px(t, 18)).TextColor(k.Text3)
			// A text input rather than ui.SearchField, whose placeholder
			// is not translated; Escape closes the palette.
			search := ui.TextInputBase(c.Key("palette-search"), local.query).AutoFocus().Grow(1).FontSize(px(t, 15)).
				Placeholder(i18n.T("Type a command or a page…")).Label(i18n.T("Search commands"))
			if search.Changed() {
				*local.sel = 0
			}
			if search.Shortcut(0, ui.KeyDown) && len(items) > 0 {
				*local.sel = (*local.sel + 1) % len(items)
			}
			if search.Shortcut(0, ui.KeyUp) && len(items) > 0 {
				*local.sel = (*local.sel - 1 + len(items)) % len(items)
			}
			if search.Submitted() && len(items) > 0 {
				actions.runCommandID(items[*local.sel].ID)
			}
			kbd(c, "Esc")
		})
		ui.Divider(c)
		ui.Scroll(c).MaxHeight(min(px(t, 380), h*0.6)).Children(func() {
			ui.Column(c).Padding(px(t, 6)).Children(func() {
				if len(items) == 0 {
					ui.Text(c, i18n.T("No matching commands")).TextColor(k.Text2).Padding(px(t, 10))
				}
				group := ""
				for i, cmd := range items {
					if cmd.Group != group {
						group = cmd.Group
						ui.Text(c, group).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text3).Padding(px(t, sp2), px(t, 10), px(t, 4))
					}
					row := ui.ButtonBase(c.Key("cmd-"+cmd.ID)).Padding(px(t, 7), px(t, 10)).Radius(px(t, radiusControl)).Label(cmd.Title)
					if i == *local.sel {
						row.Background(k.Pressed)
					} else if row.Hovered() {
						row.Background(k.Hover)
					}
					row.Children(func() {
						ui.Row(c).Grow(1).Gap(px(t, 10)).AlignItems(ui.Center).Children(func() {
							ic := icons.Command
							if p := commandIcon(cmd.ID); p != nil {
								ic = *p
							}
							ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2)
							ui.Text(c, cmd.Title).Grow(1).SingleLine().Ellipsis("…").MinWidth(0)
							if cmd.Keys != "" {
								kbd(c, cmd.Keys)
							}
						})
					})
					if row.Clicked() {
						actions.runCommandID(cmd.ID)
					}
				}
			})
		})
	})
}

// shortcutSheetView lists the keyboard shortcuts ("?"). open binds the
// sheet's visibility.
func shortcutSheetView(c *ui.Context, state *sheetViewState, open *bool) {
	t := c.Theme()
	k := tok(t)
	ui.Modal(c, open, func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Icon(c, icons.Keyboard.Linear).Size(px(t, 18), px(t, 18)).TextColor(k.Text2)
			sectionHeading(c, i18n.T("Keyboard shortcuts")).Grow(1)
		})
		// The list scrolls in a short window (680×440): Close stays in it.
		_, h := c.Size()
		ui.Scroll(c.Key("sheet-scroll")).MaxHeight(max(px(t, 160), h-px(t, 180))).Children(func() {
			ui.Column(c).Gap(px(t, sp3)).Width(px(t, 440)).MaxWidth(px(t, 440)).Children(func() {
				for _, g := range state.Groups {
					sectionTitle(c, g.Title)
					ui.Grid(c).ColumnTracks(ui.Fixed(px(t, 100)), ui.Fr(1)).GapX(px(t, sp3)).GapY(px(t, 6)).AlignItems(ui.Center).Children(func() {
						for _, r := range g.Rows {
							ui.Row(c).Children(func() { kbd(c, r.Keys) })
							ui.Text(c, r.Title)
						}
					})
				}
			})
		})
		ui.Row(c).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("sheet-close"), i18n.T("Close")).Clicked() {
				*open = false
			}
		})
	})
}

// shellShortcuts handles the window's keys (the nav model's shortcut
// pages). Each opens rather than toggles what the native menu's items
// also open, so a key that reaches both does the same thing once.
func shellShortcuts(c *ui.Context, state *navViewState, actions shellKeyActions, sheetOpen *bool) {
	if c.Shortcut(ui.Cmd, ui.KeyK) {
		actions.openPalette()
	}
	if c.Shortcut(ui.Cmd, ui.KeyComma) {
		actions.navigate("/settings")
	}
	for _, p := range state.ShortcutPages {
		if c.Shortcut(ui.Cmd, ui.Key1+ui.Key(p.Num-1)) {
			actions.navigate(p.Path)
		}
	}
	if c.Shortcut(ui.Cmd|ui.Shift, ui.KeyL) {
		actions.toggleLogsPanel()
	}
	if c.Shortcut(ui.Shift, ui.KeySlash) {
		*sheetOpen = true
	}
}
