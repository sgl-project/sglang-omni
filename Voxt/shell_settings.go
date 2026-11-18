package main

// Settings: the settings dialog (⌘,) with its sections under
// /settings/{section}, and the theme. The dialog is AudioSmith's, with the sections
// every app of the family has and VoxtNew's own: Audio
// (settings_audio.go), Overlay style (settings_overlay.go) and the table
// of Shortcuts (settings_shortcuts.go). The features' own settings live
// on their pages (Transcribe, Translate, Rewrite…).

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// settingsForm is the settings pages' local UI state: the controls bind
// to these, and their changes go to the service.
type settingsForm struct {
	systemLocale string // mygo.App.Locale(), read when the window opens

	settingsGeneralLocalUI
	settingsAppearanceLocalUI
	settingsPrivacyLocalUI
	settingsAudioLocalUI
	settingsShortcutsLocalUI
	ovStyle overlayForm // settings_overlay.go

	// loginIs is whether macOS opens VoxtNew at login, as last read or
	// set: the setting changes it only when they differ.
	loginIs     bool
	search      string
	searchFocus bool
	find        finder // Return steps through the matches (find_marks.go)
	findN       int    // the matches in the names the section showed last
}

// settingsSectionIcon is a settings section's icon.
func settingsSectionIcon(id string) icons.Icon {
	switch id {
	case "general":
		return icons.Settings
	case "permissions":
		return permissionsIcon
	case "audio":
		return icons.MicrophoneLarge
	case "overlay":
		return icons.Monitor
	case "shortcuts":
		return icons.Keyboard
	case "appearance":
		return icons.Pallete2
	case "privacy":
		return icons.ShieldCheck
	case "storage":
		return icons.Database
	}
	return icons.InfoCircle
}

// applyTheme sets the frame's theme from the appearance settings: a
// forced light or dark theme, the UI scale, which grows the text and the
// spacing (sizes of our own scale with px), and the design tokens
// (ui_tokens.go) over MyGo's colours.
func applyTheme(c *ui.Context, theme string, scale float64) {
	sys := c.Theme()
	th := *sys
	if (theme == "light" && sys.Dark) || (theme == "dark" && !sys.Dark) {
		if theme == "dark" {
			th = *ui.DarkTheme()
		} else {
			th = *ui.LightTheme()
		}
		if s := c.Preferences().TextScale; s > 0 {
			th.FontSize *= s
		}
	}
	if s := float32(scale); s > 0 && s != 1 {
		th.FontSize *= s
		th.Spacing = th.Space(1) * s
	}
	styleTheme(&th)
	c.SetTheme(&th)
}

type settingsDialogLocalUI struct {
	Open          *bool
	Router        *ui.Router
	Form          *settingsForm
	HistoryAccess *historyAccessLocalUI
	Storage       *settingsStorageLocalUI
	Updates       *updatesLocalUI
}

// settingsDialog is Settings (⌘,): a dialog over the page, its sections
// in a column on the left (AudioSmith 0.3: Typeless-style).
func settingsDialog(c *ui.Context, state settingsDialogViewState, actions settingsDialogActions, local settingsDialogLocalUI) {
	if !state.Open {
		return
	}
	if c.Shortcut(ui.Cmd, ui.KeyF) {
		local.Form.searchFocus = true
	}
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	dw, dh := settingsDialogSize(t, w, h)
	ui.DialogBase(c.Key("settings-dialog"), local.Open, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.32))
		panel.Width(dw).Height(dh).Radius(px(t, radiusCard)).Background(k.Bg).Clip().
			Shadow(0, 16, 48, 0, ui.RGBA(0, 0, 0, 0.28)).Label(i18n.T("Settings"))
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			// As wide as the longest name: 诊断与关于, Overlay style.
			ui.Column(c).Width(px(t, settingsColumnW)).Shrink(0).Background(k.Sidebar).Children(func() {
				ui.Row(c).Height(px(t, 52)).Shrink(0).AlignItems(ui.Center).Padding(0, px(t, sp4)).Children(func() {
					ui.Text(c, i18n.T("Settings")).FontSize(px(t, fsSection)).FontWeight(600).SingleLine()
				})
				ui.Scroll(c).Grow(1).Children(func() {
					ui.Column(c).Padding(0, px(t, sp2), px(t, sp2)).Gap(px(t, 2)).Children(func() {
						cur := strings.TrimPrefix(state.Path, "/")
						cur, _, _ = strings.Cut(cur, "/")
						for _, s := range state.Sections {
							sel := cur == s.ID
							b := ui.ButtonBase(c.Key("settings-"+s.ID)).Padding(px(t, 6), px(t, sp2)).Gap(px(t, sp2)).Radius(px(t, radiusControl)).
								Justify(ui.Start).Label(s.Title).Checked(sel)
							switch {
							case sel:
								b.Background(k.Pressed)
							case b.Hovered():
								b.Background(k.Hover)
							}
							b.Children(func() {
								col := k.Text2
								if sel {
									col = k.Text
								}
								ui.Icon(c, settingsSectionIcon(s.ID).Get(sel)).Size(px(t, 16), px(t, 16)).TextColor(col)
								txt := ui.Text(c, s.Title).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
								if sel {
									txt.FontWeight(600)
								} else {
									txt.TextColor(k.G[8])
								}
							})
							if b.Clicked() {
								actions.navigate("/settings/" + s.ID)
							}
						}
					})
				})
			})
			ui.Box(c).Width(1).Background(k.Line)
			ui.Column(c).Grow(1).MinWidth(0).Children(func() {
				local.Router.View(c, func(r *ui.Route) {
					title := ""
					var sectionEn string
					for _, s := range state.Sections {
						if r.Match("/" + s.ID) {
							title = s.Title
							sectionEn = s.English
						}
					}
					f := local.Form
					fq := strings.TrimSpace(f.search)
					moved := f.find.advance(f.findN)
					ui.Row(c).Height(px(t, 52)).Shrink(0).AlignItems(ui.Center).Padding(0, px(t, sp3), 0, px(t, sp6)).Gap(px(t, sp2)).Children(func() {
						ui.Text(c, title).FontSize(px(t, fsSection)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
						sf := f.find.field(c, "settings-search", &f.search, i18n.T("Find in settings"), px(t, 240))
						if f.findN > 0 { // none: a row found by a hidden tip or its English name
							f.find.count(c, fq, f.findN)
						}
						if f.searchFocus {
							f.searchFocus = false
							sf.Focus()
						}
						if iconButton(c.Key("settings-close"), icons.Close, i18n.T("Close")).Clicked() {
							*local.Open = false
						}
					})
					// The section's name matching shows it whole; else its rows that match.
					query := f.search
					if settingsTextMatches(query, title, sectionEn) {
						query = ""
					}
					ui.Scroll(c).Grow(1).MinHeight(0).Children(func() {
						ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp6)).Gap(px(t, sp4)).MaxWidth(px(t, settingsContentMaxW)).Children(func() {
							settingsSearchQuery, settingsSearchHits = query, 0
							settingsFind.query, settingsFind.n, settingsFind.cur, settingsFind.reveal = fq, 0, f.find.cur, moved
							body := ui.Column(c).Gap(px(t, sp4))
							body.Children(func() { settingsSection(c, r, state, actions, local) })
							settingsSearchQuery, settingsFind.query = "", ""
							if settingsFind.n != f.findN {
								f.findN = settingsFind.n
								c.AnimationFrame() // the count over the section shows it
							}
							if query != "" && settingsSearchHits == 0 { // its other parts (lists, cards) do not match either
								body.Invisible().Height(0)
								caption(c, i18n.T("Nothing in this section matches your search."))
							}
						})
					})
				}).Grow(1).MinHeight(0)
			})
		})
	})
}

// The settings dialog's size: settingsMaxW × settingsMaxH at most, and
// at least a margin of 5% of the window (6% of its height), sp6 at the
// least, around it: 720×458 in the 800×520 window, 612×387 in the
// smallest. Its column of sections is settingsColumnW wide, its content
// settingsContentMaxW at most (it is narrower in the default window).
const (
	settingsMaxW        = 760
	settingsMaxH        = 600
	settingsColumnW     = 176
	settingsContentMaxW = 600
)

func settingsDialogSize(t *ui.Theme, w, h float32) (dw, dh float32) {
	mx := max(px(t, sp6), w*0.05)
	my := max(px(t, sp6), h*0.06)
	return min(w-2*mx, px(t, settingsMaxW)), min(h-2*my, px(t, settingsMaxH))
}

// settingsSection builds the section of the settings dialog r shows.
func settingsSection(c *ui.Context, r *ui.Route, state settingsDialogViewState, actions settingsDialogActions, local settingsDialogLocalUI) {
	// A section action changes the router before the next model rebuild.
	if state.Path != r.Path() {
		return
	}
	for _, s := range state.Sections {
		if !r.Match("/" + s.ID) {
			continue
		}
		r.Title(s.Title)
		switch s.ID {
		case "general":
			generalSettings(c, *state.General, actions, &local.Form.settingsGeneralLocalUI)
		case "permissions":
			permissionsSettings(c, *state.Permissions, actions)
		case "audio":
			audioSettings(c, *state.Audio, actions, &local.Form.settingsAudioLocalUI)
		case "overlay":
			overlaySettings(c, state.Overlay, actions, &local.Form.ovStyle)
		case "shortcuts":
			shortcutsSettings(c, *state.Shortcuts, actions, &local.Form.settingsShortcutsLocalUI)
		case "appearance":
			appearanceSettings(c, *state.Appearance, actions, &local.Form.settingsAppearanceLocalUI)
		case "privacy":
			privacySettings(c, *state.Privacy, actions, &local.Form.settingsPrivacyLocalUI, local.HistoryAccess)
		case "storage":
			storageSettings(c, *state.Storage, actions, local.Storage)
		case "about":
			aboutDetails(c, *state.About)
			updateSettingsSection(c, *state.Updates, actions, local.Updates)
			aboutDiagnostics(c, *state.About, actions)
		}
		return
	}
	ui.Text(c, i18n.T("There is no page at %s.", state.ShownPath)).TextColor(c.Theme().TextMuted)
}
