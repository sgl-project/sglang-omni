package main

// Settings: the settings service seen from the UI (a snapshot in
// a.prefs, kept by settings.Changed), the language and the theme they
// drive, and the settings dialog (⌘,) with its sections under
// /settings/{section}. The dialog is AudioSmith's, with the sections
// every app of the family has and VoxtNew's own: Audio
// (settings_audio.go), Overlay style (settings_overlay.go) and the table
// of Shortcuts (settings_shortcuts.go). The features' own settings live
// on their pages (Transcribe, Translate, Rewrite…).

import (
	"context"
	"reflect"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// settingsState is the settings dialog's sections' view state, embedded
// in app: each section keeps its own struct in its file.
type settingsState struct {
	storageState // settings_storage.go
	aboutState   // settings_about.go
	settingsForm // shell_settings.go: the controls that edit a setting
}

// settingsForm is the settings pages' view state: the controls bind to
// these, and their changes go to the service.
type settingsForm struct {
	systemLocale string // mygo.App.Locale(), read when the window opens

	language     string // label
	mainLanguage string // label
	theme        int
	scale        int
	reduceMotion bool
	menuBar      bool
	closeToTray  bool
	openAtLogin  bool
	keepHistory  bool
	keepAudio    bool
	logText      bool
	sounds       bool
	skipSilence  bool        // !Audio.VAD.Off
	cleanSound   bool        // !Audio.Clean.Off
	muteOutput   bool        // Audio.MuteOn()
	sidebar      int         // automatic, names and icons, icons only (settings_appearance.go)
	ovStyle      overlayForm // settings_overlay.go

	clearHistoryAsk bool // Privacy's Clear the history… dialog
	// loginIs is whether macOS opens VoxtNew at login, as last read or
	// set: the setting changes it only when they differ.
	loginIs bool
}

// settingsSection is a section of the settings dialog.
type settingsSection struct{ id, title, en string }

func settingsSections() []settingsSection {
	return []settingsSection{
		{"general", i18n.T("General"), "General language menu bar login"},
		{"permissions", i18n.T("Permissions"), "Permissions microphone accessibility privacy"},
		{"audio", i18n.T("Audio"), "Audio microphone input device sounds silence"},
		{"overlay", i18n.T("Overlay style"), "Overlay style position opacity corners recording pill"},
		{"shortcuts", i18n.T("Shortcuts"), "Shortcuts hotkeys keys fn"},
		{"appearance", i18n.T("Appearance"), "Appearance theme size sidebar"},
		{"privacy", i18n.T("Privacy"), "Privacy history audio logs"},
		{"storage", i18n.T("Storage"), "Storage disk data folder reset"},
		// "About" in English: "Diagnostics and about" does not fit the
		// column in bold; Chinese keeps 诊断与关于, which does.
		{"about", i18n.X("settings section", "About"), "Diagnostics about version bug report"},
	}
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

// updateSettings changes the settings: at once in the snapshot, and in
// the service (saved in the background, then announced with Changed).
// fn must only set fields, as it runs on both.
func (a *app) updateSettings(fn func(*settings.Settings)) {
	fn(&a.prefs)
	st := a.settings
	a.async(func() {
		if err := st.Update(context.Background(), fn); err != nil {
			a.post(func() { a.toast(i18n.T("Could not save the settings: %v", err)) })
		}
	})
}

// settingsChanged applies saved settings to the app.
func (a *app) settingsChanged(old, new settings.Settings) {
	a.prefs = new
	a.familySettingsChanged(old, new) // family.go: the shared settings, the manifest
	if old.General.Language != new.General.Language {
		a.applyLocale()
	}
	if old.General.MenuBar != new.General.MenuBar {
		a.applyTray() // tray.go
	}
	if old.Updates.BetaOn(a.version) != new.Updates.BetaOn(a.version) {
		a.applyBeta() // updates.go
	}
	if !reflect.DeepEqual(old.AppEnhancement, new.AppEnhancement) {
		a.syncTrees() // appenhance.go
	}
	if !reflect.DeepEqual(old.Features, new.Features) {
		a.syncHotkeys()   // session_hotkeys.go
		a.buildTrayMenu() // the shortcuts it shows
	}
	if old.General.LaunchAtLogin != new.General.LaunchAtLogin && new.General.LaunchAtLogin != a.loginIs && a.hasMenu {
		on := new.General.LaunchAtLogin
		go func() {
			err := mygo.App.SetOpenAtLogin(on)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not change opening at login: %v", err))
					return
				}
				a.loginIs = on
			})
		}()
	}
	if old.Overlay != new.Overlay {
		a.overlayRestyle() // overlay.go
	}
	if old.Models != new.Models {
		a.applyKeepLoaded(new.Models) // models_keep.go
	}
	a.historySettingsChanged(old, new) // history_settings_ops.go
	// The problems' cards: the language of their text may have changed
	// (shell_notices.go).
	a.checkModels()
	a.checkHelper()
	a.loadSettingsForm()
}

// applyTheme sets the frame's theme from the appearance settings: a
// forced light or dark theme, the UI scale, which grows the text and the
// spacing (sizes of our own scale with px), and the design tokens
// (ui_tokens.go) over MyGo's colours.
func (a *app) applyTheme(c *ui.Context) {
	ap := a.prefs.Appearance
	sys := c.Theme()
	th := *sys
	if (ap.Theme == "light" && sys.Dark) || (ap.Theme == "dark" && !sys.Dark) {
		if ap.Theme == "dark" {
			th = *ui.DarkTheme()
		} else {
			th = *ui.LightTheme()
		}
		if s := c.Preferences().TextScale; s > 0 {
			th.FontSize *= s
		}
	}
	if s := float32(ap.Scale); s > 0 && s != 1 {
		th.FontSize *= s
		th.Spacing = th.Space(1) * s
	}
	styleTheme(&th)
	c.SetTheme(&th)
}

var scaleLabels = []string{"90%", "100%", "110%", "125%"}

// loadSettingsForm sets the controls from a.prefs.
func (a *app) loadSettingsForm() {
	s := a.prefs
	f := &a.settingsForm
	f.language = languageOptions()[0]
	for i, tag := range i18n.Supported() {
		if tag == s.General.Language {
			f.language = languageOptions()[i+1]
		}
	}
	f.mainLanguage = mainLanguageOptions()[0]
	if i := slices.Index(mainLanguages, s.General.MainLanguage); i >= 0 {
		f.mainLanguage = mainLanguageOptions()[i+1]
	}
	f.theme = max(0, slices.Index(settings.Themes, s.Appearance.Theme))
	f.scale = max(0, slices.Index(settings.Scales, s.Appearance.Scale))
	f.reduceMotion = s.Appearance.ReduceMotion
	f.menuBar, f.closeToTray, f.openAtLogin = s.General.MenuBar, s.General.CloseToTray, s.General.LaunchAtLogin
	f.keepHistory, f.keepAudio, f.logText = s.Privacy.KeepHistory, s.Privacy.KeepAudio, s.Privacy.LogText
	f.sounds = s.Audio.Sounds
	f.skipSilence = !s.Audio.VAD.Off
	f.cleanSound = !s.Audio.Clean.Off
	f.muteOutput = s.Audio.MuteOn()
	f.sidebar = sidebarChoice(s.Appearance)
	a.loadOverlayForm() // settings_overlay.go
}

// syncOpenAtLogin reads whether macOS opens VoxtNew at login (the user
// may have turned it off in System Settings) and makes the setting say
// so; it runs when the window opens.
func (a *app) syncOpenAtLogin() {
	go func() {
		on := mygo.App.OpenAtLogin()
		a.post(func() {
			a.loginIs = on
			if on != a.prefs.General.LaunchAtLogin {
				a.updateSettings(func(s *settings.Settings) { s.General.LaunchAtLogin = on })
			}
		})
	}()
}

// openSettings shows the settings dialog at a section's path ("/audio").
func (a *app) openSettings(sub string) {
	if a.setRouter.Path() != sub || !a.settingsOpen {
		a.settingsShown(sub)
	}
	if a.setRouter.Path() != sub {
		a.setRouter.Replace(sub)
	}
	a.settingsOpen = true
}

// settingsShown runs when the settings dialog goes to a section: sections
// that read something when they show start it here.
func (a *app) settingsShown(sub string) {
	switch sub {
	case "/permissions":
		a.recheckPermissions() // permissions.go
	case "/audio":
		a.loadDevices() // settings_audio.go
	}
}

// shownPath is the path of what shows: the settings dialog's section
// ("/settings/audio") while it is open, else the page's.
func (a *app) shownPath() string {
	if a.settingsOpen {
		return "/settings" + a.setRouter.Path()
	}
	return a.router.Path()
}

// settingsDialog is Settings (⌘,): a dialog over the page, its sections
// in a column on the left (AudioSmith 0.3: Typeless-style).
func (a *app) settingsDialog(c *ui.Context) {
	if !a.settingsOpen {
		return
	}
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	dw, dh := settingsDialogSize(t, w, h)
	ui.DialogBase(c.Key("settings-dialog"), &a.settingsOpen, func(backdrop, panel ui.Element) {
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
						cur := strings.TrimPrefix(a.setRouter.Path(), "/")
						cur, _, _ = strings.Cut(cur, "/")
						for _, s := range settingsSections() {
							sel := cur == s.id
							b := ui.ButtonBase(c.Key("settings-"+s.id)).Padding(px(t, 6), px(t, sp2)).Gap(px(t, sp2)).Radius(px(t, radiusControl)).
								Justify(ui.Start).Label(s.title).Checked(sel)
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
								ui.Icon(c, settingsSectionIcon(s.id).Get(sel)).Size(px(t, 16), px(t, 16)).TextColor(col)
								txt := ui.Text(c, s.title).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
								if sel {
									txt.FontWeight(600)
								} else {
									txt.TextColor(k.G[8])
								}
							})
							if b.Clicked() {
								a.navigate("/settings/" + s.id)
							}
						}
					})
				})
			})
			ui.Box(c).Width(1).Background(k.Line)
			ui.Column(c).Grow(1).MinWidth(0).Children(func() {
				a.setRouter.View(c, func(r *ui.Route) {
					title := ""
					for _, s := range settingsSections() {
						if r.Match("/" + s.id) {
							title = s.title
						}
					}
					ui.Row(c).Height(px(t, 52)).Shrink(0).AlignItems(ui.Center).Padding(0, px(t, sp3), 0, px(t, sp6)).Gap(px(t, sp2)).Children(func() {
						ui.Text(c, title).FontSize(px(t, fsSection)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
						if iconButton(c.Key("settings-close"), icons.Close, i18n.T("Close")).Clicked() {
							a.settingsOpen = false
						}
					})
					ui.Scroll(c).Grow(1).MinHeight(0).Children(func() {
						ui.Column(c).Padding(px(t, sp1), px(t, sp6), px(t, sp6)).Gap(px(t, sp4)).MaxWidth(px(t, settingsContentMaxW)).Children(func() {
							a.settingsSection(c, r)
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
func (a *app) settingsSection(c *ui.Context, r *ui.Route) {
	sections := map[string]func(*ui.Context){
		"general":     a.generalSettings,
		"permissions": a.permissionsSettings, // permissions.go
		"audio":       a.audioSettings,       // settings_audio.go
		"overlay":     a.overlaySettings,     // settings_overlay.go
		"shortcuts":   a.shortcutsSettings,   // settings_shortcuts.go
		"appearance":  a.appearanceSettings,
		"privacy":     a.privacySettings, // settings_privacy.go
		"storage":     a.storageSettings, // settings_storage.go
		"about":       a.aboutSettings,   // settings_about.go
	}
	for _, s := range settingsSections() {
		if r.Match("/" + s.id) {
			r.Title(s.title)
			a.lastSettings = s.id
			sections[s.id](c)
			return
		}
	}
	ui.Text(c, i18n.T("There is no page at %s.", a.shownPath())).TextColor(c.Theme().TextMuted)
}
