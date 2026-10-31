package main

// The menu bar icon (AudioSmith's dictation tray, for
// VoxtNew): Start transcribing, Paste the last result, the last three
// results (a click copies one), Open VoxtNew, Settings, Quit. With
// Settings → General → "Keep running when the window closes", closing
// the window hides it, so the shortcuts keep working; the Dock icon or
// the menu brings it back. Start transcribing records for the app in
// front (as the shortcut does) and turns into Stop while a session runs;
// the icon (Voxt's microphone) shows a recording dot meanwhile.

import (
	"context"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/trayicon"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// trayState is the menu bar icon.
type trayState struct {
	tray     *mygo.Tray
	trayLive bool     // the icon shows the recording one
	quitting bool     // Quit was chosen: closing the window must not hide it
	recent   []string // the last results, newest first (trayRecentN at most)
	// copyText puts text on the clipboard (mygo.Clipboard); tests record.
	copyText func(string) error
}

// trayRecentN is how many results the menu lists (each as wide as
// uitext.TrayRecentWidth); trayTipRunes how much of one the tooltip shows.
const (
	trayRecentN  = 3
	trayTipRunes = 400
)

// openTray makes the menu bar icon when the settings want one; it runs in
// App.WhenReady.
func (a *app) openTray() { a.applyTray() }

// applyTray shows or removes the icon as Settings → General says.
func (a *app) applyTray() {
	if !a.hasMenu {
		return
	}
	if !a.prefs.General.MenuBar {
		a.closeTray()
		return
	}
	if a.tray != nil {
		return
	}
	live := a.sess.State.Live()
	// Voxt's microphone, a 36 px template image MyGo shows 18 pt tall
	// (Voxt: 18×18 pt, template); while recording it has a dot.
	t, err := mygo.NewTray(mygo.TrayOptions{Icon: trayicon.Icon(live), IconIsTemplate: true, ToolTip: "VoxtNew"})
	if err != nil {
		a.toast(i18n.T("The menu bar icon could not be made: %v", err))
		return
	}
	a.tray, a.trayLive = t, live
	a.buildTrayMenu()
	a.loadTrayRecent()
}

func (a *app) closeTray() {
	if a.tray != nil {
		a.tray.Destroy()
		a.tray = nil
	}
}

// loadTrayRecent reads the last results from the history for the menu,
// then builds it again; at start and after the history changed.
func (a *app) loadTrayRecent() {
	if a.tray == nil && a.win != nil {
		return // no menu to show them in
	}
	h := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		es, err := h.List(ctx, store.HistoryQuery{Limit: 20})
		if err != nil {
			return
		}
		a.post(func() {
			a.recent = uitext.RecentResults(es, trayRecentN)
			a.buildTrayMenu()
		})
	})
}

// trayTip is a result as its item's tooltip shows it: whole, unless very
// long.
func trayTip(text string) string {
	text = strings.TrimSpace(text)
	if r := []rune(text); len(r) > trayTipRunes {
		return strings.TrimSpace(string(r[:trayTipRunes])) + "…"
	}
	return text
}

// buildTrayMenu builds the icon's menu in the current language, and
// shows the recording icon while a session runs.
func (a *app) buildTrayMenu() {
	if a.tray == nil {
		return
	}
	if live := a.sess.State.Live(); live != a.trayLive {
		if a.tray.SetIcon(trayicon.Icon(live), true) == nil {
			a.trayLive = live
		}
	}
	a.tray.SetMenu(mygo.NewMenu(a.trayItems()))
}

// trayItems are the menu's items.
func (a *app) trayItems() []*mygo.MenuItem {
	act := func(fn func()) func(*mygo.MenuItem, *mygo.Window) {
		return func(*mygo.MenuItem, *mygo.Window) { a.post(fn) }
	}
	f := a.prefs.Features
	start := i18n.T("Start transcribing")
	if a.sess.State.Live() {
		start = i18n.T("Stop transcribing")
	}
	if len(f.Transcribe.Hotkeys) > 0 {
		start += "    " + uitext.HotkeyGlyphs(f.Transcribe.Hotkeys[0].Keys)
	}
	items := []*mygo.MenuItem{
		{Label: start, Click: act(a.toggleTranscribe)},
		{Label: i18n.T("Paste the last result") + "    " + uitext.HotkeyGlyphs(f.PasteLast.Keys), Click: act(func() { a.pasteLast(a.front) })},
	}
	if len(a.recent) > 0 {
		items = append(items, mygo.Separator(), &mygo.MenuItem{Label: i18n.T("Recent results"), Disabled: true})
		for _, text := range a.recent {
			items = append(items, &mygo.MenuItem{Label: uitext.TrayLabel(text), ToolTip: trayTip(text), Click: act(func() { a.copyRecent(text) })})
		}
	}
	items = append(items,
		mygo.Separator(),
		&mygo.MenuItem{Label: i18n.T("Open VoxtNew"), Click: act(a.showMainWindow)},
		&mygo.MenuItem{Label: i18n.T("Settings…"), Click: act(func() { a.showMainWindow(); a.navigate("/settings") })},
	)
	items = append(items, a.updateMenuItems(act)...) // updates.go
	return append(items,
		mygo.Separator(),
		&mygo.MenuItem{Label: i18n.T("Quit VoxtNew"), Click: act(func() { a.quitting = true; mygo.App.Quit() })},
	)
}

// copyRecent copies a recent result (the menu bar).
func (a *app) copyRecent(text string) {
	cp := a.copyText
	if cp == nil {
		cp = func(s string) error { return mygo.Clipboard.Write(transfer.TextData(s)) }
	}
	if err := cp(text); err != nil {
		a.toast(i18n.T("Could not copy: %v", err))
	}
}

// showMainWindow brings the main window back (it may be hidden in the
// menu bar).
func (a *app) showMainWindow() {
	if a.win != nil {
		a.win.Show()
	}
}

// connectWindowLife keeps VoxtNew in the menu bar when its window closes
// (the setting, while the menu bar icon shows), and quits otherwise; it
// reads the permissions again whenever VoxtNew comes to the front
// (System Settings may have changed them).
func (a *app) connectWindowLife() {
	if a.win == nil {
		return
	}
	a.win.OnClose(func(e *mygo.CloseEvent) {
		if a.prefs.General.CloseToTray && !a.quitting && a.tray != nil {
			e.PreventDefault()
			a.win.Hide()
		}
	})
	a.win.OnClosed(func() { mygo.App.Quit() })
	mygo.App.OnActivate(func(hasVisibleWindows bool) {
		if !a.win.IsVisible() {
			a.win.Show()
		}
	})
	mygo.App.OnBeforeQuit(func(*mygo.QuitEvent) { a.quitting = true })
	mygo.App.OnDidBecomeActive(func() {
		a.post(a.recheckPermissions)
		a.post(a.familyActivated) // family.go: the family's files again (notifications are hints)
	})
}
