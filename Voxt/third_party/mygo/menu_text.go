package mygo

import (
	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/internal/platform"
)

// The labels menu items take from their role (MenuItem.Role) are MyGo's
// own strings: they say the language the app shows (App.Locale), and an
// item whose label is its role's says the new language's when it changes.

// menuMessages returns MyGo's own strings, which the labels of roles are,
// in the language the app shows; in its own language before it worked out
// which (App.Locale).
func menuMessages() l10n.Messages { return l10n.In(appLanguage()) }

// appLanguage returns the language the app shows, or "" before it worked
// out which.
func appLanguage() string {
	if s := locales.shown.Load(); s != nil {
		return s.locale
	}
	return ""
}

// platformTexts returns the words the backends say of their own in a
// language (platform.Texts).
func platformTexts(locale string) platform.Texts {
	tx := l10n.In(locale)
	return platform.Texts{OK: tx.Ok(), Cancel: tx.Cancel(), Open: tx.Open(), Save: tx.Save(),
		SelectFolder: tx.SelectFolder(), About: tx.MenuAbout("\x00")}
}

// relabelMenus gives the items whose label is their role's the role's
// label now, as the language the app shows changed, and shows the menus
// in use with them: the menus of the app and of windows anew, and the
// others, as those of trays and the Dock, item by item. Main thread only.
func relabelMenus() {
	menuMu.Lock()
	var changed []*platform.MenuItem
	for _, wp := range menuItems.byID {
		it := wp.Value()
		if it == nil || it.Role == "" || it.roleLabel == "" || it.Label != it.roleLabel {
			continue
		}
		if label, _ := roleDefaults(it.Role); label != it.Label {
			it.Label, it.roleLabel = label, label
			changed = append(changed, it.snapshotLocked())
		}
	}
	menuMu.Unlock()
	if len(changed) == 0 || !App.IsReady() {
		return // menus take their labels as Run installs them
	}
	if m := App.Menu(); m != nil {
		backend().SetApplicationMenu(m.snapshot())
	}
	for _, w := range Windows() {
		if w.menu != nil && w.native != nil {
			w.native.SetMenu(w.menu.snapshot())
		}
	}
	for _, it := range changed {
		backend().UpdateMenuItem(it)
	}
}
