package main

// Updates: MyGo's updater plugin in native UI (native.Plugin, main.go)
// checks actnow.dev once a day, shows a new version's notes in its
// own window, in the language VoxtNew shows, and installs it (a delta
// when there is one) on the user's word, then offers to restart. Its
// "Check for Updates…" is ours, in the app menu, the menu bar icon's menu,
// the palette and Settings → About, which also shows its two preferences.
// Release builds only: `updates` is put in mygo.json by the release
// workflow (docs/release.md), so other builds cannot update themselves.
//
// The updater replaces VoxtNew.app while the old version runs. Until it
// restarts, the helper on disk is the new version's: the supervisor does
// not start it (helper.FailUpdated), and a card asks to restart VoxtNew
// (appUpdated), which a timer, a check and the helper's failure look for
// (checkBundle, appbundle.Updated).
//
// Beta updates (Settings → About): betas are in the feed's beta channel
// (https://actnow.dev/updates/voxtnew/beta, whose manifest is the newer
// of the newest beta and release), which MyGo reads only while
// mygo.Updater.SetPrereleases is on (patch 0006);
// the switch is Updates.Beta in VoxtNew's settings, on by default while
// the running version is itself a beta (settings.Updates.BetaOn), and
// applied when the app starts and when it changes (applyBeta).

import (
	"os"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// updateCheck is the update check of -tags updatecheck builds
// (update_check.go, scripts/mac-update-e2e.sh); nil in others.
var updateCheck func(report string)

// argAfter is the argument after flag on the command line ("": none).
func argAfter(flag string) string {
	for i := 1; i+1 < len(os.Args); i++ {
		if os.Args[i] == flag {
			return os.Args[i+1]
		}
	}
	return ""
}

// updateService is what VoxtNew uses of the updater plugin, a fake in
// tests (the plugin's functions panic without it).
type updateService interface {
	// Prefs reads the preferences (updater.json) and whether this build
	// can update itself: I/O, on a goroutine.
	Prefs() updatePrefs
	SetAutomaticChecks(on bool)
	SetAutomaticDownloads(on bool)
	// SetPrereleases makes checks offer betas too (memory only).
	SetPrereleases(on bool)
	// CheckNow checks as the user asked: the update window shows at once.
	CheckNow()
	// OnChange calls fn on the main thread after the preferences or the
	// time of the last check changed.
	OnChange(fn func()) (off func())
}

// updatePrefs is a snapshot of the updater for Settings → About.
type updatePrefs struct {
	Enabled     bool // this build can update itself
	AutoCheck   bool // checks once a day in the background
	AutoInstall bool // installs what a background check finds without asking
	LastCheck   time.Time
}

// pluginUpdates is the updater plugin (mygo.Use(native.New(...))).
type pluginUpdates struct{}

func (pluginUpdates) Prefs() updatePrefs {
	return updatePrefs{
		Enabled: mygo.Updater.Enabled(), AutoCheck: updater.AutomaticChecks(),
		AutoInstall: updater.AutomaticDownloads(), LastCheck: updater.LastCheck(),
	}
}
func (pluginUpdates) SetAutomaticChecks(on bool)      { updater.SetAutomaticChecks(on) }
func (pluginUpdates) SetAutomaticDownloads(on bool)   { updater.SetAutomaticDownloads(on) }
func (pluginUpdates) SetPrereleases(on bool)          { mygo.Updater.SetPrereleases(on) }
func (pluginUpdates) CheckNow()                       { updater.CheckForUpdates() }
func (pluginUpdates) OnChange(fn func()) (off func()) { return updater.OnChange(fn) }

// updateState is the updater as the UI sees it.
type updateState struct {
	upd        updateService // nil: no updater (the overlay check)
	updPrefs   updatePrefs   // as Prefs last said
	updKnown   bool          // updPrefs was read
	updLoading bool
	// bundleStale reports that the app on disk is another version than
	// the one running (appbundle.Updated); nil: never (tests set it).
	bundleStale func() bool
	// bundleUpdated is set once it did: the card asks to restart.
	bundleUpdated bool
}

// The update's card: its ID, and the link of its action, which restarts
// VoxtNew (cardAction).
const (
	problemUpdate = "problem/update"
	linkRelaunch  = "action:relaunch"
)

// bundleWatch is how often the running app looks for an update installed
// in the background.
const bundleWatch = time.Minute

// connectUpdates follows the updater's preferences; main calls it with
// upd and bundleStale set.
func (a *app) connectUpdates() {
	// A card kept from before a restart: the restart finished the update.
	a.setProblem(problemUpdate, "", nil)
	if a.upd == nil {
		return
	}
	a.upd.OnChange(func() {
		a.post(func() {
			a.updKnown = false
			a.loadUpdates()
			a.checkBundle() // a check may be followed by an install
		})
	})
}

// loadUpdates reads the updater's preferences in the background.
func (a *app) loadUpdates() {
	if a.upd == nil || a.updLoading || a.updKnown {
		return
	}
	a.updLoading = true
	upd := a.upd
	a.async(func() {
		p := upd.Prefs()
		a.post(func() { a.updPrefs, a.updKnown, a.updLoading = p, true, false })
	})
}

// checkForUpdates opens the update window (the menus, the palette,
// Settings → About).
func (a *app) checkForUpdates() {
	if a.upd != nil {
		a.upd.CheckNow()
	}
}

// updateMenuItems is Check for Updates… for a menu (none without the
// updater); act runs a function on the main thread.
func (a *app) updateMenuItems(act func(func()) func(*mygo.MenuItem, *mygo.Window)) []*mygo.MenuItem {
	if a.upd == nil {
		return nil
	}
	return []*mygo.MenuItem{{Label: i18n.T("Check for Updates…"), Click: act(a.checkForUpdates)}}
}

// setAutoCheck and setAutoInstall change the updater's preferences
// (Settings → About), which it saves.
func (a *app) setAutoCheck(on bool) {
	a.updPrefs.AutoCheck = on
	upd := a.upd
	a.async(func() { upd.SetAutomaticChecks(on) })
}

// betaOn reports whether updates include betas: the user's choice, else
// whether this version is a beta.
func (a *app) betaOn() bool { return a.prefs.Updates.BetaOn(a.version) }

// applyBeta tells the updater whether to offer betas; it runs once the
// version is known (main, after openWindow) and when the choice changes.
func (a *app) applyBeta() {
	if a.upd != nil {
		a.upd.SetPrereleases(a.betaOn())
	}
}

// setBeta is the user's choice of Beta updates, which lasts.
func (a *app) setBeta(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Updates.Beta = &on })
	a.applyBeta()
}

func (a *app) setAutoInstall(on bool) {
	a.updPrefs.AutoInstall = on
	upd := a.upd
	a.async(func() { upd.SetAutomaticDownloads(on) })
}

// watchBundle looks for an update installed while VoxtNew runs, every
// bundleWatch; it runs once the window is open.
func (a *app) watchBundle() {
	if a.bundleStale == nil {
		return
	}
	stale := a.bundleStale
	go func() {
		t := time.NewTicker(bundleWatch)
		defer t.Stop()
		for range t.C {
			if stale() {
				a.post(a.appUpdated)
				return
			}
		}
	}()
}

// checkBundle looks for an installed update once, in the background.
func (a *app) checkBundle() {
	if a.bundleStale == nil || a.bundleUpdated {
		return
	}
	stale := a.bundleStale
	a.async(func() {
		if stale() {
			a.post(a.appUpdated)
		}
	})
}

// appUpdated shows that an update is installed and VoxtNew is to
// restart: a card (with Restart VoxtNew), Settings → About and Home.
func (a *app) appUpdated() {
	a.bundleUpdated = true
	a.setProblem(problemUpdate, "installed", &core.Notification{Severity: core.SeverityWarning,
		Title: i18n.T("Update installed"), Body: i18n.T("Restart VoxtNew to finish updating."), Link: linkRelaunch})
}

// relaunch restarts VoxtNew into the version on disk: the quit stops the
// helper (OnQuit), the new version starts its own.
func (a *app) relaunch() { appRelaunch() }

// updateSettingsSection is the Updates part of Settings → About.
func (a *app) updateSettingsSection(c *ui.Context) {
	if a.upd == nil {
		return
	}
	a.loadUpdates()
	p := a.updPrefs
	sectionTitle(c, i18n.T("Updates"))
	if a.bundleUpdated {
		settingRowNote(c, "about-relaunch", i18n.T("Update installed"), i18n.T("The new version runs once VoxtNew restarts."), func() {
			if primaryButton(c.Key("about-relaunch"), i18n.T("Restart VoxtNew")).Clicked() {
				a.relaunch()
			}
		})
	}
	desc := ""
	switch {
	case !a.updKnown:
	case !p.Enabled:
		desc = i18n.T("This copy cannot update itself: it is a development build, or it is in a folder VoxtNew cannot write to. Move VoxtNew to Applications.")
	case p.LastCheck.IsZero():
		desc = i18n.T("New versions come from actnow.dev, with their notes. Not checked yet.")
	default:
		desc = i18n.T("New versions come from actnow.dev, with their notes. Last checked %s.", uitext.TimeAgo(p.LastCheck))
	}
	settingRowNote(c, "about-update", i18n.T("Check for updates"), desc, func() {
		if ui.Button(c.Key("about-update"), i18n.T("Check for Updates…")).Clicked() {
			a.checkForUpdates()
		}
	})
	if !a.updKnown || !p.Enabled {
		return
	}
	settingRow(c, "about-update-auto", i18n.T("Check automatically"), i18n.T("Once a day, in the background. A new version opens a window with its notes, which asks before installing it."), func() {
		if ui.Switch(c.Key("about-update-auto-switch"), &a.updPrefs.AutoCheck).Label(i18n.T("Check for updates automatically")).Changed() {
			a.setAutoCheck(a.updPrefs.AutoCheck)
		}
	})
	beta := a.betaOn()
	note := i18n.T("Releases only. Turning it on offers betas too: new features sooner, and more bugs.")
	switch {
	case beta:
		note = i18n.T("Betas and releases, whichever is newer. Betas bring new features sooner, and more bugs.")
	case strings.Contains(a.version, "-"):
		note = i18n.T("Releases only. This beta stays until a release newer than it comes out: VoxtNew never goes back to an older version.")
	}
	settingRowNote(c, "about-update-beta", i18n.T("Beta updates"), note, func() {
		if ui.Switch(c.Key("about-update-beta-switch"), &beta).Label(i18n.T("Get beta versions")).Changed() {
			a.setBeta(beta)
		}
	})
	settingRow(c, "about-update-install", i18n.T("Install updates automatically"), i18n.T("Updates found in the background download and install without asking; VoxtNew then asks to restart. Off by default."), func() {
		if ui.Switch(c.Key("about-update-install-switch"), &a.updPrefs.AutoInstall).Label(i18n.T("Install updates without asking")).Disabled(!a.updPrefs.AutoCheck).Changed() {
			a.setAutoInstall(a.updPrefs.AutoInstall)
		}
	})
}
