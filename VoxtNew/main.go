// VoxtNew is a voice input app for macOS on local models only (speech
// recognition on sglang-omni's native MLX servers, language models in a
// process of their own): press a shortcut to transcribe what you say into
// any text field, to
// translate speech or the selection, or to rewrite the selection (or write
// new text) from a spoken instruction, with per-app scene rules. The
// window is MyGo native UI in Go; a small Swift helper (voxtnew-helper)
// records the microphone, watches the keyboard and reads browser URLs.
//
// Package main is the UI (views and their state); the domain lives in
// internal/ (see docs/architecture.md). main only wires them together:
// each step is a function in the file of the area that owns it.
package main

import (
	"log"
	"os"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/plugins/updater"
	"github.com/egoist/mygo/plugins/updater/native"

	// MyGo's own strings (a text field's menu, a dialog's buttons) in
	// Chinese, when the app shows it (showLanguage).
	_ "github.com/egoist/mygo/locales/zh-Hans"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/datadir"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
)

func main() {
	// The languages VoxtNew is written in, of which MyGo shows one
	// (showLanguage, settings_locale.go).
	mygo.App.SetLanguages(i18n.Supported()...)
	// The update check of -tags updatecheck builds (update_check.go) runs
	// instead of the app: no window, no data, no lock.
	if out := argAfter("--update-check"); out != "" && updateCheck != nil {
		updateCheck(out)
		return
	}
	// The overlay check (overlay_check.go) runs instead of the app: no
	// data, no lock, no helper.
	if dir := overlayCheckDir(); dir != "" {
		runOverlayCheck(dir)
		return
	}
	// The logs' screenshots of -tags logscheck builds (logs_check.go): a
	// window of their own over a temporary library, no lock, no helper.
	if dir := argAfter("--logs-check"); dir != "" && logsCheck != nil {
		logsCheck(dir)
		return
	}
	// The packaged app's smoke run (smoke.go, scripts/mac-regress.sh): its
	// own data, no window, no lock.
	if dir := argAfter("--smoke"); dir != "" {
		os.Exit(runSmoke(dir))
	}
	// Dev builds are "VoxtNew Dev"; share data with the installed app.
	mygo.App.SetName("VoxtNew")
	// One VoxtNew at a time (one library, one set of global shortcuts);
	// a second launch shows the first one's window.
	if !mygo.App.RequestSingleInstanceLock() {
		return
	}
	// Updates (updates.go): the update window in native UI, in the
	// language VoxtNew shows (App.Locale), checking once a day; it
	// installs only when the user says so, unless they turn on
	// Settings → About → Install updates automatically.
	mygo.Use(native.New(updater.Options{}))

	home, err := mygo.App.Path(mygo.PathUserData)
	if err != nil {
		log.Fatal(err)
	}
	// The data folder may have moved (Settings → Storage); a move or an
	// erase asked for finishes here, before the library opens.
	dataDir, err := datadir.Resolve(home) // internal/datadir
	if err != nil {
		showFatal(err)
		return
	}
	resources, _ := mygo.App.Path(mygo.PathResources)
	logDir, _ := mygo.App.Path(mygo.PathLogs)
	cacheDir, _ := mygo.App.Path(mygo.PathCache)

	// Wiring: one call per area, each defined in the file of that area.
	lib, err := openLibrary(dataDir) // library.go
	if err != nil {
		showFatal(err) // shell.go: explains, then quits
		return
	}
	h := newHelperHost(resources, logDir)         // helperlink.go
	a := newApp(lib, h)                           // app.go
	a.dataHome = home                             // settings_storage.go
	a.openIcons(cacheDir)                         // appicons.go
	a.muteJournal = outputJournal(home, cacheDir) // mute_link.go
	a.connectShell(logDir)                        // shell.go
	a.connectHelper()                             // helperlink.go
	a.upd, a.bundleStale = pluginUpdates{}, h.stale
	a.connectUpdates() // updates.go
	a.startFamily()    // family.go: the manifest, the other ActNow apps, the first start's migration

	mygo.App.WhenReady(func() {
		a.openWindow()        // shell.go
		a.applyBeta()         // updates.go: betas or not, before the first check
		a.startHelper()       // helperlink.go
		a.openTray()          // tray.go: the menu bar icon, close to the menu bar
		a.connectWindowLife() // tray.go
		a.watchBundle()       // updates.go: an update installed while it runs
		a.expireRecordings()  // history_retry.go: recordings kept for a retry, past their time
		mygo.App.OnSecondInstance(func([]string, string) { a.post(a.showMainWindow) })
	})
	mygo.App.OnQuit(func() {
		a.quit()
		lib.close()
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
