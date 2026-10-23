package main

import (
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
)

// fakeUpdates is the updater plugin for tests: a build that can update
// itself, checked two hours ago, automatic checks on and automatic
// installs off (the plugin's defaults).
type fakeUpdates struct {
	mu      sync.Mutex
	prefs   updatePrefs
	pre     bool // SetPrereleases
	checks  int
	changes []*func()
}

func newFakeUpdates() *fakeUpdates {
	return &fakeUpdates{prefs: updatePrefs{Enabled: true, AutoCheck: true, LastCheck: time.Now().Add(-2*time.Hour - time.Minute)}}
}

func (f *fakeUpdates) Prefs() updatePrefs {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.prefs
}

func (f *fakeUpdates) SetAutomaticChecks(on bool) {
	f.change(func(p *updatePrefs) { p.AutoCheck = on })
}

func (f *fakeUpdates) SetAutomaticDownloads(on bool) {
	f.change(func(p *updatePrefs) { p.AutoInstall = on })
}

func (f *fakeUpdates) SetPrereleases(on bool) {
	f.mu.Lock()
	f.pre = on
	f.mu.Unlock()
}

func (f *fakeUpdates) prereleases() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pre
}

func (f *fakeUpdates) CheckNow() {
	f.mu.Lock()
	f.checks++
	f.mu.Unlock()
}

func (f *fakeUpdates) OnChange(fn func()) func() {
	p := &fn
	f.mu.Lock()
	f.changes = append(f.changes, p)
	f.mu.Unlock()
	return func() {
		f.mu.Lock()
		f.changes = slices.DeleteFunc(f.changes, func(q *func()) bool { return q == p })
		f.mu.Unlock()
	}
}

// change changes the preferences and calls the OnChange functions, as the
// plugin does after a change.
func (f *fakeUpdates) change(fn func(*updatePrefs)) {
	f.mu.Lock()
	fn(&f.prefs)
	changes := slices.Clone(f.changes)
	f.mu.Unlock()
	for _, c := range changes {
		(*c)()
	}
}

func (f *fakeUpdates) checked() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.checks
}

// TestUpdatesSettings: Settings → About checks for updates and changes the
// updater's two preferences; a build that cannot update itself says why.
func TestUpdatesSettings(t *testing.T) {
	a, _, tt := newTestApp(t)
	f := a.upd.(*fakeUpdates)
	a.navigate("/settings/about")
	if !has(tt, "Check for updates") || !has(tt, "Last checked 2 hours ago") {
		t.Fatalf("about: %q", firstTexts(tt))
	}
	see(t, tt, "Check for Updates…")
	click(t, tt, "Check for Updates…")
	if f.checked() != 1 {
		t.Errorf("checks %d", f.checked())
	}
	click(t, tt, "Install updates without asking")
	if p := f.Prefs(); !p.AutoInstall || !p.AutoCheck {
		t.Errorf("after turning automatic installs on: %+v", p)
	}
	click(t, tt, "Check for updates automatically")
	if p := f.Prefs(); p.AutoCheck {
		t.Errorf("after turning automatic checks off: %+v", p)
	}
	// Turning automatic checks off leaves nothing to install from.
	click(t, tt, "Install updates without asking")
	if p := f.Prefs(); !p.AutoInstall {
		t.Errorf("the install switch works with checks off: %+v", p)
	}
	// A change from the update window (its checkbox) shows.
	f.change(func(p *updatePrefs) { p.AutoCheck = true; p.LastCheck = time.Time{} })
	if !has(tt, "New versions come from actnow.dev, with their notes. Not checked yet.") || !a.updPrefs.AutoCheck {
		t.Errorf("after a change: %+v %q", a.updPrefs, firstTexts(tt))
	}
	// A development build, or one in a folder it cannot write to.
	f.change(func(p *updatePrefs) { p.Enabled = false })
	frames(tt)
	if !hasPrefix(tt, "This copy cannot update itself") || has(tt, "Check automatically") {
		t.Errorf("disabled: %q", firstTexts(tt))
	}
	click(t, tt, "Check for Updates…") // the window says why
	if f.checked() != 2 {
		t.Errorf("checks %d", f.checked())
	}
}

// TestBetaUpdates: Settings → About → Beta updates is on by default in a
// beta and off in a release, tells the updater, and the user's choice
// lasts over versions.
func TestBetaUpdates(t *testing.T) {
	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	f := a.upd.(*fakeUpdates)
	a.version = "0.1.0-beta.1"
	a.applyBeta()
	if !f.prereleases() {
		t.Error("a beta does not take betas by default")
	}
	a.navigate("/settings/about")
	see(t, tt, "Get beta versions")
	if !has(tt, "Beta updates") || !hasPrefix(tt, "Betas and releases, whichever is newer.") {
		t.Fatalf("about: %q", firstTexts(tt))
	}
	click(t, tt, "Get beta versions")
	frames(tt)
	if !hasPrefix(tt, "Releases only. This beta stays until a release newer than it comes out") {
		t.Errorf("off in a beta: %q", firstTexts(tt))
	}
	if f.prereleases() || a.prefs.Updates.Beta == nil || *a.prefs.Updates.Beta {
		t.Errorf("after turning betas off: updater %t, settings %v", f.prereleases(), a.prefs.Updates.Beta)
	}
	if b := a.lib.settings.Get().Updates.Beta; b == nil || *b {
		t.Errorf("saved %v", b)
	}
	// A release turns them off by default, and the choice lasts.
	a.quit()
	a.lib.close()
	b, _, _ := newTestAppIn(t, dir)
	b.version = "0.1.0-beta.2"
	b.applyBeta()
	if b.upd.(*fakeUpdates).prereleases() {
		t.Error("the choice to turn betas off was lost")
	}
	c, _, tt3 := newTestApp(t)
	c.version = "0.1.0"
	c.applyBeta()
	if c.upd.(*fakeUpdates).prereleases() {
		t.Error("a release takes betas by default")
	}
	c.navigate("/settings/about")
	see(t, tt3, "Get beta versions")
	if !hasPrefix(tt3, "Releases only. Turning it on offers betas too") {
		t.Errorf("off in a release: %q", firstTexts(tt3))
	}
	click(t, tt3, "Get beta versions")
	if !c.upd.(*fakeUpdates).prereleases() {
		t.Error("turning betas on did not reach the updater")
	}
}

// see scrolls the settings until s shows in the middle of the window.
func see(t *testing.T, tt *ui.Tester, s string) {
	t.Helper()
	frames(tt)
	r, ok := tt.Find(s)
	if !ok {
		t.Fatalf("no %q: %q", s, firstTexts(tt))
	}
	tt.Scroll(windowWidth*0.6, windowHeight*0.5, 0, r.Y-windowHeight*0.4)
	frames(tt)
}

// TestUpdateMenus: Check for Updates… in the menu bar icon's menu and the
// palette, none without the updater.
func TestUpdateMenus(t *testing.T) {
	a, _, _ := newTestApp(t)
	f := a.upd.(*fakeUpdates)
	act := func(fn func()) func(*mygo.MenuItem, *mygo.Window) {
		return func(*mygo.MenuItem, *mygo.Window) { fn() }
	}
	labels := func() []string {
		var out []string
		for _, it := range a.trayItems() {
			out = append(out, it.Label)
			if it.Label == "Check for Updates…" {
				it.Click(it, nil)
			}
		}
		return out
	}
	tray := labels()
	if i := slices.Index(tray, "Check for Updates…"); i < 0 || tray[i-1] != "Settings…" || f.checked() != 1 {
		t.Errorf("tray %q, checks %d", tray, f.checked())
	}
	cmd := slices.IndexFunc(a.commands(), func(c command) bool { return c.id == "check-updates" })
	if cmd < 0 || !a.commands()[cmd].enabled {
		t.Fatal("no palette command")
	}
	a.runCommand(a.commands()[cmd])
	if f.checked() != 2 {
		t.Errorf("palette: checks %d", f.checked())
	}
	if len(a.updateMenuItems(act)) != 1 {
		t.Error("app menu item")
	}
	a.upd = nil
	if slices.Contains(labels(), "Check for Updates…") || len(a.updateMenuItems(act)) != 0 {
		t.Error("an item without the updater")
	}
	a.checkForUpdates() // nothing to do, no panic
}

// TestAppUpdated: an update installed while VoxtNew runs (the helper on
// disk is the new version's) shows one card, not the helper's, that
// restarts VoxtNew, on Home and in Settings → About too.
func TestAppUpdated(t *testing.T) {
	a, fs, tt := newTestApp(t)
	relaunched := 0
	old := appRelaunch
	appRelaunch = func() { relaunched++ }
	t.Cleanup(func() { appRelaunch = old })

	fs.set(helper.Status{State: helper.Failed, Reason: helper.FailUpdated, Err: "VoxtNew was updated"})
	a.navigate("/home")
	if !has(tt, "Update installed") || !has(tt, "Restart VoxtNew to finish updating.") || has(tt, "Helper stopped unexpectedly") {
		t.Fatalf("card: %q", firstTexts(tt))
	}
	if !has(tt, "VoxtNew was updated: restart it to finish.") {
		t.Errorf("home: %q", firstTexts(tt))
	}
	n := slices.IndexFunc(a.notices, func(n core.Notification) bool { return n.ID == problemUpdate })
	if n < 0 || a.notices[n].Link != linkRelaunch || slices.ContainsFunc(a.notices, func(n core.Notification) bool { return n.ID == problemHelper }) {
		t.Fatalf("notices %+v", a.notices)
	}
	a.cardAction(a.notices[n])
	if relaunched != 1 {
		t.Errorf("relaunched %d", relaunched)
	}
	// Settings over a page without the action: its button is the first.
	a.clearNotices()
	a.navigate("/logs")
	a.navigate("/settings/about")
	see(t, tt, "Restart VoxtNew")
	click(t, tt, "Restart VoxtNew")
	if relaunched != 2 {
		t.Errorf("about: relaunched %d", relaunched)
	}
	if cardActionLabel(linkRelaunch) != "Restart VoxtNew" {
		t.Error("card action label")
	}
}

// TestBundleWatch: an update installed in the background shows the card
// once the app sees it; after the restart the kept card is gone.
func TestBundleWatch(t *testing.T) {
	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	stale := false
	a.bundleStale = func() bool { return stale }
	a.checkBundle()
	if a.bundleUpdated || has(tt, "Update installed") {
		t.Fatal("updated before the update")
	}
	stale = true
	a.upd.(*fakeUpdates).change(func(p *updatePrefs) { p.LastCheck = time.Now() }) // a check, then the install
	if !a.bundleUpdated || !has(tt, "Update installed") {
		t.Fatalf("not seen: %q", firstTexts(tt))
	}
	a.quit()
	a.lib.close()
	b, _, tt2 := newTestAppIn(t, dir)
	if slices.ContainsFunc(b.notices, func(n core.Notification) bool { return n.ID == problemUpdate }) || has(tt2, "Update installed") {
		t.Errorf("the card outlived the restart: %+v", b.notices)
	}
}
