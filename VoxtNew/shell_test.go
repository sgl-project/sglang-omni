package main

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/permission"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

func TestSidebarAndShortcuts(t *testing.T) {
	a, _, tt := newTestApp(t)
	if got := a.router.Path(); got != "/home" {
		t.Fatalf("starts at %s", got)
	}
	// Every page of the sidebar shows (no card under the list: 520 tall,
	// the list then scrolls).
	frames(tt)
	a.clearNotices()
	for _, p := range pages {
		click(t, tt, pageTitle(p.path))
		if a.router.Path() != p.path {
			t.Fatalf("clicking %s went to %s", p.title, a.router.Path())
		}
	}
	// ⌘1…⌘6.
	for i, path := range shortcutPages {
		tt.Key(ui.Cmd, ui.Key1+ui.Key(i))
		frames(tt)
		if a.router.Path() != path {
			t.Errorf("⌘%d went to %s, want %s", i+1, a.router.Path(), path)
		}
	}
	if len(shortcutPages) != 6 {
		t.Errorf("%d shortcut pages", len(shortcutPages))
	}
	// Settings, at the foot of the sidebar, opens the dialog over the page.
	click(t, tt, "Settings")
	if !a.settingsOpen || a.shownPath() != "/settings/general" || a.router.Path() != "/app-enhancement" {
		t.Fatalf("settings: %v %s over %s", a.settingsOpen, a.shownPath(), a.router.Path())
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.settingsOpen {
		t.Error("Escape left the settings open")
	}
	// The rail keeps the pages.
	click(t, tt, "Show icons only")
	if !a.prefs.Appearance.Rail {
		t.Fatal("rail not on")
	}
	click(t, tt, "History")
	if a.router.Path() != "/history" {
		t.Errorf("rail: %s", a.router.Path())
	}
	click(t, tt, "Show the sidebar")
	if a.prefs.Appearance.Rail {
		t.Error("rail still on")
	}
}

func TestPaletteAndDeepLinks(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.Key(ui.Cmd, ui.KeyK)
	frames(tt)
	if !a.paletteOpen {
		t.Fatal("⌘K did not open the palette")
	}
	tt.Type("dictionary")
	frames(tt)
	items := a.paletteItems()
	if len(items) == 0 || items[0].id != "go/dictionary" {
		t.Fatalf("palette items %+v", items)
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if a.paletteOpen || a.router.Path() != "/dictionary" {
		t.Errorf("palette ran to %s (open %v)", a.router.Path(), a.paletteOpen)
	}
	// Chinese words find commands too, through their translation.
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	a.paletteQuery = "翻译"
	if items := a.paletteItems(); len(items) == 0 || items[0].id != "go/translate" {
		t.Errorf("zh palette %+v", items)
	}
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "en" })

	for link, want := range map[string]string{
		"voxtnew://history":           "/history",
		"voxtnew:///translate":        "/translate",
		"voxtnew://settings/audio":    "/settings/audio",
		"/settings/privacy":           "/settings/privacy",
		"VOXTNEW://app-enhancement/x": "/app-enhancement",
	} {
		a.settingsOpen = false
		a.openURL(link)
		if got := a.shownPath(); got != want {
			t.Errorf("%s → %s, want %s", link, got, want)
		}
	}
	a.toasts = nil
	a.openURL("voxtnew://studio")
	a.openURL("https://example.com/history")
	if len(a.toasts) != 2 || !strings.Contains(a.toasts[0], "Unknown link") {
		t.Errorf("unknown links: %q", a.toasts)
	}
	// The shortcut sheet lists the global shortcuts from the settings.
	tt.Key(ui.Shift, ui.KeySlash)
	if !has(tt, "Right ⌥ ⇧") || !has(tt, "⌃⌥V") || !has(tt, "Paste the last result") {
		t.Errorf("sheet: %q", firstTexts(tt))
	}
}

func TestSettingsPersistAndApply(t *testing.T) {
	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	a.navigate("/settings/general")
	frames(tt)
	for _, title := range []string{"App language", "Main language", "Show in the menu bar", "Keep running when the window closes", "Open at login"} {
		if !has(tt, title) {
			t.Errorf("general lacks %q", title)
		}
	}
	// The language applies at once, menu and all.
	a.settingsForm.language = i18n.Name("zh-Hans")
	a.updateSettings(func(s *settings.Settings) { s.General.Language = "zh-Hans" })
	frames(tt)
	if i18n.Locale() != "zh-Hans" || !has(tt, "设置") {
		t.Errorf("locale %s; texts %q", i18n.Locale(), firstTexts(tt))
	}
	a.updateSettings(func(s *settings.Settings) {
		s.General.MainLanguage = "ja"
		s.Appearance.Theme = "dark"
	})
	// Each section shows.
	for _, s := range settingsSections() {
		a.navigate("/settings/" + s.id)
		frames(tt)
		if a.shownPath() != "/settings/"+s.id || !has(tt, s.title) {
			t.Errorf("section %s: %s", s.id, a.shownPath())
		}
	}
	// Saved in the library: a new app over it reads them.
	a.quit()
	a.lib.close()
	b, _, _ := newTestAppIn(t, dir)
	if p := b.prefs; p.General.Language != "zh-Hans" || p.General.MainLanguage != "ja" || p.Appearance.Theme != "dark" {
		t.Errorf("reopened: %+v", p)
	}
	if b.settingsForm.mainLanguage != uitext.LangLabel("ja") {
		t.Errorf("form %q", b.settingsForm.mainLanguage)
	}
}

func TestMainLanguagesAreKnown(t *testing.T) {
	for _, code := range mainLanguages {
		if _, ok := lang.Lookup(code); !ok {
			t.Errorf("%s is not in the language table", code)
		}
	}
	if got := uitext.LangLabel("ja"); got != "Japanese · 日本語" {
		t.Errorf("label %q", got)
	}
}

// TestProblemCards: no status bar; what it showed is a card at the foot
// of the sidebar while something is wrong, with the action that fixes
// it, gone once solved; a card dismissed stays away until solved.
func TestProblemCards(t *testing.T) {
	a, fs, tt := newTestApp(t)
	frames(tt)
	if has(tt, "Helper ready") || has(tt, "No provider set up") {
		t.Errorf("a status bar: %q", firstTexts(tt))
	}
	if !hasExact(tt, "No models installed") || !has(tt, "Install a speech model and a language model to start.") {
		t.Fatalf("no models card: %q", firstTexts(tt))
	}
	click(t, tt, "Open Local models")
	if a.router.Path() != modelsPath {
		t.Errorf("its action: %s", a.router.Path())
	}
	// The helper: a card while it does not run, the newest on top.
	fs.set(helper.Status{State: helper.Failed, Reason: helper.FailMissing, Err: "voxtnew-helper was not found"})
	if !has(tt, "Helper missing") || !has(tt, "2 notifications") || a.notices[0].ID != problemHelper {
		t.Fatalf("missing helper: %q", firstTexts(tt))
	}
	click(t, tt, "Clear all")
	if len(a.notices) != 0 {
		t.Fatalf("cleared: %+v", a.notices)
	}
	a.problemDismissed = nil
	a.checkModels()
	fs.set(helper.Status{State: helper.Failed, Reason: helper.FailMissing, Err: "voxtnew-helper was not found"})
	if !has(tt, "Helper missing") {
		t.Fatalf("missing helper again: %q", firstTexts(tt))
	}
	click(t, tt, "Restart the helper")
	if a.helperStatus().State != helper.Ready || has(tt, "Helper missing") {
		t.Errorf("after restart: %+v %q", a.helperStatus(), firstTexts(tt))
	}
	// Dismissed: not again while the problem lasts, even as settings change.
	click(t, tt, "Dismiss")
	if hasExact(tt, "No models installed") || len(a.notices) != 0 {
		t.Fatalf("dismissed: %+v", a.notices)
	}
	a.updateSettings(func(s *settings.Settings) { s.Appearance.Theme = "dark" })
	if hasExact(tt, "No models installed") {
		t.Error("came back")
	}
	// Solved, then wrong again: it shows again, with the new text.
	a.local = &fakeLocal{ASRP: &fakeTranscriptions{}}
	a.checkModels() // as installing a model does
	if !hasExact(tt, "No language model") {
		t.Errorf("language model card: %+v", a.notices)
	}
	// The rail: a dot on Settings, which opens the cards.
	a.setRail(true)
	frames(tt)
	click(t, tt, "Settings")
	if !a.noticesOpen || !hasExact(tt, "No language model") {
		t.Fatalf("rail popover: %v %q", a.noticesOpen, firstTexts(tt))
	}
	click(t, tt, "Dismiss")
	if len(a.notices) != 0 || a.noticesOpen {
		t.Errorf("dismissed: %+v", a.notices)
	}
	click(t, tt, "Settings")
	if !a.settingsOpen {
		t.Error("Settings without notifications")
	}
}

func TestPermissionsSettings(t *testing.T) {
	a, fs, tt := newTestApp(t)
	a.navigate("/settings/permissions")
	frames(tt)
	if a.shownPath() != "/settings/permissions" {
		t.Fatalf("shown %s", a.shownPath())
	}
	for _, s := range []string{"Microphone", "Accessibility", "Not asked yet", "Check again"} {
		if !has(tt, s) {
			t.Errorf("section lacks %q", s)
		}
	}
	for _, s := range []string{"Browsers", "Safari"} {
		if has(tt, s) {
			t.Errorf("section still shows %q", s)
		}
	}
	// The helper's view of the permissions wins: the microphone is not
	// asked yet there.
	if !slices.Contains(fs.methods(), proto.MethodPermissionsStatus) || a.perms.mic != permission.NotDetermined || a.perms.access {
		t.Errorf("permissions %+v after %v", a.perms, fs.methods())
	}
	if slices.ContainsFunc(fs.methods(), func(m string) bool { return strings.HasPrefix(m, "automation.") }) {
		t.Errorf("asked for Automation: %v", fs.methods())
	}
	// Ask: the first is the microphone's, through the helper.
	click(t, tt, "Ask")
	if !slices.Contains(fs.methods(), proto.MethodPermissionsRequest) || a.perms.mic != permission.Granted {
		t.Errorf("microphone after asking: %+v", a.perms)
	}
	if f := a.permSys.(*permission.Fake); f.Prompted != 0 {
		t.Errorf("asked from the app: %d", f.Prompted)
	}
	// Without the helper, Accessibility is asked from the app.
	fs.set(helper.Status{State: helper.Failed, Reason: helper.FailCrashed})
	a.askPermission(proto.PermissionAccessibility)
	if f := a.permSys.(*permission.Fake); f.Prompted != 1 {
		t.Errorf("accessibility prompts: %d", f.Prompted)
	}
}

// TestPermissionsMoved: the old page's path, its deep link and the Home
// page's button open the settings section; the sidebar has no such page.
func TestPermissionsMoved(t *testing.T) {
	a, _, tt := newTestApp(t)
	if slices.ContainsFunc(pages, func(p navPage) bool { return p.path == "/permissions" }) {
		t.Error("Permissions is still a page of the sidebar")
	}
	for _, open := range []func(){
		func() { a.navigate("/permissions") },
		func() { a.openURL("voxtnew://permissions") },
		func() { a.openURL("voxtnew://settings/permissions") },
		func() { a.navigate("/home"); frames(tt); click(t, tt, "Accessibility: Open Permissions") },
	} {
		a.settingsOpen = false
		a.navigate("/home")
		open()
		frames(tt)
		if a.shownPath() != "/settings/permissions" {
			t.Errorf("shown %s", a.shownPath())
		}
	}
	if a.router.Path() != "/home" {
		t.Errorf("page under the dialog %s", a.router.Path())
	}
}

func TestPrivacyClearsHistory(t *testing.T) {
	a, _, tt := newTestApp(t)
	h := a.lib.store.History()
	for i, id := range []string{"a", "b"} {
		h.Put(context.Background(), store.Entry{ID: id, Kind: store.KindTranscribe, CreatedAt: time.Now().Add(time.Duration(i) * time.Second), Result: "x"})
	}
	a.navigate("/settings/privacy")
	click(t, tt, "Clear…")
	if !a.clearHistoryAsk {
		t.Fatal("no confirmation")
	}
	click(t, tt, "Delete the history")
	if list, _ := h.List(context.Background(), store.HistoryQuery{}); len(list) != 0 {
		t.Errorf("%d entries left", len(list))
	}
	if !slices.ContainsFunc(a.toasts, func(s string) bool { return s == "Deleted 2 entries" }) && !has(tt, "Deleted 2 entries") {
		t.Errorf("toasts %q", a.toasts)
	}
	// Keeping no history turns the recordings off too.
	a.settingsForm.keepAudio = true
	a.updateSettings(func(s *settings.Settings) { s.Privacy.KeepAudio = true })
	click(t, tt, "Save each result")
	if a.prefs.Privacy.KeepHistory || a.prefs.Privacy.KeepAudio {
		t.Errorf("privacy %+v", a.prefs.Privacy)
	}
}

func TestNotificationsAndLogs(t *testing.T) {
	a, _, tt := newTestApp(t)
	frames(tt)
	a.clearNotices() // the services' card
	a.notify(core.SeverityWarning, "Helper out of date", "Rebuild it", "/settings/about")
	frames(tt)
	if len(a.notices) != 1 || !has(tt, "Helper out of date") || !has(tt, "Rebuild it") {
		t.Fatalf("notices %+v", a.notices)
	}
	click(t, tt, "Open")
	if a.shownPath() != "/settings/about" {
		t.Errorf("notice link: %s", a.shownPath())
	}
	a.settingsOpen = false
	a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: helper.LogSource, Message: "fake helper ready"})
	a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: "app", Message: "VoxtNew started"})
	a.navigate("/logs")
	a.logsSource = helper.LogSource
	if !has(tt, "fake helper ready") || has(tt, "VoxtNew started") {
		t.Errorf("logs: %q", firstTexts(tt))
	}
	tt.Key(ui.Cmd|ui.Shift, ui.KeyL)
	frames(tt)
	if !a.logsPanel {
		t.Error("⇧⌘L")
	}
}

func TestAboutAndBundle(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/about")
	frames(tt)
	if !has(tt, "Helper protocol") || !has(tt, "0.1.0") || !has(tt, "Website") || !has(tt, "Download") {
		t.Errorf("about: %q", firstTexts(tt))
	}
	if websiteURL != "https://actnow.dev/voxtnew" || downloadURL != "https://actnow.dev/download/voxtnew" {
		t.Errorf("addresses %s %s", websiteURL, downloadURL)
	}
	d := a.diagnosticsText()
	if !strings.Contains(d, "Helper: 0.1.0 (ready)") || !strings.Contains(d, "Helper protocol: 1") {
		t.Errorf("diagnostics:\n%s", d)
	}
	bundleDir = t.TempDir()
	t.Cleanup(func() { bundleDir = "" })
	a.saveBundle()
	if ents, _ := os.ReadDir(bundleDir); len(ents) != 1 || !strings.HasSuffix(ents[0].Name(), ".zip") {
		t.Errorf("bundle: %v", ents)
	}
	if u := a.bugReportURL(); !strings.HasPrefix(u, issuesURL+"?") || !strings.Contains(u, "Diagnostics") {
		t.Errorf("bug url %s", u)
	}
}

// TestHelperProcess runs the shell over a real supervisor and the fake
// helper process: the status bar follows the handshake, Settings →
// Permissions asks it, and its hotkey events reach the log.
func TestHelperProcess(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	lib, err := openLibrary(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.close)
	queue := make(chan func(), 256)
	a := newApp(lib, &helperHost{bin: self, env: helpertest.Env("press=transcribe/0")})
	a.postQueue = func(fn func()) { queue <- fn }
	pump := func(what string, cond func() bool) {
		t.Helper()
		deadline := time.After(5 * time.Second)
		for !cond() {
			select {
			case fn := <-queue:
				fn()
			case <-deadline:
				t.Fatalf("timed out waiting for %s", what)
			}
		}
	}
	a.connectHelper()
	a.startHelper()
	t.Cleanup(a.quit)
	pump("the helper", func() bool { return a.helperStatus().State == helper.Ready })
	pump("the permissions", func() bool { return a.perms.checked && a.perms.access })
	// A hotkey.event of no request goes to helperNotified.
	raw, err := a.host.sup.Call(context.Background(), "", proto.MethodHotkeyConfigure, proto.HotkeyConfigureParams{Bindings: []proto.HotkeyBinding{
		{ID: "transcribe/0", Modifiers: map[string]string{proto.ModFn: proto.SideAny}, Behavior: proto.BehaviorTap}}}, nil)
	if err != nil || !strings.Contains(string(raw), `"installed":true`) {
		t.Fatalf("configure: %s %v", raw, err)
	}
	pump("the tap", func() bool {
		return slices.ContainsFunc(a.logs.Tail(50, helper.LogSource), func(e core.LogEntry) bool {
			return e.Message == "hotkey transcribe/0 trigger in com.tinyspeck.slackmacgap"
		})
	})
}
