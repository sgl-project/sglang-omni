package main

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// sampleRules are the groups the tests and snapshots use.
func sampleRules() appctx.Rules {
	return appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{
		{ID: "g-chat", Name: "Team chat", Enabled: true, Prompt: "This is a team chat: short and casual, no greetings or sign-offs.",
			Apps:    []appctx.App{{BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}, {BundleID: "com.tencent.xinWeChat", Name: "WeChat"}},
			AutoKey: "Return", Features: appctx.Features{Transcribe: true, Rewrite: true}},
		{ID: "g-code", Name: "Code review", Enabled: true, Prompt: "Code review comments: keep identifiers as said, wrap code in backticks.",
			URLs: []string{"github.com", "gitlab.com/acme/*"}, Features: appctx.DefaultFeatures},
		{ID: "g-work", Name: "Work browser", Enabled: false, Prompt: "Formal English for {{APP_NAME}}.",
			Apps: []appctx.App{{BundleID: "com.google.Chrome", Name: "Google Chrome"}}, Features: appctx.Features{Transcribe: true, Translate: true, Rewrite: true}},
	}}
}

// browserURLs makes the fake supervisor answer browser.url and app.list.
func browserURLs(fs *fakeSupervisor, urls map[string]string) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.handlers[proto.MethodBrowserURL] = func(raw json.RawMessage) (any, error) {
		var p proto.BrowserURLParams
		json.Unmarshal(raw, &p)
		res := proto.BrowserURLResult{BundleID: p.BundleID, Via: proto.URLViaNone, MS: 3}
		if u := urls[p.BundleID]; u != "" {
			res.URL, res.Via = u, proto.URLViaAXDocument
		} else {
			res.Error = "no focused window"
		}
		return res, nil
	}
	fs.handlers[proto.MethodAppList] = func(json.RawMessage) (any, error) {
		var apps []proto.AppInfo
		for _, app := range helpertest.Apps {
			apps = append(apps, proto.AppInfo{BundleID: app.BundleID, Name: app.Name, Running: true, PID: app.PID})
		}
		// By name, as the helper lists them: Apps is a map, and the
		// snapshots must not change with its order.
		slices.SortFunc(apps, func(x, y proto.AppInfo) int { return strings.Compare(x.Name, y.Name) })
		apps = append(apps, helpertest.Installed...)
		return proto.AppListResult{Apps: apps}, nil
	}
}

var chromeApp = proto.App{PID: 901, BundleID: "com.google.Chrome", Name: "Google Chrome"}

// aeTester is the test app on the App enhancement page, in a tall window.
func aeTester(t *testing.T) (*app, *fakeSupervisor, *ui.Tester) {
	t.Helper()
	a, fs, tt := newTestApp(t)
	browserURLs(fs, map[string]string{chromeApp.BundleID: "https://github.com/hehehai/voxtnew/pulls"})
	tt.SetSize(windowWidth, 1600)
	a.navigate("/app-enhancement")
	frames(tt)
	return a, fs, tt
}

// storedRules reads the document from the store.
func storedRules(t *testing.T, a *app) appctx.Rules {
	t.Helper()
	data, ok, err := a.lib.store.Documents().Get(context.Background(), appctx.Key)
	if err != nil || !ok {
		t.Fatalf("no document: %v", err)
	}
	r, err := appctx.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestAppEnhancementEmpty(t *testing.T) {
	a, _, tt := aeTester(t)
	for _, s := range []string{"App enhancement", "New group", "Rules for the app or website in front", "Switch to another app to see which group it falls in."} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	click(t, tt, "New group")
	if len(a.aeRules.Groups) != 1 || a.aePage.sel != a.aeRules.Groups[0].ID {
		t.Fatalf("rules %+v", a.aeRules)
	}
	g := storedRules(t, a).Groups[0]
	if g.Name != "Group 1" || !g.Enabled || g.Features != appctx.DefaultFeatures {
		t.Errorf("stored %+v", g)
	}
	// The master switch is a setting.
	click(t, tt, "Use App enhancement")
	if a.lib.settings.Get().AppEnhancement.Enabled || !has(tt, "App enhancement is off: every app gets only the features' own rules.") {
		t.Error("switch not saved")
	}
}

func TestAppEnhancementEdit(t *testing.T) {
	a, _, tt := aeTester(t)
	a.setRules(sampleRules())
	a.aePage.sel = "g-code"
	frames(tt)
	for _, s := range []string{"Team chat", "Code review", "Work browser", "2 apps", "2 websites", "Scene prompt", "github.com", "gitlab.com/acme/*", "Applies to"} {
		if !has(tt, s) {
			t.Errorf("no %q", s)
		}
	}
	// Name and prompt: typed, kept, saved.
	click(t, tt, "Group name")
	tt.Type(" 2")
	frames(tt)
	click(t, tt, "Scene prompt of the group")
	tt.Key(ui.Ctrl, ui.KeyEnd) // the prompt wraps: the click may land inside it
	tt.Type(" Thanks.")
	frames(tt)
	g, _ := storedRules(t, a).Group("g-code")
	if g.Name != "Code review 2" || !strings.HasSuffix(g.Prompt, "backticks. Thanks.") {
		t.Errorf("stored %q %q", g.Name, g.Prompt)
	}
	// Features: Translate on.
	click(t, tt, "Use the group for Translate")
	if g, _ := storedRules(t, a).Group("g-code"); !g.Features.Translate || !g.Features.Transcribe {
		t.Errorf("features %+v", g.Features)
	}
	// Website rules: an invalid one says why; a valid one is added.
	click(t, tt, "New website rule")
	tt.Type("not a site")
	click(t, tt, "Add the website rule")
	if !has(tt, "A website rule has no spaces.") {
		t.Errorf("no error: %q", firstTexts(tt))
	}
	a.aePage.urlIn = "*.notion.so"
	frames(tt)
	click(t, tt, "Add the website rule")
	if g, _ := storedRules(t, a).Group("g-code"); len(g.URLs) != 3 || g.URLs[2] != "*.notion.so" || a.aePage.urlIn != "" {
		t.Errorf("urls %q", g.URLs)
	}
	a.aePage.urlIn = "GitHub.com/"
	frames(tt)
	click(t, tt, "Add the website rule")
	if !has(tt, "This rule is in the group already.") {
		t.Error("duplicate rule added")
	}
	click(t, tt, "Remove gitlab.com/acme/*")
	if g, _ := storedRules(t, a).Group("g-code"); len(g.URLs) != 2 {
		t.Errorf("urls after remove %q", g.URLs)
	}
	// Delete the group.
	click(t, tt, "More actions")
	if err := tt.ChooseMenuItem("Delete the group…"); err != nil {
		t.Fatal(err)
	}
	click(t, tt, "Delete")
	if _, ok := storedRules(t, a).Group("g-code"); ok {
		t.Error("not deleted")
	}
}

func TestAppEnhancementAutoKey(t *testing.T) {
	a, _, tt := aeTester(t)
	a.setRules(sampleRules())
	a.aePage.sel = "g-chat"
	frames(tt)
	if !has(tt, "Return ↩") {
		t.Errorf("Return not shown: %q", firstTexts(tt))
	}
	a.aePage.keyCustom, a.aePage.keyText = true, "Shift+Fn"
	frames(tt)
	click(t, tt, "Custom key after inserting")
	tt.Key(0, ui.KeyEnter)
	if !has(tt, "One key, with modifiers or not: Return, Command+Return, Tab.") {
		t.Errorf("no error: %q", firstTexts(tt))
	}
	a.aePage.keyText = "command+return"
	frames(tt)
	click(t, tt, "Custom key after inserting")
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if g, _ := storedRules(t, a).Group("g-chat"); g.AutoKey != "Command+Return" {
		t.Errorf("auto key %q", g.AutoKey)
	}
}

func TestAppEnhancementPreview(t *testing.T) {
	a, _, tt := aeTester(t)
	a.setRules(sampleRules())
	// Chrome in front: its address matches Code review by its rule.
	a.helperNotified(appActivated(chromeApp))
	frames(tt)
	if !has(tt, "In front: Google Chrome") || !has(tt, "github.com/hehehai/voxtnew/pulls") || !has(tt, "Matches Code review, by the website rule github.com") {
		t.Errorf("preview %q", firstTexts(tt))
	}
	// VoxtNew comes to the front: the preview keeps Chrome.
	a.helperNotified(appActivated(proto.App{PID: 77, BundleID: selfBundleID, Name: "VoxtNew"}))
	frames(tt)
	if !has(tt, "In front: Google Chrome") {
		t.Errorf("preview after VoxtNew %q", firstTexts(tt))
	}
	// Slack: its group.
	a.helperNotified(appActivated(proto.App{PID: 812, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}))
	frames(tt)
	if !has(tt, "Matches Team chat, as its app") {
		t.Errorf("slack %q", firstTexts(tt))
	}
	// Safari gives no address: it says so, and nothing matches.
	a.helperNotified(appActivated(proto.App{PID: 902, BundleID: "com.apple.Safari", Name: "Safari"}))
	frames(tt)
	if !has(tt, "No group matches: only the features' own rules apply.") || !tt.HasText("does not show it to Accessibility now") {
		t.Errorf("safari %q", firstTexts(tt))
	}
	// Add Safari to the group shown, from the preview.
	a.aePage.sel = "g-work"
	frames(tt)
	click(t, tt, "Add Safari to Work browser")
	if g, _ := storedRules(t, a).Group("g-work"); !g.HasApp("com.apple.Safari") {
		t.Errorf("apps %+v", g.Apps)
	}
}

func appActivated(app proto.App) proto.Message {
	m, _ := proto.NewNotification(proto.MethodAppActivated, app)
	return m
}

func TestAppEnhancementApps(t *testing.T) {
	a, _, tt := aeTester(t)
	a.setRules(sampleRules())
	a.aePage.sel = "g-code"
	frames(tt)
	click(t, tt, "Add apps…")
	for _, s := range []string{"Add apps to Code review", "Slack", "Notes", "Running"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, firstTexts(tt))
		}
	}
	click(t, tt, "Notes")
	click(t, tt, "Bundle ID")
	tt.Type("com.example.editor")
	click(t, tt, "Add the bundle ID")
	click(t, tt, "Done")
	g, _ := storedRules(t, a).Group("g-code")
	if !g.HasApp("com.apple.Notes") || !g.HasApp("com.example.editor") {
		t.Errorf("apps %+v", g.Apps)
	}
	click(t, tt, "Remove Notes")
	if g, _ := storedRules(t, a).Group("g-code"); g.HasApp("com.apple.Notes") {
		t.Error("not removed")
	}
}

func TestAppEnhancementBrowsers(t *testing.T) {
	a, _, tt := aeTester(t)
	a.setRules(sampleRules())
	a.urlReads["com.google.Chrome"] = urlRead{ok: true}
	a.urlReads["com.apple.Safari"] = urlRead{err: "no web area"}
	frames(tt)
	click(t, tt, "Browsers")
	for _, s := range []string{"Microsoft Edge", "Brave Browser", "Arc", "Address read", "No address: app only"} {
		if !has(tt, s) {
			t.Errorf("no %q", s)
		}
	}
	a.aePage.brID, a.aePage.brName = "org.example.browser", "Example"
	frames(tt)
	click(t, tt, "Add the browser")
	if bs := a.lib.settings.Get().AppEnhancement.Browsers; len(bs) != 1 || bs[0].BundleID != "org.example.browser" {
		t.Fatalf("browsers %+v", bs)
	}
	click(t, tt, "Remove Example")
	if bs := a.lib.settings.Get().AppEnhancement.Browsers; len(bs) != 0 {
		t.Errorf("browsers %+v", bs)
	}
}

// TestAppEnhancementReload: the groups come back after a restart.
func TestAppEnhancementReload(t *testing.T) {
	dir := t.TempDir()
	a, _, _ := newTestAppIn(t, dir)
	a.setRules(sampleRules())
	a.quit()
	a.lib.close()
	b, _, _ := newTestAppIn(t, dir)
	if len(b.aeRules.Groups) != 3 || len(b.aeBook.get().Groups) != 3 {
		t.Errorf("reloaded %+v", b.aeRules)
	}
}

// TestSceneConfig: the sessions' view of app enhancement, and what the
// address reads tell the page.
func TestSceneConfig(t *testing.T) {
	a, fs, _ := newTestApp(t)
	browserURLs(fs, map[string]string{chromeApp.BundleID: "https://github.com/x"})
	a.setRules(sampleRules())
	sc := a.sceneConfig()
	if r, on := sc.Rules(); !on || len(r.Groups) != 3 {
		t.Errorf("rules %v %d", on, len(r.Groups))
	}
	if !sc.IsBrowser("com.brave.Browser") || sc.IsBrowser("com.tinyspeck.slackmacgap") {
		t.Error("IsBrowser")
	}
	a.updateSettings(func(s *settings.Settings) {
		s.AppEnhancement.Browsers = []settings.Browser{{BundleID: "org.example.browser"}}
	})
	if !sc.IsBrowser("org.example.browser") {
		t.Error("custom browser")
	}
	if u, err := sc.URL(context.Background(), chromeApp); err != nil || u != "https://github.com/x" || !a.urlReads[chromeApp.BundleID].ok {
		t.Errorf("chrome %q %v %+v", u, err, a.urlReads)
	}
	if u, err := sc.URL(context.Background(), proto.App{PID: 902, BundleID: "com.apple.Safari"}); err == nil || u != "" || a.urlReads["com.apple.Safari"].ok {
		t.Errorf("safari %q %v", u, err)
	}
	var _ session.SceneConfig = *sc
}

// appEnhanceShots renders the App enhancement page with groups, its
// dialogs, and the whole page in a tall window (TestSnapshots).
func appEnhanceShots(a *app, fs *fakeSupervisor, tt *ui.Tester, save func(string)) {
	a.clearNotices() // the README shows the page without the services' card
	browserURLs(fs, map[string]string{chromeApp.BundleID: "https://github.com/hehehai/voxtnew/pulls"})
	a.setRules(sampleRules())
	a.urlReads["com.google.Chrome"] = urlRead{ok: true}
	a.urlReads["com.apple.Safari"] = urlRead{err: "no web area"}
	a.helperNotified(appActivated(chromeApp))
	a.aePage.sel = "g-chat"
	a.navigate("/app-enhancement")
	save("page-app-enhancement-group")
	a.aePage.sel = "g-code"
	save("page-app-enhancement-websites")
	w, h := snapshotSize()
	tt.SetSize(w, 1500)
	a.aePage.sel = "g-chat"
	save("page-app-enhancement-full")
	tt.SetSize(w, h)
	a.aePage.appsOpen = true
	a.loadApps()
	save("app-enhancement-apps")
	a.aePage.appsOpen = false
	a.aePage.browsersOpen = true
	save("app-enhancement-browsers")
	a.aePage.browsersOpen = false
	a.updateSettings(func(s *settings.Settings) { s.AppEnhancement.Enabled = false })
	save("page-app-enhancement-off")
	a.updateSettings(func(s *settings.Settings) { s.AppEnhancement.Enabled = true })
}

// TestAppEnhancementHeader: the Browsers button is an icon in the page's
// header, right of New group, on its centre line and inside the window,
// at the default and the smallest size; the scene prompt's help is a ⓘ.
func TestAppEnhancementHeader(t *testing.T) {
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, lang := range []string{"en", "zh-Hans"} {
			a, _, tt := newTestApp(t)
			tt.SetSize(size[0], size[1])
			if lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
			}
			a.setRules(sampleRules())
			a.navigate("/app-enhancement")
			frames(tt)
			ng, ok1 := tt.Find(i18n.T("New group"))
			br, ok2 := tt.Find(i18n.T("Browsers"))
			if !ok1 || !ok2 {
				t.Fatalf("%v %s: New group %v, Browsers %v", size, lang, ok1, ok2)
			}
			if br.X < ng.X+ng.W || br.X+br.W > float32(size[0]) || br.W > 30 {
				t.Errorf("%v %s: Browsers at %+v, New group %+v", size, lang, br, ng)
			}
			if d := (br.Y + br.H/2) - (ng.Y + ng.H/2); d < -0.5 || d > 0.5 {
				t.Errorf("%v %s: Browsers %.1f off New group's centre line", size, lang, d)
			}
			if br.Y > float32(titleBarHeight) {
				t.Errorf("%v %s: Browsers %+v below the header", size, lang, br)
			}
			tip, ok := tt.Find(i18n.T("Added on top of the features' basic clean-up and your rules, never instead of them: say what is special here (tone, format, names)."))
			if !ok || tip.W > 20 || tip.H > 20 {
				t.Errorf("%v %s: the scene prompt's help %+v %v, not a ⓘ", size, lang, tip, ok)
			}
			click(t, tt, i18n.T("Browsers"))
			if !a.aePage.browsersOpen {
				t.Errorf("%v %s: Browsers did not open", size, lang)
			}
		}
	}
}
