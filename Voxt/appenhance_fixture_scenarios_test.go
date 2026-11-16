package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func appEnhancementFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, change func(*appEnhancementViewStateInput)) {
		input := appEnhancementFixtureInput()
		change(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: "appenhance-" + name, kind: "appenhance", appEnhancement: input})
	}
	add("empty", func(input *appEnhancementViewStateInput) {
		input.Rules = appctx.Rules{Version: appctx.Version}
		input.SelectedID = ""
		input.PreviewApp = proto.App{}
	})
	add("normal", func(input *appEnhancementViewStateInput) {})
	add("off", func(input *appEnhancementViewStateInput) { input.Enabled = false })
	add("error", func(input *appEnhancementViewStateInput) { input.Error = "fixture document read failed" })
	add("preview-loading", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "https://github.com/sgl-project/sglang-omni")
		input.Preview.reading = true
	})
	add("preview-url", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "https://github.com/sgl-project/sglang-omni")
		input.SelectedID = "g-code"
	})
	add("preview-browser", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "https://example.com/")
		input.Rules = appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{{
			ID: "g-browser", Name: "Browser work", Enabled: true, Features: appctx.DefaultFeatures,
			Apps: []appctx.App{{BundleID: chromeApp.BundleID, Name: chromeApp.Name}},
		}}}
		input.SelectedID = "g-browser"
	})
	add("preview-global", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "https://example.com/")
		input.PreviewApp = proto.App{PID: 902, BundleID: "org.mozilla.firefox", Name: "Firefox"}
		input.Preview.app = input.PreviewApp
	})
	add("preview-no-address", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "")
		input.Rules.Groups[2].Enabled = true
		input.SelectedID = "g-work"
	})
	add("preview-helper-stopped", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "")
		input.HelperReady = false
	})
	add("preview-invalid-address", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "about:blank")
	})
	add("preview-add-actions", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePreview(input, "https://example.com/chat")
	})
	add("group-empty", func(input *appEnhancementViewStateInput) {
		input.Rules = appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{{
			ID: "g-empty", Enabled: false, Features: appctx.Features{},
		}}}
		input.SelectedID = "g-empty"
		input.PreviewApp = proto.App{}
	})
	add("group-url-canonical", func(input *appEnhancementViewStateInput) {
		input.SelectedID = "g-code"
		input.Rules.Groups[1].URLs = []string{"HTTPS://GitHub.com/"}
	})
	add("url-error-empty", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePattern(input, "")
	})
	add("url-error-space", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePattern(input, "bad host")
	})
	add("url-error-scheme", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePattern(input, "ftp://example.com")
	})
	add("url-error-host", func(input *appEnhancementViewStateInput) {
		appEnhancementFixturePattern(input, "*")
	})
	add("url-duplicate", func(input *appEnhancementViewStateInput) {
		input.SelectedID = "g-code"
		input.URLInput = "GitHub.com/"
		input.URLCheck = true
	})
	add("key-custom-existing", func(input *appEnhancementViewStateInput) {
		input.Rules.Groups[0].AutoKey = "Control+Shift+Return"
		input.AutoKeyText = "Control+Shift+Return"
	})
	add("key-custom-editing", func(input *appEnhancementViewStateInput) {
		input.AutoKeyCustom = true
		input.AutoKeyText = "command+return"
	})
	add("key-error", func(input *appEnhancementViewStateInput) {
		input.AutoKeyCustom = true
		input.AutoKeyText = "Shift+Fn"
		input.AutoKeyCheck = true
	})
	add("apps-loading", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
		input.Apps = appEnhancementFixtureApps()
		input.AppsLoading = true
	})
	add("apps-normal", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
		input.Apps = appEnhancementFixtureApps()
	})
	add("apps-error", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
		input.AppsError = "fixture app list failed"
	})
	add("apps-helper-stopped", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
		input.AppsHelperStopped = true
	})
	add("apps-empty", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
	})
	add("apps-search-empty", func(input *appEnhancementViewStateInput) {
		input.AppsOpen = true
		input.Apps = appEnhancementFixtureApps()
		input.AppsQuery = "no such app"
	})
	add("browsers-normal", func(input *appEnhancementViewStateInput) {
		input.BrowsersOpen = true
		input.Apps = appEnhancementFixtureApps()
		input.CustomBrowsers = []settings.Browser{{BundleID: "org.example.browser", Name: "Example Browser"}}
		input.URLReads = map[string]urlRead{
			"com.apple.Safari":  {ok: true},
			"com.google.Chrome": {ok: false},
		}
	})
	add("browsers-installed", func(input *appEnhancementViewStateInput) {
		input.BrowsersOpen = true
		input.Apps = appEnhancementFixtureApps()[:1]
		input.CustomBrowsers = []settings.Browser{{BundleID: "org.example.browser"}}
	})
	add("delete-ask", func(input *appEnhancementViewStateInput) {
		input.DeleteAsk = true
	})
	return scenarios
}

func appEnhancementFixtureInput() appEnhancementViewStateInput {
	return appEnhancementViewStateInput{
		Enabled: true, Rules: sampleRules(), SelectedID: "g-chat", HelperReady: true,
		PreviewApp: proto.App{PID: 903, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"},
		Apps:       nil, URLReads: map[string]urlRead{},
	}
}

func appEnhancementFixturePreview(input *appEnhancementViewStateInput, rawURL string) {
	input.PreviewApp = chromeApp
	input.Preview = aePreview{app: chromeApp, read: true, url: rawURL}
}

func appEnhancementFixturePattern(input *appEnhancementViewStateInput, pattern string) {
	input.Rules = appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{{
		ID: "g-code", Name: "Website rules", Enabled: true, Features: appctx.DefaultFeatures, URLs: []string{pattern},
	}}}
	input.SelectedID = "g-code"
	input.PreviewApp = proto.App{}
}

func appEnhancementFixtureApps() []proto.AppInfo {
	return []proto.AppInfo{
		{BundleID: "com.apple.Safari", Name: "Safari", Path: "/fixtures/Safari.app"},
		{BundleID: chromeApp.BundleID, Name: chromeApp.Name, Path: "/fixtures/Chrome.app", Running: true, PID: chromeApp.PID},
		{BundleID: "org.mozilla.firefox", Name: "Firefox", Path: "/fixtures/Firefox.app"},
		{BundleID: "com.tinyspeck.slackmacgap", Name: "Slack", Path: "/fixtures/Slack.app", Running: true, PID: 903},
	}
}
