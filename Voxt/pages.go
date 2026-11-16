package main

// The pages of the sidebar: Home (home_page.go), History
// (history_page.go), Transcribe, Translate and Rewrite
// (feature_page.go), App enhancement (appenhance_page.go), Speech models
// (models_page.go), Language models (remote_page.go), Dictionary
// (dictionary_page.go) and Logs
// (shell_logs.go). Permissions is a section of the settings dialog
// (permissions.go).

import (
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// pages builds the page the router shows.
func (a *app) pages(c *ui.Context) {
	a.router.View(c, func(r *ui.Route) {
		for _, p := range pages {
			if r.Match(p.path) {
				r.Title(pageTitle(p.path))
				a.page(c, p)
				return
			}
		}
		if r.Match(llmDebugPath) {
			r.Title(i18n.T("Language model debug"))
			a.llmDebugPage(c) // llm_debug.go
			return
		}
		r.Title(i18n.T("Not found"))
		a.notFoundPage(c, r.Path())
	}).Grow(1).MinHeight(0)
}

// pageShown runs when the router goes to path: pages that read something
// when they show start it here.
func (a *app) pageShown(path string) {
	if path != "/meetings" {
		a.stopMeetingAudioWatch(a.mv.detail)
	}
	switch path {
	case "/home":
		a.hm.tryOn = false // a new visit (homeShowsTry)
		a.loadHome()       // home_page.go
	case "/history":
		a.historyShown() // history_page.go
	case "/app-enhancement":
		a.aeShown() // appenhance_page.go
	case "/dictionary":
		a.dictShown() // dictionary_page.go
	case "/notes":
		a.loadNotes() // notes_page.go
	case "/meetings":
		a.loadMeetings() // meetings_page.go
	case modelsPath:
		a.modelsShown() // models_memory.go
	}
}

// page builds one page of the sidebar.
func (a *app) page(c *ui.Context, p navPage) {
	switch p.path {
	case "/home":
		a.homePage(c) // home_page.go
		return
	case "/logs":
		a.logsPage(c)
		return
	case modelsPath:
		a.modelsPage(c) // models_page.go
		return
	case remotePath:
		a.remotePage(c) // remote_page.go
		return
	case "/transcribe":
		a.transcribePage(c) // feature_page.go
		return
	case "/translate":
		a.translatePage(c) // feature_page.go
		return
	case "/rewrite":
		a.rewritePage(c) // feature_page.go
		return
	case "/history":
		a.prepareHistoryPage()
		model := a.historyPageModel(time.Now())
		historyPage(c, &model, a, &a.histUI, a.pageWidth(c))
		historyDeleteDialog(c, &a.histUI, a)
		a.prepareHistorySettings()
		if a.histSetUI.open {
			settingsModel := a.historySettingsModel()
			historySettingsDialog(c, &settingsModel, a, &a.histSetUI)
		}
		return
	case "/app-enhancement":
		a.appEnhancementPage(c) // appenhance_page.go
		return
	case "/dictionary":
		a.dictionaryPage(c) // dictionary_page.go
		return
	case "/meetings":
		a.prepareMeetingsPage()
		model := a.meetingsPageModel(time.Now())
		meetingsPage(c, &model, a, &a.mvUI, a.pageWidth(c))
		a.meetingSettingsDialog(c)
		meetingDeleteDialog(c, &a.mvUI, a)
		return
	case "/notes":
		a.notesPage(c) // notes_page.go
		return
	}
	pageHeader(c, pageTitle(p.path), nil)
	emptyState(c, p.icon, pageTitle(p.path), "", nil)
}

func (a *app) featureShortcuts(c *ui.Context, feature string) {
	state := a.featureShortcutsModel(feature)
	featureShortcuts(c, feature, &state, a, a.fui(feature))
}

func (a *app) featureTry(c *ui.Context, feature string) {
	local := a.fui(feature)
	sectionTitle(c, i18n.T("Try it"))
	a.tryCard(c, tryOpts{
		prefix: uitext.FeaturePrefix(feature), feature: feature, mine: a.trySessionOf(feature),
		intro: a.tryIntro(feature), block: &local.block,
		start: func() *sessionNotice { _, notice := a.startTry(featureKind(feature)); return notice },
	})
}

func (a *app) tryCard(c *ui.Context, options tryOpts) {
	state, actions := a.tryCardState(options)
	tryCard(c, &state, actions)
}

func (a *app) recordFeatureKeys(row *keyRow, event ui.InputEvent, save func(string)) {
	keys, ok := pressedKeys(event)
	row.unknown = !ok
	if ok {
		row.text, row.edited = keys, false
		save(keys)
	}
}

func (a *app) customKeys(c *ui.Context, row *keyRow, label string, save func(string)) {
	t := c.Theme()
	input := ui.TextInput(c.Key("custom"), &row.text).Placeholder("Control+Option+Space").Label(label).Font("monospace").MaxWidth(px(t, 360))
	if input.Changed() {
		row.edited = true
	}
	chord, err := hotkey.Parse(row.text)
	if row.edited && err == nil && (input.Submitted() || !input.Focused()) {
		row.edited = false
		row.text = chord.String()
		save(row.text)
	}
	switch {
	case err != nil:
		warnLine(c, uitext.ParseErrorText(err))
	case row.edited:
		caption(c, i18n.T("Press Return to use %s.", uitext.HotkeyText(row.text)))
		return
	default:
		caption(c, i18n.T("Modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5."))
	}
	caption(c, i18n.T("Mouse buttons work too: Mouse3 (middle), Mouse4 and up (side), also with modifiers (Option+Mouse4)."))
}

func (a *app) notFoundPage(c *ui.Context, path string) {
	pageHeader(c, i18n.T("Not found"), nil)
	emptyState(c, icons.Magnifer, i18n.T("There is no page at %s.", path), "", func() {
		if ui.Button(c.Key("go-home"), i18n.T("Go to Home")).Clicked() {
			a.navigate("/home")
		}
	})
}
