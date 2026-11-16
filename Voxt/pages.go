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
			model := a.llmDebugModel()
			llmDebugView(c, &model, a, &a.llmDbg.llmDebugUI)
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
		a.homeUI.tryOn = false // a new visit (homeShowsTry)
		a.loadHome()           // home_controller.go
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
		a.prepareHomePage()
		model := a.homePageModel(time.Now())
		homePage(c, &model, a, &a.histUI, func(c *ui.Context) {
			a.tryCard(c, tryOpts{
				prefix: "home", feature: hotkey.Transcribe, mine: a.homeTrySession(),
				intro: a.tryIntro(hotkey.Transcribe), block: &a.homeUI.block,
				start: func() *sessionNotice { _, n := a.startTry(featureKind(hotkey.Transcribe)); return n },
			})
		})
		historyDeleteDialog(c, &a.histUI, a)
		return
	case "/logs":
		a.logsPage(c)
		return
	case modelsPath:
		a.modelsPage(c) // models_page.go
		return
	case remotePath:
		model := a.remotePageModel()
		local := remoteLocalUI{wait: &a.llmWaitLocalUI, dialog: &a.provDlg.providerDialogLocalUI}
		remotePageView(c, &model, a, &local, a.pageWidth(c))
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
		model := a.appEnhancementModel()
		appEnhancementPage(c, &model, a, &a.aeUI, a.pageWidth(c), a)
		a.flushIcons()
		return
	case "/dictionary":
		a.prepareDictionaryPage()
		model := a.dictionaryPageModel()
		dictionaryPage(c, &model, a, &a.dictUI)
		return
	case "/meetings":
		a.prepareMeetingsPage()
		model := a.meetingsPageModel(time.Now())
		meetingsPage(c, &model, a, &a.mvUI, a.pageWidth(c))
		a.meetingSettingsDialog(c)
		meetingDeleteDialog(c, &a.mvUI, a)
		return
	case "/notes":
		a.prepareNotesPage()
		model := a.notesPageModel(time.Now())
		notesPage(c, &model, a, &a.nvUI)
		return
	}
	pageHeader(c, pageTitle(p.path), nil)
	emptyState(c, p.icon, pageTitle(p.path), "", nil)
}

func (a *app) notFoundPage(c *ui.Context, path string) {
	pageHeader(c, i18n.T("Not found"), nil)
	emptyState(c, icons.Magnifer, i18n.T("There is no page at %s.", path), "", func() {
		if ui.Button(c.Key("go-home"), i18n.T("Go to Home")).Clicked() {
			a.navigate("/home")
		}
	})
}
