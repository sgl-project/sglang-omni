package main

// The pages of the sidebar: Home (home_page.go), History
// (history_page.go), Transcribe, Translate and Rewrite
// (feature_page.go), App enhancement (appenhance_page.go), Dictionary
// (dictionary_page.go) and Logs (shell_logs.go). Permissions and the
// model resources are sections of the settings dialog.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// pages builds the page the router shows.
func (a *app) pages(c *uiContext) {
	a.router.View(c, func(r *uiRoute) {
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
		notFoundPage(c, r.Path(), a)
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
	case "/snippets":
		a.loadSnippets() // snippets_controller.go
	case "/notes":
		a.loadNotes() // notes_page.go
	case "/meetings":
		a.loadMeetings() // meetings_page.go
	}
}

// page builds one page of the sidebar.
func (a *app) page(c *uiContext, p navPage) {
	switch p.path {
	case "/home":
		a.prepareHomePage()
		model := a.homePageModel(time.Now())
		homePage(c, &model, a, &a.histUI, func(c *uiContext) {
			a.tryCard(c, tryOpts{
				prefix: "home", feature: hotkey.Transcribe, mine: a.homeTrySession(),
				intro: a.tryIntro(hotkey.Transcribe), block: &a.homeUI.block,
				start: func() *sessionNotice { _, n := a.startTry(featureKind(hotkey.Transcribe)); return n },
			})
		})
		historyDeleteDialog(c, &a.histUI, a)
		return
	case "/logs":
		logsPage(c, a.logsModel(), a, a.logsLocal())
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
	case "/snippets":
		model := a.snippetsPageModel()
		snippetsPage(c, &model, a, &a.snipUI)
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

func (a *app) featureShortcuts(c *uiContext, feature string) {
	state := a.featureShortcutsModel(feature)
	featureShortcuts(c, feature, &state, a, a.fui(feature))
}

func (a *app) featurePage(c *uiContext, feature string) {
	state, tryActions := a.featurePageModel(feature)
	featurePage(c, &state, a, a.fui(feature), tryActions)
}

func (a *app) transcribePage(c *uiContext) { a.featurePage(c, hotkey.Transcribe) }
func (a *app) translatePage(c *uiContext)  { a.featurePage(c, hotkey.Translate) }
func (a *app) rewritePage(c *uiContext)    { a.featurePage(c, hotkey.Rewrite) }

func (a *app) tryCard(c *uiContext, options tryOpts) {
	state, actions := a.tryCardState(options)
	tryCard(c, &state, actions)
}

func (a *app) recordFeatureKeys(row *keyRow, event uiInputEvent, save func(string)) {
	keys, ok := pressedKeys(event)
	row.unknown = !ok
	if ok {
		row.text, row.edited = keys, false
		save(keys)
	}
}

func (a *app) meetingSettingsDialog(c *uiContext) {
	if !a.mvUI.setOpen {
		return
	}
	dialogFrame(c, "mt-set", &a.mvUI.setOpen, i18n.T("Meeting settings"), "", 560, func() {
		state := a.meetingSettingsModel()
		meetingCaptureSettings(c, state, a, a.meetingUI)
		a.featureShortcuts(c, hotkey.Meeting)
		meetingASRSettings(c, state, a)
		meetingLLMSettings(c, state, a)
		meetingCorrectionStyleSettings(c, state, a, a.meetingUI)
		meetingConsent(c)
	}, func() {
		dialogDoneButton(c, "mt-set-done", &a.mvUI.setOpen)
	})
}
