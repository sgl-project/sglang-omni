package main

import (
	"github.com/egoist/mygo"
)

// app is VoxtNew's UI state: the view state of every page, and handles
// to the domain (the library, the helper). Every field is read and
// written on the main thread: work on other goroutines reaches it through
// post.
//
// Each area keeps its fields in a struct of its own, embedded here, in the
// file of its area (docs/architecture.md):
//
//	shellState        shell.go        window chrome, navigation, notifications, logs
//	settingsState     shell_settings.go  the settings dialog's sections
//	helperLink        helperlink.go   the Swift helper's supervisor and status
//	trayState         tray.go         the menu bar icon
//	permissionsState  permissions.go  Settings → Permissions
//	familyState       family.go       the ActNow family (internal/actnow): the other apps, shared settings
//	modelsState       models_page.go  the Local models page, the local models (internal/local)
//	memoryState       models_memory.go  the loaded models' memory (the Local models page)
//	omniState         models_omni.go  the speech models' runtime (internal/omni)
//	llmState          llm_link.go     the language models' runtime (internal/llmrt)
//	sessionState      session_link.go  sessions, shortcuts, the answer card
//	overlayState      overlay.go      the recording overlay
//	transcribeState   feature_page.go  the feature pages (Transcribe, Translate, Rewrite)
//	historyState      history_page.go the History page
//	appEnhanceState   appenhance.go   App enhancement: the groups, address reads, the page
//	dictionaryState   dictionary_page.go  the Dictionary page
//	homeState         home_page.go    Home: readiness, today, recent results
//	onboardingState   onboarding.go   the setup guide
//	scPromptState     settings_shortcuts_family.go  a shortcut another ActNow app uses too: the prompt
//	audioState        settings_audio.go   Settings → Audio, the recording's sounds
//	soundsState       settings_sounds.go  the sounds' presets and previews
//	muteState         mute_link.go    the system's output, muted while recording
//	updateState       updates.go      the updater, an update installed while running
//
// Add fields to your area's struct, not here.
type app struct {
	win    *mygo.Window // nil in tests
	toasts []string     // shown by the next build of the view
	lib    *library
	// postQueue replaces the window's queue in tests that run real
	// goroutines (the helper's supervisor): they pump it on the test's
	// goroutine. nil: post runs fn at once (other tests).
	postQueue func(func())
	events    // events.go: the observers of sessions and the history

	shellState
	settingsState
	helperLink
	trayState
	permissionsState
	familyState
	modelsState
	memoryState
	omniState
	llmState
	sessionState
	shortMemoryState
	overlayState
	transcribeState
	historyState
	appEnhanceState
	appIconsState
	dictionaryState
	homeState
	onboardingState
	scPromptState
	audioState
	soundsState
	muteState
	updateState
}

func newApp(lib *library, h *helperHost) *app {
	a := &app{lib: lib, shellState: newShellState()}
	a.connectEvents() // events.go: who hears of sessions and the history
	a.helperLink.host = h
	a.openShell(lib)
	a.openPermissions()
	a.openFamily()          // family.go: before the models, whose library is the family's
	a.openModels()          // models_page.go
	a.openAppEnhancement()  // appenhance.go
	a.openAudio()           // settings_audio.go
	a.openHistorySettings() // history_settings_ops.go: the recordings folder, the clean-up
	return a
}

// post runs fn on the main thread and redraws; directly in tests.
func (a *app) post(fn func()) {
	if a.win == nil {
		if a.postQueue != nil {
			a.postQueue(fn)
			return
		}
		fn()
		return
	}
	a.win.Update(fn)
}

// async runs fn on a goroutine; in tests (no window) at once, so that
// views see its results without waiting.
func (a *app) async(fn func()) {
	if a.win == nil {
		fn()
		return
	}
	go fn()
}

// toast shows msg (already translated) as a toast.
func (a *app) toast(msg string) {
	a.toasts = append(a.toasts, msg)
}

// quit releases what the app holds; it runs in App.OnQuit, before the
// library closes.
func (a *app) quit() {
	a.closeSessions() // session_link.go
	a.closeOverlay()  // overlay.go
	a.closeTray()
	a.closeMute() // mute_link.go: the output back before the helper goes
	a.stopHelper()
	a.stopLLM() // llm_link.go
	if a.models != nil {
		a.models.Close() // a download stops; what came is kept
	}
	a.closeOmni()   // models_omni.go: the speech models' servers
	a.closeFamily() // family.go: the runtime file goes
	a.closeShell()
}
