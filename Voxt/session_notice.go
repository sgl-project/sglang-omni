package main

// Notices: the overlay's one-line messages after a session (a problem and
// a way forward, or a green check), how long each shows, and what their
// buttons do.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
)

// Notice actions: where the notice's button goes.
const (
	actionNone        = ""
	actionModels      = "models"
	actionLanguage    = "language" // the Language models page (remote_page.go)
	actionPermissions = "permissions"
	actionHelper      = "helper"
	actionRetry       = "retry"  // the session's entry, in History (history_retry.go)
	actionSignIn      = "signin" // the ChatGPT (Codex) dialog (remote_chatgpt.go)
	actionUsage       = "usage"  // the ChatGPT usage page, in the browser
)

// sessionNotice is a one-line message in the overlay, where
// the pill's waveform goes: an error and a way forward (a click), hidden
// after a few seconds.
type sessionNotice struct {
	text, more string
	action     string
	entry      string        // actionRetry's history entry
	warn       bool          // a warning (the session went on), not an error
	ok         bool          // done as asked (a term added, a note saved), not a problem
	seq        int           // which notice: its timer hides only itself
	over       bool          // shows before the card that came back as its session ended (#4013)
	hold       time.Duration // how long it shows; 0 is okTime or noticeTime
	// then runs when its time is up (not when a click or the next
	// session hides it): the notice that follows it.
	then func()
}

// noticeTime is how long a notice shows. Shorter ones have nothing more
// to read: okTime a green check (added, saved), noSpeechTime Nothing was
// heard.
const (
	noticeTime   = 5 * time.Second
	okTime       = 2 * time.Second
	noSpeechTime = 1500 * time.Millisecond
)

// showNotice shows a notice in the overlay; it hides by itself.
func (a *app) showNotice(n sessionNotice) {
	a.noticeSeq++
	seq := a.noticeSeq
	n.seq = seq
	a.notice = &n
	a.overlaySync()
	hold := n.hold
	switch {
	case hold > 0:
	case n.ok:
		hold = okTime
	default:
		hold = noticeTime
	}
	a.later(hold, func() {
		if a.notice != nil && a.notice.seq == seq {
			if then := a.notice.then; then != nil {
				then() // in its place, without hiding the overlay between
			} else {
				a.closeNotice()
			}
		}
	})
}

// closeNotice hides the notice.
func (a *app) closeNotice() {
	a.notice = nil
	a.overlaySync()
}

// noticeAction runs a notice's button.
func (a *app) noticeAction(action string) {
	entry := ""
	if a.notice != nil {
		entry = a.notice.entry
	}
	a.closeNotice()
	switch action {
	case actionRetry:
		a.retryFromNotice(entry)
	case actionModels:
		a.showMainWindow()
		a.navigate(modelsPath)
	case actionLanguage:
		a.showMainWindow()
		a.navigate(remotePath)
	case actionPermissions:
		a.showMainWindow()
		a.navigate("/settings/permissions")
	case actionHelper:
		a.restartHelper()
	case actionSignIn:
		a.showMainWindow()
		a.completeSetup(local.LLM, providers.ChatGPT)
	case actionUsage:
		go openBrowser(chatgptUsageURL)
	}
}
