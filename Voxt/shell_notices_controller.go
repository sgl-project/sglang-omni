package main

// The notifications' logic (shell_notices.go draws them): the centre's
// snapshot the sidebar shows, the problems the cards report, and the
// cards' actions.

import (
	"context"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// The problems' notifications: one ID each, so a problem shows once.
const (
	problemModels = "problem/models"
	problemHelper = "problem/helper"
)

// linkRestartHelper is the link of the helper's card: its action
// restarts the helper (noticeAction), it is not a deep link.
const linkRestartHelper = "action:restart-helper"

// noticesState is the notifications' part of the app (main thread).
type noticesState struct {
	notices []core.Notification // the centre's, newest first

	// problemCode: the form of each problem shown (setProblem);
	// problemDismissed: the forms the user dismissed, not shown again
	// this run while they last.
	problemCode, problemDismissed map[string]string

	// noticesGen counts the changes to notices; noticesView is the last
	// built state, noticesViewKey its inputs (noticesModel).
	noticesGen     int
	noticesView    []noticeViewState
	noticesViewKey noticesCacheKey
}

// noticesCacheKey is every input of a notices view state: when it
// matches the last build's, noticesModel reuses the state instead of
// rebuilding it every frame.
type noticesCacheKey struct {
	gen        int
	year       int
	location   string
	zoneOffset int
	locale     string
}

// noticePosted adds a notification to the centre's snapshot: the sidebar
// shows it at once, the rail as a dot on Settings.
func (a *app) noticePosted(n core.Notification) {
	a.notices = slices.DeleteFunc(a.notices, func(x core.Notification) bool { return x.ID == n.ID })
	a.notices = slices.Insert(a.notices, 0, n)
	a.noticesGen++
}

// notify posts a notification (on any goroutine).
func (a *app) notify(sev core.Severity, title, body, link string) {
	a.notes.Post(core.Notification{Severity: sev, Title: title, Body: body, Link: link})
}

// dismissNotice removes one notification (its card's ×).
func (a *app) dismissNotice(id string) {
	if i := slices.IndexFunc(a.notices, func(n core.Notification) bool { return n.ID == id }); i >= 0 {
		a.releaseCardGone(a.notices[i]) // updates_github.go
	}
	a.notices = slices.DeleteFunc(a.notices, func(n core.Notification) bool { return n.ID == id })
	a.problemSeen(id)
	a.noticesGen++
	notes := a.notes
	a.async(func() { notes.Dismiss(context.Background(), id) })
}

// clearNotices removes them all.
func (a *app) clearNotices() {
	for _, n := range a.notices {
		a.problemSeen(n.ID)
		a.releaseCardGone(n)
	}
	a.notices = nil
	a.noticesOpen = false
	a.noticesGen++
	notes := a.notes
	a.async(func() { notes.Clear(context.Background()) })
}

// problemSeen keeps a problem's card the user dismissed away, this run,
// while the problem stays the same (setProblem).
func (a *app) problemSeen(id string) {
	code, ok := a.problemCode[id]
	if !ok {
		return
	}
	if a.problemDismissed == nil {
		a.problemDismissed = map[string]string{}
	}
	a.problemDismissed[id] = code
}

// setProblem shows a problem's card (n, under id; code says which form
// of the problem it is), unless the same shows already or the user
// dismissed this form of it this run; nil withdraws it (solved).
func (a *app) setProblem(id, code string, n *core.Notification) {
	i := slices.IndexFunc(a.notices, func(x core.Notification) bool { return x.ID == id })
	if n == nil {
		delete(a.problemDismissed, id)
		delete(a.problemCode, id)
		if i >= 0 {
			a.notices = slices.Delete(a.notices, i, i+1)
			a.noticesGen++
			notes := a.notes
			a.async(func() { notes.Dismiss(context.Background(), id) })
		}
		return
	}
	if a.problemCode == nil {
		a.problemCode = map[string]string{}
	}
	a.problemCode[id] = code
	if a.problemDismissed[id] == code {
		return
	}
	if i >= 0 && a.notices[i].Title == n.Title && a.notices[i].Body == n.Body && a.notices[i].Link == n.Link {
		return
	}
	n.ID = id
	a.notes.Post(*n) // noticePosted adds it
}

// checkModels shows a card while no speech model is installed or no
// language model is set up (what the status bar said).
func (a *app) checkModels() {
	if a.notes == nil {
		return
	}
	asrOK := len(a.installed(local.ASR)) > 0
	llmOK := len(a.installed(local.LLM)) > 0
	var n *core.Notification
	code := ""
	link := modelsPath
	switch {
	case !asrOK && !llmOK:
		code = "both"
		n = &core.Notification{Title: i18n.T("No models"), Body: i18n.T("To start, install a speech model, then set up a language model.")}
	case !asrOK:
		code = "asr"
		n = &core.Notification{Title: i18n.T("No speech model"), Body: i18n.T("Transcribe, Translate and Rewrite need one to hear you.")}
		if p, missing, ok := a.pendingSetup(local.ASR); ok { // remote_link.go: a speech service set up but for its key
			code, link = "asr/"+p.ID+"/"+missing, linkCompleteSetup+string(local.ASR)+"/"+p.ID
			n.Body = setupNeedsText(local.ASR, p, missing)
		}
	case !llmOK:
		code, link = "llm", remotePath
		n = &core.Notification{Title: i18n.T("No language model"), Body: i18n.T("Translate and Rewrite need one. Without it, Transcribe types the text as recognised.")}
		if p, missing, ok := a.pendingSetup(local.LLM); ok {
			code, link = "llm/"+p.ID+"/"+missing, linkCompleteSetup+string(local.LLM)+"/"+p.ID
			n.Body = setupNeedsText(local.LLM, p, missing)
		}
	}
	if n != nil {
		n.Severity, n.Link = core.SeverityWarning, link
	}
	a.setProblem(problemModels, code, n)
}

// checkHelper shows a card while the helper does not run (it crashed,
// is missing or out of date, or did not restart).
func (a *app) checkHelper() {
	if a.notes == nil {
		return
	}
	st := a.helperStatus()
	if st.Reason == helper.FailUpdated {
		// Not the helper's fault: the update's card says what to do.
		a.setProblem(problemHelper, "", nil)
		a.appUpdated() // updates.go
		return
	}
	if st.State != helper.Failed {
		if st.State == helper.Ready {
			a.setProblem(problemHelper, "", nil)
		}
		return
	}
	a.setProblem(problemHelper, string(st.Reason), &core.Notification{Severity: core.SeverityError, Title: uitext.HelperFailText(st),
		Body: i18n.T("Recording and shortcuts work again once it is running."), Link: linkRestartHelper})
}

// cardAction runs a card's action: restart the helper, or open its link.
func (a *app) cardAction(n core.Notification) {
	a.noticesOpen = false
	switch n.Link {
	case linkRestartHelper:
		a.restartHelper()
		return
	case linkRelaunch:
		a.relaunch() // updates.go
		return
	case linkImportLegacy:
		a.dismissNotice(n.ID)
		a.importLegacy() // legacy_import.go
		return
	case linkAnnouncement:
		a.announceShown, a.announceShownOpen = n, true // shell_announcement_controller.go
		return
	}
	if u, ok := strings.CutPrefix(n.Link, linkDownload); ok {
		a.dismissNotice(n.ID)
		openExternal(u) // updates_github.go
		return
	}
	if setup, ok := strings.CutPrefix(n.Link, linkCompleteSetup); ok {
		kind, id, _ := strings.Cut(setup, "/")
		a.completeSetup(local.Kind(kind), id) // remote_link.go
		return
	}
	a.openURL(n.Link)
}

// noticesModel is the notifications' view state, rebuilt when the
// snapshot, the year, the time zone or the language changed.
func (a *app) noticesModel(now time.Time) []noticeViewState {
	_, zoneOffset := now.Zone()
	key := noticesCacheKey{gen: a.noticesGen, year: now.Year(), location: now.Location().String(), zoneOffset: zoneOffset, locale: i18n.Locale()}
	if key != a.noticesViewKey {
		a.noticesView = buildNoticesViewState(i18n.CurrentTranslator(), a.notices, now)
		a.noticesViewKey = key
	}
	return a.noticesView
}

// noticeCardAction runs a card's action (its link): the cards show the
// centre's snapshot, so the action looks the card up there.
func (a *app) noticeCardAction(id string) {
	for _, n := range a.notices {
		if n.ID == id {
			a.cardAction(n)
			return
		}
	}
}

// openMainSettings opens the Settings dialog (the rail popover's row, over
// which it closes).
func (a *app) openMainSettings() {
	a.noticesOpen = false
	a.navigate("/settings")
}
