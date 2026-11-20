package main

// The announcement dialog's view state: the announcement shown whole,
// under its date and the version it is about, as Voxt 1.x's dialog.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// announcementViewState is the announcement whose full notes are open.
type announcementViewState struct {
	Title string `json:"title"`
	Sub   string `json:"sub"` // its date, and the version it is about
	Body  string `json:"body"`
}

// buildAnnouncementViewState decides the dialog's line under the title:
// the announcement's date, and the version it is about when it has one
// (a relative year needs now).
func buildAnnouncementViewState(translator *i18n.Translator, n core.Notification, now time.Time) *announcementViewState {
	sub := translator.DateTimeAt(n.Time, now)
	if n.Version != "" {
		sub += " · " + translator.T("Version %s", n.Version)
	}
	return &announcementViewState{Title: n.Title, Sub: sub, Body: n.Body}
}
