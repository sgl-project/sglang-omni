package main

// The notifications' view state: a card for each of the centre's
// notifications, with the form the sidebar shows already decided (the
// severity as a code, the action's label, the time's tooltip).

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// The severities a notice card shows, as codes (the view maps them to an
// icon and a colour).
const (
	noticeLevelInfo    = "info"
	noticeLevelWarning = "warning"
	noticeLevelError   = "error"
	noticeLevelSuccess = "success"
)

// noticeViewState is one notification as the sidebar's card shows it.
type noticeViewState struct {
	ID       string `json:"id"`
	Severity string `json:"severity"`
	Title    string `json:"title"`
	Body     string `json:"body,omitempty"`
	// Summary is an announcement's short lines, shown instead of Body.
	Summary     []string `json:"summary,omitempty"`
	ActionLabel string   `json:"action_label,omitempty"`
	TimeHint    string   `json:"time_hint"`
}

// noticeSeverity is the code of a notification's severity.
func noticeSeverity(severity core.Severity) string {
	switch severity {
	case core.SeverityError:
		return noticeLevelError
	case core.SeverityWarning:
		return noticeLevelWarning
	case core.SeveritySuccess:
		return noticeLevelSuccess
	}
	return noticeLevelInfo
}

// buildNoticeViewState decides how a notification shows: its action's
// label from its link and its time's tooltip (a relative year needs now).
func buildNoticeViewState(translator *i18n.Translator, n core.Notification, now time.Time) noticeViewState {
	state := noticeViewState{
		ID:       n.ID,
		Severity: noticeSeverity(n.Severity),
		Title:    n.Title,
		Body:     n.Body,
		TimeHint: translator.DateTimeAt(n.Time, now),
	}
	if n.Link == linkAnnouncement {
		state.Summary = announcementSummary(n.Body)
	}
	if n.Link != "" {
		state.ActionLabel = cardActionLabelFor(translator, n.Link)
	}
	return state
}

// buildNoticesViewState is the stack, in the order it shows (newest
// first).
func buildNoticesViewState(translator *i18n.Translator, notices []core.Notification, now time.Time) []noticeViewState {
	if len(notices) == 0 {
		return nil
	}
	states := make([]noticeViewState, len(notices))
	for i, n := range notices {
		states[i] = buildNoticeViewState(translator, n, now)
	}
	return states
}

const (
	announcementSummaryLines = 3
	announcementSummaryRunes = 36
)

// announcementSummary keeps three short, readable lines of a Markdown
// announcement for its card; the notification keeps the whole body.
func announcementSummary(content string) []string {
	var lines []string
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(strings.TrimLeft(line, "-*+"))
		line = strings.ReplaceAll(line, "**", "")
		if line == "" {
			continue
		}
		if r := []rune(line); len(r) > announcementSummaryRunes {
			line = strings.TrimSpace(string(r[:announcementSummaryRunes-1])) + "…"
		}
		lines = append(lines, line)
		if len(lines) == announcementSummaryLines {
			break
		}
	}
	return lines
}

// cardActionLabel names a card's action by its link.
func cardActionLabel(link string) string {
	return cardActionLabelFor(i18n.CurrentTranslator(), link)
}

// cardActionLabelFor is cardActionLabel in a translator's language.
func cardActionLabelFor(translator *i18n.Translator, link string) string {
	switch link {
	case linkRestartHelper:
		return translator.T("Restart helper")
	case linkRelaunch:
		return translator.T("Restart Voxt")
	case modelsPath:
		return translator.T("Open Speech models")
	case remotePath:
		return translator.T("Open Language models")
	case linkImportLegacy:
		return translator.T("Import") // the card's title says from where
	case linkAnnouncement:
		return translator.T("What's new")
	}
	if strings.HasPrefix(link, linkDownload) {
		return translator.T("Download")
	}
	if strings.HasPrefix(link, linkCompleteSetup) {
		return translator.T("Complete setup")
	}
	return translator.T("Open")
}
