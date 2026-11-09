package main

// The sidebar's pages (shell.go): their groups, paths, titles, icons and
// badges, the ⌘1–⌘6 shortcuts and the paths that moved.

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// navPage is a page of the sidebar.
type navPage struct {
	path, title string // title: English, translated where shown (pageTitle)
	icon        icons.Icon
}

// navGroup is a section of the sidebar; the first has no title.
type navGroup struct {
	id    string // English title, translated by groupTitle; "" for none
	pages []navPage
}

// navGroups are the sidebar's sections and pages, in order.
// pages.go keeps the pages' views; their icons are below.
var navGroups = []navGroup{
	{"", []navPage{
		{"/home", "Home", homeIcon},
		{"/history", "History", historyIcon},
	}},
	{"Features", []navPage{
		{"/transcribe", "Transcribe", transcribeIcon},
		{"/translate", "Translate", translateIcon},
		{"/rewrite", "Rewrite", rewriteIcon},
		{"/notes", "Notes", notesIcon},
		{"/meetings", "Meetings", meetingIcon},
		{"/app-enhancement", "App enhancement", appEnhancementIcon},
	}},
	{"Resources", []navPage{
		{modelsPath, "Speech models", modelsIcon},
		{remotePath, "Language models", remoteIcon},
		{"/dictionary", "Dictionary", dictionaryIcon},
	}},
	{"System", []navPage{
		{"/logs", "Logs", logsIcon},
	}},
}

// shortcutPages are the pages ⌘1…⌘6 open, in that order.
var shortcutPages = []string{"/home", "/history", "/transcribe", "/translate", "/rewrite", "/app-enhancement"}

// pageHidden reports whether path is the page of a feature the user hid
// (settings.Features.Hidden, whose ids are the pages' paths).
func (a *app) pageHidden(path string) bool {
	return a.prefs.Features.IsHidden(strings.TrimPrefix(path, "/"))
}

// pages are every page of the sidebar, in its order.
var pages = func() []navPage {
	var out []navPage
	for _, g := range navGroups {
		out = append(out, g.pages...)
	}
	return out
}()

// pageKeys is the shortcut of a page ("⌘1"), "" when it has none.
func pageKeys(path string) string {
	for i, p := range shortcutPages {
		if p == path {
			return "⌘" + itoa(i+1)
		}
	}
	return ""
}

// groupTitle translates a sidebar section's title.
func groupTitle(id string) string {
	switch id {
	case "Features":
		return i18n.T("Features")
	case "Resources":
		return i18n.T("Resources")
	case "System":
		return i18n.T("System")
	}
	return id
}

// navBadge is what a page's sidebar item shows after its title: a count
// or a state, "" for nothing. Areas answer for their pages.
func (a *app) navBadge(path string) string {
	if path == "/dictionary" && len(a.suggestions) > 0 {
		return itoa(len(a.suggestions))
	}
	return ""
}

// pageTitle translates a page's title (literal calls, for the key scan).
func pageTitle(path string) string {
	switch path {
	case "/home":
		return i18n.T("Home")
	case "/history":
		return i18n.T("History")
	case "/transcribe":
		return i18n.T("Transcribe")
	case "/translate":
		return i18n.T("Translate")
	case "/rewrite":
		return i18n.T("Rewrite")
	case "/notes":
		return i18n.T("Notes")
	case "/meetings":
		return i18n.T("Meetings")
	case "/app-enhancement":
		return i18n.T("App enhancement")
	case modelsPath:
		return i18n.T("Speech models")
	case remotePath:
		return i18n.T("Language models")
	case "/dictionary":
		return i18n.T("Dictionary")
	case "/logs":
		return i18n.T("Logs")
	case "/settings":
		return i18n.T("Settings")
	}
	return path
}

// movedPaths are pages that moved, and where they are now: links,
// notifications and old settings may still name them.
var movedPaths = map[string]string{
	"/permissions": "/settings/permissions", // a page of the sidebar until 10-2026
}

// The pages' and the shell's icons (Solar; internal/icons/names.txt).
var (
	homeIcon           = icons.Home2
	historyIcon        = icons.History
	transcribeIcon     = icons.Microphone3
	translateIcon      = icons.Global
	rewriteIcon        = icons.PenNewRound
	appEnhancementIcon = icons.Target
	modelsIcon         = icons.Soundwave
	remoteIcon         = icons.Cloud
	dictionaryIcon     = icons.NotebookBookmark
	notesIcon          = icons.Notes
	meetingIcon        = icons.MicrophoneLarge
	permissionsIcon    = icons.ShieldKeyhole
	logsIcon           = icons.ClipboardList
	searchIcon         = icons.Magnifer
)
