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
		{"/dictionary", "Dictionary", dictionaryIcon},
	}},
	{"System", []navPage{
		{"/logs", "Logs", logsIcon},
		{settingsPath, "Settings", settingsIcon}, // the dialog over the page, not a page of the router
	}},
}

// settingsPath opens the settings dialog (navigate), at the section shown
// last.
const settingsPath = "/settings"

// shortcutPages are the pages ⌘1…⌘6 open, in that order.
var shortcutPages = []string{"/home", "/history", "/transcribe", "/translate", "/rewrite", "/app-enhancement"}

// pageHidden reports whether path is the page of a feature the user hid
// (settings.Features.Hidden, whose ids are the pages' paths).
func (a *app) pageHidden(path string) bool {
	return a.prefs.Features.IsHidden(strings.TrimPrefix(path, "/"))
}

// pages are every page of the sidebar, in its order: the router's pages,
// so not Settings, whose item opens its dialog.
var pages = func() []navPage {
	var out []navPage
	for _, g := range navGroups {
		for _, p := range g.pages {
			if p.path != settingsPath {
				out = append(out, p)
			}
		}
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
func groupTitle(id string) string { return groupTitleFor(i18n.CurrentTranslator(), id) }

func groupTitleFor(translator *i18n.Translator, id string) string {
	switch id {
	case "Features":
		return translator.T("Features")
	case "Resources":
		return translator.T("Resources")
	case "System":
		return translator.T("System")
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
func pageTitle(path string) string { return pageTitleFor(i18n.CurrentTranslator(), path) }

func pageTitleFor(translator *i18n.Translator, path string) string {
	switch path {
	case "/home":
		return translator.T("Home")
	case "/history":
		return translator.T("History")
	case "/transcribe":
		return translator.T("Transcribe")
	case "/translate":
		return translator.T("Translate")
	case "/rewrite":
		return translator.T("Rewrite")
	case "/notes":
		return translator.T("Notes")
	case "/meetings":
		return translator.T("Meetings")
	case "/app-enhancement":
		return translator.T("App enhancement")
	case modelsPath:
		return translator.T("Speech models")
	case remotePath:
		return translator.T("Language models")
	case "/dictionary":
		return translator.T("Dictionary")
	case "/logs":
		return translator.T("Logs")
	case settingsPath:
		return translator.T("Settings")
	}
	return path
}

// movedPaths are pages that moved, and where they are now: links,
// notifications and old settings may still name them.
var movedPaths = map[string]string{
	"/permissions":     "/settings/permissions", // a page of the sidebar until 10-2026
	"/models":          modelsPath,
	"/models/network":  networkPath,
	"/language-models": remotePath,
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
	settingsIcon       = icons.Settings
	searchIcon         = icons.Magnifer
)
