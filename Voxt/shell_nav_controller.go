package main

// The sidebar's logic (shell_nav.go draws it): the layout choice (the
// rail or the full sidebar), the width and the closed sections the
// settings remember, and the view state's cache.

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// autoRailWidth: below this window width (DIPs at 100 %) the automatic
// sidebar shows icons only.
const autoRailWidth = 740

// navState is the sidebar's part of the app (main thread).
type navState struct {
	// narrowWindow: the window was narrower than autoRailWidth at the
	// last build (the view measures it, railMode reads it).
	narrowWindow bool

	navUI navLocalUI // the views' local state

	// navView is the last built view state, navViewKey its inputs
	// (navModel).
	navView    navViewState
	navViewKey navCacheKey
}

// navCacheKey is every input of a navViewState: when it matches the last
// build's, navModel reuses the state instead of rebuilding it every
// frame.
type navCacheKey struct {
	settings     uint64
	suggestions  int
	current, nav string
	settingsOpen bool
	notices      int
	locale       string
}

// railMode reports whether the sidebar shows icons only: the user's
// choice (Settings → Appearance, the sidebar's button), else automatic,
// while the window is narrower than autoRailWidth (narrowWindow, set by
// each build). It is the only place that decides for the app's other
// readers; the sidebar's own view applies the same rule (shell_nav.go).
func (a *app) railMode() bool {
	ap := a.prefs.Appearance
	if ap.Rail || ap.RailChosen {
		return ap.Rail
	}
	return a.narrowWindow
}

// setRail is the user's choice of the sidebar's layout: it stays,
// whatever the window's width.
func (a *app) setRail(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Appearance.Rail, s.Appearance.RailChosen = on, true })
}

// sidebarWidthChanged is the user's resize of the sidebar, once the drag
// rested (the view times it): the settings keep the width.
func (a *app) sidebarWidthChanged(width float64) {
	if width != a.prefs.Appearance.SidebarWidth {
		a.updateSettings(func(s *settings.Settings) { s.Appearance.SidebarWidth = width })
	}
}

// saveClosedSections saves the sections the user closed or opened.
func (a *app) saveClosedSections() {
	var closed []string
	for _, g := range navGroups {
		if p, ok := a.navUI.sections[g.id]; ok && !*p {
			closed = append(closed, g.id)
		}
	}
	if slices.Equal(closed, a.prefs.Appearance.ClosedSections) {
		return
	}
	a.updateSettings(func(s *settings.Settings) { s.Appearance.ClosedSections = closed })
}

// navModel is the sidebar's view state, rebuilt when the settings, the
// badges, the page, the dialog, the notifications or the language
// changed.
func (a *app) navModel() *navViewState {
	key := navCacheKey{
		settings: a.settingsRevision, suggestions: len(a.suggestions),
		current: section(a.router.Path()), nav: a.nav, settingsOpen: a.settingsOpen,
		notices: len(a.notices), locale: i18n.Locale(),
	}
	if key != a.navViewKey {
		badges := map[string]string{}
		for _, g := range navGroups {
			for _, p := range g.pages {
				if b := a.navBadge(p.path); b != "" {
					badges[p.path] = b
				}
			}
		}
		a.navView = buildNavViewState(i18n.CurrentTranslator(), navViewStateInput{
			Appearance:   a.prefs.Appearance,
			Features:     a.prefs.Features,
			Closed:       a.prefs.Appearance.ClosedSections,
			Current:      section(a.router.Path()),
			SettingsOpen: a.settingsOpen,
			Notices:      len(a.notices),
			Badges:       badges,
		})
		a.navViewKey = key
	}
	return &a.navView
}

// navLocal returns the views' local state, wired to the app's on first
// use.
func (a *app) navLocal() *navLocalUI {
	if a.navUI.nav == nil {
		a.navUI.nav, a.navUI.noticesOpen, a.navUI.narrow = &a.nav, &a.noticesOpen, &a.narrowWindow
	}
	return &a.navUI
}
