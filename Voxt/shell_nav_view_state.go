package main

// The sidebar's view state: its sections and pages (titles translated,
// hidden pages dropped, badges and the chosen page decided), the
// Settings item with the notifications' count, the layout choice and the
// sidebar's width.

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// navPageViewState is one page's item in the sidebar or the rail.
type navPageViewState struct {
	Path     string `json:"path"`
	Title    string `json:"title"`
	Keys     string `json:"keys,omitempty"`  // its shortcut for the tooltip, "⌘1"
	Badge    string `json:"badge,omitempty"` // a count or state after the title
	Selected bool   `json:"selected"`        // the router shows it (the rail's rule)
}

// navGroupViewState is one section of the sidebar; the first has no
// title.
type navGroupViewState struct {
	ID    string             `json:"id"`
	Title string             `json:"title,omitempty"`
	Pages []navPageViewState `json:"pages"`
}

// navShortcutViewState is one of the ⌘1–⌘6 pages: hidden ones keep
// their numbers (hiding Translate leaves ⌘5 on Rewrite).
type navShortcutViewState struct {
	Path string `json:"path"`
	Num  int    `json:"num"`
}

// navViewState is the sidebar's display data.
type navViewState struct {
	Groups           []navGroupViewState    `json:"groups"`
	Current          string                 `json:"current"` // the router's section ("/home")
	ShortcutPages    []navShortcutViewState `json:"shortcut_pages,omitempty"`
	Settings         navPageViewState       `json:"settings"` // the Settings item (the dialog, not a page)
	SettingsSelected bool                   `json:"settings_selected"`
	NoticesCount     int                    `json:"notices_count"`
	NoticesLabel     string                 `json:"notices_label,omitempty"` // "3 notifications" (the rail's tooltip)
	ClosedSections   []string               `json:"closed_sections,omitempty"`
	RailChosen       bool                   `json:"rail_chosen"`   // the user chose the layout
	Rail             bool                   `json:"rail"`          // their choice: icons only
	SidebarWidth     float64                `json:"sidebar_width"` // DIPs at 100 %, 0 for the default
	MinSidebarWidth  float64                `json:"min_sidebar_width"`
	MaxSidebarWidth  float64                `json:"max_sidebar_width"`
}

// navViewStateInput is what a navViewState is built from.
type navViewStateInput struct {
	Appearance   settings.Appearance
	Features     settings.Features
	Closed       []string // the sections the settings keep closed
	Current      string   // the router's section ("/home")
	SettingsOpen bool
	Notices      int
	Badges       map[string]string
}

// buildNavViewState decides the sidebar's content in translator's language.
func buildNavViewState(translator *i18n.Translator, input navViewStateInput) navViewState {
	state := navViewState{
		Current:          input.Current,
		Settings:         navPageViewState{Path: settingsPath, Title: pageTitleFor(translator, settingsPath), Keys: "⌘,"},
		SettingsSelected: input.SettingsOpen,
		NoticesCount:     input.Notices,
		ClosedSections:   input.Closed,
		RailChosen:       input.Appearance.Rail || input.Appearance.RailChosen,
		Rail:             input.Appearance.Rail,
		SidebarWidth:     input.Appearance.SidebarWidth,
		MinSidebarWidth:  settings.MinSidebarWidth,
		MaxSidebarWidth:  settings.MaxSidebarWidth,
	}
	if input.Notices > 0 {
		state.NoticesLabel = translator.N(input.Notices, "%d notification", "%d notifications", input.Notices)
	}
	for i, path := range shortcutPages {
		if input.Features.IsHidden(path[1:]) {
			continue
		}
		state.ShortcutPages = append(state.ShortcutPages, navShortcutViewState{Path: path, Num: i + 1})
	}
	for _, g := range navGroups {
		group := navGroupViewState{ID: g.id}
		if g.id != "" {
			group.Title = groupTitleFor(translator, g.id)
		}
		for _, p := range g.pages {
			if input.Features.IsHidden(p.path[1:]) {
				continue
			}
			group.Pages = append(group.Pages, navPageViewState{
				Path:     p.path,
				Title:    pageTitleFor(translator, p.path),
				Keys:     pageKeys(p.path),
				Badge:    input.Badges[p.path],
				Selected: !input.SettingsOpen && input.Current == p.path,
			})
		}
		state.Groups = append(state.Groups, group)
	}
	return state
}
