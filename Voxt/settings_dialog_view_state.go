package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// settingsSectionCatalogEntry is a section of the settings dialog.
type settingsSectionCatalogEntry struct{ id, title, en string }

func settingsSections() []settingsSectionCatalogEntry {
	return buildSettingsSections(i18n.CurrentTranslator())
}

func buildSettingsSections(translator *i18n.Translator) []settingsSectionCatalogEntry {
	return []settingsSectionCatalogEntry{
		{"general", translator.T("General"), "General language menu bar login"},
		{"permissions", translator.T("Permissions"), "Permissions microphone accessibility privacy"},
		{"audio", translator.T("Audio"), "Audio microphone input device sounds silence"},
		{"overlay", translator.T("Overlay style"), "Overlay style position opacity corners recording pill"},
		{"shortcuts", translator.T("Shortcuts"), "Shortcuts hotkeys keys fn"},
		{"appearance", translator.T("Appearance"), "Appearance theme size sidebar"},
		{"privacy", translator.T("Privacy"), "Privacy history audio logs"},
		{"storage", translator.T("Storage"), "Storage disk data folder reset"},
		// "About" in English: "Diagnostics and about" does not fit the
		// column in bold; Chinese keeps 诊断与关于, which does.
		{"about", translator.X("settings section", "About"), "Diagnostics about version bug report"},
	}
}

type settingsDialogSectionViewState struct {
	ID      string `json:"id"`
	Title   string `json:"title"`
	English string `json:"english"`
}

type settingsDialogViewState struct {
	Open        bool                             `json:"open"`
	Path        string                           `json:"path"`
	ShownPath   string                           `json:"shown_path"`
	Sections    []settingsDialogSectionViewState `json:"sections"`
	General     *settingsGeneralViewState        `json:"general,omitempty"`
	Permissions *permissionsViewState            `json:"permissions,omitempty"`
	Audio       *settingsAudioViewState          `json:"audio,omitempty"`
	Overlay     *settingsOverlayViewState        `json:"overlay,omitempty"`
	Shortcuts   *settingsShortcutsViewState      `json:"shortcuts,omitempty"`
	Appearance  *settingsAppearanceViewState     `json:"appearance,omitempty"`
	Privacy     *settingsPrivacyViewState        `json:"privacy,omitempty"`
	Storage     *settingsStorageViewState        `json:"storage,omitempty"`
	About       *settingsAboutViewState          `json:"about,omitempty"`
	Updates     *settingsUpdatesViewState        `json:"updates,omitempty"`
}

func buildSettingsDialogViewState(translator *i18n.Translator, open bool, currentPath, shownPath string) settingsDialogViewState {
	state := settingsDialogViewState{Open: open, Path: currentPath, ShownPath: shownPath}
	for _, section := range buildSettingsSections(translator) {
		state.Sections = append(state.Sections, settingsDialogSectionViewState{
			ID: section.id, Title: section.title, English: section.en,
		})
	}
	return state
}
