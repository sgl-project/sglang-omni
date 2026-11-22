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
		{"transcribe", translator.T("Transcribe"), "Transcribe dictation clean-up shortcut"},
		{"translate", translator.T("Translate"), "Translate target language shortcut"},
		{"rewrite", translator.T("Rewrite"), "Rewrite answer shortcut"},
		{"models", translator.T("Speech models"), "Speech models download delete install recognition"},
		{"language-models", translator.T("Language models"), "Language models services providers API"},
		{"permissions", translator.T("Permissions"), "Permissions microphone accessibility privacy"},
		{"audio", translator.T("Audio"), "Audio microphone input device sounds silence"},
		{"overlay", translator.T("Overlay style"), "Overlay style position opacity corners recording pill"},
		{"shortcuts", translator.T("Shortcuts"), "Shortcuts hotkeys keys fn"},
		{"appearance", translator.T("Appearance"), "Appearance theme color palette transparency size sidebar"},
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
	Feature     *featureViewState                `json:"feature,omitempty"`
	Models      *settingsModelsViewState         `json:"models,omitempty"`
	Remote      *remoteViewState                 `json:"language_models,omitempty"`
	Permissions *permissionsViewState            `json:"permissions,omitempty"`
	Audio       *settingsAudioViewState          `json:"audio,omitempty"`
	Overlay     *settingsOverlayViewState        `json:"overlay,omitempty"`
	Shortcuts   *settingsShortcutsViewState      `json:"shortcuts,omitempty"`
	Appearance  *settingsAppearanceViewState     `json:"appearance,omitempty"`
	Privacy     *settingsPrivacyViewState        `json:"privacy,omitempty"`
	Storage     *settingsStorageViewState        `json:"storage,omitempty"`
	About       *settingsAboutViewState          `json:"about,omitempty"`
	Updates     *updatesViewState                `json:"updates,omitempty"`
}

// settingsModelsViewState is Settings → Speech models: the models of
// this Mac, the setup guide's way back and the remote speech services.
type settingsModelsViewState struct {
	Models      modelsViewState         `json:"models"`
	BackBar     onboardingBackViewState `json:"back_bar"`
	RemoteCards remoteCardsViewState    `json:"remote_cards"`
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
