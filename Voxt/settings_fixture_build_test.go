package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func buildSettingsFixture(t *testing.T, translator *i18n.Translator, input settingsFixtureInput) settingsDialogViewState {
	t.Helper()
	currentPath := "/" + input.section
	state := buildSettingsDialogViewState(translator, true, currentPath, "/settings"+currentPath)
	switch input.section {
	case "general":
		section := buildSettingsGeneralViewState(translator, input.preferences)
		state.General = &section
	case "permissions":
		section := buildPermissionsViewState(translator, input.permissions)
		state.Permissions = &section
	case "audio":
		audioInput := input.audio
		audioInput.Sounds = buildSettingsSoundsViewState(translator, input.sounds)
		section := buildSettingsAudioViewState(translator, audioInput)
		state.Audio = &section
	case "overlay":
		section := buildSettingsOverlayViewState(translator, input.preferences.Overlay)
		state.Overlay = &section
	case "shortcuts":
		section := buildSettingsShortcutsViewState(translator, input.shortcuts)
		state.Shortcuts = &section
	case "appearance":
		section := buildSettingsAppearanceViewState(translator, input.preferences.Appearance)
		state.Appearance = &section
	case "privacy":
		privacyInput := input.privacy
		privacyInput.HistoryAccess = buildHistoryAccessViewState(translator, input.historyAccess)
		section := buildSettingsPrivacyViewState(translator, privacyInput)
		state.Privacy = &section
	case "storage":
		storageInput := input.storage
		if len(input.keptModelIDs) > 0 {
			storageInput.KeptModelsText = keptModelsTextWithTranslator(translator, input.keptModelIDs, input.keptModelBytes)
		}
		section := buildSettingsStorageViewState(translator, storageInput)
		state.Storage = &section
	case "about":
		section := buildSettingsAboutViewState(translator, input.about)
		state.About = &section
		updates := buildSettingsUpdatesViewState(translator, input.updates, input.now)
		state.Updates = &updates
	default:
		t.Fatalf("unknown settings section %q", input.section)
	}
	return state
}
