package main

import (
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

func TestSettingsFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		t.Run(locale, func(t *testing.T) {
			fixtures := readMeetingFixtures(t, locale)
			checkSettingsFixtureEnvelopes(t, locale, fixtures)
			checkSettingsFixtureValues(t, locale, fixtures)
		})
	}
}

func checkSettingsFixtureEnvelopes(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for name, fixture := range fixtures {
		states := map[string]bool{
			"overlay": fixture.Overlay != nil, "page": fixture.Page != nil, "start": fixture.Start != nil,
			"history": fixture.History != nil, "history-settings": fixture.HistorySettings != nil,
			"history-access": fixture.HistoryAccess != nil, "settings": fixture.Settings != nil,
			"meeting-settings": fixture.MeetingSettings != nil, "shortcut-prompt": fixture.ShortcutPrompt != nil,
		}
		for kind, present := range states {
			if want := kind == fixture.Kind; present != want {
				t.Errorf("%s/%s: %s state present=%t, want %t", locale, name, kind, present, want)
			}
		}
		if fixture.Settings != nil {
			checkSettingsDialogFixture(t, locale, name, fixture.Settings)
		}
	}
}

func checkSettingsDialogFixture(t *testing.T, locale, name string, state *settingsDialogViewState) {
	t.Helper()
	if !state.Open {
		t.Errorf("%s/%s: settings dialog is closed", locale, name)
	}
	sectionID := strings.TrimPrefix(state.Path, "/")
	if state.Path != "/"+sectionID || state.ShownPath != "/settings/"+sectionID {
		t.Errorf("%s/%s: paths %q and %q do not select %q", locale, name, state.Path, state.ShownPath, sectionID)
	}
	translator := i18n.NewTranslator(locale)
	sections := buildSettingsSections(translator)
	knownSections := map[string]bool{}
	var expectedIDs, actualIDs []string
	for _, section := range sections {
		knownSections[section.id] = true
		expectedIDs = append(expectedIDs, section.id)
	}
	for _, section := range state.Sections {
		actualIDs = append(actualIDs, section.ID)
	}
	if !slices.Equal(actualIDs, expectedIDs) {
		t.Errorf("%s/%s: section IDs %v, want %v", locale, name, actualIDs, expectedIDs)
	}
	if !knownSections[sectionID] {
		t.Errorf("%s/%s: unexpected selected section %q", locale, name, sectionID)
	}
	sectionStates := map[string]bool{
		"general": state.General != nil, "permissions": state.Permissions != nil,
		"audio": state.Audio != nil, "overlay": state.Overlay != nil,
		"shortcuts": state.Shortcuts != nil, "appearance": state.Appearance != nil,
		"privacy": state.Privacy != nil, "storage": state.Storage != nil, "about": state.About != nil,
	}
	for id, present := range sectionStates {
		if want := id == sectionID; present != want {
			t.Errorf("%s/%s: %s section present=%t, want %t", locale, name, id, present, want)
		}
	}
	if present := state.Updates != nil; present != (sectionID == "about") {
		t.Errorf("%s/%s: updates state present=%t, want %t", locale, name, present, sectionID == "about")
	}
}

func checkSettingsFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	translator := i18n.NewTranslator(locale)
	for _, name := range []string{"settings-empty", "settings-loading", "settings-normal", "settings-error"} {
		requireSettingsFixture(t, locale, fixtures, name)
	}
	for _, section := range buildSettingsSections(translator) {
		state := requireSettingsDialogFixture(t, locale, fixtures, settingsNormalFixtureName(section.id))
		if strings.TrimPrefix(state.Path, "/") != section.id {
			t.Errorf("%s: normal fixture selects %q, want %q", locale, state.Path, section.id)
		}
	}

	general := requireSettingsDialogFixture(t, locale, fixtures, "settings-normal").General
	if len(general.OtherLanguages) == 0 {
		t.Errorf("%s: general normal fixture has no other languages", locale)
	}
	if !slices.ContainsFunc(general.Features, func(feature settingsFeatureViewState) bool { return !feature.Shown }) {
		t.Errorf("%s: general normal fixture has no hidden feature", locale)
	}
	emptyLanguages := requireSettingsDialogFixture(t, locale, fixtures, "settings-general-empty-languages").General
	if len(emptyLanguages.OtherLanguages) != 0 {
		t.Errorf("%s: empty-languages fixture has %d other languages", locale, len(emptyLanguages.OtherLanguages))
	}

	accessibilityStates := map[bool]bool{}
	for state := permission.Unknown; state <= permission.Granted; state++ {
		name := "settings-permissions-state-" + settingsPermissionStateName(state)
		permissions := requireSettingsDialogFixture(t, locale, fixtures, name).Permissions
		expectedStatus, _ := permissionStatusText(translator, state)
		if permissions.Microphone.Status != expectedStatus {
			t.Errorf("%s/%s: microphone status %q, want %q", locale, name, permissions.Microphone.Status, expectedStatus)
		}
		expectedSpeechStatus, _ := permissionStatusText(translator, permission.Granted-state)
		if permissions.SpeechRecognition.Status != expectedSpeechStatus {
			t.Errorf("%s/%s: speech status %q, want %q", locale, name, permissions.SpeechRecognition.Status, expectedSpeechStatus)
		}
		expectedAccessibility := state == permission.Granted
		if permissions.Accessibility.Allowed != expectedAccessibility {
			t.Errorf("%s/%s: accessibility allowed=%t, want %t", locale, name, permissions.Accessibility.Allowed, expectedAccessibility)
		}
		accessibilityStates[permissions.Accessibility.Allowed] = true
	}
	assertSettingsFixtureSet(t, locale+" accessibility states", map[bool]bool{false: true, true: true}, accessibilityStates)

	checkSettingsAudioFixtures(t, locale, fixtures)
	checkSettingsAppearanceFixtures(t, locale, fixtures)
	checkSettingsShortcutFixtures(t, locale, fixtures)
	checkSettingsPrivacyFixtures(t, locale, fixtures)
	checkSettingsStorageFixtures(t, locale, fixtures)
	checkSettingsAboutFixtures(t, locale, fixtures)
	checkSettingsMeetingAndPromptFixtures(t, locale, fixtures)
}

func checkSettingsAudioFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	normal := requireSettingsDialogFixture(t, locale, fixtures, "settings-audio-normal").Audio
	if len(normal.Devices) != 2 {
		t.Errorf("%s: audio normal fixture has %d selected devices, want 2", locale, len(normal.Devices))
	}
	loading := requireSettingsDialogFixture(t, locale, fixtures, "settings-loading").Audio
	if !loading.DevicesLoading {
		t.Errorf("%s: audio loading fixture is not loading", locale)
	}
	empty := requireSettingsDialogFixture(t, locale, fixtures, "settings-empty").Audio
	if !empty.DevicesRead || len(empty.Devices) != 0 || len(empty.OtherDevices) != 0 {
		t.Errorf("%s: audio empty fixture has read=%t, %d selected and %d other devices", locale, empty.DevicesRead, len(empty.Devices), len(empty.OtherDevices))
	}
	helperError := requireSettingsDialogFixture(t, locale, fixtures, "settings-error").Audio
	if helperError.DevicesError == "" {
		t.Errorf("%s: audio helper-error fixture has no error", locale)
	}
	noHelper := requireSettingsDialogFixture(t, locale, fixtures, "settings-audio-no-helper").Audio
	if noHelper.HelperReady {
		t.Errorf("%s: no-helper fixture says the helper is ready", locale)
	}

	sourceIDs := map[string]bool{}
	for _, fixture := range fixtures {
		if fixture.Settings != nil && fixture.Settings.Audio != nil {
			sourceIDs[fixture.Settings.Audio.Sounds.SourceID] = true
		}
	}
	requiredSourceIDs := map[string]bool{}
	for _, source := range []soundSource{soundBuiltIn, soundSystem, soundNone} {
		requiredSourceIDs[soundSourceID(source)] = true
	}
	assertSettingsFixtureSet(t, locale+" sound sources", requiredSourceIDs, sourceIDs)

	customPair := requireSettingsDialogFixture(t, locale, fixtures, "settings-audio-source-built-in-custom-playing").Audio
	if !slices.ContainsFunc(customPair.Sounds.Pairs, func(pair settingsSoundPairViewState) bool {
		return pair.Start == "chime" && pair.Stop == "tink" && pair.Playing && pair.Remaining == settingsFixtureSoundRemaining
	}) {
		t.Errorf("%s: custom pair is not playing for the deterministic remaining time", locale)
	}
}

func checkSettingsAppearanceFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	themes, scales := map[string]bool{}, map[float64]bool{}
	for _, fixture := range fixtures {
		if fixture.Settings == nil || fixture.Settings.Appearance == nil {
			continue
		}
		appearance := fixture.Settings.Appearance
		if appearance.Theme >= 0 && appearance.Theme < len(appearance.ThemeValues) {
			themes[appearance.ThemeValues[appearance.Theme]] = true
		}
		if appearance.Scale >= 0 && appearance.Scale < len(appearance.ScaleValues) {
			scales[appearance.ScaleValues[appearance.Scale]] = true
		}
	}
	requiredThemes, requiredScales := map[string]bool{}, map[float64]bool{}
	for _, theme := range settings.Themes {
		requiredThemes[theme] = true
	}
	for _, scale := range settings.Scales {
		requiredScales[scale] = true
	}
	assertSettingsFixtureSet(t, locale+" themes", requiredThemes, themes)
	assertSettingsFixtureSet(t, locale+" scales", requiredScales, scales)

	variant := requireSettingsDialogFixture(t, locale, fixtures, "settings-appearance-dark-icons-reduce-motion").Appearance
	if variant.Theme < 0 || variant.Theme >= len(variant.ThemeValues) || variant.ThemeValues[variant.Theme] != "dark" {
		t.Errorf("%s: appearance variant does not select dark", locale)
	}
	if variant.Sidebar < 0 || variant.Sidebar >= len(variant.SidebarValues) || variant.SidebarValues[variant.Sidebar] != settingsSidebarIcons {
		t.Errorf("%s: appearance variant does not select icons", locale)
	}
	if !variant.ReduceMotion {
		t.Errorf("%s: appearance variant does not reduce motion", locale)
	}
	overlay := requireSettingsDialogFixture(t, locale, fixtures, "settings-overlay-top-live-lines").Overlay
	if overlay.Position != 1 || overlay.LiveLayout != 1 {
		t.Errorf("%s: overlay variant has position=%d and live layout=%d", locale, overlay.Position, overlay.LiveLayout)
	}
}

func checkSettingsShortcutFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	translator := i18n.NewTranslator(locale)
	variant := requireSettingsDialogFixture(t, locale, fixtures, "settings-shortcuts-duplicate-problem-strict-fn-no-accessibility").Shortcuts
	if !variant.StrictShortcuts || variant.AccessibilityInstalled {
		t.Errorf("%s: shortcut variant has strict=%t and accessibility=%t", locale, variant.StrictShortcuts, variant.AccessibilityInstalled)
	}
	problems := hotkey.Check(settingsFixtureProblemFeatures())
	problemCodes := map[string]bool{}
	expectedProblems := map[string][]string{}
	for _, problem := range problems {
		problemCodes[problem.Code] = true
		expectedProblems[hotkey.ID(problem.Feature, problem.Index)] = append(
			expectedProblems[hotkey.ID(problem.Feature, problem.Index)], uitext.HotkeyProblemTextFor(translator, problem))
		if problem.Code == hotkey.ProblemConflict {
			id := hotkey.ID(problem.Other, problem.OtherIndex)
			expectedProblems[id] = append(expectedProblems[id], translator.T(
				"%s has the same keys and trigger, so only one of them works.", uitext.FeatureTitleFor(translator, problem.Feature)))
		}
	}
	assertSettingsFixtureSet(t, locale+" shortcut problem codes", map[string]bool{
		hotkey.ProblemInvalid: true, hotkey.ProblemDuplicate: true, hotkey.ProblemConflict: true,
		hotkey.ProblemSystem: true, hotkey.ProblemBareKey: true,
	}, problemCodes)
	fnRows := 0
	for _, row := range variant.Rows {
		id := hotkey.ID(row.Feature, row.Index)
		if !slices.Equal(row.Problems, expectedProblems[id]) {
			t.Errorf("%s: shortcut row %s has problems %q, want %q", locale, id, row.Problems, expectedProblems[id])
		}
		delete(expectedProblems, id)
		if strings.Contains(strings.ToLower(row.KeyText), "fn") {
			fnRows++
		}
	}
	if len(expectedProblems) > 0 || fnRows == 0 {
		t.Errorf("%s: shortcut variant is missing problem rows %v and has %d fn rows", locale, expectedProblems, fnRows)
	}

	family := requireSettingsDialogFixture(t, locale, fixtures, "settings-shortcuts-family-conflict").Shortcuts
	conflict := settingsFixtureConflict()
	expectedLine := alsoUsedByFor(translator, conflict.Other, conflict.Theirs)
	if !slices.ContainsFunc(family.Rows, func(row settingsShortcutRowViewState) bool {
		return row.Feature == hotkey.Transcribe && slices.Contains(row.Problems, expectedLine)
	}) {
		t.Errorf("%s: family-conflict fixture lacks %q", locale, expectedLine)
	}
}

func checkSettingsPrivacyFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	normal := requireSettingsDialogFixture(t, locale, fixtures, "settings-privacy-normal").Privacy
	offline := requireSettingsDialogFixture(t, locale, fixtures, "settings-privacy-offline-disabled").Privacy
	if normal.Online == offline.Online {
		t.Errorf("%s: online and offline privacy fixtures have the same network text", locale)
	}
	if offline.KeepHistory || offline.KeepAudio || offline.KeepAudioEnabled || offline.LogText || offline.HistoryAccess.On {
		t.Errorf("%s: offline/disabled privacy fixture has an enabled feature: %+v", locale, offline)
	}
	clearHistory := requireSettingsDialogFixture(t, locale, fixtures, "settings-privacy-clear-confirmation").Privacy
	if !clearHistory.ClearHistoryAsk {
		t.Errorf("%s: privacy clear-confirmation fixture does not seed the dialog", locale)
	}
}

func checkSettingsStorageFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	loading := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-loading").Storage
	if loading.Report != nil {
		t.Errorf("%s: storage loading fixture has a report", locale)
	}
	normal := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-normal").Storage
	if normal.Report == nil || len(normal.Report.Categories) == 0 {
		t.Errorf("%s: storage normal fixture has no usage report", locale)
	}
	moveError := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-move-error").Storage
	if moveError.MoveError == "" {
		t.Errorf("%s: storage move-error fixture has no error", locale)
	}
	dataMove := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-data-move-progress").Storage
	if !dataMove.Moving || dataMove.MoveProgress != 0.5 || !dataMove.CanCancelMove {
		t.Errorf("%s: data move has moving=%t, progress=%v and cancel=%t", locale, dataMove.Moving, dataMove.MoveProgress, dataMove.CanCancelMove)
	}
	modelMove := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-model-move-progress").Storage
	if !modelMove.ModelMoving || modelMove.ModelMoveText == "" || !modelMove.CanCancelModelMove {
		t.Errorf("%s: model move has moving=%t, text=%q and cancel=%t", locale, modelMove.ModelMoving, modelMove.ModelMoveText, modelMove.CanCancelModelMove)
	}
	erase := requireSettingsDialogFixture(t, locale, fixtures, "settings-storage-erase-confirmation").Storage
	if !strings.Contains(erase.KeptModelsText, "fixture-shared-model") || erase.ResetText != "DELETE" {
		t.Errorf("%s: erase confirmation has kept text %q and confirmation %q", locale, erase.KeptModelsText, erase.ResetText)
	}

	resetKinds := map[string]bool{}
	for _, fixture := range fixtures {
		if fixture.Settings != nil && fixture.Settings.Storage != nil && fixture.Settings.Storage.ResetAsk != "" {
			resetKinds[fixture.Settings.Storage.ResetAsk] = true
		}
	}
	assertSettingsFixtureSet(t, locale+" reset confirmations", map[string]bool{"settings": true, "everything": true}, resetKinds)
}

func checkSettingsAboutFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	translator := i18n.NewTranslator(locale)
	normal := requireSettingsDialogFixture(t, locale, fixtures, "settings-about-normal").About
	loading := requireSettingsDialogFixture(t, locale, fixtures, "settings-about-loading").About
	if len(normal.Rows) <= len(loading.Rows) {
		t.Errorf("%s: about normal and loading fixtures both have %d and %d rows", locale, len(normal.Rows), len(loading.Rows))
	}
	selfChecking := requireSettingsDialogFixture(t, locale, fixtures, "settings-about-self-checking").About
	if !selfChecking.SelfChecking {
		t.Errorf("%s: about self-checking fixture is not checking", locale)
	}
	for _, name := range []string{"settings-about-normal", "settings-about-loading", "settings-about-self-checking", "settings-about-results"} {
		if updates := requireSettingsDialogFixture(t, locale, fixtures, name).Updates; updates.Present {
			t.Errorf("%s/%s: no-updater fixture has an updater", locale, name)
		}
	}
	expectedDescription := translator.T("Last checked %s.", uitext.TimeAgoFor(translator, settingsFixtureLastCheck, settingsFixtureNow))
	updates := requireSettingsDialogFixture(t, locale, fixtures, "settings-about-updates").Updates
	if !updates.Present || !updates.Known || !updates.Enabled || updates.CheckOnly || updates.BundleUpdated ||
		!updates.AutoCheck || updates.AutoInstall || updates.Beta || updates.AutoInstallDisabled {
		t.Errorf("%s: full updater fixture has unexpected state %+v", locale, updates)
	}
	if updates.Description != expectedDescription || updates.AutoCheckText != translator.T("Once a day. Voxt asks before installing.") || updates.BetaNote != translator.T("Releases only.") {
		t.Errorf("%s: full updater fixture has unexpected text %+v", locale, updates)
	}
	checkOnly := requireSettingsDialogFixture(t, locale, fixtures, "settings-about-updates-check-only").Updates
	if !checkOnly.Present || !checkOnly.Known || !checkOnly.Enabled || !checkOnly.CheckOnly || !checkOnly.BundleUpdated ||
		!checkOnly.AutoCheck || checkOnly.AutoInstall || !checkOnly.Beta {
		t.Errorf("%s: check-only updater fixture has unexpected state %+v", locale, checkOnly)
	}
	if checkOnly.Description != expectedDescription || checkOnly.AutoCheckText != translator.T("Once a day. A card says when a new version is out.") ||
		checkOnly.BetaNote != translator.T("Betas and releases, whichever is newer.") {
		t.Errorf("%s: check-only updater fixture has unexpected text %+v", locale, checkOnly)
	}
	statuses := map[string]bool{}
	for _, fixture := range fixtures {
		if fixture.Settings != nil && fixture.Settings.About != nil {
			for _, check := range fixture.Settings.About.Checks {
				statuses[check.Status] = true
			}
		}
	}
	assertSettingsFixtureSet(t, locale+" self-check statuses", map[string]bool{"pass": true, "warn": true, "fail": true}, statuses)
}

func checkSettingsMeetingAndPromptFixtures(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	captureModes := map[string]bool{}
	for _, fixture := range fixtures {
		if fixture.MeetingSettings != nil {
			state := fixture.MeetingSettings
			captureModes[settingsLanguageCode(state.CaptureModeSelected, state.CaptureModeOptions)] = true
		}
	}
	assertSettingsFixtureSet(t, locale+" meeting capture modes", map[string]bool{"": true, "subtitles": true, "recording": true}, captureModes)
	normal := requireSettingsFixture(t, locale, fixtures, "meeting-settings-normal").MeetingSettings
	if normal.ASRDisabled || normal.NoLLM || normal.LLMSelected != "Fixture Language Two" {
		t.Errorf("%s: meeting normal fixture has ASR disabled=%t, no LLM=%t and LLM %q", locale, normal.ASRDisabled, normal.NoLLM, normal.LLMSelected)
	}
	noModels := requireSettingsFixture(t, locale, fixtures, "meeting-settings-no-models").MeetingSettings
	if !noModels.ASRDisabled || !noModels.NoLLM {
		t.Errorf("%s: meeting no-models fixture has ASR disabled=%t and no LLM=%t", locale, noModels.ASRDisabled, noModels.NoLLM)
	}

	prompt := requireSettingsFixture(t, locale, fixtures, "shortcut-prompt-conflict").ShortcutPrompt
	expected := buildShortcutPromptViewState(i18n.NewTranslator(locale), settings.Defaults().Features, settingsFixtureConflict())
	if *prompt != expected {
		t.Errorf("%s: shortcut prompt is %+v, want %+v", locale, *prompt, expected)
	}
}

func requireSettingsFixture(t *testing.T, locale string, fixtures map[string]meetingFixtureState, name string) meetingFixtureState {
	t.Helper()
	fixture, ok := fixtures[name]
	if !ok {
		t.Fatalf("%s: missing required settings fixture %s", locale, name)
	}
	return fixture
}

func requireSettingsDialogFixture(t *testing.T, locale string, fixtures map[string]meetingFixtureState, name string) *settingsDialogViewState {
	t.Helper()
	fixture := requireSettingsFixture(t, locale, fixtures, name)
	if fixture.Settings == nil {
		t.Fatalf("%s/%s: missing settings dialog state", locale, name)
	}
	return fixture.Settings
}

func assertSettingsFixtureSet[T comparable](t *testing.T, label string, required, actual map[T]bool) {
	t.Helper()
	for value := range required {
		if !actual[value] {
			t.Errorf("%s: missing %v", label, value)
		}
	}
	for value := range actual {
		if !required[value] {
			t.Errorf("%s: unexpected %v", label, value)
		}
	}
}
