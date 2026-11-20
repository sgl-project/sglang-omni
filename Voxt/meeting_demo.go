package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingDemoFixture struct {
	locale string
	name   string
	state  meetingFixtureState
}

type meetingDemoHost struct {
	fixtures              []meetingDemoFixture
	selected              int
	calls                 []string
	pageUI                meetingsLocalUI
	overlayUI             meetingOverlayLocalUI
	historyUI             historyLocalUI
	historySettingsUI     historySettingsLocalUI
	historyAccessUI       historyAccessLocalUI
	updatesUI             updatesLocalUI
	onboardingUI          onboardingLocalUI
	appEnhancementUI      appEnhancementLocalUI
	appEnhancementDetails map[string][]appEnhancementGroupDetail
	appEnhancementApps    map[string][]appEnhancementAppChoice
	dictionaryUI          dictionaryLocalUI
	notesUI               notesLocalUI
	remoteWaitUI          llmWaitLocalUI
	remoteDialogUI        providerDialogLocalUI
	remoteDebugUI         llmDebugUI
	modelsUI              modelsLocalUI
	feature               featureDemoState
	overlayLook           settings.Overlay
	sessionState          sessionOverlayViewState
	sessionSeed           sessionOverlayFixtureUI
	sessionUI             sessionOverlayLocalUI
	sessionCardUI         answerCardUI
	sessionLayout         sessionOverlayLayout
	sessionMemory         sessionOverlayLayoutMemory
	sessionMeasurements   map[ovSized]ovMeasured
	settingsOpen          bool
	settingsRouter        *settingsDemoRouter
	settingsForm          settingsForm
	settingsStorageUI     settingsStorageLocalUI
	settingsMeetingUI     meetingSettingsUI
	settingsDialogUI      settingsDialogLocalUI
	keyRow                keyRow
	recorderActive        bool
	shellDemoState        // shell_demo_host.go
}

func loadMeetingDemoHost(directory string) (*meetingDemoHost, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	host := &meetingDemoHost{
		overlayLook:           settings.Defaults().Overlay,
		appEnhancementDetails: map[string][]appEnhancementGroupDetail{},
		appEnhancementApps:    map[string][]appEnhancementAppChoice{},
	}
	for _, localeEntry := range entries {
		if !localeEntry.IsDir() || !slices.Contains(i18n.Supported(), localeEntry.Name()) {
			return nil, fmt.Errorf("meeting demo: unsupported entry %s", localeEntry.Name())
		}
		locale := localeEntry.Name()
		files, err := os.ReadDir(filepath.Join(directory, locale))
		if err != nil {
			return nil, err
		}
		for _, file := range files {
			if file.IsDir() || filepath.Ext(file.Name()) != ".json" {
				continue
			}
			name := strings.TrimSuffix(file.Name(), ".json")
			path := filepath.Join(directory, locale, file.Name())
			fixture, err := readMeetingDemoFixture(path)
			if err != nil {
				return nil, fmt.Errorf("meeting demo: %s: %w", path, err)
			}
			if fixture.Name != name {
				return nil, fmt.Errorf("meeting demo: %s has name %q", path, fixture.Name)
			}
			host.fixtures = append(host.fixtures, meetingDemoFixture{locale: locale, name: locale + "/" + name, state: fixture})
			if fixture.AppEnhancementDetails != nil {
				host.appEnhancementDetails[locale] = fixture.AppEnhancementDetails
			}
			if name == "appenhance-apps-normal" {
				host.appEnhancementApps[locale] = fixture.AppEnhancement.Apps.Rows
			}
		}
	}
	if len(host.fixtures) == 0 {
		return nil, fmt.Errorf("meeting demo: no fixtures in %s", directory)
	}
	sort.Slice(host.fixtures, func(i, j int) bool { return host.fixtures[i].name < host.fixtures[j].name })
	for index, fixture := range host.fixtures {
		if index > 0 && host.fixtures[index-1].name == fixture.name {
			return nil, fmt.Errorf("meeting demo: duplicate fixture %s", fixture.name)
		}
	}
	initial := 0
	for index, fixture := range host.fixtures {
		state := fixture.state.AppEnhancement
		if state != nil && (state.Apps != nil || state.Browsers != nil || state.Delete != nil) {
			continue
		}
		initial = index
		break
	}
	host.selectMeetingFixture(initial)
	return host, nil
}

func (host *meetingDemoHost) current() meetingDemoFixture {
	return host.fixtures[host.selected]
}

func (host *meetingDemoHost) selectMeetingFixture(delta int) {
	count := len(host.fixtures)
	host.selected = (host.selected + delta%count + count) % count
	fixture := host.current()
	i18n.SetLocale(fixture.locale)
	host.pageUI = meetingsLocalUI{}
	if fixture.state.Page != nil {
		host.pageUI.query = fixture.state.Page.Query
		host.pageUI.page = fixture.state.Page.Page
	}
	host.overlayUI = meetingOverlayLocalUI{}
	if fixture.state.Overlay != nil {
		host.overlayUI.collapsed = fixture.state.Overlay.Collapsed
	}
	host.historyUI = historyLocalUI{}
	if fixture.state.History != nil {
		host.historyUI.tab = len(store.Kinds)
		for index, kind := range store.Kinds {
			if string(kind) == fixture.state.History.Kind {
				host.historyUI.tab = index
			}
		}
		host.historyUI.query = fixture.state.History.Query
	}
	host.historySettingsUI = historySettingsLocalUI{}
	if state := fixture.state.HistorySettings; state != nil {
		host.historySettingsUI = historySettingsLocalUI{
			open: state.Open, cleanOn: state.CleanOn, daysSel: state.DaysSelection, countSel: state.CountSelection,
			exportSel: state.RecordingScope, pruneAsk: state.PruneAsk, delAudioAsk: state.DeleteAudioAsk,
			textScope: state.TextScope, textFormat: state.TextFormat,
		}
	}
	host.selectFeatureFixture(fixture.state.Feature)
	host.historyAccessUI = historyAccessLocalUI{}
	if state := fixture.state.HistoryAccess; state != nil {
		host.historyAccessUI = historyAccessLocalUI{
			historyAccess: state.On, historySelection: state.Selection, ask: state.Ask, listOpen: state.ListOpen,
		}
	}
	host.resetSessionFixture(fixture.state)
	host.updatesUI = updatesLocalUI{relOpen: fixture.state.UpdatesRelease != nil}
	if state := fixture.state.UpdatesSettings; state != nil {
		host.updatesUI.autoCheck, host.updatesUI.autoInstall, host.updatesUI.beta = state.AutoCheck, state.AutoInstall, state.Beta
	}
	host.onboardingUI = onboardingLocalUI{}
	if state := fixture.state.Onboarding; state != nil {
		host.onboardingUI.open = state.Open
	}
	host.appEnhancementUI = appEnhancementLocalUI{}
	if state := fixture.state.AppEnhancement; state != nil {
		host.appEnhancementUI.sel = state.Selected
		host.appEnhancementUI.appsOpen = state.Apps != nil
		host.appEnhancementUI.browsersOpen = state.Browsers != nil
		host.appEnhancementUI.delAsk = state.Delete != nil
		if state.Apps != nil {
			host.appEnhancementUI.appsQuery = state.Apps.Query
			host.appEnhancementUI.bundleIn = state.Apps.BundleInput
		}
		if state.Browsers != nil {
			host.appEnhancementUI.brID = state.Browsers.BundleInput
			host.appEnhancementUI.brName = state.Browsers.NameInput
		}
	}
	host.dictionaryUI = dictionaryLocalUI{}
	if state := fixture.state.Dictionary; state != nil {
		host.dictionaryUI.query = state.Query
		host.dictionaryUI.tab = state.Tab
		host.dictionaryUI.scanOpen = state.ScanOpen
		host.dictionaryUI.notes = speechContextState{text: state.NotesText, seen: state.NotesText}
		host.dictionaryUI.notesQuery = state.NotesQuery
		host.dictionaryUI.notesFind.cur = state.NotesFindCurrent
		host.dictionaryUI.editOpen = state.Edit.Open
		host.dictionaryUI.edit = termEdit{
			id: state.Edit.ID, term: state.Edit.Term, aliases: state.Edit.Aliases, note: state.Edit.Note,
			enabled: state.Edit.Enabled, err: state.Edit.Error, saving: state.Edit.Saving,
			original: state.Edit.Original, suggestion: state.Edit.Suggestion,
		}
		host.dictionaryUI.editWas = host.dictionaryUI.edit
		edit := state.ProfileEdit
		host.dictionaryUI.profileEdit = dictionaryProfileEdit{open: edit.Open, saving: edit.Saving, id: edit.ID, name: edit.Name, err: edit.Error}
		host.dictionaryUI.profileDeleteOpen, host.dictionaryUI.profileDeleteID = state.ProfileDelete != "", state.ProfileDelete
	}
	host.notesUI = notesLocalUI{}
	if state := fixture.state.Notes; state != nil {
		host.notesUI.editID, host.notesUI.draft, host.notesUI.saving = state.Edit.ID, state.Edit.Draft, state.Edit.Saving
		shortcut := state.Shortcut
		host.notesUI.keys = keyRow{
			sel: shortcut.Selection, custom: shortcut.Custom, typing: shortcut.Typing, edited: shortcut.Edited,
			text: shortcut.Text, recFocused: shortcut.Custom, recordDenied: shortcut.RecordDenied, unknown: shortcut.Unknown,
		}
		if host.notesUI.keys.sel == "" {
			host.notesUI.keys.sel = shortcut.CurrentText
		}
	}
	host.remoteWaitUI = llmWaitLocalUI{}
	host.remoteDialogUI = providerDialogLocalUI{}
	if state := fixture.state.Remote; state != nil {
		host.remoteWaitUI = llmWaitLocalUI{llmWaitSel: state.Wait.Selected, llmWaitFor: state.Wait.Token}
		if state.Dialog != nil {
			host.remoteDialogUI = remoteDialogLocalState(*state.Dialog)
		}
	}
	if state := fixture.state.RemoteDialog; state != nil {
		host.remoteDialogUI = remoteDialogLocalState(*state)
	}
	host.remoteDebugUI = llmDebugUI{}
	if state := fixture.state.RemoteDebug; state != nil {
		host.remoteDebugUI = llmDebugLocalState(*state)
	}
	host.modelsUI = modelsLocalUI{}
	if state := fixture.state.Models; state != nil {
		host.modelsUI.modelAdvanced = state.Advanced.Open
		host.modelsUI.modelKeepSel = state.Advanced.Selection
		host.modelsUI.modelKeepFor = state.Advanced.Value + "\x00" + state.Advanced.Locale
		host.modelsUI.modelHiddenOpen = state.HiddenOpen
		host.modelsUI.netOpen = state.Network.Open
		host.modelsUI.netForm = networkForm{
			proxyMode: state.Network.ProxyMode, proxyErr: state.Network.ProxyError,
			source: state.Network.Source, sourceURL: state.Network.SourceURL,
		}
		host.modelsUI.netWas = host.modelsUI.netForm
		host.modelsUI.speechContext = speechContextState{text: state.SpeechContext, seen: state.SpeechContext}
		if state.Delete.Plain != nil {
			host.modelsUI.modelAsk = state.Delete.Plain.ID
		}
		if state.Delete.Shared != nil {
			host.modelsUI.modelShared = &sharedDelete{id: state.Delete.Shared.ID, forAll: state.Delete.Shared.ForAll}
		}
		if state.Assignment != nil {
			host.modelsUI.assign = modelAssignmentLocalUI{
				open: state.Assignment.Open, kind: remoteKind(state.Assignment.Kind), id: state.Assignment.ID,
				selected: state.Assignment.Checked,
			}
		}
	}
	host.resetShellDemo(fixture)
	host.keyRow = keyRow{}
	host.recorderActive = false
	if state := fixture.state.KeyRecorder; state != nil {
		host.keyRow = keyRow{
			custom: state.Custom, typing: state.Typing, text: state.RawKeys, edited: state.Edited,
			unknown: state.Unknown, recordDenied: state.RecordDenied, recFocused: state.Focused,
		}
	}
	host.resetSettingsDemo(fixture)
}

func (host *meetingDemoHost) featureViewState(fixture *featureViewState) featureViewState {
	state := *fixture
	state.Try.Block = host.feature.block
	state.Shortcuts.Rows = slices.Clone(state.Shortcuts.Rows)
	for index := range state.Shortcuts.Rows {
		if index < len(host.feature.local.rows) {
			state.Shortcuts.Rows[index].Custom = buildFeatureCustomKeysViewState(i18n.CurrentTranslator(), host.feature.local.rows[index])
		}
	}
	if state.Shortcuts.Paste != nil {
		row := *state.Shortcuts.Paste
		row.Custom = buildFeatureCustomKeysViewState(i18n.CurrentTranslator(), host.feature.local.paste)
		state.Shortcuts.Paste = &row
	}
	if state.Shortcuts.Cancel != nil {
		row := *state.Shortcuts.Cancel
		row.Custom = buildFeatureCustomKeysViewState(i18n.CurrentTranslator(), host.feature.local.cancel)
		state.Shortcuts.Cancel = &row
	}
	state.Prompt.Rules.Matches = findFolded(host.feature.local.prompt, host.feature.local.find)
	return state
}

func (host *meetingDemoHost) faceStyle(model meetingOverlayModel) meetingOverlayFaceStyle {
	mode, height := meetingOverlayFace(model)
	return meetingOverlayFaceStyle{radius: faceRadius(host.overlayLook, mode, height), alpha: layout.FaceAlpha(host.overlayLook)}
}

func (host *meetingDemoHost) record(format string, arguments ...any) {
	call := fmt.Sprintf(format, arguments...)
	host.calls = append(host.calls, call)
	fmt.Printf("meeting demo action: %s\n", call)
}

func (host *meetingDemoHost) startMeeting()  { host.record("startMeeting") }
func (host *meetingDemoHost) meetingFinish() { host.record("meetingFinish") }
func (host *meetingDemoHost) loadMeetings()  { host.record("loadMeetings") }
func (host *meetingDemoHost) searchMeetings(query string) {
	host.pageUI.query, host.pageUI.page = query, 0
	host.record("searchMeetings %q", query)
}
func (host *meetingDemoHost) showMeetingsPage(page int) {
	host.pageUI.page = page
	host.record("showMeetingsPage %d", page)
}
func (host *meetingDemoHost) openMeetingDetail(id string) { host.record("openMeetingDetail %s", id) }
func (host *meetingDemoHost) closeMeetingDetail()         { host.record("closeMeetingDetail") }
func (host *meetingDemoHost) deleteMeeting(id string)     { host.record("deleteMeeting %s", id) }
func (host *meetingDemoHost) meetingReprocess(id string)  { host.record("meetingReprocess %s", id) }
func (host *meetingDemoHost) meetingCorrect(id string)    { host.record("meetingCorrect %s", id) }
func (host *meetingDemoHost) showMeetingInFinder(id string) {
	host.record("showMeetingInFinder %s", id)
}
func (host *meetingDemoHost) regenerateMeetingSummary(id string) {
	host.record("regenerateMeetingSummary %s", id)
}
func (host *meetingDemoHost) openMeetingLanguageModels() { host.record("openMeetingLanguageModels") }
func (host *meetingDemoHost) openMeetingSpeechModels()   { host.record("openMeetingSpeechModels") }
func (host *meetingDemoHost) copyMeetingTranscript(id string) {
	host.record("copyMeetingTranscript %s", id)
}
func (host *meetingDemoHost) toggleMeetingPlayback(id string) {
	host.record("toggleMeetingPlayback %s", id)
}
func (host *meetingDemoHost) seekMeetingPlayback(id string, ms int64) {
	host.record("seekMeetingPlayback %s %d", id, ms)
}
func (host *meetingDemoHost) askMeetingFollowUp(id, question string) {
	host.record("askMeetingFollowUp %s %q", id, question)
}
func (host *meetingDemoHost) saveMeetingExport(id string, format meetingExportFormat) {
	host.record("saveMeetingExport %s %s", id, format)
}
func (host *meetingDemoHost) meetingPauseResume() { host.record("meetingPauseResume") }
func (host *meetingDemoHost) meetingSave()        { host.record("meetingSave") }
func (host *meetingDemoHost) meetingDiscard()     { host.record("meetingDiscard") }
func (host *meetingDemoHost) meetingSetModeID(id string) {
	host.overlayUI.menu = meetingMenuNone
	host.record("meetingSetModeID %s", id)
}
func (host *meetingDemoHost) toggleMeetingTranslation(id string) {
	host.overlayUI.menu = meetingMenuNone
	host.record("toggleMeetingTranslation %s", id)
}
func (host *meetingDemoHost) copyLiveMeetingTranscript() {
	host.overlayUI.menu = meetingMenuNone
	host.record("copyLiveMeetingTranscript")
}
func (host *meetingDemoHost) setMeetingOverlayCardSize(size int) {
	host.record("setMeetingOverlayCardSize %d", size)
}
func (host *meetingDemoHost) toggleMeetingMenu(menu meetingMenu) {
	if host.overlayUI.menu == menu {
		menu = meetingMenuNone
	}
	host.overlayUI.menu = menu
	host.record("toggleMeetingMenu %d", menu)
}
func (host *meetingDemoHost) setMeetingOverlayCollapsed(collapsed bool) {
	host.overlayUI.collapsed = collapsed
	host.record("setMeetingOverlayCollapsed %t", collapsed)
}
func (host *meetingDemoHost) requestMeetingDiscard() {
	host.overlayUI.menu = meetingMenuNone
	host.overlayUI.abandon = true
	host.record("requestMeetingDiscard")
}

func (host *meetingDemoHost) loadHistory() { host.record("loadHistory") }
func (host *meetingDemoHost) searchHistory(query string) {
	host.historyUI.query = query
	host.record("searchHistory %q", query)
}
func (host *meetingDemoHost) showHistoryTab(tab int) {
	host.historyUI.tab, host.historyUI.sel = tab, ""
	host.record("showHistoryTab %d", tab)
}
func (host *meetingDemoHost) openHistoryDetail(id string) {
	host.historyUI.sel = id
	host.record("openHistoryDetail %s", id)
}
func (host *meetingDemoHost) closeHistoryDetail() {
	host.historyUI.sel = ""
	host.record("closeHistoryDetail")
}
func (host *meetingDemoHost) openHistoryFeature(feature string) {
	host.record("openHistoryFeature %s", feature)
}
func (host *meetingDemoHost) openHistoryPrivacy() { host.record("openHistoryPrivacy") }
func (host *meetingDemoHost) saveHistoryDetailWidth(width float64) {
	host.record("saveHistoryDetailWidth %g", width)
}
func (host *meetingDemoHost) openHistorySettingsDialog() {
	host.historySettingsUI.open = true
	host.record("openHistorySettingsDialog")
}
func (host *meetingDemoHost) copyHistoryText(text string) { host.record("copyHistoryText %q", text) }
func (host *meetingDemoHost) showHistoryRecording(id string) {
	host.record("showHistoryRecording %s", id)
}
func (host *meetingDemoHost) retryHistoryEntry(id string) {
	host.record("retryHistoryEntry %s", id)
}
func (host *meetingDemoHost) deleteHistoryEntry(id string) {
	host.record("deleteHistoryEntry %s", id)
}
func (host *meetingDemoHost) chooseHistoryRecordingFolder() {
	host.record("chooseHistoryRecordingFolder")
}
func (host *meetingDemoHost) showHistoryRecordingFolder() {
	host.record("showHistoryRecordingFolder")
}
func (host *meetingDemoHost) useDefaultHistoryRecordingFolder() {
	host.record("useDefaultHistoryRecordingFolder")
}
func (host *meetingDemoHost) exportHistoryRecordings(shown bool) {
	host.record("exportHistoryRecordings %t", shown)
}
func (host *meetingDemoHost) exportShownHistoryText(shown bool, format historyTextFormat) {
	host.record("exportShownHistoryText %t %s", shown, format)
}
func (host *meetingDemoHost) setHistoryCleanup(on bool) {
	host.historySettingsUI.cleanOn = on
	host.record("setHistoryCleanup %t", on)
}
func (host *meetingDemoHost) setHistoryCleanupDays(keepDays int) {
	host.record("setHistoryCleanupDays %d", keepDays)
}
func (host *meetingDemoHost) setHistoryCleanupCount(keepCount int) {
	host.record("setHistoryCleanupCount %d", keepCount)
}
func (host *meetingDemoHost) savePendingHistoryCleanup() {
	host.record("savePendingHistoryCleanup")
}
func (host *meetingDemoHost) deleteHistoryRecordings() {
	host.record("deleteHistoryRecordings")
}
func (host *meetingDemoHost) cancelHistoryWork() { host.record("cancelHistoryWork") }
func (host *meetingDemoHost) setHistoryAccess(on bool) {
	host.historyAccessUI.historyAccess = on
	host.record("setHistoryAccess %t", on)
}
func (host *meetingDemoHost) setHistorySelection(on bool) {
	host.historyAccessUI.historySelection = on
	host.record("setHistorySelection %t", on)
}
func (host *meetingDemoHost) copyHistoryAccessCommand() {
	host.record("copyHistoryAccessCommand")
}
func (host *meetingDemoHost) openHistoryAccessDocs() { host.record("openHistoryAccessDocs") }

func (host *meetingDemoHost) checkForUpdates() { host.record("checkForUpdates") }
func (host *meetingDemoHost) setAutoCheck(on bool) {
	host.updatesUI.autoCheck = on
	host.record("setAutoCheck %t", on)
}
func (host *meetingDemoHost) setAutoInstall(on bool) {
	host.updatesUI.autoInstall = on
	host.record("setAutoInstall %t", on)
}
func (host *meetingDemoHost) setBeta(on bool) {
	host.updatesUI.beta = on
	host.record("setBeta %t", on)
}
func (host *meetingDemoHost) relaunch() { host.record("relaunch") }
func (host *meetingDemoHost) checkReleasesNow() {
	host.updatesUI.relOpen = true
	host.record("checkReleasesNow")
}
func (host *meetingDemoHost) downloadRelease(address string) {
	host.record("downloadRelease %q", address)
}

func (host *meetingDemoHost) openHomeGuide()   { host.record("openHomeGuide") }
func (host *meetingDemoHost) openHomeHistory() { host.record("openHomeHistory") }
func (host *meetingDemoHost) openHomeRecent(id string) {
	host.record("openHomeRecent %s", id)
}
func (host *meetingDemoHost) copyHomeRecent(id string) {
	host.record("copyHomeRecent %s", id)
}
func (host *meetingDemoHost) addHomeShortcut() { host.record("addHomeShortcut") }
func (host *meetingDemoHost) fixHomeReadiness(fix string) {
	host.record("fixHomeReadiness %s", fix)
}

func (host *meetingDemoHost) openOnboarding()  { host.record("openOnboarding") }
func (host *meetingDemoHost) closeOnboarding() { host.record("closeOnboarding") }
func (host *meetingDemoHost) finishOnboarding(path string) {
	host.record("finishOnboarding %s", path)
}
func (host *meetingDemoHost) dismissOnboardingBack() { host.record("dismissOnboardingBack") }
func (host *meetingDemoHost) obGo(step int)          { host.record("obGo %d", step) }
func (host *meetingDemoHost) obLeave(path string)    { host.record("obLeave %s", path) }
func (host *meetingDemoHost) obTry(text string) *sessionNotice {
	host.record("obTry %q", text)
	return nil
}
func (host *meetingDemoHost) cancelSession() { host.record("cancelSession") }
func (host *meetingDemoHost) copyOnboardingResult(text string) {
	host.record("copyOnboardingResult %q", text)
}
func (host *meetingDemoHost) clearOnboardingBlock() { host.record("clearOnboardingBlock") }
func (host *meetingDemoHost) onboardingNoticeAction(action string) {
	host.record("onboardingNoticeAction %s", action)
}
func (host *meetingDemoHost) onboardingTryLanguage() { host.record("onboardingTryLanguage") }
func (host *meetingDemoHost) onboardingDownloadAndContinue() {
	host.record("onboardingDownloadAndContinue")
}
func (host *meetingDemoHost) obUse(ids ...string) {
	host.record("obUse %s", strings.Join(ids, ","))
}
func (host *meetingDemoHost) askRequired() { host.record("askRequired") }
func (host *meetingDemoHost) askPermission(which string, then func()) {
	host.record("askPermission %s", which)
}
func (host *meetingDemoHost) recheckPermissions() { host.record("recheckPermissions") }
func (host *meetingDemoHost) onboardingOpenPrivacy(url string) {
	host.record("onboardingOpenPrivacy %s", url)
}
func (host *meetingDemoHost) completeOnboardingLegacyTodo(kind, id string) {
	host.record("completeOnboardingLegacyTodo %s %s", kind, id)
}
func (host *meetingDemoHost) applyShortcutPreset(preset string) {
	host.record("applyShortcutPreset %s", preset)
}
