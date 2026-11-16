package main

import (
	"encoding/json"
	"fmt"
	"io"
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
	fixtures          []meetingDemoFixture
	selected          int
	calls             []string
	pageUI            meetingsLocalUI
	overlayUI         meetingOverlayLocalUI
	historyUI         historyLocalUI
	historySettingsUI historySettingsLocalUI
	historyAccessUI   historyAccessLocalUI
	onboardingUI      onboardingLocalUI
	overlayLook       settings.Overlay
}

func loadMeetingDemoHost(directory string) (*meetingDemoHost, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	host := &meetingDemoHost{overlayLook: settings.Defaults().Overlay}
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
	host.selectMeetingFixture(0)
	return host, nil
}

func readMeetingDemoFixture(path string) (meetingFixtureState, error) {
	file, err := os.Open(path)
	if err != nil {
		return meetingFixtureState{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var fixture meetingFixtureState
	if err := decoder.Decode(&fixture); err != nil {
		return meetingFixtureState{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return meetingFixtureState{}, fmt.Errorf("trailing JSON value")
	}
	valid := fixture.Kind == "overlay" && fixture.Overlay != nil ||
		fixture.Kind == "page" && fixture.Page != nil ||
		fixture.Kind == "start" && fixture.Start != nil ||
		fixture.Kind == "history" && fixture.History != nil ||
		fixture.Kind == "history-settings" && fixture.HistorySettings != nil ||
		fixture.Kind == "history-access" && fixture.HistoryAccess != nil ||
		fixture.Kind == "onboarding" && fixture.Onboarding != nil
	if !valid {
		return meetingFixtureState{}, fmt.Errorf("kind %q has no matching state", fixture.Kind)
	}
	return fixture, nil
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
		fixture.state.Page.SettingsAvailable = false
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
	host.historyAccessUI = historyAccessLocalUI{}
	if state := fixture.state.HistoryAccess; state != nil {
		host.historyAccessUI = historyAccessLocalUI{
			historyAccess: state.On, historySelection: state.Selection, ask: state.Ask, listOpen: state.ListOpen,
		}
	}
	host.onboardingUI = onboardingLocalUI{}
	if state := fixture.state.Onboarding; state != nil {
		host.onboardingUI.open = state.Open
	}
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
func (host *meetingDemoHost) stopSession()   { host.record("stopSession") }
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
func (host *meetingDemoHost) downloadModel(id string) { host.record("downloadModel %s", id) }
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
func (host *meetingDemoHost) importLegacy() { host.record("importLegacy") }
func (host *meetingDemoHost) completeOnboardingLegacyTodo(kind, id string) {
	host.record("completeOnboardingLegacyTodo %s %s", kind, id)
}
func (host *meetingDemoHost) applyShortcutPreset(preset string) {
	host.record("applyShortcutPreset %s", preset)
}
