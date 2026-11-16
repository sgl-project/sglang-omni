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
)

type meetingDemoFixture struct {
	locale string
	name   string
	state  meetingFixtureState
}

type meetingDemoHost struct {
	fixtures    []meetingDemoFixture
	selected    int
	calls       []string
	pageUI      meetingsLocalUI
	overlayUI   meetingOverlayLocalUI
	overlayLook settings.Overlay
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
	valid := fixture.Kind == "overlay" && fixture.Overlay != nil || fixture.Kind == "page" && fixture.Page != nil || fixture.Kind == "start" && fixture.Start != nil
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
		host.pageUI.query = fixture.state.Page.Query
		host.pageUI.page = fixture.state.Page.Page
	}
	host.overlayUI = meetingOverlayLocalUI{}
	if fixture.state.Overlay != nil {
		host.overlayUI.collapsed = fixture.state.Overlay.Collapsed
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
