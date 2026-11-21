package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func TestSessionDemoSwitchesFixtures(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/card-short")
	stored := host.current().state.Session
	storedText := stored.Card.Blocks[0].Text
	host.sessionState.Card.Blocks[0].Text = "local-only edit"
	if stored.Card.Blocks[0].Text != storedText {
		t.Fatal("the selected DTO aliases the stored fixture")
	}
	host.sessionState.Card.Blocks[0].Text = storedText
	host.sessionState.Card.Copied = true
	host.sessionUI.bubbleCopied = "stale"
	host.sessionUI.bubbleSeq = 9
	host.sessionUI.cardScrollKey = "stale"
	host.sessionUI.langScroll.Y = 42
	host.sessionUI.lines.id = "stale"
	host.sessionCardUI.langOpen = true
	host.sessionCardUI.quotesOpen = map[int]bool{99: true}
	host.sessionMemory = sessionOverlayLayoutMemory{LiveID: "stale", LiveWidth: 999, StreamHeight: 99}
	host.sessionMeasurements[liveTextKey("stale")] = ovMeasured{w: 99, h: 99}
	host.sessionLayout.Width = 999
	initial := host.selected
	host.selectMeetingFixture(1)
	if host.selected != initial+1 {
		t.Fatalf("selected %d, want %d", host.selected, initial+1)
	}
	if !reflect.DeepEqual(&host.sessionUI, &sessionOverlayLocalUI{}) {
		t.Fatalf("session overlay local state survived switching: %+v", &host.sessionUI)
	}
	if host.sessionCardUI.langOpen || host.sessionCardUI.thinkMenu || host.sessionCardUI.pillMenu || len(host.sessionCardUI.quotesOpen) != 0 {
		t.Fatalf("session card local state survived switching: %+v", host.sessionCardUI)
	}
	if host.sessionMemory != (sessionOverlayLayoutMemory{}) || len(host.sessionMeasurements) != 0 || host.sessionLayout.Width != 0 {
		t.Fatal("session layout or measurements survived switching")
	}
	if host.sessionState.Card == host.current().state.Session.Card {
		t.Fatal("the next selected DTO aliases its stored card")
	}
	if len(host.calls) != 0 {
		t.Fatalf("fixture switch recorded actions: %v", host.calls)
	}

	selectMeetingDemoFixture(t, host, "en/card-short")
	if host.sessionState.Card.Copied || host.sessionState.Card.Blocks[0].Text != storedText {
		t.Fatal("switching back did not restore the fixture DTO")
	}
	tester := newMeetingDemoTester(t, host)
	initial = host.selected
	if err := tester.ClickControl("Continue"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if host.selected != (initial+1)%len(host.fixtures) {
		t.Fatalf("Continue selected %d, want %d", host.selected, initial+1)
	}
	if err := tester.ClickControl("Back"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if host.selected != initial || len(host.calls) != 0 {
		t.Fatalf("Back selected %d or recorded calls %v", host.selected, host.calls)
	}
}

func TestSessionDemoHiddenFixtures(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for index, fixture := range host.fixtures {
		if fixture.state.Kind != "session-overlay" || fixture.state.Session.Mode != ovHidden {
			continue
		}
		t.Run(fixture.name, func(t *testing.T) {
			host.selectMeetingFixture(index - host.selected)
			tester := newMeetingDemoTester(t, host)
			checked++
			if !tester.HasText(fixture.name) {
				t.Fatalf("hidden fixture does not show its selector name %q", fixture.name)
			}
			if sentinel := sessionOverlayContentSentinel(host.sessionState); sentinel != "" {
				t.Fatalf("hidden fixture has visible overlay sentinel %q", sentinel)
			}
			wantLabels := []string{i18n.T("Back"), i18n.T("Continue"), fixture.name}
			if labels := tester.AccessLabels(); !slices.Equal(labels, wantLabels) {
				t.Fatalf("hidden fixture is not selector-only: got %q, want %q", labels, wantLabels)
			}
		})
	}
	if checked == 0 {
		t.Fatal("no hidden session fixtures")
	}
}

func TestSessionDemoRejectsInvalidSeed(t *testing.T) {
	directory := t.TempDir()
	if err := os.Mkdir(filepath.Join(directory, "en"), 0o755); err != nil {
		t.Fatal(err)
	}
	data := []byte(`{"name":"bad","kind":"session-overlay","session":{"mode":"hidden"},"session_ui":{"overlay":{},"menu":"language"}}`)
	if err := os.WriteFile(filepath.Join(directory, "en", "bad.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadMeetingDemoHost(directory); err == nil {
		t.Fatal("accepted a session fixture with a stale menu seed")
	}
}

func TestSessionOverlayDemoMatchesProduction(t *testing.T) {
	for _, name := range []string{"overlay-live-short", "card-conversation", "card-follow-up-selected-open", "session-problem-microphone", "session-markdown-conversation"} {
		t.Run(name, func(t *testing.T) {
			scenario := sessionFixtureScenarioNamed(t, name)
			a, generated, _ := driveSessionFixture(t, scenario, "en")
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, "en/"+name)
			// The fixture was written with macOS-rendered texts; compare
			// platform-neutralized JSON on every platform.
			if data, err := json.Marshal(host.sessionState); err != nil {
				t.Fatal(err)
			} else {
				normalized := *generated
				if err := json.Unmarshal([]byte(platformText(string(data))), &host.sessionState); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(normalized, host.sessionState) {
					t.Fatal("JSON round trip changed the production DTO")
				}
			}
			a.measureHook = nil
			a.ov.live.id, a.ov.live.w = "", 0
			a.ov.stream.h = 0
			a.ov.textW = nil
			a.ov.projection.pillReady = false
			a.refreshSessionOverlay(meetingFixtureNow)
			width, height, _ := a.sessionOverlaySize(&a.ov.projection.view)
			a.ov.mode, a.ov.w, a.ov.h = a.ov.projection.view.Mode, width, height
			placement := host.sessionDemoPlacement()
			if placement.overlay.Width != width+2*layout.Pad || placement.overlay.Height != height+2*layout.Pad {
				t.Fatalf("production size %dx%d, JSON size %dx%d", width, height, placement.overlay.Width, placement.overlay.Height)
			}
			production := newSessionProductionComparisonTester(a, placement.overlay.Width, placement.overlay.Height)
			fromJSON := newSessionJSONComparisonTester(host, placement.overlay.Width, placement.overlay.Height)
			frames(production)
			frames(fromJSON)
			if production.Image().Bounds() != fromJSON.Image().Bounds() || !bytes.Equal(production.Image().Pix, fromJSON.Image().Pix) {
				t.Fatalf("production and JSON fixture rendered different pixels\nproduction layout %+v texts %q\nJSON layout %+v texts %q", a.currentSessionOverlayLayout(), production.Texts(), host.sessionLayout, fromJSON.Texts())
			}
			sentinel := sessionOverlayContentSentinel(host.sessionState)
			if sentinel == "" || !production.HasText(sentinel) || !fromJSON.HasText(sentinel) {
				t.Fatalf("missing rendered sentinel %q", sentinel)
			}
		})
	}
}

func sessionFixtureScenarioNamed(t *testing.T, name string) sessionFixtureScenario {
	t.Helper()
	for _, scenario := range sessionFixtureScenarios() {
		if scenario.name == name {
			return scenario
		}
	}
	t.Fatalf("no session scenario %q", name)
	return sessionFixtureScenario{}
}

func newSessionProductionComparisonTester(a *app, width, height int) *ui.Tester {
	tester := ui.NewTester(func(context *ui.Context) {
		sessionOverlayView(context, &a.ov.projection.view, a.currentSessionOverlayLayout(), a,
			sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui}, sessionOverlayAnimation{}, a.postSessionOverlayMeasurements, nil)
	}, width, height)
	tester.SetLocale("en")
	tester.SetFontSmoothing(hudSmoothing)
	return tester
}

func newSessionJSONComparisonTester(host *meetingDemoHost, width, height int) *ui.Tester {
	tester := ui.NewTester(func(context *ui.Context) {
		host.rebuildSessionOverlayLayout()
		sessionOverlayView(context, &host.sessionState, host.sessionLayout, host, host.sessionLocal(),
			sessionOverlayAnimation{}, host.applySessionOverlayMeasurements, nil)
	}, width, height)
	tester.SetLocale(host.current().locale)
	tester.SetFontSmoothing(hudSmoothing)
	return tester
}
