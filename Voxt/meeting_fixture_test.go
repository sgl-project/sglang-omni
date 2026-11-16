package main

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func TestMeetingViewStateGolden(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	scenarios := meetingFixtureScenarios()
	names := map[string]bool{}
	for _, scenario := range scenarios {
		if names[scenario.name] {
			t.Fatalf("duplicate scenario %q", scenario.name)
		}
		names[scenario.name] = true
	}
	for _, locale := range i18n.Supported() {
		i18n.SetLocale(locale)
		for _, scenario := range scenarios {
			t.Run(locale+"/"+scenario.name, func(t *testing.T) {
				fixture := buildMeetingFixture(t, scenario, locale)
				data, err := json.MarshalIndent(fixture, "", "  ")
				if err != nil {
					t.Fatal(err)
				}
				data = append(data, '\n')
				path := filepath.Join("testdata", "meeting", locale, scenario.name+".json")
				if os.Getenv("VOXTNEW_UPDATE_GOLDEN") == "1" {
					if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, data, 0o644); err != nil {
						t.Fatal(err)
					}
					return
				}
				want, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(data, want) {
					t.Fatalf("fixture differs; regenerate with VOXTNEW_UPDATE_GOLDEN=1 go test -run '^TestMeetingViewStateGolden$' .\n--- want\n%s\n--- got\n%s", want, data)
				}
			})
		}
	}
}

func buildMeetingFixture(t *testing.T, scenario meetingFixtureScenario, locale string) meetingFixtureState {
	t.Helper()
	fixture := meetingFixtureState{Name: scenario.name, Kind: scenario.kind}
	translator := i18n.NewTranslator(locale)
	switch scenario.kind {
	case "overlay":
		state := buildMeetingOverlayViewState(translator, scenario.snapshot, scenario.collapsed, scenario.cardSize)
		fixture.Overlay = &state
	case "page":
		state := buildMeetingsViewState(translator, scenario.page)
		fixture.Page = &state
	case "start":
		state := buildMeetingStartViewState(translator, scenario.err)
		fixture.Start = &state
	case "history":
		if scenario.historyProblem != "" {
			scenario.history = historyFixtureProblemInput(scenario.historyProblem)
		}
		state := buildHistoryViewState(translator, scenario.history)
		fixture.History = &state
	case "history-settings":
		if scenario.historySettings.DaysSelection == "" {
			scenario.historySettings.DaysSelection = historyKeepDayOptions(translator)[3].Label
		}
		if scenario.historySettings.CountSelection == "" {
			scenario.historySettings.CountSelection = historyKeepCountOptions(translator)[0].Label
		}
		state := buildHistorySettingsViewState(translator, scenario.historySettings)
		fixture.HistorySettings = &state
	case "history-access":
		state := buildHistoryAccessViewState(translator, scenario.historyAccess)
		fixture.HistoryAccess = &state
	case "models":
		if scenario.modelsProblem != "" {
			_, code := networkProxyValueFor(scenario.models.NetworkForm)
			if code != scenario.modelsProblem {
				t.Fatalf("%s: proxy validation returned %q, want %q", scenario.name, code, scenario.modelsProblem)
			}
			scenario.models.NetworkForm.proxyErr = networkProxyProblemTextFor(translator, code)
		}
		state := buildModelsViewState(translator, scenario.models)
		fixture.Models = &state
	default:
		t.Fatalf("unknown fixture kind %q", scenario.kind)
	}
	return fixture
}

func TestMeetingFixtureCoverage(t *testing.T) {
	scenarios := meetingFixtureScenarios()
	scenarioNames := map[string]bool{}
	for _, scenario := range scenarios {
		scenarioNames[scenario.name] = true
	}
	for _, locale := range i18n.Supported() {
		fixtures := readMeetingFixtures(t, locale)
		if len(fixtures) != len(scenarios) {
			t.Errorf("%s has %d fixtures, want %d", locale, len(fixtures), len(scenarios))
		}
		for name := range scenarioNames {
			if _, ok := fixtures[name]; !ok {
				t.Errorf("%s: missing fixture %s.json", locale, name)
			}
		}
		for name, fixture := range fixtures {
			if !scenarioNames[name] {
				t.Errorf("%s: orphan fixture %s.json", locale, name)
			}
			if fixture.Name != name {
				t.Errorf("%s/%s: embedded name is %q", locale, name, fixture.Name)
			}
		}
		checkMeetingFixtureValues(t, locale, fixtures)
	}
}

func readMeetingFixtures(t *testing.T, locale string) map[string]meetingFixtureState {
	t.Helper()
	directory := filepath.Join("testdata", "meeting", locale)
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	fixtures := map[string]meetingFixtureState{}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		name := strings.TrimSuffix(entry.Name(), ".json")
		file, err := os.Open(filepath.Join(directory, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		decoder := json.NewDecoder(file)
		decoder.DisallowUnknownFields()
		var fixture meetingFixtureState
		decodeErr := decoder.Decode(&fixture)
		closeErr := file.Close()
		if decodeErr != nil {
			t.Fatalf("%s/%s: %v", locale, entry.Name(), decodeErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		fixtures[name] = fixture
	}
	return fixtures
}

func checkMeetingFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	states := map[string]bool{}
	stages := map[string]bool{}
	modes := map[string]bool{}
	sources := map[string]bool{}
	problems := map[string]bool{}
	startGuidance := map[string]bool{}
	statuses := map[string]bool{}
	storeModes := map[string]bool{}
	segmentSources := map[string]bool{}
	for _, fixture := range fixtures {
		if overlay := fixture.Overlay; overlay != nil {
			states[overlay.State] = true
			if overlay.Stage != "" {
				stages[overlay.Stage] = true
			}
			if overlay.Mode != "" {
				modes[overlay.Mode] = true
			}
			for _, row := range overlay.Rows {
				sources[row.Source] = true
			}
			if overlay.ProblemKey != "" {
				problems[overlay.ProblemKey] = true
			}
		}
		if page := fixture.Page; page != nil {
			for _, item := range page.Items {
				if item.Meeting != nil {
					statuses[item.Meeting.StatusCode] = true
					storeModes[item.Meeting.ModeCode] = true
				}
			}
			if page.Detail != nil {
				for _, segment := range page.Detail.Segments {
					segmentSources[segment.Source] = true
				}
			}
		}
		if fixture.Start != nil {
			startGuidance[fixture.Start.Code] = true
		}
	}
	delete(states, "")
	assertMeetingFixtureSet(t, locale+" states", meetingTypedConstants(t, "internal/meeting/types.go", "State"), states)
	assertMeetingFixtureSet(t, locale+" stages", meetingTypedConstants(t, "internal/meeting/types.go", "Stage"), stages)
	assertMeetingFixtureSet(t, locale+" modes", meetingTypedConstants(t, "internal/meeting/types.go", "Mode"), modes)
	assertMeetingFixtureSet(t, locale+" sources", meetingTypedConstants(t, "internal/meeting/types.go", "Source"), sources)
	assertMeetingFixtureSet(t, locale+" store statuses", meetingTypedConstants(t, "internal/store/store.go", "MeetingStatus"), statuses)
	assertMeetingFixtureSet(t, locale+" store modes", meetingTypedConstants(t, "internal/store/store.go", "MeetingMode"), storeModes)
	assertMeetingFixtureSet(t, locale+" segment sources", meetingTypedConstants(t, "internal/store/store.go", "SegmentSource"), segmentSources)
	problemValues := meetingNamedConstants(t, "internal/meeting/problems.go", "ProblemKey")
	degradedPrefix := meeting.ProblemKeyTrackDegraded
	delete(problemValues, meeting.ProblemKeyTrackDegraded)
	for source := range meetingTypedConstants(t, "internal/meeting/types.go", "Source") {
		problemValues[degradedPrefix+source] = true
	}
	assertMeetingFixtureSet(t, locale+" problem codes", problemValues, problems)
	guidanceValues := meetingNamedConstants(t, "internal/meeting/types.go", "Guidance")
	for required := range guidanceValues {
		if !startGuidance[required] {
			t.Errorf("%s: no start fixture for %s", locale, required)
		}
	}
}

func assertMeetingFixtureSet(t *testing.T, label string, required, actual map[string]bool) {
	t.Helper()
	for value := range required {
		if !actual[value] {
			t.Errorf("%s: missing %s", label, value)
		}
	}
	for value := range actual {
		if !required[value] {
			t.Errorf("%s: unexpected %s", label, value)
		}
	}
}

func meetingTypedConstants(t *testing.T, path, typeName string) map[string]bool {
	t.Helper()
	return meetingConstants(t, path, func(spec *ast.ValueSpec) bool {
		identifier, ok := spec.Type.(*ast.Ident)
		return ok && identifier.Name == typeName
	})
}

func meetingNamedConstants(t *testing.T, path, prefix string) map[string]bool {
	t.Helper()
	return meetingConstants(t, path, func(spec *ast.ValueSpec) bool {
		for _, name := range spec.Names {
			if strings.HasPrefix(name.Name, prefix) {
				return true
			}
		}
		return false
	})
}

func meetingConstants(t *testing.T, path string, include func(*ast.ValueSpec) bool) map[string]bool {
	t.Helper()
	syntax, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	values := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.CONST {
			continue
		}
		for _, rawSpec := range general.Specs {
			spec := rawSpec.(*ast.ValueSpec)
			if !include(spec) {
				continue
			}
			for _, value := range spec.Values {
				literal, ok := value.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					continue
				}
				text, err := strconv.Unquote(literal.Value)
				if err != nil {
					t.Fatal(err)
				}
				values[text] = true
			}
		}
	}
	if len(values) == 0 {
		t.Fatalf("%s: no constants found", path)
	}
	return values
}
