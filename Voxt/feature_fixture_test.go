package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

var tryFixtureActiveStateScenarios = []struct {
	featureName, onboardingName string
	state                       session.State
}{
	{featureName: "feature-try-state-recording", onboardingName: "session-recording", state: session.Recording},
	{featureName: "feature-try-state-recognizing", onboardingName: "session-recognizing", state: session.Recognizing},
	{featureName: "feature-try-state-composing", onboardingName: "session-composing", state: session.Composing},
	{featureName: "feature-try-state-done", onboardingName: "session-done", state: session.Done},
	{featureName: "feature-try-state-cancelled", onboardingName: "session-cancelled", state: session.Cancelled},
	{featureName: "feature-try-state-failed", onboardingName: "try-error", state: session.Failed},
}

func TestFeatureViewStateUsesExplicitLocale(t *testing.T) {
	previousLocale := i18n.Locale()
	i18n.SetLocale("en")
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	input := featureFixtureTryInput(hotkey.Transcribe, session.Done, session.ProblemASR)
	input.Snapshot.Problem.Degraded = true
	input.Local.rules, input.Local.preview = true, true
	state := buildFeatureFixtureViewState(i18n.NewTranslator("zh-Hans"), input)
	if state.Title != "转录" {
		t.Errorf("title %q", state.Title)
	}
	if got := state.Models.ASR.Options[0].Label; got != "自动（现在用：Fixture speech model）" {
		t.Errorf("automatic model label %q", got)
	}
	if got := state.Shortcuts.Rows[0].Value; got != "右 ⌥" {
		t.Errorf("shortcut label %q", got)
	}
	if got := state.Shortcuts.Rows[0].Custom.ErrorText; got != "请输入按键，例如 Control+Option+Space。" {
		t.Errorf("shortcut parse error %q", got)
	}
	if got := state.Try.Ended.Warning.Text; got != "完整识别失败，用了实时识别的文字。" {
		t.Errorf("problem text %q", got)
	}
	if got := state.Try.Ended.TimingsText; !strings.HasPrefix(got, "录音 3.2 秒") {
		t.Errorf("timings %q", got)
	}
	if got := state.Prompt.Rules.Prompt; !strings.Contains(got, "你是语音输入应用 VoxtNew 的听写整理器") {
		t.Errorf("prompt locale %q", got)
	}
}

// TestFeatureProblemFixtureScenarios checks that every problem fixture can come from its session kind.
func TestFeatureProblemFixtureScenarios(t *testing.T) {
	tests := map[string]struct {
		kind     store.Kind
		state    session.State
		degraded bool
	}{
		"feature-problem-noASR":        {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-speechWait":   {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-microphone":   {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-device":       {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-helper":       {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-asr":          {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-llm":          {kind: store.KindTranslate, state: session.Failed},
		"feature-problem-noSpeech":     {kind: store.KindTranscribe, state: session.Done},
		"feature-problem-tooShort":     {kind: store.KindTranscribe, state: session.Cancelled},
		"feature-problem-noLLM":        {kind: store.KindTranslate, state: session.Failed},
		"feature-problem-empty":        {kind: store.KindRewrite, state: session.Failed},
		"feature-problem-truncated":    {kind: store.KindTranslate, state: session.Done, degraded: true},
		"feature-problem-internal":     {kind: store.KindTranscribe, state: session.Failed},
		"feature-problem-asr-degraded": {kind: store.KindTranscribe, state: session.Done, degraded: true},
		"feature-problem-llm-degraded": {kind: store.KindTranscribe, state: session.Done, degraded: true},
	}
	scenarios := map[string]featureFixtureScenarioInput{}
	for _, scenario := range featureFixtureScenarios() {
		if strings.HasPrefix(scenario.name, "feature-problem-") {
			scenarios[scenario.name] = scenario.feature
		}
	}
	if len(scenarios) != len(tests) {
		t.Fatalf("got %d problem scenarios, want %d", len(scenarios), len(tests))
	}
	for name, want := range tests {
		input, ok := scenarios[name]
		if !ok {
			t.Errorf("missing %s", name)
			continue
		}
		snapshot := input.Snapshot
		if snapshot.Kind != want.kind || snapshot.State != want.state || snapshot.Problem == nil || snapshot.Problem.Degraded != want.degraded {
			t.Errorf("%s: kind %s, state %s, problem %+v", name, snapshot.Kind, snapshot.State, snapshot.Problem)
		}
		if name == "feature-problem-empty" && (snapshot.Raw == "" || snapshot.Result != "" || snapshot.Draft != "") {
			t.Errorf("%s: empty Rewrite needs a raw instruction and no result or draft: %+v", name, snapshot)
		}
		try := buildFeatureFixtureViewState(i18n.NewTranslator("en"), input).Try
		if try.State != string(want.state) || try.Ended == nil {
			t.Errorf("%s: view %+v", name, try)
			continue
		}
		if try.Ended.Cancelled != (want.state == session.Cancelled) {
			t.Errorf("%s: cancelled %v", name, try.Ended.Cancelled)
		}
		if (try.Ended.Warning != nil) != (want.state == session.Done) {
			t.Errorf("%s: warning %+v", name, try.Ended.Warning)
		}
	}
}

func assertTryFixtureSnapshot(t *testing.T, name string, snapshot session.Snapshot, mine bool) {
	t.Helper()
	switch snapshot.State {
	case "":
		if !reflect.DeepEqual(snapshot, session.Snapshot{}) || mine {
			t.Errorf("%s: idle must be a zero snapshot with no Try owner: %+v, mine %t", name, snapshot, mine)
		}
		return
	case session.Idle:
		t.Errorf("%s: Manager.Start never starts in Idle; use a zero snapshot", name)
		return
	case session.Reading:
		t.Errorf("%s: Try cannot read a selection, so Reading is unreachable", name)
		return
	case session.Delivering:
		t.Errorf("%s: Try skips delivery, so Delivering is unreachable", name)
		return
	}
	legal := false
	for _, scenario := range tryFixtureActiveStateScenarios {
		legal = legal || scenario.state == snapshot.State
	}
	if !legal {
		t.Errorf("%s: unexpected active Try state %s", name, snapshot.State)
		return
	}
	if snapshot.ID == "" || !snapshot.Try || snapshot.Rev == 0 || !mine {
		t.Errorf("%s: active Try needs an ID, Try, a revision, and an owner: %+v, mine %t", name, snapshot, mine)
	}
	if snapshot.SelectionVia != "" {
		t.Errorf("%s: Try cannot read a selection: %+v", name, snapshot)
	}
	if snapshot.Delivery != "" || snapshot.Inserted || snapshot.Card {
		t.Errorf("%s: Try skips delivery and the answer card: %+v", name, snapshot)
	}
}

func TestTryFixtureSnapshotSuppliedSelection(t *testing.T) {
	suppliedSnapshot := func(state session.State, via string) session.Snapshot {
		snapshot := session.Snapshot{
			Rev: 1, ID: "sample-text", Kind: store.KindRewrite, State: state, Try: true,
			Raw: "rewrite this", Selection: "sample text", SelectionVia: via,
		}
		switch state {
		case session.Composing:
			snapshot.Draft = "partial answer"
		case session.Done:
			snapshot.Result = "answer"
		case session.Failed:
			snapshot.Problem = &session.Problem{Code: session.ProblemEmpty}
		}
		return snapshot
	}
	for _, state := range []session.State{session.Composing, session.Done, session.Failed} {
		t.Run(string(state), func(t *testing.T) {
			assertTryFixtureSnapshot(t, "supplied sample text", suppliedSnapshot(state, ""), true)
		})
	}
	t.Run("selection read", func(t *testing.T) {
		passed := testing.RunTests(func(string, string) (bool, error) { return true, nil }, []testing.InternalTest{{
			Name: "selection via ax",
			F: func(t *testing.T) {
				assertTryFixtureSnapshot(t, "selection read", suppliedSnapshot(session.Composing, "ax"), true)
			},
		}})
		if passed {
			t.Fatal("selection read passed Try fixture validation")
		}
	})
}

func expectedTryFixtureStateCoverage(t *testing.T) map[string]bool {
	t.Helper()
	expected := meetingTypedConstants(t, "internal/session/session.go", "State")
	delete(expected, string(session.Idle))       // Start has no Idle state; reachability tests check the zero snapshot.
	delete(expected, string(session.Reading))    // canSelect rejects Try.
	delete(expected, string(session.Delivering)) // finish bypasses delivery for Try.
	return expected
}

func TestFeatureTryFixtureStateReachability(t *testing.T) {
	expected := map[string]bool{"feature-try-state-idle": true}
	for _, scenario := range tryFixtureActiveStateScenarios {
		expected[scenario.featureName] = true
	}
	actual := map[string]bool{}
	for _, scenario := range featureFixtureScenarios() {
		if !strings.HasPrefix(scenario.name, "feature-try-state-") {
			continue
		}
		actual[scenario.name] = true
		assertTryFixtureSnapshot(t, scenario.name, scenario.feature.Snapshot, scenario.feature.Mine)
	}
	assertMeetingFixtureSet(t, "Feature Try scenarios", expected, actual)
}

func TestFeatureFixtureCoverage(t *testing.T) {
	for _, locale := range i18n.Supported() {
		checkFeatureFixtureValues(t, locale, readMeetingFixtures(t, locale))
	}
}

func checkFeatureFixtureValues(t *testing.T, locale string, fixtures map[string]meetingFixtureState) {
	t.Helper()
	for _, name := range []string{"feature-empty", "feature-loading", "feature-normal", "feature-error"} {
		if _, ok := fixtures[name]; !ok {
			t.Errorf("%s: missing required feature fixture %s", locale, name)
		}
	}
	features := map[string]bool{}
	states := map[string]bool{}
	problems := map[string]bool{}
	shortcutProblems := map[string]bool{}
	cardSizes := map[int]bool{}
	streaming := map[string]bool{}
	for _, scenario := range meetingFixtureScenarios() {
		if scenario.kind == "feature" {
			streaming[scenario.name] = featureFixtureStreams(scenario.feature)
		}
	}
	for name, fixture := range fixtures {
		feature := fixture.Feature
		if feature == nil {
			continue
		}
		features[feature.Feature] = true
		states[feature.Try.State] = true
		if feature.Try.ProblemCode != "" {
			problems[feature.Try.ProblemCode] = true
			if _, ok := fixtures["feature-problem-"+feature.Try.ProblemCode]; !ok {
				t.Errorf("%s: missing fixture for problem code %s", locale, feature.Try.ProblemCode)
			}
		}
		cardSizes[feature.Prompt.Card.Size] = true
		if feature.Feature == hotkey.Transcribe && streaming[name] != (feature.Shortcuts.EndWord != nil) {
			t.Errorf("%s/%s: end word visibility does not match the selected model", locale, name)
		}
		for _, row := range feature.Shortcuts.Rows {
			for _, code := range row.ProblemCodes {
				shortcutProblems[code] = true
			}
		}
		for _, row := range []*featureKeyRowViewState{feature.Shortcuts.Paste, feature.Shortcuts.Cancel} {
			if row == nil {
				continue
			}
			for _, code := range row.ProblemCodes {
				shortcutProblems[code] = true
			}
		}
		if name == "feature-shortcut-problems" && len(shortcutProblems) == 0 {
			t.Errorf("%s: shortcut problem fixture has no codes", locale)
		}
	}
	delete(states, "")
	assertMeetingFixtureSet(t, locale+" feature names", map[string]bool{hotkey.Transcribe: true, hotkey.Translate: true, hotkey.Rewrite: true}, features)
	assertMeetingFixtureSet(t, locale+" feature session states", expectedTryFixtureStateCoverage(t), states)
	expectedProblems := meetingNamedConstants(t, "internal/session/session.go", "Problem")
	// Supported Feature Try sessions cannot produce an unknown-kind error.
	delete(expectedProblems, session.ProblemUnsupported)
	assertMeetingFixtureSet(t, locale+" feature problem codes", expectedProblems, problems)
	assertMeetingFixtureSet(t, locale+" feature shortcut problem codes", meetingNamedConstants(t, "internal/hotkey/check.go", "Problem"), shortcutProblems)
	for size := range settings.CardSizeSteps {
		if !cardSizes[size] {
			t.Errorf("%s: missing card size %d", locale, size)
		}
	}
}
