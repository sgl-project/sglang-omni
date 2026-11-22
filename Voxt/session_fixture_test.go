package main

import (
	"sort"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func driveSessionFixture(t *testing.T, scenario sessionFixtureScenario, locale string) (*app, *sessionOverlayViewState, *sessionOverlayFixtureUI) {
	t.Helper()
	a, _, _ := newTestApp(t)
	i18n.SetLocale(locale)
	a.prefs.Overlay.ShowTime, a.prefs.Overlay.LiveLines = false, false
	switch scenario.source {
	case sessionFixtureBenchmark:
		recipe := sessionOverlayBenchmarkStates(a, meetingFixtureNow)[scenario.benchmarkIndex]
		if recipe.name == "overlay-rewrite-conversation" {
			a.answer = answerCard{open: true, id: "first", kind: store.KindRewrite, conversation: true}
			a.answer.turns = []compose.Turn{{Instruction: "写一句话告诉同事我会晚到十分钟", Answer: compose.Answer{Title: "晚到通知", Content: "大家好，我这边路上有点堵，大概会晚到十分钟，会议请先开始，不用等我。"}}}
			a.followUp = "first"
		}
		if recipe.setup != nil {
			recipe.setup()
		}
		id := "shot" + itoa(scenario.benchmarkIndex)
		final := recipe.snapshot
		final.ID = id
		live := final
		live.State = session.Recording
		live.Problem, live.Result, live.Inserted, live.Card = nil, "", false, false
		snap(a, live)
		snap(a, final)
		if recipe.after != nil {
			recipe.after(id)
			a.overlaySync()
		}
		if recipe.name == "overlay-translate-card-menu" {
			a.toggleLangMenu()
		}
	case sessionFixtureCardPrefix:
		steps := cardSteps()
		for prior := 0; prior <= scenario.cardPrefix; prior++ {
			steps[prior].apply(a)
			a.overlaySync()
		}
	case sessionFixtureCustom:
		scenario.apply(a)
	}
	if scenario.after != nil {
		scenario.after(a)
	}
	// Pin the fixture's line layout after session callbacks that may save settings.
	a.prefs.Overlay.LiveLines = scenario.liveLines
	a.ov.projection.pillReady = false
	a.refreshSessionOverlay(meetingFixtureNow)
	state := cloneSessionOverlayViewState(a.ov.projection.view)
	seed := buildSessionFixtureUI(a, state, locale)
	return a, &state, &seed
}

func buildSessionFixtureUI(a *app, state sessionOverlayViewState, locale string) sessionOverlayFixtureUI {
	overlay := a.prefs.Overlay
	seed := sessionOverlayFixtureUI{
		Overlay: sessionOverlayLayoutStyle{
			Position: overlay.Position, Opacity: overlay.Opacity, Radius: overlay.Radius, Margin: overlay.Margin,
			ShowTime: overlay.ShowTime, LiveLines: overlay.LiveLines,
		},
	}
	if state.Menu != nil {
		seed.Menu = state.Menu.Kind
	}
	for turn, open := range a.answer.ui.quotesOpen {
		if open {
			seed.OpenQuotes = append(seed.OpenQuotes, turn)
		}
	}
	sort.Ints(seed.OpenQuotes)
	if state.Card != nil && state.Card.CanLanguage {
		seed.LanguageMenu = sessionFixtureMenu(a, sessionMenuLanguage, locale)
	}
	if state.Card != nil && state.Card.CanThink {
		seed.CardThinkingMenu = sessionFixtureMenu(a, sessionMenuCardThinking, locale)
	}
	if state.Pill.CanThink {
		seed.PillThinkingMenu = sessionFixtureMenu(a, sessionMenuPillThinking, locale)
	}
	return seed
}

func sessionFixtureMenu(a *app, kind, locale string) *sessionMenuViewState {
	local := a.answer.ui
	defer func() { a.answer.ui = local }()
	a.answer.ui.langOpen = true
	a.answer.ui.thinkMenu = kind == sessionMenuCardThinking
	a.answer.ui.pillMenu = kind == sessionMenuPillThinking
	input, provider := a.sessionMenuProjectionInput()
	if kind != sessionMenuLanguage {
		input.ThinkingLevels = thinkLevels(provider)
	}
	menu := buildSessionMenuViewState(i18n.NewTranslator(locale), input)
	return &menu
}

func TestSessionFixtureCoverage(t *testing.T) {
	scenarios := sessionFixtureScenarios()
	expectedNames := map[string]bool{}
	benchmarkFixtures, cardFixtures := 0, 0
	for _, scenario := range scenarios {
		if expectedNames[scenario.name] {
			t.Errorf("duplicate session scenario %q", scenario.name)
		}
		expectedNames[scenario.name] = true
		if scenario.variant {
			continue
		}
		switch scenario.source {
		case sessionFixtureBenchmark:
			benchmarkFixtures++
		case sessionFixtureCardPrefix:
			cardFixtures++
		}
	}
	if benchmarkFixtures != len(sessionOverlayBenchmarkStates(&app{}, meetingFixtureNow)) {
		t.Errorf("covered %d benchmark fixtures, want all %d", benchmarkFixtures, len(sessionOverlayBenchmarkStates(&app{}, meetingFixtureNow)))
	}
	if cardFixtures != len(cardSteps()) {
		t.Errorf("covered %d card prefixes, want all %d", cardFixtures, len(cardSteps()))
	}
	for _, name := range []string{"session-empty-idle", "session-loading-reading", "session-normal-recording", "session-error-failed", "session-markdown-conversation"} {
		if !expectedNames[name] {
			t.Errorf("missing required scenario %q", name)
		}
	}

	for _, locale := range i18n.Supported() {
		allFixtures := readMeetingFixtures(t, locale)
		fixtures := map[string]meetingFixtureState{}
		states := map[string]bool{}
		problems := map[string]bool{}
		for name, fixture := range allFixtures {
			if fixture.Kind != "session-overlay" {
				continue
			}
			if expectedNames[fixture.Name] && fixtures[fixture.Name].Session != nil {
				t.Errorf("%s: duplicate session fixture %q in %s.json", locale, fixture.Name, name)
			}
			fixtures[name] = fixture
			if fixture.Session == nil || fixture.SessionUI == nil {
				t.Errorf("%s/%s: session payload or UI seed is missing", locale, name)
				continue
			}
			state, seed := fixture.Session, fixture.SessionUI
			if state.Card != nil && state.Card.CanLanguage && seed.LanguageMenu == nil {
				t.Errorf("%s/%s: language control has no menu seed", locale, name)
			}
			if state.Card != nil && state.Card.CanThink && seed.CardThinkingMenu == nil {
				t.Errorf("%s/%s: card thinking control has no menu seed", locale, name)
			}
			if state.Pill.CanThink && seed.PillThinkingMenu == nil {
				t.Errorf("%s/%s: pill thinking control has no menu seed", locale, name)
			}
			states[fixture.Session.State] = true
			if fixture.Session.ProblemCode != "" {
				problems[fixture.Session.ProblemCode] = true
			}
			if fixture.Session.Card != nil && fixture.Session.Card.ProblemCode != "" {
				problems[fixture.Session.Card.ProblemCode] = true
			}
		}
		if len(fixtures) != len(scenarios) {
			t.Errorf("%s has %d session fixtures, want %d", locale, len(fixtures), len(scenarios))
		}
		for name := range expectedNames {
			if _, ok := fixtures[name]; !ok {
				t.Errorf("%s: missing session fixture %s.json", locale, name)
			}
		}
		for name, fixture := range fixtures {
			if !expectedNames[name] {
				t.Errorf("%s: orphan session fixture %s.json", locale, name)
			}
			if fixture.Name != name {
				t.Errorf("%s/%s: embedded name is %q", locale, name, fixture.Name)
			}
		}
		assertMeetingFixtureSet(t, locale+" session states", meetingTypedConstants(t, "internal/session/session.go", "State"), states)
		assertMeetingFixtureSet(t, locale+" session problem codes", meetingNamedConstants(t, "internal/session/session.go", "Problem"), problems)
	}
}
