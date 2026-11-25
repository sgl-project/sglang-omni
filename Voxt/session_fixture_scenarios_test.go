package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type sessionFixtureSource int

const (
	sessionFixtureBenchmark sessionFixtureSource = iota
	sessionFixtureCardPrefix
	sessionFixtureCustom
)

type sessionFixtureScenario struct {
	name           string
	source         sessionFixtureSource
	benchmarkIndex int
	cardPrefix     int
	apply          func(*app)
	after          func(*app)
	variant        bool
}

func appendSessionFixtureScenarios(scenarios []meetingFixtureScenario) []meetingFixtureScenario {
	for _, sessionScenario := range sessionFixtureScenarios() {
		sessionScenario := sessionScenario
		scenarios = append(scenarios, meetingFixtureScenario{
			name: sessionScenario.name, kind: "session-overlay", session: &sessionScenario,
		})
	}
	return scenarios
}

func sessionFixtureScenarios() []sessionFixtureScenario {
	var scenarios []sessionFixtureScenario
	for index, recipe := range sessionOverlayBenchmarkStates(&app{}, meetingFixtureNow) {
		scenarios = append(scenarios, sessionFixtureScenario{name: recipe.name, source: sessionFixtureBenchmark, benchmarkIndex: index})
	}
	for index, step := range cardSteps() {
		scenarios = append(scenarios, sessionFixtureScenario{name: "card-" + step.name, source: sessionFixtureCardPrefix, cardPrefix: index})
	}

	scenarios = append(scenarios,
		sessionFixtureScenario{name: "session-empty-idle", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-empty", State: session.Idle})
		}},
		sessionFixtureScenario{name: "session-loading-reading", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-loading", Kind: store.KindTranslate, State: session.Reading})
		}},
		sessionFixtureScenario{name: "session-normal-recording", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-normal", State: session.Recording, StartedAt: meetingFixtureNow.Add(-time.Minute), HeardAt: meetingFixtureNow.Add(-time.Minute), Partial: "A normal recording", Levels: wave()})
		}},
		sessionProblemFixture("session-error-failed", session.ProblemInternal, nil),
		sessionFixtureScenario{name: "session-state-delivering", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-delivering", State: session.Recording})
			snap(a, session.Snapshot{ID: "session-delivering", State: session.Delivering, Result: "Delivery in progress"})
		}},
		sessionFixtureScenario{name: "session-state-cancelled", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-cancelled", State: session.Recording})
			snap(a, session.Snapshot{ID: "session-cancelled", State: session.Cancelled})
		}},
	)

	scenarios = append(scenarios,
		sessionProblemFixture("session-problem-no-asr", session.ProblemNoASR, nil),
		sessionFixtureScenario{name: "session-problem-speech-wait", source: sessionFixtureCustom, apply: func(a *app) {
			downloading(a, a.obSpeech())
			problem := &session.Problem{Code: session.ProblemSpeechWait, Err: errors.New(a.speechWaits(store.KindTranslate).text)}
			snap(a, session.Snapshot{ID: "session-problem", State: session.Recording, StartedAt: meetingFixtureNow})
			snap(a, session.Snapshot{ID: "session-problem", State: session.Failed, Problem: problem})
		}},
		sessionProblemFixture("session-problem-microphone", session.ProblemMicrophone, nil),
		sessionProblemFixture("session-problem-device", session.ProblemDevice, nil),
		sessionProblemFixture("session-problem-helper", session.ProblemHelper, nil),
		sessionProblemFixture("session-problem-asr", session.ProblemASR, func(snapshot *session.Snapshot) {
			snapshot.Problem.Err = &asr.Error{Kind: asr.KindServer}
		}),
		sessionProblemFixture("session-problem-asr-retry", session.ProblemASR, func(snapshot *session.Snapshot) {
			snapshot.Recording = true
			snapshot.Problem.Err = &asr.Error{Kind: asr.KindServer}
		}),
		sessionProblemFixture("session-problem-asr-degraded", session.ProblemASR, func(snapshot *session.Snapshot) {
			snapshot.State, snapshot.Result, snapshot.Inserted = session.Done, "Live text", true
			snapshot.Problem.Degraded = true
			snapshot.Problem.Err = &asr.Error{Kind: asr.KindTimeout}
		}),
		sessionProblemFixture("session-problem-llm", session.ProblemLLM, func(snapshot *session.Snapshot) {
			snapshot.Kind, snapshot.Card = store.KindRewrite, true
			snapshot.Raw, snapshot.Selection = "make this formal", "the release slips"
		}),
		sessionProblemFixture("session-problem-llm-degraded", session.ProblemLLM, func(snapshot *session.Snapshot) {
			snapshot.State, snapshot.Result, snapshot.Inserted = session.Done, "Recognised text", true
			snapshot.Problem.Degraded = true
		}),
		sessionProblemFixture("session-problem-llm-sign-in", session.ProblemLLM, func(snapshot *session.Snapshot) {
			snapshot.Problem.Err = llm.SignInError(providers.ChatGPT)
		}),
		sessionProblemFixture("session-problem-llm-usage", session.ProblemLLM, func(snapshot *session.Snapshot) {
			snapshot.Problem.Err = &llm.Error{Code: llm.CodeUsageLimit, Params: map[string]string{"provider": providers.ChatGPT}}
		}),
		sessionProblemFixture("session-problem-no-speech", session.ProblemNoSpeech, func(snapshot *session.Snapshot) {
			snapshot.State = session.Done
		}),
		sessionProblemFixture("session-problem-too-short", session.ProblemTooShort, func(snapshot *session.Snapshot) {
			snapshot.State = session.Cancelled
		}),
		sessionProblemFixture("session-problem-no-llm", session.ProblemNoLLM, nil),
		sessionProblemFixture("session-problem-empty", session.ProblemEmpty, func(snapshot *session.Snapshot) {
			snapshot.Kind, snapshot.Card = store.KindTranslate, true
			snapshot.Selection, snapshot.TargetLanguage = "Source text", "zh-Hans"
		}),
		sessionProblemFixture("session-problem-truncated", session.ProblemTruncated, func(snapshot *session.Snapshot) {
			snapshot.State, snapshot.Kind, snapshot.Card = session.Done, store.KindTranslate, true
			snapshot.Result, snapshot.Delivery = "Incomplete answer", deliver.ReasonCard
			snapshot.TargetLanguage = "zh-Hans"
			snapshot.Problem.Degraded = true
		}),
		sessionProblemFixture("session-problem-unsupported", session.ProblemUnsupported, nil),
	)

	scenarios = append(scenarios,
		sessionFixtureScenario{name: "session-notice-over-card", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "held-card", State: session.Recording})
			snap(a, session.Snapshot{ID: "held-card", State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Result: "Held answer"})
			a.setAside(false)
			snap(a, session.Snapshot{ID: "notice-session", State: session.Recording})
			snap(a, session.Snapshot{ID: "notice-session", State: session.Failed, Problem: &session.Problem{Code: session.ProblemDevice, Err: errors.New("no input")}})
		}},
		sessionNoticeChainFixture("session-notice-chain-problem", false),
		sessionNoticeChainFixture("session-notice-chain-copied", true),
		sessionFixtureScenario{name: "session-live-privacy", source: sessionFixtureCustom, apply: func(a *app) {
			a.prefs.Overlay.HideLiveText = true
			snap(a, session.Snapshot{ID: "session-privacy", State: session.Recording, StartedAt: meetingFixtureNow, HeardAt: meetingFixtureNow, Partial: "private live words"})
		}},
		sessionFixtureScenario{name: "session-pill-note", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-pill-note", State: session.Recording, StartedAt: meetingFixtureNow, HeardAt: meetingFixtureNow, Partial: "original live words"})
			snap(a, session.Snapshot{ID: "session-pill-note", State: session.Recording, StartedAt: meetingFixtureNow, HeardAt: meetingFixtureNow, Partial: "original live words", Switched: 1, Device: "Fixture microphone"})
		}},
		sessionFixtureScenario{name: "session-inserting-hidden", source: sessionFixtureCustom, apply: func(a *app) {
			snap(a, session.Snapshot{ID: "session-inserting", State: session.Recording})
			snap(a, session.Snapshot{ID: "session-inserting", State: session.Done, Card: true, Delivery: deliver.ReasonNotEditable, Result: "Insert this answer"})
			a.answer.inserting = true
			a.overlaySync()
		}},
		// A conversation whose answers carry Markdown (a bubble renders
		// it), while the user's own markers stay plain text.
		sessionFixtureScenario{name: "session-markdown-conversation", source: sessionFixtureCustom, apply: func(a *app) {
			a.answer = answerCard{open: true, id: "first", kind: store.KindRewrite, conversation: true}
			a.answer.turns = []compose.Turn{{
				Instruction: "keep the **stars** and `ticks` as they are",
				Answer:      compose.Answer{Title: "Release notes", Content: "## Plan\n\n- **Bold** step\n- `code` span"},
			}}
			a.followUp = "first"
			snap(a, session.Snapshot{ID: "second", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "second", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard, Raw: "shorter", Title: "Release notes", Result: "1. **Ship it** — see `main.go`."})
		}},
	)

	pillMenu := sessionBenchmarkFixture("overlay-composing-think")
	pillMenu.name = "session-menu-pill-thinking"
	pillMenu.after = func(a *app) { a.togglePillThinkMenu() }
	pillMenu.variant = true
	cardMenu := sessionCardPrefixFixture("thinking")
	cardMenu.name = "session-menu-card-thinking"
	cardMenu.after = func(a *app) { a.toggleThinkMenu() }
	cardMenu.variant = true
	return append(scenarios, pillMenu, cardMenu)
}

func sessionBenchmarkFixture(name string) sessionFixtureScenario {
	for index, recipe := range sessionOverlayBenchmarkStates(&app{}, meetingFixtureNow) {
		if recipe.name == name {
			return sessionFixtureScenario{name: name, source: sessionFixtureBenchmark, benchmarkIndex: index}
		}
	}
	panic("unknown session benchmark fixture " + name)
}

func sessionCardPrefixFixture(name string) sessionFixtureScenario {
	for index, step := range cardSteps() {
		if step.name == name {
			return sessionFixtureScenario{name: "card-" + name, source: sessionFixtureCardPrefix, cardPrefix: index}
		}
	}
	panic("unknown session card fixture " + name)
}

func sessionProblemFixture(name, code string, configure func(*session.Snapshot)) sessionFixtureScenario {
	return sessionFixtureScenario{name: name, source: sessionFixtureCustom, apply: func(a *app) {
		final := session.Snapshot{
			ID: "session-problem", State: session.Failed,
			Problem: &session.Problem{Code: code, Err: errors.New("fixture problem detail")},
		}
		if configure != nil {
			configure(&final)
		}
		snap(a, session.Snapshot{ID: final.ID, Kind: final.Kind, State: session.Recording, StartedAt: meetingFixtureNow})
		snap(a, final)
	}}
}

func sessionNoticeChainFixture(name string, advance bool) sessionFixtureScenario {
	return sessionFixtureScenario{name: name, source: sessionFixtureCustom, apply: func(a *app) {
		timers := new(fakeTimers).install(a)
		snap(a, session.Snapshot{ID: "session-notice-chain", State: session.Recording})
		snap(a, session.Snapshot{
			ID: "session-notice-chain", State: session.Done, Result: "Recognised text", Delivery: deliver.ReasonNotEditable,
			Problem: &session.Problem{Code: session.ProblemLLM, Degraded: true, Err: errors.New("fixture timeout")},
		})
		if advance {
			timers.advance(noticeTime)
		}
	}}
}
