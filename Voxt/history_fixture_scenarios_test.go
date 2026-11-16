package main

import (
	"errors"
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/historyapi"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func historyFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	addHistory := func(name string, mutate func(*historyViewStateInput)) {
		input := historyFixturePageInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "history", history: input})
	}
	addHistory("history-page-loading", func(input *historyViewStateInput) {
		input.Read, input.Loading, input.Entries = false, true, nil
	})
	for _, kind := range append([]store.Kind{""}, store.Kinds...) {
		name := "all"
		if kind != "" {
			name = string(kind)
		}
		addHistory("history-page-empty-"+name, func(input *historyViewStateInput) {
			input.Kind, input.Entries = kind, nil
		})
	}
	addHistory("history-page-normal", func(input *historyViewStateInput) {
		input.Kind = store.KindTranscribe
		input.Entries = historyFixtureEntriesOfKind(store.KindTranscribe)
	})
	addHistory("history-page-all", func(input *historyViewStateInput) { input.Kind = "" })
	addHistory("history-page-error", func(input *historyViewStateInput) { input.Error = "fixture read failed" })
	addHistory("history-page-error-initial", func(input *historyViewStateInput) {
		input.Error, input.Entries = "fixture read failed", nil
	})
	addHistory("history-page-search-match", func(input *historyViewStateInput) {
		entry := historyFixtureBaseEntry()
		input.Kind, input.Query, input.Entries = store.KindTranscribe, "agenda", []store.Entry{entry}
	})
	addHistory("history-page-search-empty", func(input *historyViewStateInput) {
		input.Query, input.Entries = "nothing like this", nil
	})
	addHistory("history-page-search-empty-long", func(input *historyViewStateInput) {
		input.Query, input.Entries = "a query whose visible form is longer than forty runes", nil
	})
	addHistory("history-page-off", func(input *historyViewStateInput) { input.KeepHistory = false })
	addHistory("history-page-audio-missing", func(input *historyViewStateInput) {
		input.AudioMissing = "/fixture/home/External Recordings"
	})
	for _, stage := range []session.State{session.Recognizing, session.Composing} {
		addHistory("history-page-retry-"+string(stage), func(input *historyViewStateInput) {
			retry := historyFixtureRetryEntry()
			input.Kind, input.Entries = store.KindTranscribe, []store.Entry{retry}
			input.Retry = entryRetry{id: retry.ID, stage: stage}
		})
	}
	addHistory("history-page-capped", func(input *historyViewStateInput) {
		input.Kind, input.Entries = store.KindTranscribe, historyFixtureManyEntries(historyLimit)
	})

	addDetail := func(name string, entry store.Entry) {
		input := historyFixturePageInput()
		input.Kind, input.Entries, input.SelectedID = entry.Kind, []store.Entry{entry}, entry.ID
		input.Detail = &historyDetailStateInput{Entry: entry, ASRModel: "Fixture speech model", LLMModel: "Fixture language model"}
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "history", history: input})
	}
	base := historyFixtureBaseEntry()
	addDetail("history-detail-normal", base)
	translateSpeech := historyFixtureEntries()[1]
	translateSpeech.SourceText = ""
	addDetail("history-detail-translate-speech", translateSpeech)
	addDetail("history-detail-translate-selection", historyFixtureEntries()[1])
	addDetail("history-detail-rewrite-selection", historyFixtureEntries()[2])
	rewriteScratch := historyFixtureEntries()[2]
	rewriteScratch.SourceText = ""
	addDetail("history-detail-rewrite-scratch", rewriteScratch)
	addDetail("history-detail-failed", historyFixtureEntries()[3])
	failedRaw := historyFixtureEntries()[3]
	failedRaw.RawText = "recognised before the failure"
	addDetail("history-detail-failed-raw", failedRaw)
	addDetail("history-detail-retry", historyFixtureRetryEntry())
	cancelledRetry := historyFixtureRetryEntry()
	cancelledRetry.Status = store.StatusCancelled
	addDetail("history-detail-cancelled-retry", cancelledRetry)
	addDetail("history-detail-transcribe-again", base)
	doneProblem := base
	doneProblem.Error = "the answer may be incomplete"
	doneProblem.Delivery = "card"
	addDetail("history-detail-done-with-problem", doneProblem)
	noText := historyFixtureRetryEntry()
	noText.Status, noText.Error = store.StatusDone, ""
	noText.RawText, noText.SourceText, noText.PromptText = "", "", ""
	addDetail("history-detail-no-text-retry", noText)
	elsewhere := base
	elsewhere.InsertedName, elsewhere.InsertedBundle = "TextEdit", "com.example.textedit"
	addDetail("history-detail-inserted-elsewhere", elsewhere)
	for _, source := range []string{"url", "browserApp", "app", ""} {
		entry := base
		entry.RuleSource = source
		name := source
		if name == "" {
			name = "none"
		}
		addDetail("history-detail-rule-"+name, entry)
	}
	for _, variant := range []struct {
		name string
		dsp  string
		gain float64
	}{
		{"none", "", 0},
		{"highpass", "highpass", 0},
		{"level", "level", 3},
		{"level-unchanged", "level", 0},
		{"both-unchanged", "highpass+level", 0},
	} {
		entry := base
		entry.DSP, entry.GainDB = variant.dsp, variant.gain
		addDetail("history-detail-sound-"+variant.name, entry)
	}
	for _, code := range historyFixtureProblemCodes() {
		scenarios = append(scenarios, meetingFixtureScenario{name: "history-problem-" + code, kind: "history", historyProblem: code})
	}

	addSettings := func(name string, mutate func(*historySettingsStateInput)) {
		input := historyFixtureSettingsInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "history-settings", historySettings: input})
	}
	addSettings("history-settings-normal", func(input *historySettingsStateInput) {})
	addSettings("history-settings-loading", func(input *historySettingsStateInput) { input.UsageFor = "loading" })
	addSettings("history-settings-error", func(input *historySettingsStateInput) { input.UsageN = -1 })
	addSettings("history-settings-missing", func(input *historySettingsStateInput) {
		input.AudioMissing = input.AudioDir
	})
	addSettings("history-settings-move", func(input *historySettingsStateInput) {
		input.Busy, input.Done, input.Total, input.CanCancel = "move", 1, 3, true
	})
	addSettings("history-settings-export", func(input *historySettingsStateInput) {
		input.Busy, input.Done, input.Total, input.CanCancel = "export", 2, 3, true
	})
	addSettings("history-settings-prune-ask", func(input *historySettingsStateInput) {
		input.PruneAsk, input.PruneN = true, 12
	})
	addSettings("history-settings-delete-audio-ask", func(input *historySettingsStateInput) {
		input.DeleteAudioAsk = true
	})
	addSettings("history-settings-filtered-jsonl", func(input *historySettingsStateInput) {
		input.Kind, input.Query, input.TextFormat = store.KindTranslate, "agenda", "JSONL"
	})

	addAccess := func(name string, mutate func(*historyAccessStateInput)) {
		input := historyAccessFixtureInput()
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "history-access", historyAccess: input})
	}
	addAccess("history-access-off", func(input *historyAccessStateInput) {
		input.On, input.Selection, input.Reads = false, false, nil
	})
	addAccess("history-access-on", func(input *historyAccessStateInput) {})
	addAccess("history-access-ask", func(input *historyAccessStateInput) {
		input.On, input.Ask = false, true
	})
	addAccess("history-access-log", func(input *historyAccessStateInput) { input.ListOpen = true })
	return scenarios
}

func historyFixturePageInput() historyViewStateInput {
	return historyViewStateInput{
		Entries: historyFixtureEntries(), Kind: store.KindTranscribe, Read: true, KeepHistory: true,
		SettingsAvailable: true, DefaultAudioDir: "/fixture/home/Default Recordings", HomeDir: "/fixture/home", Now: meetingFixtureNow,
	}
}

func historyFixtureBaseEntry() store.Entry {
	return store.Entry{
		ID: "history-inserted", Kind: store.KindTranscribe, Status: store.StatusDone,
		CreatedAt: meetingFixtureNow.Add(-5 * time.Minute), Result: "Weekly plan\nI'll send the new agenda tonight.",
		RawText: "weekly plan\nI'll send the new agenda tonight", AppName: "Notes", Delivery: "insert",
		ASRMS: 850, LLMMS: 1200, TotalMS: 2400, AudioMS: 7500, SentMS: 2500,
		Audio: "history-inserted.wav", DSP: "highpass+level", GainDB: 1.5,
		RuleName: "Fixture rule", RuleSource: "app",
	}
}

func historyFixtureRetryEntry() store.Entry {
	entry := historyFixtureBaseEntry()
	entry.ID, entry.Status, entry.Error = "history-retry", store.StatusFailed, "recognition failed"
	entry.Result = ""
	entry.AudioUntil = meetingFixtureNow.Add(24 * time.Hour)
	return entry
}

func historyFixtureEntries() []store.Entry {
	inserted := historyFixtureBaseEntry()
	translated := inserted
	translated.ID, translated.Kind, translated.Delivery = "history-translated", store.KindTranslate, "card"
	translated.Result, translated.RawText, translated.SourceText = "晚上好", "good evening", "selected source"
	translated.TargetLang, translated.RuleSource, translated.DSP, translated.GainDB = "zh-Hans", "url", "highpass", 0
	rewritten := inserted
	rewritten.ID, rewritten.Kind, rewritten.Delivery = "history-rewritten", store.KindRewrite, "copy"
	rewritten.Result, rewritten.RawText = "A formal answer", "make this formal"
	rewritten.SourceText, rewritten.PromptText = "an informal source", "Rewrite formally"
	rewritten.RuleSource, rewritten.DSP, rewritten.GainDB = "browserApp", "level", 2
	failed := inserted
	failed.ID, failed.Status, failed.Error = "history-failed", store.StatusFailed, "speech service failed"
	failed.Result, failed.RawText, failed.Audio = "", "", ""
	failed.Delivery, failed.RuleName, failed.RuleSource, failed.DSP, failed.GainDB = "", "", "", "", 0
	cancelled := inserted
	cancelled.ID, cancelled.Status = "history-cancelled", store.StatusCancelled
	cancelled.Result, cancelled.RawText, cancelled.Delivery = "", "", ""
	cancelled.AudioUntil = meetingFixtureNow.Add(24 * time.Hour)
	degraded := translated
	degraded.ID, degraded.Error = "history-degraded", "the answer may be incomplete"
	return []store.Entry{inserted, translated, rewritten, failed, cancelled, degraded}
}

func historyFixtureEntriesOfKind(kind store.Kind) []store.Entry {
	var entries []store.Entry
	for _, entry := range historyFixtureEntries() {
		if entry.Kind == kind {
			entries = append(entries, entry)
		}
	}
	return entries
}

func historyFixtureManyEntries(count int) []store.Entry {
	entries := make([]store.Entry, count)
	for index := range entries {
		entry := historyFixtureBaseEntry()
		entry.ID = fmt.Sprintf("history-%04d", index)
		entry.Result = fmt.Sprintf("History result %04d", index)
		entries[index] = entry
	}
	return entries
}

func historyFixtureProblemCodes() []string {
	return []string{
		session.ProblemNoASR, session.ProblemSpeechWait, session.ProblemMicrophone, session.ProblemDevice,
		session.ProblemHelper, session.ProblemASR, session.ProblemLLM, session.ProblemNoSpeech,
		session.ProblemTooShort, session.ProblemNoLLM, session.ProblemEmpty, session.ProblemTruncated,
		session.ProblemUnsupported, session.ProblemInternal,
	}
}

func historyFixtureProblemInput(code string) historyViewStateInput {
	entry := historyFixtureBaseEntry()
	entry.ID = "history-problem-" + code
	entry.Status, entry.Result, entry.Audio = store.StatusFailed, "", ""
	problem := &session.Problem{Code: code, Err: errors.New("fixture problem"), Degraded: code == session.ProblemASR || code == session.ProblemLLM}
	entry.Error, _, _ = sessionProblemText(problem)
	if code == session.ProblemNoSpeech || code == session.ProblemTruncated || problem.Degraded {
		entry.Status, entry.Delivery = store.StatusDone, "card"
	}
	if code == session.ProblemTooShort {
		entry.Status = store.StatusCancelled
	}
	if code == session.ProblemNoSpeech || code == session.ProblemTooShort {
		entry.RawText, entry.SourceText, entry.PromptText = "", "", ""
	}
	input := historyFixturePageInput()
	input.Kind, input.Entries, input.SelectedID = entry.Kind, []store.Entry{entry}, entry.ID
	input.Detail = &historyDetailStateInput{Entry: entry}
	return input
}

func historyFixtureSettingsInput() historySettingsStateInput {
	return historySettingsStateInput{
		Open: true, AudioDir: "/fixture/home/Custom Recordings", DefaultAudioDir: "/fixture/home/Default Recordings",
		HomeDir: "/fixture/home", UsageN: 2, UsageBytes: 2048, UsageFor: "done", Kind: store.KindTranscribe, CleanOn: true,
	}
}

func historyAccessFixtureInput() historyAccessStateInput {
	return historyAccessStateInput{
		On: true, Selection: true, Exe: "/fixture/Voxt", Now: meetingFixtureNow,
		Reads: []historyapi.Read{
			{At: meetingFixtureNow.Add(-time.Minute), Op: historyapi.OpExport, Count: 1, As: "daily-editor", Program: "fixture-shell"},
			{At: meetingFixtureNow.Add(-2 * time.Minute), Op: historyapi.OpStatus},
		},
	}
}
