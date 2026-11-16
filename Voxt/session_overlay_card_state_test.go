package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestSessionCardBuilderLocaleRedoAndActions(t *testing.T) {
	low := &llm.Thinking{Mode: llm.ThinkEffort, Effort: "low"}
	answer := answerCard{open: true, id: "translate-1", kind: store.KindTranslate, text: "  See you Thursday.  ",
		target: proto.App{Name: "Arc"}, lang: "en", think: low, redo: "translate-2", redoLang: "ja", redoThink: low}
	input := sessionCardProjectionInput{
		Snapshot: session.Snapshot{ID: "translate-2", Kind: store.KindTranslate, State: session.Composing},
		Answer:   answer, Own: true, ShownDraft: "途中", CanFollowUp: false,
		ThinkingProvider: settings.Provider{ID: "deepseek"}, Generation: settings.Generation{},
	}
	english := buildSessionCardViewState(i18n.NewTranslator("en"), input)
	if !english.Busy || english.Language != "ja" || english.LanguageLabel != "Japanese" || english.Thinking != sessionThinkingValueOf(low) {
		t.Fatalf("redo projection: %+v", english)
	}
	if english.CanInsert || english.CanLanguage || english.CanThink || !english.CanCopy {
		t.Errorf("redo actions: %+v", english)
	}
	if len(english.Blocks) != 1 || english.Blocks[0].Text != "See you Thursday." || english.LiveCount != 0 {
		t.Errorf("redo must keep the committed answer: %+v", english)
	}
	chinese := buildSessionCardViewState(i18n.NewTranslator("zh-Hans"), input)
	if chinese.Title != chineseLanguageTitle(t, "Translation") || chinese.LanguageLabel != "日语" {
		t.Errorf("Chinese card labels: %+v", chinese)
	}

	input.Snapshot = session.Snapshot{ID: "translate-1", Kind: store.KindTranslate, State: session.Done}
	input.Own, input.CanFollowUp = false, true
	english = buildSessionCardViewState(i18n.NewTranslator("en"), input)
	if !english.CanInsert || !english.CanLanguage || !english.CanFollowUp || !english.CanThink || english.Busy {
		t.Errorf("settled actions: %+v", english)
	}
	if english.InsertTooltip != "Insert into Arc" || !strings.Contains(english.FollowTooltip, "Follow up by voice") {
		t.Errorf("action labels: %+v", english)
	}
}

func chineseLanguageTitle(t *testing.T, title string) string {
	t.Helper()
	return i18n.NewTranslator("zh-Hans").T(title)
}

func TestSessionCardBuilderFollowUpQuotesAndLiveTail(t *testing.T) {
	longQuote := strings.Repeat("selected text\n", 40)
	answer := answerCard{open: true, id: "rewrite-1", kind: store.KindRewrite, text: "First answer", conversation: true,
		turns: []compose.Turn{{Instruction: "first", Answer: compose.Answer{Title: "One", Content: " First answer "}},
			{Instruction: "second", Selection: longQuote, Answer: compose.Answer{Content: "Second answer"}}}}
	input := sessionCardProjectionInput{Answer: answer,
		Snapshot: session.Snapshot{ID: "rewrite-2", Kind: store.KindRewrite, State: session.Composing,
			Raw: "third", Selection: "new selection", Title: "Three", Draft: "live answer"},
		Own: true, ShownDraft: "live answer"}
	cut := buildSessionCardViewState(i18n.NewTranslator("en"), input)
	if len(cut.Blocks) != 4 || cut.LiveCount != 2 || !cut.Conversation || !cut.Busy {
		t.Fatalf("fixed/live blocks: %+v", cut)
	}
	quoted := cut.Blocks[2]
	if !quoted.Cut || quoted.Open || !strings.HasSuffix(quoted.Quote, "…") || quoted.FullQuote != strings.TrimSpace(longQuote) {
		t.Errorf("cut quote does not retain full text: %+v", quoted)
	}
	if cut.Blocks[1].Text != "First answer" || cut.Live[0].Text != "third" || cut.Live[1].Text != "live answer" || !cut.Live[1].Live {
		t.Errorf("committed or live text changed: %+v", cut)
	}
	answer.ui.quotesOpen = map[int]bool{1: true}
	input.Answer = answer
	open := buildSessionCardViewState(i18n.NewTranslator("en"), input)
	if !open.Blocks[2].Open || !open.Blocks[2].Cut || open.Blocks[2].Quote != strings.TrimSpace(longQuote) {
		t.Errorf("expanded quote: %+v", open.Blocks[2])
	}

	input.Snapshot.State = session.Recording
	recording := buildSessionCardViewState(i18n.NewTranslator("en"), input)
	if !recording.Recording || !recording.CanStop || recording.LiveCount != 1 || recording.Live[0].Text != "third" {
		t.Errorf("recording tail or actions: %+v", recording)
	}
}

func TestSessionCardBuilderFailureSourcesAndProblem(t *testing.T) {
	answer := answerCard{open: true, id: "rewrite-failed", kind: store.KindRewrite,
		problem: &session.Problem{Code: session.ProblemLLM, Err: errors.New("server unavailable")},
		input:   session.TextInput{Raw: "make it formal", Selection: "original text"}}
	state := buildSessionCardViewState(i18n.NewTranslator("en"), sessionCardProjectionInput{
		Snapshot: session.Snapshot{ID: "rewrite-failed", State: session.Failed}, Answer: answer,
	})
	if !state.Failed || state.ProblemCode != session.ProblemLLM || !state.CanRetry || state.CanInsert || state.CanLanguage {
		t.Fatalf("failure metadata or actions: %+v", state)
	}
	if len(state.Blocks) != 4 || state.Blocks[0].Kind != sessionCardProblem || state.Blocks[1].Kind != sessionCardProblemMore ||
		state.Blocks[2].Text != "make it formal" || state.Blocks[3].Text != "original text" {
		t.Errorf("failure and source blocks: %+v", state.Blocks)
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{`"Answer"`, `"Snapshot"`, `"Err"`, `"Generation"`, `"ThinkingProvider"`, `"input"`, `"turns"`} {
		if strings.Contains(string(encoded), forbidden) {
			t.Errorf("domain field %s crossed into card JSON: %s", forbidden, encoded)
		}
	}
}

func TestSessionMenuBuilderLocaleAndThinkingValues(t *testing.T) {
	language := buildSessionMenuViewState(i18n.NewTranslator("zh-Hans"), sessionMenuProjectionInput{Kind: sessionMenuLanguage, CurrentLanguage: "ja"})
	if language.Label != chineseLanguageTitle(t, "Translate into") || language.SelectedIndex < 0 || !language.Options[language.SelectedIndex].Selected || language.Options[language.SelectedIndex].Language != "ja" {
		t.Fatalf("language menu: %+v", language)
	}
	low := &llm.Thinking{Mode: llm.ThinkEffort, Effort: "low"}
	thinking := buildSessionMenuViewState(i18n.NewTranslator("en"), sessionMenuProjectionInput{
		Kind: sessionMenuCardThinking, CurrentThinking: low, ThinkingLevels: []*llm.Thinking{nil, low},
		Generation: settings.Generation{Thinking: llm.ThinkEffort, ThinkingEffort: "high"},
	})
	if len(thinking.Options) != 2 || thinking.Options[0].Label != "High (default)" || thinking.Options[0].Thinking != (sessionThinkingValue{}) ||
		thinking.Options[1].Label != "Low" || !thinking.Options[1].Selected || thinking.Options[1].Thinking != sessionThinkingValueOf(low) {
		t.Errorf("thinking menu: %+v", thinking)
	}
	if thinking.Options[0].Thinking.thinking() != nil || *thinking.Options[1].Thinking.thinking() != *low {
		t.Errorf("JSON-safe values did not convert inside the action boundary: %+v", thinking.Options)
	}
}

func TestSessionOverlayProjectionReusesCommittedBlocksAndMenuOptions(t *testing.T) {
	a := &app{prefs: settings.Defaults()}
	a.answer = answerCard{open: true, id: "rewrite-1", kind: store.KindRewrite, conversation: true, lang: "en",
		turns: []compose.Turn{{Instruction: "first", Answer: compose.Answer{Content: "committed"}}}}
	a.followUp = a.answer.id
	a.sess = session.Snapshot{ID: "rewrite-2", Kind: store.KindRewrite, State: session.Composing, Raw: "next", Draft: "first live"}
	a.answer.ui.langOpen = true
	projection := &a.ov.projection
	projection.translator = i18n.NewTranslator("en")
	var next sessionOverlayViewState
	a.refreshSessionCardProjection(projection, &next)
	a.refreshSessionMenuProjection(projection, &next, "en")
	committed := &projection.card.Blocks[0]
	options := &projection.menu.Options[0]

	a.sess.Draft = "second live"
	a.sess.Levels = []float64{0.25, 0.75}
	next = sessionOverlayViewState{}
	a.refreshSessionCardProjection(projection, &next)
	a.refreshSessionMenuProjection(projection, &next, "en")
	if committed != &projection.card.Blocks[0] || options != &projection.menu.Options[0] {
		t.Fatal("live levels or draft rebuilt committed storage")
	}
	if projection.card.Blocks[1].Text != "committed" || projection.card.Live[1].Text != "second live" || projection.card.Levels[session.Bars-1] != 0.75 {
		t.Errorf("cached projection did not update its live fields: %+v", projection.card)
	}
}

func TestSessionOverlayProjectionUpdatesWritingTitle(t *testing.T) {
	a := &app{prefs: settings.Defaults()}
	a.sess = session.Snapshot{ID: "rewrite-stream", Kind: store.KindRewrite, State: session.Composing, Card: true, Draft: "answer"}
	projection := &a.ov.projection
	projection.translator = i18n.NewTranslator("en")
	var next sessionOverlayViewState
	a.refreshSessionCardProjection(projection, &next)
	if projection.card.Title != "AI answer" {
		t.Fatalf("initial writing title: %q", projection.card.Title)
	}
	a.sess.Title = "Streamed title"
	a.sess.Target.Name = "Arc"
	a.refreshSessionCardProjection(projection, &next)
	if projection.card.Title != "Streamed title" || projection.card.TargetName != "Arc" {
		t.Fatalf("writing card kept stale metadata: %+v", projection.card)
	}
}

func TestSessionCardBuilderUsesTranslatorForModelErrors(t *testing.T) {
	i18n.SetLocale("en")
	t.Cleanup(func() { i18n.SetLocale("en") })
	translator := i18n.NewTranslator("zh-Hans")
	answer := answerCard{open: true, id: "failed", kind: store.KindRewrite,
		problem: &session.Problem{Code: session.ProblemLLM, Err: local.ErrNotInstalled}}
	input := sessionCardProjectionInput{Snapshot: session.Snapshot{ID: "failed", State: session.Failed}, Answer: answer}
	first := buildSessionCardViewState(translator, input)
	i18n.SetLocale("zh-Hans")
	second := buildSessionCardViewState(translator, input)
	want := translator.T("The language model failed: %s", translator.T("The model is not installed."))
	if first.Blocks[0].Text != want || second.Blocks[0].Text != want {
		t.Fatalf("failure text used the global locale: %q, %q; want %q", first.Blocks[0].Text, second.Blocks[0].Text, want)
	}
}

func TestSessionOverlayCopyBubbleUsesDisplayedBytes(t *testing.T) {
	var copied string
	a := &app{copyText: func(text string) error { copied = text; return nil }}
	a.copyBubble(sessionCardBlock{Title: "Answer", Text: "…"})
	if copied != "Answer\n\n…" || a.ov.ui.bubbleCopied != "…" {
		t.Fatalf("empty displayed block copy: %q, mark %q", copied, a.ov.ui.bubbleCopied)
	}
	a.copyBubble(sessionCardBlock{Text: "trimmed display text"})
	if copied != "trimmed display text" {
		t.Errorf("displayed block copy: %q", copied)
	}
}
