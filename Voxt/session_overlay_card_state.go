package main

import (
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

const (
	sessionCardBody          = "body"
	sessionCardProblem       = "problem"
	sessionCardProblemMore   = "problem-more"
	sessionCardSource        = "source"
	sessionCardBubble        = "bubble"
	sessionCardLiveUserKey   = "live-user"
	sessionCardLiveAnswerKey = "live-answer"
)

// sessionCardViewState is the JSON-safe display state of one answer card.
type sessionCardViewState struct {
	ID            string                `json:"id"`
	Kind          string                `json:"kind"`
	Title         string                `json:"title"`
	Message       string                `json:"message,omitempty"`
	Why           string                `json:"why,omitempty"`
	TargetName    string                `json:"target_name,omitempty"`
	Language      string                `json:"language,omitempty"`
	LanguageLabel string                `json:"language_label,omitempty"`
	Thinking      sessionThinkingValue  `json:"thinking"`
	ThinkingLabel string                `json:"thinking_label,omitempty"`
	FollowTooltip string                `json:"follow_tooltip,omitempty"`
	InsertTooltip string                `json:"insert_tooltip,omitempty"`
	ProblemCode   string                `json:"problem_code,omitempty"`
	SizeStep      int                   `json:"size_step"`
	Levels        [session.Bars]float64 `json:"levels"`
	Blocks        []sessionCardBlock    `json:"blocks"`
	Live          [2]sessionCardBlock   `json:"live"`
	LiveCount     int                   `json:"live_count"`
	Busy          bool                  `json:"busy"`
	Recording     bool                  `json:"recording"`
	Writing       bool                  `json:"writing"`
	Conversation  bool                  `json:"conversation"`
	Failed        bool                  `json:"failed"`
	Copied        bool                  `json:"copied"`
	Heard         bool                  `json:"heard"`
	CanInsert     bool                  `json:"can_insert"`
	CanCopy       bool                  `json:"can_copy"`
	CanFollowUp   bool                  `json:"can_follow_up"`
	CanStop       bool                  `json:"can_stop"`
	CanStopAnswer bool                  `json:"can_stop_answer"`
	CanRetry      bool                  `json:"can_retry"`
	CanLanguage   bool                  `json:"can_language"`
	ShowThinking  bool                  `json:"show_thinking"`
	CanThink      bool                  `json:"can_think"`
}

// sessionCardBlock is one laid-out body, failure, source, or conversation block.
type sessionCardBlock struct {
	Key        string `json:"key"`
	Kind       string `json:"kind"`
	Turn       int    `json:"turn"`
	Title      string `json:"title,omitempty"`
	Text       string `json:"text"`
	Quote      string `json:"quote,omitempty"`
	FullQuote  string `json:"full_quote,omitempty"`
	Live       bool   `json:"live,omitempty"`
	User       bool   `json:"user,omitempty"`
	Selectable bool   `json:"selectable"`
	Cut        bool   `json:"cut,omitempty"`
	Open       bool   `json:"open,omitempty"`
}

// sessionCardProjectionInput contains the domain inputs used by the pure card builder.
type sessionCardProjectionInput struct {
	Snapshot         session.Snapshot
	Answer           answerCard
	Own              bool
	Writing          bool
	HideLiveText     bool
	ShownDraft       string
	FollowKeys       string
	CanFollowUp      bool
	ThinkingProvider settings.Provider
	Generation       settings.Generation
	SizeStep         int
}

func buildSessionCardViewState(translator *i18n.Translator, input sessionCardProjectionInput) sessionCardViewState {
	snapshot, card := input.Snapshot, input.Answer
	if input.HideLiveText {
		snapshot.Partial, snapshot.Draft = "", ""
	}
	if !card.open {
		state := sessionCardViewState{SizeStep: input.SizeStep}
		if input.Writing {
			state.ID, state.Kind = snapshot.ID, string(snapshot.Kind)
			state.Title = sessionCardTitleFor(translator, snapshot.Kind, snapshot.Title)
			state.TargetName = snapshot.Target.Name
			state.Busy, state.Writing, state.CanStopAnswer = true, true, snapshot.CanStop()
			state.Live[0] = sessionCardWritingBlock(input.ShownDraft)
			state.LiveCount = 1
		}
		return state
	}

	redo := input.Own && snapshot.ID == card.redo
	follow := input.Own && !redo
	lang, think := card.lang, card.think
	if redo {
		lang, think = card.redoLang, card.redoThink
	}
	state := sessionCardViewState{
		ID: card.id, Kind: string(card.kind), Title: sessionCardTitleFor(translator, card.kind, card.title),
		Message: card.msg, Why: card.why, TargetName: card.target.Name,
		Language: lang, LanguageLabel: compose.LanguageName(lang, translator.Locale()),
		Thinking: sessionThinkingValueOf(think), SizeStep: input.SizeStep,
		Busy: input.Own, Conversation: card.conversation || follow, Failed: card.problem != nil,
		Copied: card.copied, CanCopy: card.text != "",
		CanInsert:   card.problem == nil && card.text != "" && !input.Own,
		CanLanguage: card.kind == store.KindTranslate && card.problem == nil && !snapshot.State.Live(),
		CanFollowUp: input.CanFollowUp, CanStop: follow && snapshot.State == session.Recording,
		CanStopAnswer: follow && snapshot.CanStop(), CanRetry: card.problem != nil && !snapshot.State.Live(),
		FollowTooltip: continueTipFor(translator, input.FollowKeys),
	}
	if card.problem != nil {
		state.ProblemCode = card.problem.Code
	}
	thinkingLevels := thinkLevels(input.ThinkingProvider)
	state.ShowThinking = card.problem == nil && !state.Conversation && len(card.turns) <= 1 && len(thinkingLevels) > 1
	state.CanThink = state.ShowThinking && !snapshot.State.Live()
	if state.ShowThinking {
		state.ThinkingLabel = askAgainTextFor(translator, think, input.Generation)
	}
	if !state.Failed {
		target := cmpOr(state.TargetName, translator.T("the app"))
		state.InsertTooltip = translator.T("Insert into %s", target)
		if state.Why != "" {
			state.InsertTooltip = state.Why
		}
	}
	if state.Failed && !state.Conversation {
		state.Blocks = sessionCardFailureBlocks(translator, card)
	} else if !state.Conversation {
		state.Blocks = []sessionCardBlock{{Key: "answer", Kind: sessionCardBody, Text: strings.TrimSpace(card.text), Selectable: strings.TrimSpace(card.text) != ""}}
	} else {
		state.Blocks = sessionCardConversationBlocks(translator, card)
		state.Recording = follow && snapshot.State == session.Recording
		state.Heard = !snapshot.HeardAt.IsZero()
		copySessionLevels(&state.Levels, snapshot.Levels)
	}
	state.Live, state.LiveCount = buildSessionCardLiveTail(translator, input, follow)
	if state.Conversation && card.problem != nil && state.Message == "" {
		state.Message, _, _ = sessionProblemTextFor(translator, card.problem)
	}
	return state
}

func sessionCardTitleFor(translator *i18n.Translator, kind store.Kind, title string) string {
	if title = strings.TrimSpace(title); title != "" {
		return title
	}
	switch kind {
	case store.KindTranslate:
		return translator.T("Translation")
	case store.KindRewrite:
		return translator.T("AI answer")
	}
	return translator.T("Transcription")
}

func sessionCardFailureBlocks(translator *i18n.Translator, card answerCard) []sessionCardBlock {
	text, more, _ := sessionProblemTextFor(translator, card.problem)
	blocks := []sessionCardBlock{{Key: "problem", Kind: sessionCardProblem, Text: text, Selectable: false}}
	if more != "" {
		blocks = append(blocks, sessionCardBlock{Key: "problem-more", Kind: sessionCardProblemMore, Text: more, Selectable: true})
	}
	input := card.input
	if card.kind == store.KindRewrite && input.Raw != "" && input.Selection != "" {
		blocks = append(blocks, sessionCardBlock{Key: "source-0", Kind: sessionCardSource, Text: strings.TrimSpace(input.Raw), Selectable: true})
	}
	if source := cmpOr(input.Selection, input.Raw); source != "" {
		blocks = append(blocks, sessionCardBlock{Key: "source-last", Kind: sessionCardSource, Text: strings.TrimSpace(source), Selectable: true})
	}
	return blocks
}

func sessionCardConversationBlocks(translator *i18n.Translator, card answerCard) []sessionCardBlock {
	blocks := make([]sessionCardBlock, 0, 2*len(card.turns)+1)
	for turn, answer := range card.turns {
		if user, ok := sessionCardUserBlock(translator, card.ui.quotesOpen, turn, answer.Instruction, answer.Selection, false); ok {
			blocks = append(blocks, user)
		}
		blocks = append(blocks, sessionCardAnswerBlock(sessionCardTurnKey(turn, "answer"), answer.Answer.Title, answer.Answer.Content, false))
	}
	if card.problem != nil {
		if user, ok := sessionCardUserBlock(translator, card.ui.quotesOpen, len(card.turns), card.input.Raw, card.input.Selection, false); ok {
			blocks = append(blocks, user)
		}
	}
	return blocks
}

func buildSessionCardLiveTail(translator *i18n.Translator, input sessionCardProjectionInput, follow bool) ([2]sessionCardBlock, int) {
	var live [2]sessionCardBlock
	if !input.Answer.open {
		if input.Writing {
			live[0] = sessionCardWritingBlock(input.ShownDraft)
			return live, 1
		}
		return live, 0
	}
	if !follow {
		return live, 0
	}
	snapshot := input.Snapshot
	shownDraft := input.ShownDraft
	if input.HideLiveText {
		snapshot.Partial, snapshot.Draft, shownDraft = "", "", ""
	}
	count := 0
	said := cmpOr(strings.TrimSpace(snapshot.Raw), strings.TrimSpace(snapshot.Partial))
	if user, ok := sessionCardUserBlock(translator, input.Answer.ui.quotesOpen, len(input.Answer.turns), said, snapshot.Selection, true); ok {
		live[0], count = user, 1
	}
	if snapshot.State != session.Recording {
		live[count] = sessionCardAnswerBlock(sessionCardLiveAnswerKey, snapshot.Title, shownDraft, true)
		count++
	}
	return live, count
}

func sessionCardUserBlock(translator *i18n.Translator, quotesOpen map[int]bool, turn int, said, selection string, live bool) (sessionCardBlock, bool) {
	block := sessionCardBlock{Key: sessionCardTurnKey(turn, "user"), Kind: sessionCardBubble, Turn: turn,
		Title: translator.T("You"), Text: strings.TrimSpace(said), Live: live, User: true, Selectable: !live}
	if turn > 0 {
		block.FullQuote = strings.TrimSpace(selection)
		block.Open = quotesOpen[turn]
		block.Quote, block.Cut = quoteText(selection, block.Open)
	}
	if block.Text == "" {
		block.Text = "…"
	}
	return block, strings.TrimSpace(said) != "" || block.Quote != ""
}

func sessionCardWritingBlock(text string) sessionCardBlock {
	text = strings.TrimSpace(text)
	return sessionCardBlock{Key: sessionCardLiveAnswerKey, Kind: sessionCardBody, Text: text, Live: true, Selectable: text != ""}
}

func sessionCardAnswerBlock(key, title, text string, live bool) sessionCardBlock {
	text = strings.TrimSpace(text)
	if text == "" {
		text = "…"
	}
	return sessionCardBlock{Key: key, Kind: sessionCardBubble, Title: strings.TrimSpace(title), Text: text, Live: live, Selectable: !live}
}

func sessionCardTurnKey(turn int, side string) string {
	return "turn-" + strconv.Itoa(turn) + "-" + side
}

// quoteMax is how much of a follow-up's selection its bubble shows before Show all.
const quoteMax, quoteLines = 160, 3

func quoteText(selection string, open bool) (string, bool) {
	selection = strings.TrimSpace(selection)
	short := selection
	if lines := strings.SplitN(short, "\n", quoteLines+1); len(lines) > quoteLines {
		short = strings.Join(lines[:quoteLines], "\n")
	}
	if runes := []rune(short); len(runes) > quoteMax {
		short = string(runes[:quoteMax])
	}
	if short == selection {
		return selection, false
	}
	if open {
		return selection, true
	}
	return strings.TrimSpace(short) + "…", true
}

func continueTipFor(translator *i18n.Translator, keys string) string {
	if keys == "" {
		return translator.T("Follow up by voice. Text you select first goes with it.")
	}
	return translator.T("Follow up by voice (%s). Text you select first goes with it.", keys)
}

func updateSessionCardLiveState(translator *i18n.Translator, card *sessionCardViewState, input sessionCardProjectionInput) {
	follow := input.Own && input.Snapshot.ID != input.Answer.redo
	card.Recording = follow && input.Snapshot.State == session.Recording
	card.CanStop = follow && input.Snapshot.State == session.Recording
	card.CanStopAnswer = follow && input.Snapshot.CanStop()
	if !input.Answer.open && input.Writing {
		card.Title = sessionCardTitleFor(translator, input.Snapshot.Kind, input.Snapshot.Title)
		card.TargetName = input.Snapshot.Target.Name
		card.CanStopAnswer = input.Snapshot.CanStop()
	}
	card.Live, card.LiveCount = buildSessionCardLiveTail(translator, input, follow)
}

type sessionCardProjectionKey struct {
	sessionID, answerID, kind, state      string
	answerText, title, message, why       string
	target, language, redo, redoLanguage  string
	providerID, providerModel             string
	thinking, redoThinking                sessionThinkingValue
	generationThinking, generationEffort  string
	generationBudget, sizeStep, turnCount int
	followKeys                            string
	own, writing, hideLiveText            bool
	canFollowUp, copied, conversation     bool
	problemCode                           string
}

type sessionCardLiveProjectionKey struct {
	sessionID, state, raw, partial string
	selection, title, shownDraft   string
	target                         string
	own, writing, hideLiveText     bool
	canStop                        bool
}

func sessionCardKeyFor(input sessionCardProjectionInput) sessionCardProjectionKey {
	card, snapshot := input.Answer, input.Snapshot
	problemCode := ""
	if card.problem != nil {
		problemCode = card.problem.Code
	}
	return sessionCardProjectionKey{
		sessionID: snapshot.ID, answerID: card.id, kind: string(card.kind), state: string(snapshot.State),
		answerText: card.text, title: card.title, message: card.msg, why: card.why,
		target: card.target.Name, language: card.lang, redo: card.redo, redoLanguage: card.redoLang,
		providerID: input.ThinkingProvider.ID, providerModel: input.ThinkingProvider.Model,
		thinking: sessionThinkingValueOf(card.think), redoThinking: sessionThinkingValueOf(card.redoThink),
		generationThinking: input.Generation.Thinking, generationEffort: input.Generation.ThinkingEffort,
		generationBudget: input.Generation.ThinkingBudget, sizeStep: input.SizeStep,
		turnCount: len(card.turns), followKeys: input.FollowKeys,
		own: input.Own, writing: input.Writing, hideLiveText: input.HideLiveText,
		canFollowUp: input.CanFollowUp, copied: card.copied, conversation: card.conversation, problemCode: problemCode,
	}
}

func sessionCardLiveKeyFor(input sessionCardProjectionInput) sessionCardLiveProjectionKey {
	snapshot := input.Snapshot
	return sessionCardLiveProjectionKey{sessionID: snapshot.ID, state: string(snapshot.State),
		raw: snapshot.Raw, partial: snapshot.Partial, selection: snapshot.Selection, title: snapshot.Title,
		target: snapshot.Target.Name, shownDraft: input.ShownDraft, own: input.Own, writing: input.Writing,
		hideLiveText: input.HideLiveText, canStop: snapshot.CanStop()}
}

func (a *app) sessionCardProjectionInput() sessionCardProjectionInput {
	snapshot := a.sess
	kind := a.answer.kind
	if !a.answer.open {
		kind = snapshot.Kind
	}
	input := sessionCardProjectionInput{
		Snapshot: snapshot, Answer: a.answer, Own: a.cardSessionLive(), Writing: a.writingAnswer(),
		HideLiveText: a.prefs.Overlay.HideLiveText, ShownDraft: a.shownDraft(snapshot),
		SizeStep: a.prefs.Features.CardSize(string(kind)),
	}
	if !a.answer.open {
		return input
	}
	input.FollowKeys = a.featureKeys(hotkey.Rewrite)
	input.CanFollowUp = a.canFollowUp()
	input.ThinkingProvider, _ = a.llmService(a.answer.llmID)
	input.Generation = a.llmGeneration(a.answer.llmID)
	return input
}

func (a *app) refreshSessionCardProjection(projection *sessionOverlayProjection, next *sessionOverlayViewState) {
	input := a.sessionCardProjectionInput()
	key, liveKey := sessionCardKeyFor(input), sessionCardLiveKeyFor(input)
	if !projection.cardReady || key != projection.cardKey {
		projection.card = buildSessionCardViewState(projection.translator, input)
		projection.cardKey, projection.cardReady = key, true
	} else if liveKey != projection.cardLiveKey {
		updateSessionCardLiveState(projection.translator, &projection.card, input)
	}
	projection.cardLiveKey = liveKey
	projection.card.Heard = !a.sess.HeardAt.IsZero()
	copySessionLevels(&projection.card.Levels, a.sess.Levels)
	next.Card = &projection.card
}

func cmpOr(first, second string) string {
	if first != "" {
		return first
	}
	return second
}

func (a *app) writingAnswer() bool {
	snapshot := a.pillSession()
	return !a.answer.open && snapshot.Kind == store.KindRewrite && snapshot.State == session.Composing && snapshot.Card && !snapshot.Try && strings.TrimSpace(snapshot.Draft) != ""
}

func (a *app) featureKeys(feature string) string {
	hotkeys := hotkey.HotkeysOf(a.prefs.Features, feature)
	if len(hotkeys) == 0 {
		return ""
	}
	return uitext.HotkeyText(hotkeys[0].Keys)
}
