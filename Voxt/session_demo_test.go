package main

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func TestSessionDemoRejectsInvalidCardFixture(t *testing.T) {
	for _, test := range []struct {
		name string
		card sessionCardViewState
		want string
	}{
		{name: "size step", card: sessionCardViewState{SizeStep: 99}, want: "size step 99"},
		{name: "live count", card: sessionCardViewState{LiveCount: 3}, want: "live count 3"},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := meetingFixtureState{Name: "invalid", Kind: "session-overlay", Session: &sessionOverlayViewState{Card: &test.card}, SessionUI: &sessionOverlayFixtureUI{}}
			data, err := json.Marshal(fixture)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "invalid.json")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := readMeetingDemoFixture(path); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("invalid fixture error %v, want text %q", err, test.want)
			}
		})
	}
}

func newSessionDemoActionTester(t *testing.T, host *meetingDemoHost) *ui.Tester {
	t.Helper()
	placement := host.sessionDemoPlacement()
	width := max(placement.width, layout.CardSizes[len(layout.CardSizes)-1].W+2*layout.Pad)
	tester := ui.NewTester(func(context *ui.Context) { sessionDemoOverlayView(context, host) }, width, placement.height)
	tester.SetLocale(host.current().locale)
	tester.SetFontSmoothing(hudSmoothing)
	frames(tester)
	return tester
}

func TestAnswerCardMathRequestsPreview(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/card-rewrite")
	host.sessionState.Card.Blocks[0].Text = "$x^2$"
	newSessionDemoActionTester(t, host)
	if !slices.Contains(host.calls, `overlayTranscriptResult "$x^2$"`) {
		t.Fatalf("math answer did not ask for the transcript preview: %q", host.calls)
	}
}

type sessionDemoActionCase struct {
	name           string
	fixture        string
	methods        []string
	steps          []meetingDemoStep
	notice         bool
	bubble         bool
	disabledLabels []string
	want           []string
	check          func(*testing.T, *meetingDemoHost, *ui.Tester)
}

func sessionDemoActionCases() []sessionDemoActionCase {
	return []sessionDemoActionCase{
		{name: "skip", fixture: "en/overlay-composing-skip", methods: []string{"skipLLM"}, steps: steps("Use transcript"), want: []string{"skipLLM"}},
		{name: "stop session", fixture: "en/card-follow-up-recording", methods: []string{"stopSession"}, steps: steps("Done"), want: []string{"stopSession"}},
		{name: "stop answer", fixture: "en/card-follow-up-writing", methods: []string{"stopLLM"}, steps: steps("Stop"), want: []string{"stopLLM"}},
		{name: "follow up", fixture: "en/card-short", methods: []string{"followUpAnswer"}, steps: steps("Continue"), want: []string{`followUpAnswer ""`}},
		{name: "retry", fixture: "en/card-failed", methods: []string{"retryAnswer"}, steps: steps("Try again"), want: []string{`retryAnswer ""`}},
		{name: "insert", fixture: "en/card-short", methods: []string{"insertAnswer"}, steps: steps("Insert"), want: []string{"insertAnswer"}},
		{name: "copy", fixture: "en/card-short", methods: []string{"copyAnswer"}, steps: steps("Copy"), want: []string{"copyAnswer"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if !host.sessionState.Card.Copied {
				t.Fatal("copy did not update the local card")
			}
		}},
		{name: "close", fixture: "en/card-short", methods: []string{"closeAnswer"}, steps: steps("Close"), want: []string{"closeAnswer"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.sessionState.Mode != ovHidden || host.sessionState.Card != nil {
				t.Fatal("close did not hide the local card")
			}
		}},
		{name: "notice models", fixture: "en/session-problem-no-asr", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction models"}},
		{name: "notice language", fixture: "en/session-problem-no-llm", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction language"}},
		{name: "notice permissions", fixture: "en/session-problem-microphone", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction permissions"}},
		{name: "notice helper", fixture: "en/session-problem-helper", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction helper"}},
		{name: "notice retry", fixture: "en/session-problem-asr-retry", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction retry"}},
		{name: "notice sign in", fixture: "en/session-problem-llm-sign-in", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction signin"}},
		{name: "notice usage", fixture: "en/session-problem-llm-usage", methods: []string{"noticeAction"}, notice: true, want: []string{"noticeAction usage"}},
		{name: "language open and choose", fixture: "en/card-short", methods: []string{"toggleLangMenu", "chooseLanguage"}, steps: steps("Translate into", "English"), want: []string{"toggleLangMenu", "chooseLanguage en"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			card := host.sessionState.Card
			if card.Language != "en" || card.LanguageLabel != "English" || host.sessionState.Menu != nil {
				t.Fatalf("language choice did not update and close the local menu: %+v", card)
			}
		}},
		{name: "card thinking open and choose", fixture: "en/card-thinking", methods: []string{"toggleThinkMenu", "chooseThinking"}, steps: steps("Thinking level", "Default"), want: []string{"toggleThinkMenu", `chooseThinking "" "" 0`}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.sessionState.Card.Thinking != (sessionThinkingValue{}) || host.sessionState.Menu != nil {
				t.Fatal("thinking choice did not update and close the local menu")
			}
		}},
		{name: "pill thinking open and choose", fixture: "en/overlay-composing-think", methods: []string{"togglePillThinkMenu", "chooseThinking"}, steps: steps("Thinking level: Off", "Default"), want: []string{"togglePillThinkMenu", `chooseThinking "" "" 0`}},
		{name: "open card thinking choose", fixture: "en/session-menu-card-thinking", methods: []string{"chooseThinking"}, steps: steps("Default"), want: []string{`chooseThinking "" "" 0`}},
		{name: "open pill thinking choose", fixture: "en/session-menu-pill-thinking", methods: []string{"chooseThinking"}, steps: steps("Default"), want: []string{`chooseThinking "" "" 0`}},
		{name: "quote open", fixture: "en/card-follow-up-selected-long", methods: []string{"toggleQuote"}, steps: steps("Show all"), want: []string{"toggleQuote 3"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			block := sessionDemoQuoteBlock(host, 3)
			if block == nil || !block.Open || block.Quote != block.FullQuote {
				t.Fatal("Show all did not expand the full local quote")
			}
		}},
		{name: "quote close", fixture: "en/card-follow-up-selected-open", methods: []string{"toggleQuote"}, steps: steps("Show less"), want: []string{"toggleQuote 3"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			block := sessionDemoQuoteBlock(host, 3)
			if block == nil || block.Open || !block.Cut || block.Quote == block.FullQuote {
				t.Fatal("Show less did not truncate the local quote")
			}
		}},
		{name: "bubble copy", fixture: "en/card-conversation", methods: []string{"copyBubble"}, bubble: true, want: []string{`copyBubble "Late\n\nTen minutes late: start without me."`}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.sessionUI.bubbleCopied != "Ten minutes late: start without me." {
				t.Fatalf("bubble copy mark has %q", host.sessionUI.bubbleCopied)
			}
		}},
		{name: "size cycle wraps", fixture: "en/card-short", methods: []string{"cycleOverlayCardSize"}, steps: steps("Make the card larger", "Make the card larger", "Make the card smaller"), want: []string{"cycleOverlayCardSize", "cycleOverlayCardSize", "cycleOverlayCardSize"}, check: func(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
			if host.sessionState.Card.SizeStep != 0 {
				t.Fatalf("size step %d, want wrapped step 0", host.sessionState.Card.SizeStep)
			}
		}},
		{name: "disabled controls", fixture: "en/card-redo", disabledLabels: []string{"Insert", "Translate into"}},
	}
}

func TestSessionDemoActionControls(t *testing.T) {
	for _, test := range sessionDemoActionCases() {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			original := host.current().state.Session.Card
			originalText := ""
			if original != nil && len(original.Blocks) > 0 {
				originalText = original.Blocks[0].Text
			}
			tester := newSessionDemoActionTester(t, host)
			if test.notice {
				notice := host.sessionState.Notice
				if notice == nil {
					t.Fatal("notice fixture has no notice")
				}
				test.steps = steps(notice.Text + " · " + notice.ActionLabel)
			}
			if test.bubble {
				clickSessionDemoBubbleCopy(t, host, tester)
			}
			for _, step := range test.steps {
				if err := tester.ClickControl(step.label); err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			for _, label := range test.disabledLabels {
				if err := tester.Click(label); err != nil {
					t.Fatalf("disabled %s: %v", label, err)
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.check != nil {
				test.check(t, host, tester)
			}
			if original != nil && len(original.Blocks) > 0 && original.Blocks[0].Text != originalText {
				t.Fatal("local action mutated the stored fixture")
			}
			if test.notice && host.sessionState.Mode != ovHidden {
				t.Fatal("notice action did not hide the local notice")
			}
		})
	}
}

func TestSessionDemoMenuChoicesSurviveReopen(t *testing.T) {
	for _, test := range []struct {
		name, fixture, kind, choose string
	}{
		{name: "language", fixture: "en/card-short", kind: sessionMenuLanguage, choose: "English"},
		{name: "card thinking", fixture: "en/card-thinking", kind: sessionMenuCardThinking, choose: "Default"},
		{name: "pill thinking", fixture: "en/overlay-composing-think", kind: sessionMenuPillThinking, choose: "Default"},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newSessionDemoActionTester(t, host)
			open := "Translate into"
			if test.kind != sessionMenuLanguage {
				open = "Thinking level"
				if test.kind == sessionMenuPillThinking {
					open = host.sessionState.Pill.ThinkingLabel
				}
			}
			for step := range 2 {
				if err := tester.ClickControl(open); err != nil {
					t.Fatalf("open %s: %v", open, err)
				}
				frames(tester)
				if host.sessionState.Menu == nil {
					t.Fatal("menu did not open")
				}
				if step == 0 {
					if err := tester.ClickControl(test.choose); err != nil {
						t.Fatalf("choose %s: %v", test.choose, err)
					}
					frames(tester)
				}
				if test.kind == sessionMenuPillThinking {
					open = host.sessionState.Pill.ThinkingLabel
				}
			}
			menu := host.sessionState.Menu
			if menu == nil {
				t.Fatal("menu did not reopen")
			}
			selected := -1
			for index, option := range menu.Options {
				if !option.Selected {
					continue
				}
				if selected >= 0 {
					t.Fatal("multiple options selected after reopen")
				}
				selected = index
				if test.kind == sessionMenuLanguage && option.Language != "en" || test.kind != sessionMenuLanguage && option.Thinking != (sessionThinkingValue{}) {
					t.Fatalf("stale option after reopen: %+v", option)
				}
			}
			if selected < 0 || menu.SelectedIndex != selected {
				t.Fatalf("selection after reopen: index %d, selected %d", menu.SelectedIndex, selected)
			}
		})
	}
}

func sessionDemoQuoteBlock(host *meetingDemoHost, turn int) *sessionCardBlock {
	card := host.sessionState.Card
	if card == nil {
		return nil
	}
	for index := range card.Blocks {
		if card.Blocks[index].Turn == turn && card.Blocks[index].Kind == sessionCardBubble && card.Blocks[index].User {
			return &card.Blocks[index]
		}
	}
	return nil
}

func clickSessionDemoBubbleCopy(t *testing.T, host *meetingDemoHost, tester *ui.Tester) {
	t.Helper()
	card := host.sessionState.Card
	block := card.Blocks[len(card.Blocks)-1]
	textBounds, ok := tester.Find(block.Text)
	if !ok {
		t.Fatalf("bubble text %q is not visible", block.Text)
	}
	titleBounds, ok := tester.Find(block.Title)
	if !ok {
		t.Fatalf("bubble title %q is not visible", block.Title)
	}
	tester.Move(textBounds.X+textBounds.W/2, textBounds.Y+textBounds.H/2)
	frames(tester)
	text, _, _ := sessionCardBlockText(card, block)
	tester.ClickAt(textBounds.X+text.wrap+layout.BubblePadX-35, titleBounds.Y-layout.BubblePadY+19)
	frames(tester)
}

func sessionOverlayContentSentinel(state sessionOverlayViewState) string {
	if state.Mode == ovHidden {
		return ""
	}
	if state.Menu != nil && state.Menu.Label != "" {
		return state.Menu.Label
	}
	if state.Notice != nil && state.Notice.Text != "" {
		return state.Notice.Text
	}
	if card := state.Card; card != nil {
		if card.Message != "" {
			return card.Message
		}
		for index := 0; index < sessionCardBlockCount(card); index++ {
			if block := sessionCardBlockAt(card, index); block.Text != "" {
				if block.Markdown { // drawn a paragraph per text element
					return strings.SplitN(block.Text, "\n\n", 2)[0]
				}
				return block.Text
			}
		}
		if card.Title != "" {
			return card.Title
		}
	}
	if state.Pill.LiveText != "" {
		return state.Pill.LiveText
	}
	if state.Mode == ovWorking && state.Pill.WorkingLabel != "" {
		return state.Pill.WorkingLabel
	}
	if state.Pill.Starting && state.Pill.WaitingLabel != "" {
		return state.Pill.WaitingLabel
	}
	if state.Pill.Title != "" {
		return state.Pill.Title
	}
	return state.Pill.WorkingLabel
}

func TestSessionDemoActionCoverage(t *testing.T) {
	syntax, err := parser.ParseFile(token.NewFileSet(), "session_overlay_state.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	required := map[string]bool{}
	for _, declaration := range syntax.Decls {
		general, ok := declaration.(*ast.GenDecl)
		if !ok || general.Tok != token.TYPE {
			continue
		}
		for _, specification := range general.Specs {
			typeSpecification := specification.(*ast.TypeSpec)
			if typeSpecification.Name.Name != "sessionOverlayActions" {
				continue
			}
			for _, method := range typeSpecification.Type.(*ast.InterfaceType).Methods.List {
				for _, name := range method.Names {
					required[name.Name] = true
				}
			}
		}
	}
	if len(required) == 0 {
		t.Fatal("sessionOverlayActions has no methods")
	}
	covered := map[string]bool{
		"overlayTranscriptResult": true, // TestAnswerCardMathRequestsPreview covers this automatic action.
	}
	for _, test := range sessionDemoActionCases() {
		for _, method := range test.methods {
			covered[method] = true
		}
	}
	for method := range required {
		if !covered[method] {
			t.Errorf("no session demo control covers %s", method)
		}
	}
	for method := range covered {
		if !required[method] {
			t.Errorf("session demo coverage names unknown method %s", method)
		}
	}
}
