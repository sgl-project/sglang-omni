package session

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

var vscode = proto.App{PID: 900, BundleID: "com.microsoft.VSCode", Name: "Code"}

// TestWrite runs Translate and Rewrite sessions: speech or a selection
// (read, missing, failing, by ⌘C where allowed), the strict retry, the
// card (by choice, a target that takes no text, a long answer), the JSON
// answer and its fallback, failures and the text given at Start.
func TestWrite(t *testing.T) {
	type check func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector)
	for _, tc := range []struct {
		name    string
		kind    Kind
		req     StartRequest // Kind and Target filled in
		sel     *fakeSelector
		llm     *fakeLLM
		focus   *proto.FocusResult
		set     func(*settings.Settings)
		speak   bool
		states  string
		state   State
		result  string
		pasted  bool
		card    deliver.Reason // "" with no card
		problem string
		check   check
	}{
		{
			name: "translate speech", kind: store.KindTranslate, sel: &fakeSelector{}, llm: &fakeLLM{text: "你好，VoxtNew。", chunk: 2, pause: 15 * time.Millisecond},
			speak: true, states: "reading recording recognizing composing delivering done", state: Done, result: "你好，VoxtNew。", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				req := model.requests()[0]
				sys, user := req.Messages[0].Content, req.Messages[len(req.Messages)-1].Content
				if req.Skill != llm.SkillTranslateText || req.OnDelta == nil || !strings.Contains(sys, "Basic cleanup") || len(req.Messages) != 4 ||
					!strings.Contains(sys, "into Simplified Chinese") || user != "<input>\nhello VoxtNew\n</input>" {
					t.Errorf("request %+v", req)
				}
				if p := sel.params(); len(p) != 1 || !p[0].AllowCopy {
					t.Errorf("selection params %+v", p)
				}
				e := h.hist.all()[0]
				if e.Kind != store.KindTranslate || e.RawText != "hello aura link" || e.SourceText != "" || e.TargetLang != "zh-Hans" || e.Delivery != "insert" || e.LLMProvider != "deepseek" {
					t.Errorf("entry %+v", e)
				}
				h.rec.waitFor(t, "a draft", func(s Snapshot) bool { return s.State == Composing && s.Draft != "" && s.Draft != end.Result })
			},
		},
		{
			name: "translate a selection", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{text: "早上好"},
			states: "reading composing delivering done", state: Done, result: "早上好", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if h.audio.recorded() != 0 || h.audio.stopped != 0 {
					t.Errorf("recorded %d frames", h.audio.recorded())
				}
				req := model.requests()[0]
				if sys := req.Messages[0].Content; strings.Contains(sys, "Basic cleanup") || !strings.Contains(sys, "text the user selected") || req.Messages[1].Content != "<input>\nGood morning\n</input>" {
					t.Errorf("request %+v", req.Messages)
				}
				e := h.hist.all()[0]
				if end.Selection != "Good morning" || end.SelectionVia != proto.ViaAX || e.SourceText != "Good morning" || e.RawText != "" || e.AudioMS != 0 || e.ASRProvider != "" {
					t.Errorf("end %+v entry %+v", end, e)
				}
			},
		},
		{
			name: "a selection on a web page goes to the card", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{text: "早上好"},
			focus:  &proto.FocusResult{PID: 812, Role: "AXWebArea"},
			states: "reading composing done", state: Done, result: "早上好", card: deliver.ReasonNotEditable,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				h.rec.waitFor(t, "streaming into the card", func(s Snapshot) bool { return s.State == Composing && s.Card })
			},
		},
		{
			name: "reading the selection fails: it records", kind: store.KindTranslate, sel: &fakeSelector{err: errors.New("AX timeout")}, llm: &fakeLLM{text: "你好"},
			speak: true, states: "reading recording recognizing composing delivering done", state: Done, result: "你好", pasted: true,
		},
		{
			name: "⌘C reads the selection", kind: store.KindTranslate, sel: &fakeSelector{copyText: "Good night"}, llm: &fakeLLM{text: "晚安"},
			states: "reading composing delivering done", state: Done, result: "晚安", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if end.SelectionVia != proto.ViaCopy {
					t.Errorf("via %q", end.SelectionVia)
				}
			},
		},
		{
			name: "no ⌘C in VS Code", kind: store.KindTranslate, req: StartRequest{Target: vscode}, sel: &fakeSelector{copyText: "a whole line"}, llm: &fakeLLM{text: "你好"},
			focus: &proto.FocusResult{PID: 900, Role: "AXTextArea", Editable: true},
			speak: true, states: "reading recording recognizing composing delivering done", state: Done, result: "你好", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if p := sel.params(); len(p) != 1 || p[0].AllowCopy || end.Selection != "" {
					t.Errorf("params %+v, selection %q", p, end.Selection)
				}
			},
		},
		{
			name: "untranslated: asked again strictly", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{texts: []string{"Good morning.", "早上好"}},
			states: "reading composing delivering done", state: Done, result: "早上好", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				reqs := model.requests()
				if len(reqs) != 2 || strings.Contains(reqs[0].Messages[0].Content, "Strict") || !strings.Contains(reqs[1].Messages[0].Content, "Strict:") {
					t.Fatalf("%d requests", len(reqs))
				}
				if !end.Retried || !strings.Contains(end.Profile, "strict") {
					t.Errorf("end %+v", end)
				}
			},
		},
		{
			name: "still untranslated: the second answer goes on", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{text: "Good morning"},
			states: "reading composing delivering done", state: Done, result: "Good morning", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if len(model.requests()) != 2 {
					t.Errorf("%d requests", len(model.requests()))
				}
			},
		},
		{
			name: "translate into a card", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{text: "早上好"},
			set:    func(s *settings.Settings) { s.Features.Translate.ShowCard = true; s.Features.Translate.Target = "ja" },
			states: "reading composing done", state: Done, result: "早上好", card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if end.TargetLanguage != "ja" || !strings.Contains(model.requests()[0].Messages[0].Content, "into Japanese") {
					t.Errorf("target %q", end.TargetLanguage)
				}
				if e := h.hist.all()[0]; e.Delivery != "card" {
					t.Errorf("entry %+v", e)
				}
			},
		},
		{
			name: "the model fails: the card offers to try again", kind: store.KindTranslate, sel: &fakeSelector{text: "Good morning"}, llm: &fakeLLM{err: errors.New("503")},
			states: "reading composing failed", state: Failed, card: deliver.ReasonNone, problem: ProblemLLM,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				e := h.hist.all()
				if !end.Card || end.Problem.Degraded || len(e) != 1 || e[0].Status != store.StatusFailed || e[0].SourceText != "Good morning" || e[0].Error == "" {
					t.Errorf("end %+v, entries %+v", end, e)
				}
			},
		},
		{
			name: "no language model", kind: store.KindTranslate, sel: &fakeSelector{},
			states: "reading recording failed", state: Failed, problem: ProblemNoLLM,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				n := h.audio.recorded()
				time.Sleep(20 * time.Millisecond)
				if h.audio.recorded() != n {
					t.Error("still recording")
				}
			},
		},
		{
			// Into the card, which replaces the selection when asked (Voxt's).
			name: "rewrite the selection", kind: store.KindRewrite, sel: &fakeSelector{text: "hey team, ship friday?", delay: 30 * time.Millisecond}, llm: &fakeLLM{text: "Team: we ship on Friday."},
			speak: true, states: "recording recognizing composing done", state: Done, result: "Team: we ship on Friday.", card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				req := model.requests()[0]
				if req.Skill != llm.SkillRewrite || req.Format != llm.FormatText ||
					req.Messages[1].Content != "<instruction>\nhello VoxtNew\n</instruction>\n\n<selected_text>\nhey team, ship friday?\n</selected_text>" ||
					!strings.Contains(req.Messages[0].Content, "Reading the instruction") {
					t.Errorf("request %+v", req)
				}
				e := h.hist.all()[0]
				if e.PromptText != "hello aura link" || e.RawText != "" || e.SourceText != "hey team, ship friday?" || e.Result != "Team: we ship on Friday." {
					t.Errorf("entry %+v", e)
				}
			},
		},
		{
			// A short answer into an app that takes text: the card all the
			// same (Voxt's; 10-15 it was typed, with no way to follow up).
			name: "write from scratch: a JSON answer", kind: store.KindRewrite, sel: &fakeSelector{}, llm: &fakeLLM{text: `{"title": "Friday", "content": "We ship on Friday."}`, chunk: 5, pause: 15 * time.Millisecond},
			speak: true, states: "recording recognizing composing done", state: Done, result: "We ship on Friday.", card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				req := model.requests()[0]
				if req.Format != llm.FormatJSON || req.Schema == nil || !strings.Contains(req.Messages[0].Content, "JSON object") {
					t.Errorf("request %+v", req)
				}
				if end.Title != "Friday" {
					t.Errorf("title %q", end.Title)
				}
				h.rec.waitFor(t, "a partial title", func(s Snapshot) bool { return s.State == Composing && s.Title != "" && s.Draft == "" })
			},
		},
		{
			name: "streams into the card", kind: store.KindRewrite, sel: &fakeSelector{}, llm: &fakeLLM{text: `{"title":"Long","content":"` + strings.Repeat("word ", 20) + `"}`, chunk: 8, pause: 5 * time.Millisecond},
			speak: true, states: "recording recognizing composing done", state: Done, result: strings.TrimSpace(strings.Repeat("word ", 20)), card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				h.rec.waitFor(t, "the card while it streams", func(s Snapshot) bool { return s.State == Composing && s.Card && s.Draft != "" })
			},
		},
		{
			name: "not JSON: taken as text", kind: store.KindRewrite, sel: &fakeSelector{}, llm: &fakeLLM{text: "We ship on Friday."},
			speak: true, states: "recording recognizing composing done", state: Done, result: "We ship on Friday.", card: deliver.ReasonCard,
		},
		{
			name: "an empty answer", kind: store.KindRewrite, sel: &fakeSelector{}, llm: &fakeLLM{text: `{"title":"x","content":""}`},
			speak: true, states: "recording recognizing composing failed", state: Failed, problem: ProblemEmpty,
		},
		{
			name: "speech into nothing", kind: store.KindTranslate, sel: &fakeSelector{}, llm: &fakeLLM{text: " "},
			speak: true, states: "reading recording recognizing composing done", state: Done, problem: ProblemNoSpeech,
		},
		{
			name: "again with the text, into another language", kind: store.KindTranslate, sel: &fakeSelector{text: "unused"}, llm: &fakeLLM{text: "おはよう"},
			req:    StartRequest{Text: &TextInput{Selection: "Good morning"}, TargetLanguage: "ja"},
			states: "composing delivering done", state: Done, result: "おはよう", pasted: true,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if len(sel.params()) != 0 || h.audio.recorded() != 0 || end.TargetLanguage != "ja" || end.Selection != "Good morning" {
					t.Errorf("end %+v, %d reads", end, len(sel.params()))
				}
			},
		},
		{
			// The card's language menu: the answer stays in the card, though
			// the app takes text.
			name: "into another language, in the card", kind: store.KindTranslate, sel: &fakeSelector{text: "unused"}, llm: &fakeLLM{text: "おはよう"},
			req:    StartRequest{Text: &TextInput{Selection: "Good morning"}, TargetLanguage: "ja", Card: true},
			states: "composing done", state: Done, result: "おはよう", card: deliver.ReasonCard,
		},
		{
			// A follow-up answers in plain text, a conversation's turn (Voxt's).
			name: "a follow-up", kind: store.KindRewrite, sel: &fakeSelector{text: "unused"}, llm: &fakeLLM{text: "Friday."},
			req:   StartRequest{Previous: []compose.Turn{{Instruction: "write it", Answer: compose.Answer{Title: "Friday", Content: "We ship on Friday."}}}},
			speak: true, states: "recording recognizing composing done", state: Done, result: "Friday.", card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				req := model.requests()[0]
				msgs := req.Messages
				if len(msgs) != 4 || msgs[2].Role != llm.Assistant || msgs[2].Content != "We ship on Friday." || !strings.Contains(msgs[0].Content, "Follow-up") ||
					req.Format == llm.FormatJSON || len(sel.params()) != 0 || end.Title != "" {
					t.Errorf("request %+v, end %+v", req, end)
				}
			},
		},
		{
			// A rewritten selection takes follow-ups too: its turn holds it.
			name: "a follow-up on a selection", kind: store.KindRewrite, sel: &fakeSelector{text: "unused"}, llm: &fakeLLM{text: "Team: Friday."},
			req:   StartRequest{Previous: []compose.Turn{{Instruction: "more formal", Selection: "ship friday", Answer: compose.Answer{Content: "Team: we ship on Friday."}}}},
			speak: true, states: "recording recognizing composing done", state: Done, result: "Team: Friday.", card: deliver.ReasonCard,
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				msgs := model.requests()[0].Messages
				if len(msgs) != 4 || !strings.Contains(msgs[1].Content, "<selected_text>\nship friday") || msgs[3].Content != "<instruction>\nhello VoxtNew\n</instruction>" {
					t.Errorf("messages %+v", msgs)
				}
			},
		},
		{
			name: "a try reads no selection", kind: store.KindTranslate, sel: &fakeSelector{text: "unused"}, llm: &fakeLLM{text: "你好"},
			req:   StartRequest{Try: true},
			speak: true, states: "recording recognizing composing done", state: Done, result: "你好",
			check: func(t *testing.T, h *harness, end Snapshot, model *fakeLLM, sel *fakeSelector) {
				if len(sel.params()) != 0 || end.Card || len(h.hist.all()) != 0 {
					t.Errorf("end %+v", end)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var model LLM
			if tc.llm != nil {
				model = tc.llm
			}
			h := newHarness(t, &fakeFile{text: " hello aura link "}, model, func(c *Config, s *settings.Settings) {
				c.Select = tc.sel
				if tc.set != nil {
					tc.set(s)
				}
			})
			if tc.focus != nil {
				h.text.focus = *tc.focus
			}
			req := tc.req
			req.Kind = tc.kind
			if req.Target.PID == 0 {
				req.Target = slack
			}
			id, err := h.m.Start(req)
			if err != nil {
				t.Fatal(err)
			}
			if tc.speak {
				h.speak(id, 5)
			}
			end := h.rec.wait(t)
			if got := statesString(h.states()); got != tc.states {
				t.Errorf("states %q, want %q", got, tc.states)
			}
			if end.State != tc.state || end.Result != tc.result {
				t.Errorf("end %s %q, want %s %q (problem %+v)", end.State, end.Result, tc.state, tc.result, end.Problem)
			}
			if (end.Problem == nil) != (tc.problem == "") || end.Problem != nil && end.Problem.Code != tc.problem {
				t.Errorf("problem %+v, want %q", end.Problem, tc.problem)
			}
			pastes := h.text.pastes()
			if tc.pasted != (len(pastes) == 1) || tc.pasted && (pastes[0] != tc.result || !end.Inserted) {
				t.Errorf("pasted %q, inserted %v", pastes, end.Inserted)
			}
			if wantCard := tc.card != "" || tc.problem == ProblemLLM || tc.problem == ProblemEmpty; end.Card != wantCard && !req.Try || end.Delivery != tc.card && tc.card != "" {
				t.Errorf("card %v (%s), want %q", end.Card, end.Delivery, tc.card)
			}
			if tc.state == Done && tc.result != "" && !req.Try && h.m.Last() != tc.result {
				t.Errorf("last %q", h.m.Last())
			}
			if tc.check != nil {
				tc.check(t, h, end, tc.llm, tc.sel)
			}
		})
	}
}

// TestTranslateHint: a translation model that takes no rules gets speech
// without its fillers, and a selection as it is.
func TestTranslateHint(t *testing.T) {
	for _, c := range []struct {
		name string
		sel  *fakeSelector
		want string
	}{{"speech", &fakeSelector{}, "那个报告周四之前发"}, {"selection", &fakeSelector{text: "嗯，好的"}, "嗯，好的"}} {
		model := &fakeLLM{text: "Thursday."}
		h := newHarness(t, &fakeFile{text: "呃，那个报告周四之前发"}, model, func(cfg *Config, _ *settings.Settings) { cfg.Select = c.sel })
		id, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
		if err != nil {
			t.Fatal(err)
		}
		if c.sel.text == "" {
			h.speak(id, 5)
		}
		h.rec.wait(t)
		if reqs := model.requests(); len(reqs) == 0 || reqs[0].Translate == nil || reqs[0].Translate.Text != c.want {
			t.Errorf("%s: hint %+v, want %q", c.name, reqs, c.want)
		}
	}
}

// TestWriteCancel cancels Translate while it reads the selection and
// while the answer streams: nothing is pasted or kept.
func TestWriteCancel(t *testing.T) {
	t.Run("reading", func(t *testing.T) {
		sel := &fakeSelector{text: "x", delay: time.Second}
		h := newHarness(t, &fakeFile{text: "x"}, &fakeLLM{text: "y"}, func(c *Config, _ *settings.Settings) { c.Select = sel })
		id, _ := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
		h.rec.waitFor(t, "reading", func(s Snapshot) bool { return s.State == Reading })
		h.m.Cancel(id)
		if end := h.rec.wait(t); end.State != Cancelled || h.audio.recorded() != 0 || len(h.hist.all()) != 0 {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("streaming", func(t *testing.T) {
		sel := &fakeSelector{text: "x"}
		h := newHarness(t, &fakeFile{text: "x"}, &fakeLLM{hang: true}, func(c *Config, _ *settings.Settings) { c.Select = sel })
		id, _ := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
		h.rec.waitFor(t, "a draft", func(s Snapshot) bool { return s.Draft != "" })
		h.m.Cancel(id)
		if end := h.rec.wait(t); end.State != Cancelled || len(h.text.pastes()) != 0 || len(h.hist.all()) != 0 || end.Card {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("the answer times out", func(t *testing.T) {
		sel := &fakeSelector{text: "x"}
		h := newHarness(t, &fakeFile{text: "x"}, &fakeLLM{hang: true}, func(c *Config, _ *settings.Settings) {
			c.Select = sel
			c.LLMFirstToken, c.WriteTimeout = 50*time.Millisecond, 50*time.Millisecond
		})
		h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
		end := h.rec.wait(t)
		if end.State != Failed || end.Problem.Code != ProblemLLM || !end.Card || !errors.Is(end.Problem.Err, context.DeadlineExceeded) {
			t.Errorf("end %+v %v", end, end.Problem)
		}
	})
}

func TestTranslateTarget(t *testing.T) {
	s := settings.Defaults()
	for _, tc := range []struct {
		main, target, locale, override, want string
	}{
		{"", "", "en", "", "zh-Hans"},
		{"", "", "zh-Hans", "", "en"},
		{"zh", "", "en", "", "en"},
		{"en", "", "zh-Hans", "", "zh-Hans"},
		{"ja", "", "en", "", "en"},
		{"zh", "ja", "en", "", "ja"},
		{"zh", "ja", "en", "fr", "fr"},
	} {
		s.General.MainLanguage, s.Features.Translate.Target = tc.main, tc.target
		if got := TranslateTarget(s, tc.locale, tc.override); got != tc.want {
			t.Errorf("%+v: %q", tc, got)
		}
	}
}

func TestAllowCopy(t *testing.T) {
	for id, want := range map[string]bool{
		"com.apple.TextEdit": true, "com.tinyspeck.slackmacgap": true, "": true,
		"com.microsoft.VSCode": false, "com.todesktop.230313mzl4w4u92": false, "com.jetbrains.goland": false,
		"md.obsidian": false, "dev.zed.Zed": false, "com.example.MyVSCodeFork": false,
	} {
		if got := AllowCopy(id); got != want {
			t.Errorf("AllowCopy(%q) = %v", id, got)
		}
	}
}

// prefillLLM is a fakeLLM with a prompt cache (llm.Prefiller).
type prefillLLM struct {
	*fakeLLM
	mu       sync.Mutex
	prefills []llm.Request
}

func (p *prefillLLM) Prefill(_ context.Context, req llm.Request) error {
	p.mu.Lock()
	p.prefills = append(p.prefills, req)
	p.mu.Unlock()
	return nil
}

// TestWarmLLM: while the user speaks, the session warms a model with a
// prompt cache up with the prompt it will send, but for the text:
// Transcribe's clean-up and Translate of speech, not Rewrite (its prompt
// depends on the selection) or a Transcribe without the language model.
func TestWarmLLM(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kind    Kind
		enhance bool
		warm    bool
	}{
		{"transcribe", store.KindTranscribe, true, true},
		{"transcribe, no clean-up", store.KindTranscribe, false, false},
		{"translate speech", store.KindTranslate, true, true},
		{"rewrite", store.KindRewrite, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &prefillLLM{fakeLLM: &fakeLLM{text: `{"title":"Hi","content":"Hello."}`}}
			h := newHarness(t, &fakeFile{text: "hello aura link"}, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = tc.enhance
			})
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.rec.waitFor(t, "recording", func(s Snapshot) bool { return s.State == Recording })
			h.speak(id, 3)
			h.rec.wait(t)
			var prefills []llm.Request
			for deadline := time.Now().Add(time.Second); ; time.Sleep(time.Millisecond) {
				model.mu.Lock()
				prefills = model.prefills
				model.mu.Unlock()
				if len(prefills) > 0 || time.Now().After(deadline) {
					break
				}
			}
			if len(prefills) != map[bool]int{true: 1, false: 0}[tc.warm] {
				t.Fatalf("%d prefills", len(prefills))
			}
			if !tc.warm {
				return
			}
			chat := model.reqs[0]
			p := prefills[0]
			n := len(chat.Messages)
			if p.Skill != chat.Skill || len(p.Messages) != n || n < 2 || !reflect.DeepEqual(p.Messages[:n-1], chat.Messages[:n-1]) ||
				p.Messages[n-1].Content == chat.Messages[n-1].Content {
				t.Errorf("prefill %+v\nchat %+v", p, chat)
			}
		})
	}
}
