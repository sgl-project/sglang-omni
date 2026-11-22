package session

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestTranslateKeepsInputTags: Translate never applies Rewrite's echoed-tag
// cleanup, even when the whole answer is one tagged value.
func TestTranslateKeepsInputTags(t *testing.T) {
	for _, answer := range []string{
		"<instruction>Hello</instruction>",
		"<selected_text>Good morning</selected_text>",
	} {
		t.Run(answer, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: answer}, nil)
			if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, TargetLanguage: "en", Card: true,
				Text: &TextInput{Selection: "<指令>你好</指令>"}}); err != nil {
				t.Fatal(err)
			}
			if end := h.rec.wait(t); end.State != Done || end.Result != answer {
				t.Errorf("end %s %q, want %q (problem %+v)", end.State, end.Result, answer, end.Problem)
			}
		})
	}
}

// TestRewriteDropsEchoedInputTags: an <instruction> or <selected_text>
// pair the model echoes from any text the request sent (this turn's or an
// earlier turn's) never reaches the answer card; the text between stays.
// Tags the user said or asked for stay.
func TestRewriteDropsEchoedInputTags(t *testing.T) {
	const fox = "The quick brown fox jumps over the lazy dog."
	earlier := []compose.Turn{{Instruction: "make it shorter", Selection: fox, Answer: compose.Answer{Content: "The quick fox."}}}
	templateEarlier := []compose.Turn{{Instruction: "write an instruction template", Selection: fox, Answer: compose.Answer{Content: "The quick fox."}}}
	asJSON := func(content string) string {
		b, _ := json.Marshal(compose.Answer{Title: "T", Content: content})
		return string(b)
	}
	template := "Use:\n<instruction>\n{task}\n</instruction>\n<selected_text>\n{text}\n</selected_text>"
	instructionTemplate := "<instruction>\n{task}\n</instruction>"
	for _, tc := range []struct {
		name, raw, sel, typed string
		previous              []compose.Turn
		answer, want          string
	}{
		{name: "the instruction, after a selection", raw: "make it shorter", sel: fox,
			answer: "<instruction>\nmake it shorter\n</instruction>\n\nDone: The quick fox.", want: "make it shorter\n\nDone: The quick fox."},
		{name: "the selection", raw: "make it shorter", sel: fox,
			answer: "OK: <selected_text>\n" + fox + "\n</selected_text>", want: "OK: " + fox},
		{name: "an earlier turn's selection, in a follow-up", raw: "again", previous: earlier,
			answer: "OK: <selected_text>\n" + fox + "\n</selected_text>", want: "OK: " + fox},
		{name: "an earlier turn's instruction, in a follow-up", raw: "again", previous: earlier,
			answer: "<instruction>\nmake it shorter\n</instruction> Done.", want: "make it shorter Done."},
		{name: "the instruction, nothing selected", raw: "make it shorter",
			answer: asJSON("<instruction>make it shorter</instruction> Please give me the text."), want: "make it shorter Please give me the text."},
		{name: "the instruction, nothing selected, not JSON", raw: "make it shorter",
			answer: "<instruction>\nmake it shorter\n</instruction>\nPlease give me the text.", want: "make it shorter\nPlease give me the text."},
		{name: "tags the user said stay", raw: "wrap it in <selected_text> tags", sel: "ship friday",
			answer: "<selected_text>ship friday</selected_text> sent", want: "<selected_text>ship friday</selected_text> sent"},
		{name: "a requested template stays", raw: "write a prompt template",
			answer: asJSON(template), want: template},
		{name: "a whole instruction template stays", raw: "write an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a requested template stays, after a selection", raw: "turn it into a template", sel: fox,
			answer: template, want: template},
		{name: "a whole instruction wrapper around the answer", raw: "make it formal", sel: "see you tmrw",
			answer: "<instruction>I look forward to seeing you tomorrow.</instruction>", want: "I look forward to seeing you tomorrow."},
		{name: "a whole selected_text template stays", raw: "write a selected_text template",
			answer: asJSON("<selected_text>{text}</selected_text>"), want: "<selected_text>{text}</selected_text>"},
		{name: "generic prompt wording does not request tags", raw: "rewrite this prompt to be shorter",
			answer: asJSON("<selected_text>Use a shorter prompt.</selected_text>"), want: "Use a shorter prompt."},
		{name: "editing a named prompt does not request tags", raw: "rewrite this instruction prompt to be shorter",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a later creation verb does not change prompt editing", raw: "rewrite this instruction prompt to make it shorter",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "make with a definite prompt is editing", raw: "make this instruction prompt shorter",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "want with a definite prompt is editing", raw: "I want this instruction prompt to be shorter",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an instructional prompt is not a tag name", raw: "rewrite this instructional prompt to be shorter",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a negated template mention is not a request", raw: "Do not use an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a template question is not a request", raw: "Can I use an instruction template?",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a later negated clause is not a request", raw: "Make this shorter, do not use an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a conjoined negated clause is not a request", raw: "Make this shorter and do not use an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an exclamation starts a negated clause", raw: "Make this shorter! Do not use an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a negation covers coordinated verbs", raw: "Do not use or write an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a contracted negation is not a request", raw: "I didn't want an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a negated want covers a later use", raw: "I do not want you to use an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an adverb does not hide negation", raw: "I do not really want an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an adverb after did not does not hide negation", raw: "I didn't really want an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an adverb after do not does not hide negation", raw: "I don't really want an instruction template",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an instruction is not always a template", raw: "Write an instruction for onboarding new engineers",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "an instruction purpose is not a template", raw: "Give me an instruction to reset the router",
			answer: asJSON("<instruction>Use a shorter prompt.</instruction>"), want: "Use a shorter prompt."},
		{name: "a later negated Chinese clause is not a request", raw: "写短一点，不要用 instruction 模板",
			answer: asJSON("<instruction>换个更短的说法。</instruction>"), want: "换个更短的说法。"},
		{name: "a Chinese need negation is not a request", raw: "不需要用 instruction 模板",
			answer: asJSON("<instruction>换个更短的说法。</instruction>"), want: "换个更短的说法。"},
		{name: "a Chinese no-need negation is not a request", raw: "无需用 instruction 模板",
			answer: asJSON("<instruction>换个更短的说法。</instruction>"), want: "换个更短的说法。"},
		{name: "a Chinese generation negation is not a request", raw: "不生成 instruction 模板",
			answer: asJSON("<instruction>换个更短的说法。</instruction>"), want: "换个更短的说法。"},
		{name: "a Chinese use negation is not a request", raw: "不使用 instruction 模板",
			answer: asJSON("<instruction>换个更短的说法。</instruction>"), want: "换个更短的说法。"},
		{name: "a different requested tag does not keep the wrapper", raw: "write an instruction template",
			answer: asJSON("<selected_text>A real answer</selected_text>"), want: "A real answer"},
		{name: "template intent can follow the template", raw: "An instruction template is what I want",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "template intent can have a later writer", raw: "An instruction template is what I want you to write",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "分别 is not a Chinese negation", raw: "分别用 instruction 模板写",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a positive clause survives another negation", raw: "Write it, not as plain text, but as an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a positive request follows a negated preference", raw: "I do not want plain text so write an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a subordinate negation does not negate the request", raw: "Write an instruction template that does not include examples",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a later negative sentence does not negate the request", raw: "Can you write an instruction template? Do not use plain text",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "do not forget is a positive request", raw: "Do not forget to write an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "do not just want is a positive request", raw: "I don't just want an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "did not just want is a positive request", raw: "I didn't just want an instruction template",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a negative-form polite request is positive", raw: "Can't you write an instruction template?",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "a Chinese negative-form polite request is positive", raw: "你不能写个 instruction 模板吗？",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "不错 is not a Chinese negation", raw: "写一份不错的 instruction 模板",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "告别 is not a Chinese negation", raw: "用 instruction 模板写告别邮件",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "prompt intent can have a later generator", raw: "An instruction prompt is what I want you to generate",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "prompt intent can have a later creator", raw: "An instruction prompt is what I want you to create",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "turning text into a prompt keeps the tags", raw: "turn this into an instruction prompt",
			answer: asJSON(instructionTemplate), want: instructionTemplate},
		{name: "template prose in the selection", raw: "make it formal", sel: "The template is attached.",
			answer: "<instruction>I look forward to seeing you tomorrow.</instruction>", want: "I look forward to seeing you tomorrow."},
		{name: "a template request only in an earlier turn", raw: "make it formal", previous: templateEarlier,
			answer: "<instruction>I look forward to seeing you tomorrow.</instruction>", want: "I look forward to seeing you tomorrow."},
		{name: "a typed selected_text template request", raw: "do it", sel: "see you tmrw", typed: "write a selected_text template",
			answer: "<selected_text>{text}</selected_text>", want: "<selected_text>{text}</selected_text>"},
		{name: "template prose in typed input", raw: "make it formal", sel: "see you tmrw", typed: "The template is attached.",
			answer: "<instruction>I look forward to seeing you tomorrow.</instruction>", want: "I look forward to seeing you tomorrow."},
		{name: "a repeated Chinese template keyword", raw: "照做", sel: "see you tmrw", typed: "这个模板太啰嗦，重新写一个 selected_text 模板",
			answer: "<selected_text>{text}</selected_text>", want: "<selected_text>{text}</selected_text>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: tc.answer}, func(c *Config, s *settings.Settings) { c.Select = &fakeSelector{} })
			if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack, Card: true, Previous: tc.previous,
				Text: &TextInput{Raw: tc.raw, Selection: tc.sel, Typed: tc.typed}}); err != nil {
				t.Fatal(err)
			}
			if end := h.rec.wait(t); end.State != Done || end.Result != tc.want || !end.Card {
				t.Errorf("end %s %q, card %v, want %q (problem %+v)", end.State, end.Result, end.Card, tc.want, end.Problem)
			}
		})
	}
}

func TestRewriteDraftHidesEchoedInputTags(t *testing.T) {
	for _, response := range []string{"<selected_text>Bonjour</selected_text>", "<selected_text>Bonjour"} {
		t.Run(response, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: response, chunk: 1}, nil)
			if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack, Card: true, Text: &TextInput{Raw: "rewrite", Selection: "text"}}); err != nil {
				t.Fatal(err)
			}
			if end := h.rec.wait(t); end.State != Done || end.Result != "Bonjour" {
				t.Fatalf("end %+v, want Bonjour", end)
			}
			for _, snapshot := range h.rec.all() {
				if strings.Contains(snapshot.Draft, "selected_text") || strings.Contains(snapshot.Draft, "<selected") {
					t.Errorf("draft exposed an echoed input tag: %q", snapshot.Draft)
				}
			}
		})
	}
}

func TestOtherFeaturesKeepLiteralInputTags(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranslate, store.KindTranscribe} {
		t.Run(string(kind), func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "Good morning"}, &fakeLLM{text: "<selected_text>早上好</selected_text>", chunk: 1}, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = true
			})
			req := StartRequest{Kind: kind, Target: slack, Card: true}
			if kind == store.KindTranslate {
				req.Text = &TextInput{Raw: "Good morning"}
			}
			id, err := h.m.Start(req)
			if err != nil {
				t.Fatal(err)
			}
			if kind == store.KindTranscribe {
				h.speak(id, 3)
			}
			end := h.rec.wait(t)
			if end.Problem != nil {
				t.Fatalf("session failed: %+v", *end.Problem)
			}
			if end.State != Done || end.Result != "<selected_text>早上好</selected_text>" {
				t.Errorf("end %+v, want the literal tags kept", end)
			}
		})
	}
}
