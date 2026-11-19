package session

import (
	"encoding/json"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRewriteDropsEchoedInputTags: an <instruction> or <selected_text>
// pair the model echoes from any text the request sent (this turn's or an
// earlier turn's) never reaches the answer card; the text between stays.
// Tags the user said or asked for stay.
func TestRewriteDropsEchoedInputTags(t *testing.T) {
	const fox = "The quick brown fox jumps over the lazy dog."
	earlier := []compose.Turn{{Instruction: "make it shorter", Selection: fox, Answer: compose.Answer{Content: "The quick fox."}}}
	asJSON := func(content string) string {
		b, _ := json.Marshal(compose.Answer{Title: "T", Content: content})
		return string(b)
	}
	template := "Use:\n<instruction>\n{task}\n</instruction>\n<selected_text>\n{text}\n</selected_text>"
	for _, tc := range []struct {
		name, raw, sel string
		previous       []compose.Turn
		answer, want   string
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
		{name: "a requested template stays, after a selection", raw: "turn it into a template", sel: fox,
			answer: template, want: template},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: tc.answer}, func(c *Config, s *settings.Settings) { c.Select = &fakeSelector{} })
			if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack, Card: true, Previous: tc.previous,
				Text: &TextInput{Raw: tc.raw, Selection: tc.sel}}); err != nil {
				t.Fatal(err)
			}
			if end := h.rec.wait(t); end.State != Done || end.Result != tc.want || !end.Card {
				t.Errorf("end %s %q, card %v, want %q (problem %+v)", end.State, end.Result, end.Card, tc.want, end.Problem)
			}
		})
	}
}
