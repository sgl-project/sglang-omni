package compose

import (
	"strings"
	"testing"
)

// TestExample: speech with clean-up gets the example turn, in the main
// language's set, its answer in the target language when the set has
// one; a selection, Rewrite or no clean-up get none.
func TestExample(t *testing.T) {
	for _, c := range []struct {
		in     Input
		answer string // a part of the answer, "" for no example
	}{
		{Input{Task: Transcribe, Cleanup: true, MainLanguage: "zh"}, "发给小李，今天下班前"},
		{Input{Task: Transcribe, Cleanup: true, MainLanguage: "en"}, "to Anna before 5"},
		{Input{Task: Transcribe, Cleanup: true, MainLanguage: "fr"}, "to Anna before 5"},
		{Input{Task: Transcribe, MainLanguage: "zh"}, ""},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "en"}, "Xiao Li before the end of today"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "zh-Hans"}, "把周报发给 Anna"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "en", TargetLanguage: "zh-Hans"}, "把周报发给 Anna"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "en", TargetLanguage: "en-GB"}, "Xiao Li before the end of today"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "ja"}, ""},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "zh-Hant"}, ""},
		{Input{Task: Translate, Cleanup: true, Selected: true, MainLanguage: "zh", TargetLanguage: "en"}, ""},
		{Input{Task: Rewrite, Cleanup: true, MainLanguage: "zh", Instruction: "x"}, ""},
	} {
		c.in.Text = "x"
		_, answer, ok := Example(c.in)
		out, err := Build(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if c.answer == "" {
			if ok || len(out.History) != 0 {
				t.Errorf("%+v: an example %q", c.in, answer)
			}
			continue
		}
		if !ok || !strings.Contains(answer, c.answer) {
			t.Errorf("%+v: example %q, want %q", c.in, answer, c.answer)
		}
		// Transcribe's contract asks for the <output> tags, so its
		// example turn answers with them.
		inHistory := answer
		if c.in.Task == Transcribe {
			inHistory = "<output>\n" + answer + "\n</output>"
		}
		if len(out.History) != 2 || out.History[0].Role != "user" || !strings.HasPrefix(out.History[0].Content, "<input>") ||
			out.History[1].Role != "assistant" || out.History[1].Content != inHistory {
			t.Errorf("%+v: history %+v", c.in, out.History)
		}
		if m := out.Messages(); len(m) != 4 || m[3].Content != UserMessage("x") {
			t.Errorf("messages %+v", m)
		}
	}
}

// TestTranslateIntoMainLanguage: Translate into the user's main language
// (issue #4153: Chinese speech, English main language and target) neither
// forbids the target in the system prompt nor shows a same-language
// clean-up as its example, which made the model return the source.
func TestTranslateIntoMainLanguage(t *testing.T) {
	for _, c := range []struct {
		main, target, text string
		forbid             string // the clause that forbids the main language
		input, answer      string // a part of the example's input and answer
	}{
		{"en", "en", "嗯那个周报帮我发给小李", "never as a language to translate into", "发给小王", "Xiao Li before the end of today"},
		{"zh", "zh-Hans", "um send the weekly report to Anna", "不是翻译的目标语言", "the weekly report to Tom", "把周报发给 Anna"},
	} {
		in := Input{Task: Translate, Cleanup: true, MainLanguage: c.main, TargetLanguage: c.target, Text: c.text}
		out, err := Build(in)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(out.System, c.forbid) {
			t.Errorf("%s→%s: the system prompt forbids the target:\n%s", c.main, c.target, out.System)
		}
		if len(out.History) != 2 || !strings.Contains(out.History[0].Content, c.input) || !strings.Contains(out.History[1].Content, c.answer) {
			t.Errorf("%s→%s: example %+v, want a translation into the target", c.main, c.target, out.History)
		}
		in.Strict = true
		if out, _ := Build(in); strings.Contains(out.System, c.forbid) {
			t.Errorf("%s→%s strict: the system prompt forbids the target", c.main, c.target)
		}
	}
}
