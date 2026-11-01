package compose

import (
	"strings"
	"testing"
)

// TestExample: speech with clean-up gets the example turn, in the main
// language's set, its answer in the target language when the set has
// it; a selection, Rewrite or no clean-up get none.
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
		{Input{Task: Translate, Cleanup: true, MainLanguage: "zh", TargetLanguage: "zh-Hans"}, "发给小李"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "en", TargetLanguage: "zh-Hans"}, "把周报发给 Anna"},
		{Input{Task: Translate, Cleanup: true, MainLanguage: "en", TargetLanguage: "en-GB"}, "to Anna before 5"},
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
		if len(out.History) != 2 || out.History[0].Role != "user" || !strings.HasPrefix(out.History[0].Content, "<input>") ||
			out.History[1].Role != "assistant" || out.History[1].Content != answer {
			t.Errorf("%+v: history %+v", c.in, out.History)
		}
		if m := out.Messages(); len(m) != 4 || m[3].Content != UserMessage("x") {
			t.Errorf("messages %+v", m)
		}
	}
}

func TestStripFillers(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"呃，今天天气真好，嗯，我们下午见。", "今天天气真好，我们下午见。"},
		{"嗯那个报告呃周四之前", "那个报告周四之前"},
		{"Um, so we need to, uh, fix the bug.", "So we need to, fix the bug."},
		{"umbrella and hummus, um", "umbrella and hummus,"},
		{"不对，是杭州", "不对，是杭州"},
		{"嗯，那个，我们下周二，呃不对，下周三开会。", "我们下周二，不对，下周三开会。"},
		{"那个报告的话，嗯，周五之前发给我就行，啊不对，周四之前。", "那个报告的话，周五之前发给我就行，不对，周四之前。"},
		{"就是说，这个方案呢，好啊。", "这个方案呢，好啊。"},
		{"那个人说的", "那个人说的"},
	} {
		if got := StripFillers(c.in); got != c.want {
			t.Errorf("StripFillers(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
