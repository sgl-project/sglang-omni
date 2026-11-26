package compose

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestAnswerDropsRewriteTags: a model that echoes the <selected_text>,
// <typed_text> or <instruction> tags RewriteMessage wraps its input in
// never shows them on the answer card; the text between them stays.
func TestAnswerDropsRewriteTags(t *testing.T) {
	const selection, instruction, typed = "The quick brown fox", "make it formal", "type this"
	for _, tc := range []struct{ out, want string }{
		{"OK: <selected_text>\nThe quick brown fox\n</selected_text>", "OK: The quick brown fox"},
		{"<instruction>\nmake it formal\n</instruction>\n\nDone.", "make it formal\n\nDone."},
		{"Fixed: <selected_text>The quick brown fox</selected_text>", "Fixed: The quick brown fox"},
		{"<selected_text>\nA quick brown fox.\n</selected_text>", "A quick brown fox."},
		{"OK: <typed_text>\ntype this\n</typed_text>", "OK: type this"},
		{"<typed_text>\nA typed note.\n</typed_text>", "A typed note."},
		{"<typed_text>\ntype this", "type this"},
		{"<instruction>I look forward to seeing you tomorrow.</instruction>", "I look forward to seeing you tomorrow."},
		{"OK: <selected_text>The quick brown fox", "OK: The quick brown fox"},
	} {
		checkAnswer(t, tc.out, tc.want, selection, instruction, typed)
	}
	checkAnswer(t, "<instruction>Use 230 V.</instruction>", "Use 230 V.", "", "fix the voltage")
	checkAnswer(t, "<instruction>I look forward to seeing you tomorrow.</instruction>", "I look forward to seeing you tomorrow.", "The template is attached.", "make it formal")
	checkAnswer(t, "<instruction>I look forward to seeing you tomorrow.</instruction>", "I look forward to seeing you tomorrow.", "see you tmrw", "The template is attached.\nmake it formal", "The template is attached.")
	checkAnswer(t, "<selected_text>Use a shorter prompt.</selected_text>", "Use a shorter prompt.", "", "rewrite this prompt to be shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "rewrite this instruction prompt to be shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "rewrite this instruction prompt to make it shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "rewrite this instruction prompt because I want it shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "make this instruction prompt shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I want this instruction prompt to be shorter")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "rewrite this instructional prompt to be shorter")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "把这个 instruction 提示词改短些")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Do not use an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Do not write an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Can I use an instruction template?")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Do I need an instruction template?")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "不要使用 instruction 模板")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Make this shorter, do not use an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Make this shorter and do not use an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Make this shorter! Do not use an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Do not use or write an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I didn't want an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I do not really want an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I didn't really want an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I don't really want an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "I do not want you to use an instruction template")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Write an instruction for onboarding new engineers")
	checkAnswer(t, "<instruction>Use a shorter prompt.</instruction>", "Use a shorter prompt.", "", "Give me an instruction to reset the router")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "写短一点，不要用 instruction 模板")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "不需要用 instruction 模板")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "无需用 instruction 模板")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "不生成 instruction 模板")
	checkAnswer(t, "<instruction>换个更短的说法。</instruction>", "换个更短的说法。", "", "不使用 instruction 模板")
	checkAnswer(t, "<instruction>A real answer</instruction>", "A real answer", "", "No need to use an instruction template")
	checkAnswer(t, "<instruction>A real answer</instruction>", "A real answer", "", "You don't have to use an instruction template")
	checkAnswer(t, "<instruction>A real answer</instruction>", "A real answer", "", "Write plain text, not an instruction template")
	checkAnswer(t, "<instruction>真实回答</instruction>", "真实回答", "", "没必要用 instruction 模板")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "An instruction template is not what I want")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "An instruction template isn't required")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "An instruction template is not needed")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "instruction 模板不需要")
	checkAnswer(t, "<instruction>An instruction template is a reusable prompt.</instruction>", "An instruction template is a reusable prompt.", "", "What is an instruction template?")
	checkAnswer(t, "<instruction>A real answer</instruction>", "A real answer", "", "Could you explain an instruction template?")
	checkAnswer(t, "<instruction>A real answer</instruction>", "A real answer", "", "Could you tell me what an instruction template is?")
	checkAnswer(t, "<instruction>真实回答</instruction>", "真实回答", "", "你能解释一下 instruction 模板吗？")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "No need to use <instruction> tags")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "能不能不用 instruction 模板？")
	checkAnswer(t, "<instruction>{task}</instruction>", "{task}", "", "Use <instruction> tags, then do not use <instruction> tags")
	checkAnswer(t, "<selected_text>A real answer</selected_text>", "A real answer", "", "write an instruction template")
}

// TestAnswerKeepsRequestedTags: tags that are part of the answer, such as
// a prompt template the user asked for, stay; so do tags the user's own
// selection has.
func TestAnswerKeepsRequestedTags(t *testing.T) {
	template := "Use this template:\n<instruction>\n{task}\n</instruction>\n<selected_text>\n{text}\n</selected_text>\n<typed_text>\n{typed}\n</typed_text>"
	checkAnswer(t, template, template, "", "Write a prompt template with instruction, selected_text and typed_text tags")
	sentence := "Wrap the task in <instruction></instruction> tags, the input in <selected_text> tags."
	checkAnswer(t, sentence, sentence, "", "Rephrase it")
	own := "Put it in <instruction>\nthe task\n</instruction>, then answer."
	checkAnswer(t, own, own, own, "fix")
	checkAnswer(t, "<selected_text>"+own+"</selected_text>", own, own, "fix")
}

func TestAnswerKeepsSingleInstructionPair(t *testing.T) {
	template := "<instruction>\n{task}\n</instruction>"
	checkAnswer(t, template, template, "", "Write an instruction template")
	checkAnswer(t, template, template, "", "Write an instruction prompt")
	checkAnswer(t, template, template, "", "An instruction template is what I want")
	checkAnswer(t, template, template, "", "An instruction template is what I want you to write")
	checkAnswer(t, template, template, "", "An instruction prompt is what I want you to generate")
	checkAnswer(t, template, template, "", "An instruction prompt is what I want you to create")
	checkAnswer(t, template, template, "", "turn this into an instruction prompt")
	checkAnswer(t, template, template, "", "convert this to an instruction prompt")
	checkAnswer(t, template, template, "", "rewrite this as an instruction prompt")
	checkAnswer(t, template, template, "", "Can you write an instruction template?")
	checkAnswer(t, template, template, "", "Could you write an instruction template?")
	checkAnswer(t, template, template, "", "Can you use an instruction template?")
	checkAnswer(t, template, template, "", "Could you use an instruction template?")
	checkAnswer(t, template, template, "", "你能用 instruction 模板吗？")
	checkAnswer(t, template, template, "", "你可以用 instruction 模板吗？")
	checkAnswer(t, template, template, "", "把文字改成 instruction 提示词")
	checkAnswer(t, template, template, "", "分别用 instruction 模板写")
	checkAnswer(t, template, template, "", "Write it, not as plain text, but as an instruction template")
	checkAnswer(t, template, template, "", "I do not want plain text so write an instruction template")
	checkAnswer(t, template, template, "", "Write an instruction template that does not include examples")
	checkAnswer(t, template, template, "", "Can you write an instruction template? Do not use plain text")
	checkAnswer(t, template, template, "", "Do not forget to write an instruction template")
	checkAnswer(t, template, template, "", "I don't just want an instruction template")
	checkAnswer(t, template, template, "", "I didn't just want an instruction template")
	checkAnswer(t, template, template, "", "Can't you write an instruction template?")
	checkAnswer(t, template, template, "", "你不能写个 instruction 模板吗？")
	checkAnswer(t, template, template, "", "写一份不错的 instruction 模板")
	checkAnswer(t, template, template, "", "用 instruction 模板写告别邮件")
	checkAnswer(t, template, template, "", "Draft an instruction template")
	checkAnswer(t, template, template, "", "起草一个 instruction 模板")
	checkAnswer(t, template, template, "", "写一个特别的 instruction 模板")
	checkAnswer(t, template, template, "", "能不能起草一个 instruction 模板？")
	checkAnswer(t, template, template, "", "Do not hesitate to draft an instruction template")
	checkAnswer(t, template, template, "", "Use an instruction template. Can you do that?")
	checkAnswer(t, template, template, "", "用 instruction 模板。明白吗？")
	checkAnswer(t, template, template, "", "Draft an instruction template that is not generic")
	checkAnswer(t, template, template, "", "Draft an instruction template that isn't generic")
	checkAnswer(t, template, template, "", "Draft an instruction template that is no longer than five lines")
	checkAnswer(t, template, template, "", "写一个 instruction 模板不要包含示例")
	checkAnswer(t, template, template, "", "Do not use <instruction> tags, then draft an <instruction> template")
}

func TestAnswerIgnoresEarlierTemplateRequest(t *testing.T) {
	const out = "<instruction>I look forward to seeing you tomorrow.</instruction>"
	const want = "I look forward to seeing you tomorrow."
	checkAnswerInputs(t, out, want, "make it formal", "write an instruction template")
	checkAnswerInputs(t, out, want, "current source\nmake it formal", "make it formal", "<instruction>{task}</instruction>")
}

func TestAnswerKeepsControlMarkupWithCurrentRequest(t *testing.T) {
	const out = "Use <|im_start|> literally"
	input := []string{"Explain <|im_start|>\nexplain this marker", "explain this marker"}
	for _, partial := range []bool{false, true} {
		if got := CleanRewrite(out, input, partial); got != out {
			t.Errorf("CleanRewrite(partial %v) = %q, want %q", partial, got, out)
		}
	}
	checkAnswerInputs(t, out, out, input...)
}

func TestAnswerKeepsRequestedTemplatePrefixes(t *testing.T) {
	const open = "<instruction>"
	const template = open + "\n{task}\n"
	for i := len(open); i <= len(template); i++ {
		prefix := template[:i]
		want := strings.TrimSpace(prefix)
		input := []string{"Draft an instruction template"}
		if got := CleanRewrite(prefix, input, true); got != want {
			t.Errorf("CleanRewrite(%q) = %q, want %q", prefix, got, want)
		}
		if got := PartialAnswer(prefix, input...); got.Content != want {
			t.Errorf("PartialAnswer(%q) = %q, want %q", prefix, got.Content, want)
		}
	}
}

func TestAnswerKeepsSingleSelectedTextPair(t *testing.T) {
	template := "<selected_text>{text}</selected_text>"
	checkAnswer(t, template, template, "", "Write a selected_text template")
	checkAnswer(t, template, template, "see you tmrw", "write a selected_text template\ndo it", "write a selected_text template")
	checkAnswer(t, template, template, "see you tmrw", "这个模板太啰嗦，重新写一个 selected_text 模板\n照做", "这个模板太啰嗦，重新写一个 selected_text 模板")
}

func TestAnswerTitleDropsWholeInstructionPair(t *testing.T) {
	raw, _ := json.Marshal(Answer{Title: "<instruction>Greeting</instruction>", Content: "Hello"})
	for _, input := range [][]string{{"write a greeting"}, {"Draft an <instruction> template"}} {
		if got, ok := ParseAnswer(string(raw), input...); !ok || got.Title != "Greeting" {
			t.Errorf("ParseAnswer title = %q, want Greeting", got.Title)
		}
		if got := PartialAnswer(string(raw), input...); got.Title != "Greeting" {
			t.Errorf("PartialAnswer title = %q, want Greeting", got.Title)
		}
	}
}

func checkAnswer(t *testing.T, out, want string, input ...string) {
	t.Helper()
	if len(input) > 1 {
		input = append([]string{input[0] + "\n" + input[1]}, input[1:]...)
	}
	checkAnswerInputs(t, out, want, input...)
}

func checkAnswerInputs(t *testing.T, out, want string, input ...string) {
	t.Helper()
	if got := CleanRewrite(out, input, false); got != want {
		t.Errorf("CleanRewrite(%q) = %q, want %q", out, got, want)
	}
	b, _ := json.Marshal(Answer{Title: "T", Content: out})
	if got, ok := ParseAnswer(string(b), input...); !ok || got.Content != want {
		t.Errorf("ParseAnswer(%s) = %q, want %q", b, got.Content, want)
	}
	if got := PartialAnswer(string(b), input...); got.Content != want {
		t.Errorf("PartialAnswer(%s) = %q, want %q", b, got.Content, want)
	}
	if got := PartialAnswer(out, input...); got.Content != want {
		t.Errorf("PartialAnswer(%q) = %q, want %q", out, got.Content, want)
	}
}

// TestAnswerDropsRewriteTagsChangelog keeps the user-visible fix in the release notes.
func TestAnswerDropsRewriteTagsChangelog(t *testing.T) {
	data, err := os.ReadFile("../../CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range []string{
		"Rewrite answers no longer show echoed input tags",
		"改写回答不再显示重复的输入标签",
	} {
		// A release moves the entry out of Unreleased, so any section counts.
		if !strings.Contains(string(data), entry) {
			t.Errorf("changelog missing %q", entry)
		}
	}
}
