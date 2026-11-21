package compose

import (
	"strconv"
	"strings"
	"testing"
)

func assertAnswerCleanup(t *testing.T, text, input, want string) {
	t.Helper()
	if got := Clean(text, input); got != want {
		t.Errorf("Clean(%q, %q) = %q, want %q", text, input, got, want)
	}
	if got := PartialClean(text, input); got != want {
		t.Errorf("PartialClean(%q, %q) = %q, want %q", text, input, got, want)
	}
	structured := `{"title":"T","content":` + strconv.Quote(text) + `}`
	wantAnswer := Answer{Title: "T", Content: want}
	if got, ok := ParseAnswer(structured, input); !ok || got != wantAnswer {
		t.Errorf("ParseAnswer(%q, %q) = %+v, %v, want %+v", structured, input, got, ok, wantAnswer)
	}
	if got := PartialAnswer(structured, input); got != wantAnswer {
		t.Errorf("PartialAnswer(%q, %q) = %+v, want %+v", structured, input, got, wantAnswer)
	}
}

func assertAnswerCleanupWithPrevious(t *testing.T, text, previous, want string) {
	t.Helper()
	inputs := []string{"x", previous}
	if got := CleanWithInputs(text, inputs); got != want {
		t.Errorf("CleanWithInputs(%q, %q) = %q, want %q", text, previous, got, want)
	}
	if got := PartialCleanWithInputs(text, inputs); got != want {
		t.Errorf("PartialCleanWithInputs(%q, %q) = %q, want %q", text, previous, got, want)
	}
	structured := `{"title":"T","content":` + strconv.Quote(text) + `}`
	wantAnswer := Answer{Title: "T", Content: want}
	if got, ok := ParseAnswer(structured, inputs...); !ok || got != wantAnswer {
		t.Errorf("ParseAnswer(%q, %q) = %+v, %v, want %+v", structured, previous, got, ok, wantAnswer)
	}
	if got := PartialAnswer(structured, inputs...); got.Title != wantAnswer.Title || strings.TrimSpace(got.Content) != wantAnswer.Content {
		t.Errorf("PartialAnswer(%q, %q) = %+v, want %+v", structured, previous, got, wantAnswer)
	}
}

func TestAnswerCleanupWithPreviousInput(t *testing.T) {
	for _, tc := range []struct{ name, text, previous, want string }{
		{name: "real tool call", text: `Visible<|tool_call|>{"name":"view_image"}`, previous: "explain <|tool_call|>", want: "Visible"},
		{name: "real role marker", text: "Visible<|im_start|>user", previous: "explain <|im_start|>", want: "Visible"},
		{name: "spaced tool call", text: `Visible <|tool_call|> {"name":"view_image"}`, previous: "explain <|tool_call|>", want: "Visible"},
		{name: "spaced role marker", text: "Visible <|im_start|> user", previous: "explain <|im_start|>", want: "Visible"},
		{name: "padded trailing token", text: "Visible <|im_end|> ", previous: "<|im_end|>", want: "Visible"},
		{name: "unclosed real thinking block", text: "Shorter. <think>private", previous: "<think>", want: "Shorter."},
		{name: "spaced real thinking block", text: "Visible <think> private</think>", previous: "explain <think>", want: "Visible"},
		{name: "unclosed spaced real thinking block", text: "Visible <think> private", previous: "explain <think>", want: "Visible"},
		{name: "previous end token before a role marker", text: "Visible <|im_end|> <|im_start|>user", previous: "explain <|im_end|>", want: "Visible"},
		{name: "literal opener before real block", text: "The <think> tag <think>private</think>", previous: "<think>", want: "The"},
		{name: "two inline literals", text: "Use <|im_end|> <think> tag", previous: "explain <|im_end|> and <think>", want: "Use"},
		{name: "literal channel opener", text: "The <|channel> tag wraps reasoning.", previous: "explain <|channel>", want: "The"},
		{name: "literal tool call opener", text: "The <|tool_call|> tag starts a call.", previous: "explain <|tool_call|>", want: "The <|tool_call|> tag starts a call."},
		{name: "literal thinking opener at answer start", text: "<think> tag wraps reasoning.", previous: "explain <think>", want: ""},
		{name: "literal thinking opener after a newline", text: "Markers:\n<think> tag opens reasoning.", previous: "explain <think>", want: "Markers:"},
		{name: "literal thinking closer at answer start", text: "</think> tag ends reasoning.", previous: "explain </think>", want: "</think> tag ends reasoning."},
		{name: "literal thinking tag pair", text: "The <think> tag and </think> tag are markers.", previous: "explain <think> and </think>", want: "The <think> tag and </think> tag are markers."},
		{name: "literal thinking pair without prose hints", text: "Keep <think>this</think>", previous: "Explain <think> and </think> literally", want: "Keep <think>this</think>"},
		{name: "real close after a literal thinking tag pair", text: "The <think> tag and </think> tag are markers.</think>", previous: "explain <think> and </think>", want: "The <think> tag and </think> tag are markers."},
		{name: "literal thinking opener after an orphan close", text: "reasoning</think> <think> tag", previous: "explain <think>", want: "reasoning"},
		{name: "literal tool call after an orphan close", text: "reasoning</think> <|tool_call|> tag", previous: "explain </think> and <|tool_call|>", want: "reasoning"},
		{name: "literal trailing token only", text: "<|im_end|>", previous: "<|im_end|>", want: "<|im_end|>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertAnswerCleanupWithPrevious(t, tc.text, tc.previous, tc.want)
		})
	}
}

func TestPartialAnswerWithPreviousInput(t *testing.T) {
	inputs := []string{"x", "<think>"}
	text := "The <think> tag <think>private</think>"
	for index := 1; index <= len(text); index++ {
		if got := PartialCleanWithInputs(text[:index], inputs); strings.Contains(got, "private") {
			t.Fatalf("at %d: reasoning leaked in %q", index, got)
		}
	}
	for _, tc := range []struct{ text, previous, want string }{
		{text: "Shorter.<|im_end", previous: "<|im_end|>", want: "Shorter."},
		{text: "The <thi", previous: "<think>", want: "The"},
	} {
		if got := PartialCleanWithInputs(tc.text, []string{"x", tc.previous}); got != tc.want {
			t.Errorf("PartialCleanWithInputs(%q, %q) = %q, want %q", tc.text, tc.previous, got, tc.want)
		}
	}
}

func TestAnswerCleanupPreservesLiteralControlMarkup(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input string
		text  string
	}{
		{"end token", "Explain <|im_end|> literally", "Use <|im_end|> literally"},
		{"thinking tags", "Explain <think> and </think> literally", "Keep <think>this</think> whole"},
		{"dictated tool call token", "Use the <|tool_call|> token carefully.", "Use the <|tool_call|> token carefully."},
		{"Chinese tool call token", "解释 <|tool_call|> 这个标记。", "解释 <|tool_call|> 这个标记。"},
		{"Chinese tool call token without spaces", "解释<|tool_call|>这个标记。", "解释<|tool_call|>这个标记。"},
		{"Chinese tool call token with right space", "解释<|tool_call|> 这个标记。", "解释<|tool_call|> 这个标记。"},
		{"Chinese tool call token with left space", "解释 <|tool_call|>这个标记。", "解释 <|tool_call|>这个标记。"},
		{"English tool call token at sentence end", "Use the <|tool_call|> token.", "Use the <|tool_call|> token."},
		{"Chinese tool call token after a colon", "注意：<|tool_call|> 标记。", "注意：<|tool_call|> 标记。"},
		{"Chinese tool call token after a comma", "例如，<|tool_call|> 这个标记。", "例如，<|tool_call|> 这个标记。"},
		{"plural English tool call token", "Use the <|tool_call|> tokens carefully.", "Use the <|tool_call|> tokens carefully."},
		{"English cue before the tool call token", "The token <|tool_call|> stays.", "The token <|tool_call|> stays."},
		{"Chinese label after the tool call token", "说明 <|tool_call|> 这个标签。", "说明 <|tool_call|> 这个标签。"},
		{"parenthesized English cue", "Use the <|tool_call|> (token) carefully.", "Use the <|tool_call|> (token) carefully."},
		{"bracketed documentation after the tool call token", "Use the <|tool_call|> token, see [docs].", "Use the <|tool_call|> token, see [docs]."},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Clean(tc.text, tc.input); got != tc.text {
				t.Errorf("Clean(%q, %q) = %q", tc.text, tc.input, got)
			}
			if got := PartialClean(tc.text, tc.input); got != tc.text {
				t.Errorf("PartialClean(%q, %q) = %q", tc.text, tc.input, got)
			}
			if got := CleanOutput(tc.text, tc.input); got != tc.text {
				t.Errorf("CleanOutput(%q, %q) = %q", tc.text, tc.input, got)
			}
			if got := PartialOutput(tc.text, tc.input); got != tc.text {
				t.Errorf("PartialOutput(%q, %q) = %q", tc.text, tc.input, got)
			}
			structured := `{"title":"T","content":"` + tc.text + `"}`
			want := Answer{Title: "T", Content: tc.text}
			if got, ok := ParseAnswer(structured, tc.input); !ok || got != want {
				t.Errorf("ParseAnswer(%q, %q) = %+v, %v", structured, tc.input, got, ok)
			}
			if got := PartialAnswer(structured, tc.input); got != want {
				t.Errorf("PartialAnswer(%q, %q) = %+v", structured, tc.input, got)
			}
		})
	}
}

func TestAnswerCleanupPreservesLiteralToolMarkerBeforeTurnEnd(t *testing.T) {
	const said = "Use the <|tool_call|> token carefully."
	for _, end := range []string{"<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>"} {
		t.Run(end, func(t *testing.T) {
			assertAnswerCleanup(t, said+end, said, said)
			wrapped := "<output>" + said + "</output>" + end
			if got := CleanOutput(wrapped, said); got != said {
				t.Errorf("CleanOutput(%q) = %q, want %q", wrapped, got, said)
			}
			if got := PartialOutput(wrapped, said); got != said {
				t.Errorf("PartialOutput(%q) = %q, want %q", wrapped, got, said)
			}
		})
	}
	assertAnswerCleanup(t, `Use the <|tool_call|> "token" carefully.`, `Use the <|tool_call|> "token" carefully.`, `Use the <|tool_call|> "token" carefully.`)
}

func TestFinalAnswerKeepsPartialControlMarker(t *testing.T) {
	for _, tc := range []struct{ text, want string }{
		{"Check that a<c", "Check that a<c"},
		{"Answer<|tool_", "Answer<|tool_"},
	} {
		if got := Clean(tc.text, "x"); got != tc.want {
			t.Errorf("Clean(%q) = %q, want %q", tc.text, got, tc.want)
		}
		structured := `{"title":"T","content":"` + tc.text + `"}`
		want := Answer{Title: "T", Content: tc.want}
		if got, ok := ParseAnswer(structured); !ok || got != want {
			t.Errorf("ParseAnswer(%q) = %+v, %v, want %+v", structured, got, ok, want)
		}
	}
	if got := PartialClean("Answer<|tool_", "x"); got != "Answer" {
		t.Errorf("PartialClean held back %q", got)
	}
	if got := PartialAnswer("Answer<|tool_").Content; got != "Answer" {
		t.Errorf("PartialAnswer held back %q", got)
	}
}

func TestAnswerCleanupKeepsTextAfterOrphanThinkClose(t *testing.T) {
	for _, tc := range []struct{ text, want string }{
		{"推理…</think>正式回答", "正式回答"},
		{"<think>推理…</think>正式回答", "正式回答"},
		{"Visible</think>", "Visible"},
	} {
		if got := Clean(tc.text, "x"); got != tc.want {
			t.Errorf("Clean(%q) = %q, want %q", tc.text, got, tc.want)
		}
		if got := PartialClean(tc.text, "x"); got != tc.want {
			t.Errorf("PartialClean(%q) = %q, want %q", tc.text, got, tc.want)
		}
		structured := `{"title":"T","content":"` + tc.text + `"}`
		want := Answer{Title: "T", Content: tc.want}
		if got, ok := ParseAnswer(structured); !ok || got != want {
			t.Errorf("ParseAnswer(%q) = %+v, %v, want %+v", structured, got, ok, want)
		}
		if got := PartialAnswer(structured); got != want {
			t.Errorf("PartialAnswer(%q) = %+v, want %+v", structured, got, want)
		}
	}
}

func TestAnswerCleanupStripsTurnEndLeak(t *testing.T) {
	// A turn-end marker followed by a fresh sentence is the model
	// writing into the next turn: cut there, as with a role marker.
	const leak = "Visible <|im_end|> Next turn."
	if got := Clean(leak, "x"); got != "Visible" {
		t.Errorf("Clean(%q) = %q, want %q", leak, got, "Visible")
	}
	if got := PartialClean(leak, "x"); got != "Visible" {
		t.Errorf("PartialClean(%q) = %q, want %q", leak, got, "Visible")
	}
	structured := `{"title":"T","content":` + strconv.Quote(leak) + `}`
	if got, ok := ParseAnswer(structured, "x"); !ok || got.Content != "Visible" {
		t.Errorf("ParseAnswer(%q) = %+v, %v", structured, got, ok)
	}
	if got := PartialAnswer(structured, "x"); got.Content != "Visible " {
		t.Errorf("PartialAnswer(%q) = %+v", structured, got)
	}
	// Fragments and continuations without sentence shape stay literal.
	assertAnswerCleanup(t, "把 <|im_end|> 放在句尾", "x", "把 <|im_end|> 放在句尾")
}

func TestAnswerCleanupStripsReasoningBeforeFinalAnswer(t *testing.T) {
	// A complete thinking pair the input never quoted is reasoning,
	// whatever words follow the open tag; the final answer stays.
	const reasoning = "The model emits <think> before it checks the request</think> Final answer."
	for name, got := range map[string]string{
		"Clean":         Clean(reasoning, "x"),
		"PartialClean":  PartialClean(reasoning, "x"),
		"CleanOutput":   CleanOutput(reasoning, "x"),
		"PartialOutput": PartialOutput(reasoning, "x"),
	} {
		if strings.Contains(got, "checks the request") {
			t.Errorf("%s preserved the reasoning as prose: %q", name, got)
		}
		if !strings.Contains(got, "Final answer.") {
			t.Errorf("%s lost the visible answer: %q", name, got)
		}
	}
	// Unclosed prose is intentionally cut under the stricter contract.
	assertAnswerCleanupWithPrevious(t, "Put <think> at the top of the prompt. That is all.", "history", "Put")
	assertAnswerCleanupWithPrevious(t, "在提示词开头写 <think> 就可以让模型先思考。", "history", "在提示词开头写")
}

func TestAnswerCleanupPreservesInlineControlToken(t *testing.T) {
	const controlRun = "Visible <|im_end|> <|im_end|> <|im_start|>user"
	if got := Clean(controlRun, "x"); got != "Visible" {
		t.Errorf("Clean(%q) = %q", controlRun, got)
	}
	if got := PartialClean(controlRun, "x"); got != "Visible" {
		t.Errorf("PartialClean(%q) = %q", controlRun, got)
	}
	structuredRun := `{"title":"T","content":"` + controlRun + `"}`
	if got, ok := ParseAnswer(structuredRun); !ok || got.Content != "Visible" {
		t.Errorf("ParseAnswer(%q) = %+v, %v", structuredRun, got, ok)
	}
	if got := PartialAnswer(structuredRun); got.Content != "Visible " {
		t.Errorf("PartialAnswer(%q) = %+v", structuredRun, got)
	}
	assertAnswerCleanup(t, "a <|im_end|> <|im_end|> x", "x", "a <|im_end|> <|im_end|> x")
	assertAnswerCleanup(t, "a <|im_end|> x <|im_end|> y <|im_end|> z", "x", "a <|im_end|> x <|im_end|> y <|im_end|> z")
	assertAnswerCleanup(t, "Use <|im_end|> <|im_start|> is a start marker", "Explain <|im_start|> literally", "Use <|im_end|> <|im_start|> is a start marker")
	assertAnswerCleanup(t, "reason</think><|im_start", "Explain <|im_start|> literally", "<|im_start")
	assertAnswerCleanup(t, "reason</think><|im_end|>", "Explain <|im_end|> literally", "<|im_end|>")
	const text = "把 <|im_end|> 放在句尾"
	if got := Clean(text, "x"); got != text {
		t.Errorf("Clean(%q) = %q", text, got)
	}
	if got := PartialClean(text, "x"); got != text {
		t.Errorf("PartialClean(%q) = %q", text, got)
	}
	structured := `{"title":"T","content":"` + text + `"}`
	want := Answer{Title: "T", Content: text}
	if got, ok := ParseAnswer(structured); !ok || got != want {
		t.Errorf("ParseAnswer(%q) = %+v, %v, want %+v", structured, got, ok, want)
	}
	if got := PartialAnswer(structured); got != want {
		t.Errorf("PartialAnswer(%q) = %+v, want %+v", structured, got, want)
	}
}

func TestAnswerCleanupLiteralMarkerBeforeRealMarker(t *testing.T) {
	assertAnswerCleanup(t, "Use <|im_end|> literally<|im_end|>", "x", "Use <|im_end|> literally")
	const text = "Visible <|im_end|> "
	if got := Clean(text, "x"); got != "Visible" {
		t.Errorf("Clean(%q) = %q", text, got)
	}
	if got := PartialClean(text, "x"); got != "Visible" {
		t.Errorf("PartialClean(%q) = %q", text, got)
	}
	structured := `{"title":"T","content":"` + text + `"}`
	if got, ok := ParseAnswer(structured); !ok || got.Content != "Visible" {
		t.Errorf("ParseAnswer(%q) = %+v, %v", structured, got, ok)
	}
	if got := PartialAnswer(structured); got.Content != "Visible " {
		t.Errorf("PartialAnswer(%q) = %+v", structured, got)
	}
}

func TestAnswerCleanupPreservesLiteralThinkingPair(t *testing.T) {
	assertAnswerCleanup(t, "Keep <think>this</think> whole", "Explain <think> literally", "Keep <think>this</think> whole")
	assertAnswerCleanup(t, "Keep <think>this</think>", "Explain <think> and </think> literally", "Keep <think>this</think>")
	assertAnswerCleanup(t, "The <think> tag and </think> tag are markers.</think>", "Explain <think> and </think> literally", "The <think> tag and </think> tag are markers.")
	assertAnswerCleanup(t, "The <think> tag wraps reasoning.</think>", "Explain <think> and </think> literally", "The <think> tag wraps reasoning.")
	assertAnswerCleanup(t, "Keep <|channel>this<channel|>", "Explain <|channel> and <channel|> literally", "Keep <|channel>this<channel|>")
}

func TestAnswerCleanupOrphanThinkCloseWithControlSuffix(t *testing.T) {
	assertAnswerCleanup(t, "Visible</think><|im_end|>", "x", "Visible")
	assertAnswerCleanup(t, "Visible</think>Answer", "x", "Answer")
	assertAnswerCleanup(t, "Visible</think><|im_", "x", "Visible")
	assertAnswerCleanup(t, "Visible</think><", "x", "Visible")
	const nextTurn = "Visible <|im_end|> <|im_start|>user"
	if got := Clean(nextTurn, "x"); got != "Visible" {
		t.Errorf("Clean(%q) = %q", nextTurn, got)
	}
	if got := PartialClean(nextTurn, "x"); got != "Visible" {
		t.Errorf("PartialClean(%q) = %q", nextTurn, got)
	}
	structured := `{"title":"T","content":"` + nextTurn + `"}`
	if got, ok := ParseAnswer(structured); !ok || got.Content != "Visible" {
		t.Errorf("ParseAnswer(%q) = %+v, %v", structured, got, ok)
	}
	if got := PartialAnswer(structured); got.Content != "Visible " {
		t.Errorf("PartialAnswer(%q) = %+v", structured, got)
	}
}

func TestAnswerControlTokenVariants(t *testing.T) {
	for _, tc := range []struct{ name, text, want string }{
		{"Llama end", "Visible<|eot_id|>", "Visible"},
		{"Gemma end", "Visible<end_of_turn>", "Visible"},
		{"assistant prefix without newline", "<|im_start|>assistant Answer<|im_end|>", "Answer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Clean(tc.text, "x"); got != tc.want {
				t.Errorf("Clean(%q) = %q, want %q", tc.text, got, tc.want)
			}
			if got := PartialClean(tc.text, "x"); got != tc.want {
				t.Errorf("PartialClean(%q) = %q, want %q", tc.text, got, tc.want)
			}
			structured := `{"title":"T","content":"` + tc.text + `"}`
			want := Answer{Title: "T", Content: tc.want}
			if got, ok := ParseAnswer(structured); !ok || got != want {
				t.Errorf("ParseAnswer(%q) = %+v, %v, want %+v", structured, got, ok, want)
			}
			if got := PartialAnswer(structured); got != want {
				t.Errorf("PartialAnswer(%q) = %+v, want %+v", structured, got, want)
			}
		})
	}
}

func TestAnswerCleanupKeepsProseControlTokens(t *testing.T) {
	for _, tc := range []struct{ name, text, want string }{
		{name: "ChatML markers in an explanation", text: "ChatML uses <|im_start|> to begin a turn and <|im_end|> to end it.", want: "ChatML uses"},
		{name: "thinking marker before a description", text: "The model emits <think> before it reasons.", want: "The model emits"},
		{name: "backtick quoted closing marker", text: "Reasoning ends with `</think>`. Then the answer follows.", want: "Reasoning ends with `</think>`. Then the answer follows."},
		{name: "full width quoted closing marker", text: "中文用“</think>”表示结束。然后显示回答。", want: "中文用“</think>”表示结束。然后显示回答。"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertAnswerCleanupWithPrevious(t, tc.text, "history", tc.want)
			if got := CleanOutput(tc.text, "x"); got != tc.want {
				t.Errorf("CleanOutput(%q) = %q, want %q", tc.text, got, tc.want)
			}
			if got := PartialOutput(tc.text, "x"); got != tc.want && got != tc.want+" " {
				t.Errorf("PartialOutput(%q) = %q, want %q or %q", tc.text, got, tc.want, tc.want+" ")
			}
		})
	}
	assertAnswerCleanupWithPrevious(t, "The model emits <think> before it reasons.</think>", "history", "The model emits")
}

func TestAnswerCleanupKeepsEverydayControlTokenProse(t *testing.T) {
	for _, tc := range []struct{ name, text, want, previous string }{
		{name: "role marker in a sentence", text: "Each turn starts with <|im_start|> and a role name.", want: "Each turn starts with"},
		{name: "tool marker in a sentence", text: "The marker <|tool_call|> starts a tool call.", want: "The marker"},
		{name: "channel marker in a sentence", text: "Gemma uses <|channel> for its thoughts. Ok.", want: "Gemma uses"},
		{name: "thinking marker in Chinese prose", text: "在提示词里加上 <think> 模型就会先推理。完。", want: "在提示词里加上"},
		{name: "thinking marker at the start of prose", text: "<think> is the tag that starts reasoning.", want: ""},
		{name: "thinking pair requested as an example", text: "Wrap it as <think>step one</think> and go on.", want: "Wrap it as <think>step one</think> and go on.", previous: "how do I write a <think> block?"},
		{name: "closing marker in a sentence", text: "Reasoning ends with </think> and then the answer follows.", want: "Reasoning ends with </think> and then the answer follows."},
		{name: "prompting instruction", text: "To start, put <think> in your prompt and the model reasons first. That is all.", want: "To start, put"},
		{name: "reworded prompting instruction", text: "To start, add <think> to your prompt and the model reasons first. That is all.", want: "To start, add"},
		{name: "thinking marker at the prompt top", text: "Put <think> at the top of the prompt. That is all.", want: "Put"},
		{name: "thinking marker before a question", text: "You can write <think> before the question to make it reason.", want: "You can write"},
		{name: "thinking marker in reworded Chinese prose", text: "在提示词开头写 <think> 就可以让模型先思考。", want: "在提示词开头写"},
		{name: "channel marker in reworded prose", text: "Gemma writes <|channel> before its thoughts. Ok.", want: "Gemma writes"},
		{name: "closing marker in reworded prose", text: "The reasoning stops at </think> and the answer comes next.", want: "The reasoning stops at </think> and the answer comes next."},
		{name: "thinking pair in reworded prose", text: "Write it as <think>step one</think> and continue.", want: "Write it as <think>step one</think> and continue.", previous: "how do I write a <think> block?"},
		{name: "tool marker in reworded prose", text: "A tool call begins with <|tool_call|> in this template.", want: "A tool call begins with"},
		{name: "channel marker ahead of thoughts", text: "Gemma writes <|channel> ahead of its thoughts. Ok.", want: "Gemma writes"},
		{name: "thinking marker prior to a question", text: "You should write <think> prior to the question.", want: "You should write"},
		{name: "tool marker inside a template", text: "A tool call begins with <|tool_call|> inside this template.", want: "A tool call begins with"},
		{name: "closing marker before a comma", text: "The reasoning stops at </think>, and the answer comes next.", want: "The reasoning stops at </think>, and the answer comes next."},
		{name: "thinking marker in fresh Chinese prose", text: "在提示词开头加 <think> 会让模型先思考。", want: "在提示词开头加"},
		{name: "thinking marker near the prompt top", text: "Just add <think> near the top of your prompt.", want: "Just add"},
		{name: "thinking pair before then", text: "Write it as <think>step one</think> then continue.", want: "Write it as <think>step one</think> then continue.", previous: "how do I write a <think> block?"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			previous := tc.previous
			if previous == "" {
				previous = "history"
			}
			assertAnswerCleanupWithPrevious(t, tc.text, previous, tc.want)
			input := "q"
			if tc.previous != "" {
				input = tc.previous
			}
			if got := CleanOutput(tc.text, input); got != tc.want {
				t.Errorf("CleanOutput(%q) = %q, want %q", tc.text, got, tc.want)
			}
			if got := PartialOutput(tc.text, input); got != tc.want && got != tc.want+" " {
				t.Errorf("PartialOutput(%q) = %q, want %q or %q", tc.text, got, tc.want, tc.want+" ")
			}
		})
	}
}

func TestAnswerCleanupDoesNotPreserveReasoningAsProse(t *testing.T) {
	assertStripped := func(text, want string) {
		t.Helper()
		if got := Clean(text, "x"); got != want {
			t.Errorf("Clean(%q) = %q, want %q", text, got, want)
		}
		if got := PartialClean(text, "x"); got != want {
			t.Errorf("PartialClean(%q) = %q, want %q", text, got, want)
		}
		if got := CleanOutput(text, "x"); got != want {
			t.Errorf("CleanOutput(%q) = %q, want %q", text, got, want)
		}
		if got := PartialOutput(text, "x"); got != want+" " {
			t.Errorf("PartialOutput(%q) = %q, want %q", text, got, want+" ")
		}
		structured := `{"title":"T","content":` + strconv.Quote(text) + `}`
		if got, ok := ParseAnswer(structured, "x"); !ok || got.Content != want {
			t.Errorf("ParseAnswer(%q) = %+v, %v, want %q", structured, got, ok, want)
		}
		if got := PartialAnswer(structured, "x"); got.Content != want+" " {
			t.Errorf("PartialAnswer(%q) = %+v, want %q", structured, got, want+" ")
		}
	}
	assertStripped("Visible <think> before private reasoning", "Visible")
	assertStripped("Visible <think> before it checks the request</think>", "Visible")
	assertStripped("The model emits tokens. Visible <think> before it checks the request</think>", "The model emits tokens. Visible")
	assertStripped("The model emits tokens, Visible <think> before it checks the request</think>", "The model emits tokens, Visible")
	assertStripped("The model emits tokens：Visible <think> before it checks the request</think>", "The model emits tokens：Visible")
	assertAnswerCleanup(t, "reasoning </think> Final answer", "x", "Final answer")
	const text = "Visible <think> before private reasoning"
	for index := 1; index <= len(text); index++ {
		if got := PartialClean(text[:index], "x"); strings.Contains(got, "private") {
			t.Fatalf("at %d: reasoning leaked in %q", index, got)
		}
	}
}

func TestAnswerCleanupStripsStructuralMarkupFromLiteralInput(t *testing.T) {
	for _, tc := range []struct{ name, text, input string }{
		{name: "ChatML wrapper", text: "<|im_start|>assistant\nVisible<|im_end|>", input: "Explain <|im_start|> and <|im_end|>"},
		{name: "JSON tool call", text: `Visible<|tool_call|>{"name":"view_image"}`, input: "Explain <|tool_call|>"},
		{name: "plain tool payload", text: "Visible<|tool_call|>internal", input: "Explain <|tool_call|>"},
		{name: "trailing end token", text: "Visible<|im_end|>", input: "Explain <|im_end|>"},
		{name: "thinking block at content start", text: "<think>secret plan</think>Visible", input: "Explain <think>"},
		{name: "thinking block after assistant prefix", text: "<|im_start|>assistant\n<think>secret plan</think>Visible", input: "Explain <think>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertAnswerCleanup(t, tc.text, tc.input, "Visible")
			if got := CleanOutput(tc.text, tc.input); got != "Visible" {
				t.Errorf("CleanOutput(%q) = %q, want Visible", tc.text, got)
			}
			if got := PartialOutput(tc.text, tc.input); got != "Visible" {
				t.Errorf("PartialOutput(%q) = %q, want Visible", tc.text, got)
			}
		})
	}
	for _, tc := range []struct{ text, want string }{
		{"Visible <|tool_call|> search images now", "Visible"},
		{"Visible <|tool_call|> search images now. 这个标记会显示。", "Visible"},
		{"Visible <|tool_call|> token carefully. {\"name\":\"view_image\"}", "Visible"},
		{"Use the <|tool_call|> token carefully. {\"name\":\"view_image\"}", "Use the"},
	} {
		assertAnswerCleanupWithPrevious(t, tc.text, "explain <|tool_call|>", tc.want)
		if got := CleanOutput(tc.text, "explain <|tool_call|>"); got != tc.want {
			t.Errorf("CleanOutput(%q) = %q, want %q", tc.text, got, tc.want)
		}
		if got := PartialOutput(tc.text, "explain <|tool_call|>"); got != tc.want && got != tc.want+" " {
			t.Errorf("PartialOutput(%q) = %q, want %q or %q", tc.text, got, tc.want, tc.want+" ")
		}
	}
}

func TestPartialAnswerStripsLeadingThinkingWithLiteralInput(t *testing.T) {
	plain := "<think>secret plan</think>Visible"
	for index := 1; index <= len(plain); index++ {
		if got := PartialClean(plain[:index], "Explain <think>"); strings.Contains(got, "secret") {
			t.Fatalf("at %d: reasoning leaked in %q", index, got)
		}
	}
	structured := `{"title":"T","content":"<think>secret plan</think>Visible"}`
	for index := 1; index <= len(structured); index++ {
		if got := PartialAnswer(structured[:index], "Explain <think>"); strings.Contains(got.Content, "secret") {
			t.Fatalf("at %d: reasoning leaked in %q", index, got.Content)
		}
	}
}

func TestAnswerCleanupKeepsStructuralControlMarkup(t *testing.T) {
	assertAnswerCleanup(t, "Visible<think>secret plan</think> Final answer.", "x", "Visible Final answer.")
	assertAnswerCleanup(t, "Visible<|channel>secret plan<channel|> Final answer.", "x", "Visible Final answer.")
	assertAnswerCleanupWithPrevious(t, "Visible <|im_start|> user\nNext turn.", "history", "Visible")
	assertAnswerCleanupWithPrevious(t, `Visible <|tool_call|> view_image {"path":"secret.png"}`, "history", "Visible")
	assertAnswerCleanup(t, "secret plan </think> Final answer.", "x", "Final answer.")
	assertAnswerCleanupWithPrevious(t, "Visible <think>private</think>", "Example <think>public</think>", "Visible")
	assertAnswerCleanup(t, "Keep <think> this</think>", "Keep <think> this</think>", "Keep <think> this</think>")
	assertAnswerCleanup(t, "<think> The user is asking for medical advice. I should avoid diagnosis", "x", "")
	assertAnswerCleanup(t, "<think> to inspect private data", "x", "")
	assertAnswerCleanupWithPrevious(t, `Visible <|tool_call|> tokens ["/private/secret.png"]`, "history", "Visible")
	assertAnswerCleanupWithPrevious(t, "ChatML places <|im_start|> before every role.", "history", "ChatML places")
	assertAnswerCleanup(t, "The <think> tag pairs with </think> on every answer.", "Explain <think> and </think>", "The <think> tag pairs with </think> on every answer.")
	assertAnswerCleanupWithPrevious(t, "Visible <think>The user wants private medical advice</think>", "history", "Visible")
	assertAnswerCleanup(t, "The user asked which confidential details to respond with </think> Final answer.", "x", "Final answer.")
	assertAnswerCleanup(t, "I will respond as <think>The user needs confidential medical guidance</think> a numbered list.", "x", "I will respond as  a numbered list.")
	assertAnswerCleanupWithPrevious(t, `Visible <|tool_call|> view_image ["/private/secret.png"]`, "history", "Visible")
	assertAnswerCleanupWithPrevious(t, "Visible <|tool_call|> view_image secret.png", "history", "Visible")
	assertAnswerCleanupWithPrevious(t, `The marker <|tool_call|> starts a tool call. Example: {"name":"view_image"}.`, "history", "The marker")
	if got := PartialClean("Visible <thi", "Explain <think>"); got != "Visible" {
		t.Errorf("PartialClean exposed a marker fragment: %q", got)
	}
	if got := PartialClean("Visible <|im_end", "Explain <|im_end|>"); got != "Visible" {
		t.Errorf("PartialClean exposed a control fragment: %q", got)
	}
	plain := "Visible<think>secret plan</think> Final answer."
	for index := 1; index <= len(plain); index++ {
		if got := PartialClean(plain[:index], "x"); strings.Contains(got, "secret") {
			t.Fatalf("at %d: reasoning leaked in %q", index, got)
		}
	}
}

func TestAnswerCleanupStripsClosedReasoningBeforeLowercaseAnswer(t *testing.T) {
	const text = "Sure! <think>the user's note says the password is hunter2</think> here is the summary."
	const want = "Sure!  here is the summary."
	assertAnswerCleanupWithPrevious(t, text, "history", want)
	if got := CleanOutput(text, "x"); got != want {
		t.Errorf("CleanOutput(%q) = %q, want %q", text, got, want)
	}
	if got := PartialOutput(text, "x"); got != want {
		t.Errorf("PartialOutput(%q) = %q, want %q", text, got, want)
	}
	const example = "Wrap it as <think>step one</think> and go on."
	assertAnswerCleanup(t, example, "how do I write a <think> block?", example)
	assertAnswerCleanupWithPrevious(t, example, "how do I write a <think> block?", example)
}

func TestAnswerCleanupRejectsUnclosedAndStructuredMarkup(t *testing.T) {
	for _, tc := range []struct{ name, text, want string }{
		{name: "unclosed reasoning with private data", text: "Visible <think> to inspect private data, I will read the user's file.", want: "Visible"},
		{name: "unclosed reasoning with a secret", text: "Sure. <think> in this case the user wants the secret; I must refuse.", want: "Sure."},
		{name: "leading unclosed reasoning", text: "<think> is the user testing me? I should hide the key.", want: ""},
		{name: "tool payload", text: `Done <|tool_call|> for get_weather {"city":"x"}`, want: "Done"},
		{name: "accepted unclosed prose cut", text: "Models print <think> when they reason. That is all.", want: "Models print"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertAnswerCleanupWithPrevious(t, tc.text, "history", tc.want)
			if got := CleanOutput(tc.text, "x"); got != tc.want {
				t.Errorf("CleanOutput(%q) = %q, want %q", tc.text, got, tc.want)
			}
			if got := PartialOutput(tc.text, "x"); got != tc.want && got != tc.want+" " {
				t.Errorf("PartialOutput(%q) = %q, want %q or %q", tc.text, got, tc.want, tc.want+" ")
			}
		})
	}
}

func TestAnswerCleanupThinkingClose(t *testing.T) {
	for _, tc := range []struct{ name, text, want string }{
		{"answer after thinking close", "reasoning here</think>Final answer", "Final answer"},
		{"answer after newline-wrapped thinking close", "reasoning\n</think>\n\nFinal answer", "Final answer"},
		{"orphan thinking close before a newline", "reasoning </think>\nFinal answer", "Final answer"},
		{"answer after channel close", "reasoning here<channel|>Final answer", "Final answer"},
		{"answer after duplicate thinking closes", "reasoning</think></think>Final answer", "Final answer"},
		{"answer after mixed thinking closes", "reasoning</think><channel|>Final answer", "Final answer"},
		{"answer after reversed thinking closes", "reasoning<channel|></think>Final answer", "Final answer"},
		{"answer after mixed thinking block", "reasoning</think><|channel>private<channel|>Final answer", "Final answer"},
		{"trailing thinking close", "Visible</think>", "Visible"},
		{"thinking close before control ending", "Visible</think><|im_end|>", "Visible"},
		{"thinking close before tool call", `Visible</think><|tool_call|>{"name":"view_image"}`, "Visible"},
		{"end marker before tool call", `Visible<|endoftext|><|tool_call|>{"name":"view_image"}`, "Visible"},
		{"end marker before role marker", "Visible<|eot_id|><|im_start|>user", "Visible"},
		{"stacked control endings", "Visible<|channel|><|im_end|>", "Visible"},
		{"control token mentioned in prose", "Answer: use <|channel|> tokens", "Answer: use <|channel|> tokens"},
		{"channel close mentioned in prose", "Answer: use <channel|> tokens", "Answer: use <channel|> tokens"},
		{"channel close before full stop", "Answer: use <channel|>.", "Answer: use <channel|>."},
		{"channel close before comma", "Answer: use <channel|>, then stop", "Answer: use <channel|>, then stop"},
		{"channel close before parenthesis", "Answer: (use <channel|>).", "Answer: (use <channel|>)."},
		{"thinking close mentioned in prose", "使用 </think> 标记结尾", "使用 </think> 标记结尾"},
		{"literal channel close before thinking block", "Use <channel|> literally. <|channel>private<channel|>Visible", "Use <channel|> literally. Visible"},
		{"end token mentioned in prose", "把 <|im_end|> 放在句尾", "把 <|im_end|> 放在句尾"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Clean(tc.text, "x"); got != tc.want {
				t.Errorf("Clean(%q) = %q, want %q", tc.text, got, tc.want)
			}
			if got := PartialClean(tc.text, "x"); got != tc.want {
				t.Errorf("PartialClean(%q) = %q, want %q", tc.text, got, tc.want)
			}
			structured := `{"title":"T","content":` + strconv.Quote(tc.text) + `}`
			want := Answer{Title: "T", Content: tc.want}
			if got, ok := ParseAnswer(structured); !ok || got != want {
				t.Errorf("ParseAnswer(%q) = %+v, %v, want %+v", structured, got, ok, want)
			}
			if got := PartialAnswer(structured); got != want {
				t.Errorf("PartialAnswer(%q) = %+v, want %+v", structured, got, want)
			}
			if got := PartialAnswer(tc.text); got.Content != tc.want {
				t.Errorf("PartialAnswer(%q) = %+v, want content %q", tc.text, got, tc.want)
			}
		})
	}
}
