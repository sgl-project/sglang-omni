package compose

import (
	"strconv"
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
	structured := `{"title":"T","content":"` + text + `"}`
	wantAnswer := Answer{Title: "T", Content: want}
	if got, ok := ParseAnswer(structured, input); !ok || got != wantAnswer {
		t.Errorf("ParseAnswer(%q, %q) = %+v, %v, want %+v", structured, input, got, ok, wantAnswer)
	}
	if got := PartialAnswer(structured, input); got != wantAnswer {
		t.Errorf("PartialAnswer(%q, %q) = %+v, want %+v", structured, input, got, wantAnswer)
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

func TestAnswerCleanupThinkingClose(t *testing.T) {
	for _, tc := range []struct{ name, text, want string }{
		{"answer after thinking close", "reasoning here</think>Final answer", "Final answer"},
		{"answer after newline-wrapped thinking close", "reasoning\n</think>\n\nFinal answer", "Final answer"},
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
