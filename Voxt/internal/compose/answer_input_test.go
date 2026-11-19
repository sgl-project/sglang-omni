package compose

import "testing"

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
