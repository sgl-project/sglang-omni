package compose

import (
	"strings"
	"testing"
)

func TestPartialAnswerControlMarkupPrefixes(t *testing.T) {
	full := "Answer<think>private</think>Visible"
	for i := 1; i <= len(full); i++ {
		got := PartialAnswer(full[:i]).Content
		if strings.Contains(got, "<") || strings.Contains(got, "private") {
			t.Fatalf("at %d: control markup in %q", i, got)
		}
	}
	if got := PartialAnswer(full).Content; got != "AnswerVisible" {
		t.Fatalf("final %q", got)
	}
}

func TestPartialAnswerControlPrefixBeforeFence(t *testing.T) {
	text := "<|im_start|>assistant\n```json\n{\"title\":\"T\",\"content\":\"Visible answer\"}\n```<|im_end|>"
	want := Answer{Title: "T", Content: "Visible answer"}
	for i := 1; i <= len(text); i++ {
		got := PartialAnswer(text[:i])
		if strings.Contains(got.Content, "```") || strings.Contains(got.Content, `"title"`) {
			t.Fatalf("at %d: raw fenced answer in %q", i, got.Content)
		}
	}
	if got := PartialAnswer(text); got != want {
		t.Fatalf("PartialAnswer(%q) = %+v, want %+v", text, got, want)
	}
}

func TestPartialAnswerThinkingPrefixBeforeFence(t *testing.T) {
	text := "<think>private</think>```json\n{\"title\":\"T\",\"content\":\"Visible answer\"}\n```"
	want := Answer{Title: "T", Content: "Visible answer"}
	for i := 1; i <= len(text); i++ {
		got := PartialAnswer(text[:i])
		if strings.Contains(got.Content, "```") || strings.Contains(got.Content, `"title"`) || strings.Contains(got.Content, "private") {
			t.Fatalf("at %d: control markup or raw fenced answer in %q", i, got.Content)
		}
	}
	if got := PartialAnswer(text); got != want {
		t.Fatalf("PartialAnswer(%q) = %+v, want %+v", text, got, want)
	}
}

func TestPartialClean(t *testing.T) {
	for _, tc := range []struct{ out, in, want string }{
		{"Answer<th", "x", "Answer"},
		{"<|im_start|>assistant\nVisible answer<|im_end|>", "x", "Visible answer"},
		{"<|im_start|>assistant", "x", ""},
		{"Visible<|endoftext|>", "x", "Visible"},
		{"Visible<|endo", "x", "Visible"},
		{"Visible</think>", "x", "Visible"},
		{"Visible</th", "x", "Visible"},
		{"Visible<channel|>", "x", "Visible"},
		{"Visible<channel", "x", "Visible"},
		{"Answer<|tool_", "x", "Answer"},
		{"a <|x", "x", "a <|x"},
		{"Use <b>bold</b>", "format this as HTML", "Use <b>bold</b>"},
	} {
		if got := PartialClean(tc.out, tc.in); got != tc.want {
			t.Errorf("PartialClean(%q) = %q, want %q", tc.out, got, tc.want)
		}
	}
}
