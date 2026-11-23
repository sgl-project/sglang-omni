package compose

import (
	"strings"
	"testing"
)

func TestParseAnswer(t *testing.T) {
	for _, tc := range []struct {
		text string
		want Answer
		ok   bool
	}{
		{`{"title":"Update","content":"Ships Friday."}`, Answer{"Update", "Ships Friday."}, true},
		{"```json\n{\"title\": \" **Update** \", \"content\": \"Line 1\\nLine 2\"}\n```", Answer{"Update", "Line 1\nLine 2"}, true},
		{`Sure! {"title":"T","content":"x {y} z"} hope it helps`, Answer{"T", "x {y} z"}, true},
		{`{"title":"T","content":"No image was provided.<|tool_call|>internal"}`, Answer{"T", "No image was provided."}, true},
		{`{"title":"T","content":"<|im_start|>assistant\nVisible answer<|im_end|>"}`, Answer{"T", "Visible answer"}, true},
		{`{"title":"T","content":"Visible<|endoftext|>"}`, Answer{"T", "Visible"}, true},
		{`{"title":"T","content":"Visible</think>"}`, Answer{"T", "Visible"}, true},
		{`{"title":"T","content":"Visible<channel|>"}`, Answer{"T", "Visible"}, true},
		{`{"title":"T","content":"a <"}`, Answer{"T", "a <"}, true},
		{`{"title":"T","content":"  "}`, Answer{Title: "T"}, true},
		{`{"title":"T","text":"x"}`, Answer{}, false},
		{`{"title":"T"`, Answer{}, false},
		{"just text", Answer{}, false},
		{`{bad} {"content":"second"}`, Answer{Content: "second"}, true},
	} {
		got, ok := ParseAnswer(tc.text)
		if ok != tc.ok || got != tc.want {
			t.Errorf("ParseAnswer(%q) = %+v %v", tc.text, got, ok)
		}
	}
}

func TestPartialAnswer(t *testing.T) {
	full := `{"title": "周五发布", "content": "大家好：\n- VoxtNew \"0.1\" 周五发布 \ud83d\ude80 \u4f60"}`
	var last Answer
	for i := 1; i <= len(full); i++ {
		a := PartialAnswer(full[:i])
		if !strings.HasPrefix(full, full[:i]) || len(a.Content) < len(last.Content) && !strings.HasPrefix(last.Content, a.Content) {
			t.Fatalf("at %d: %+v after %+v", i, a, last)
		}
		if strings.ContainsRune(a.Content, '\ufffd') {
			t.Fatalf("at %d: a broken rune in %q", i, a.Content)
		}
		last = a
	}
	if last.Title != "周五发布" || last.Content != "大家好：\n- VoxtNew \"0.1\" 周五发布 🚀 你" {
		t.Errorf("final %+v", last)
	}
	for text, want := range map[string]Answer{
		"":                            {},
		"Plain text answer":           {Content: "Plain text answer"},
		"Answer<|tool_call|>internal": {Content: "Answer"},
		"Answer<|tool_":               {Content: "Answer"},
		"Answer<th":                   {Content: "Answer"},
		"a <|x":                       {Content: "a <|x"},
		`{"title":"T","content":"A\u003cthink\u003eprivate\u003c/think\u003e"}`:             {Title: "T", Content: "A"},
		`{"title":"T<|tool_call|>x","content":"Visible"}`:                                   {Title: "T", Content: "Visible"},
		`<think>private</think>{"title":"T","content":"Visible"}`:                           {Title: "T", Content: "Visible"},
		`<|channel>private<channel|>{"title":"T","content":"Visible"}`:                      {Title: "T", Content: "Visible"},
		"<|im_start|>assistant\nVisible answer<|im_end|>":                                   {Content: "Visible answer"},
		"<|im_start|>assistant\n{\"title\":\"T\",\"content\":\"Visible answer\"}<|im_end|>": {Title: "T", Content: "Visible answer"},
		"Visible<|endoftext|>":           {Content: "Visible"},
		"Visible</think>":                {Content: "Visible"},
		"Visible<channel|>":              {Content: "Visible"},
		"Use <b>bold</b>":                {Content: "Use <b>bold</b>"},
		"```":                            {},
		"```json\n{\"title\":\"T\",\"co": {Title: "T"},
		`{"content":"a\`:                 {Content: "a"},
		`{"content": "x", "title": "`:    {Content: "x"},
	} {
		if got := PartialAnswer(text); got != want {
			t.Errorf("PartialAnswer(%q) = %+v", text, got)
		}
	}
}

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

func TestClean(t *testing.T) {
	for _, tc := range []struct{ out, in, want string }{
		{"  Hello.  ", "hello", "Hello."},
		{"<input>\nHello.\n</input>", "hello", "Hello."},
		{`"Hello."`, "hello", "Hello."},
		{"“你好。”", "你好", "你好。"},
		{`"Hello," she said, "bye."`, "x", `"Hello," she said, "bye."`},
		{`"Quoted."`, `"quoted"`, `"Quoted."`},
		{"", "x", ""},
		{"No image was provided.<|tool_call|>{\"name\":\"view_image\"}<|im_end|>", "help me explain this picture", "No image was provided."},
		{"Answer<think>private</think>", "x", "Answer"},
		{"<|im_start|>assistant\nVisible answer<|im_end|>", "x", "Visible answer"},
		{"Visible<|endoftext|>", "x", "Visible"},
		{"Visible</think>", "x", "Visible"},
		{"Visible<channel|>", "x", "Visible"},
		{"Answer<|tool_", "x", "Answer<|tool_"},
		{"a <|x", "x", "a <|x"},
		{"a <", "x", "a <"},
		{"Use <b>bold</b> and 2 < 3.", "format this as HTML", "Use <b>bold</b> and 2 < 3."},
		// Only Transcribe's clean-up asks for <output> tags: a Rewrite
		// or Translate answer whose text itself has them (HTML it was
		// asked for) keeps them.
		{"Use <output>result</output> for results.", "format this as HTML", "Use <output>result</output> for results."},
	} {
		if got := Clean(tc.out, tc.in); got != tc.want {
			t.Errorf("Clean(%q) = %q, want %q", tc.out, got, tc.want)
		}
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

// TestCleanOutput: the Transcribe clean-up's contract — only the
// <output> tags' content, never the model's commentary about its edits
// (#4086); text that has the tags because the user said them keeps
// them, less one outermost pair the model wrapped it in.
func TestCleanOutput(t *testing.T) {
	for _, tc := range []struct{ out, in, want string }{
		{"  Hello.  ", "hello", "Hello."},
		{"<output>\nHello.\n</output>", "hello", "Hello."},
		{"<output>\nHello.\n</output>\n\nI removed the filler “um”.", "hello", "Hello."},
		{"Here you go: <output>Hello.</output> Hope it helps.", "hello", "Hello."},
		{"<output>\n算计算量（FLOPs）—— 用 bf16 算和 int8 有区别吗？\n</output>\n\n清理时删掉了“就是说”。", "x", "算计算量（FLOPs）—— 用 bf16 算和 int8 有区别吗？"},
		{"<output>\nHello.", "hello", "Hello."}, // cut short: the rest is the text
		{"<output></output>", "hello", ""},
		{"<output>“你好。”</output>", "你好", "你好。"},
		// The tags were in the input itself (dictated literally): the
		// answer is used as it is…
		{`use <output> and </output> here`, "use <output> and </output> here", `use <output> and </output> here`},
		{"<output>a</output><output>b</output>", "<output>a</output><output>b</output>", "<output>a</output><output>b</output>"},
		// …but a contract-abiding model wrapped it in one more pair:
		// exactly that pair goes.
		{"<output>use <output> and </output> here</output>", "use <output> and </output> here", "use <output> and </output> here"},
	} {
		if got := CleanOutput(tc.out, tc.in); got != tc.want {
			t.Errorf("CleanOutput(%q) = %q, want %q", tc.out, got, tc.want)
		}
	}
}

// TestPartialOutput: a streaming clean-up answer never shows the
// contract's <output> tags, whole or as far as they came.
func TestPartialOutput(t *testing.T) {
	full := "<output>\n大家好，周五发布。\n</output>\n\n我删掉了语气词。"
	for i := 1; i <= len(full); i++ {
		got := PartialOutput(full[:i], "周五发布")
		if strings.Contains(got, "<output>") || strings.Contains(got, "</") || strings.HasSuffix(got, "<") {
			t.Fatalf("at %d: a tag in %q", i, got)
		}
	}
	// Commentary before the opening tag goes too, as Clean drops it.
	prefixed := "Here you go: <output>Hello.</output>"
	for i := 1; i <= len(prefixed); i++ {
		got := PartialOutput(prefixed[:i], "hello")
		if strings.Contains(got, "<output>") || strings.Contains(got, "</") || strings.HasSuffix(got, "<") {
			t.Fatalf("at %d: a tag in %q", i, got)
		}
	}
	for text, want := range map[string]string{
		"":                            "",
		"<":                           "",
		"<outpu":                      "",
		"<output>":                    "",
		"plain text":                  "plain text",
		"Answer<|tool_call|>internal": "Answer",
		"Answer<|tool_":               "Answer",
		"Answer<th":                   "Answer",
		"a <|x":                       "a <|x",
		"<output>\nHello.":            "Hello.",
		"<output>Hello.</output>\n":   "Hello.",
		"Here you go: <output>Hello.": "Hello.",
		// A stream ending in the opening tag's start holds back: the
		// tag may complete, and the commentary before it then goes.
		"Here you go: <outpu": "",
		// Without an opening tag a closing one is the user's text and
		// shows as it is, whole or as far as it came (CleanOutput
		// keeps it too).
		"use </output> here": "use </output> here",
		"Hello.</out":        "Hello.</out",
	} {
		if got := PartialOutput(text, "hello"); got != want {
			t.Errorf("PartialOutput(%q) = %q, want %q", text, got, want)
		}
	}
	// The tags the user said stay: CleanOutput keeps that text too,
	// less one outermost pair a contract-abiding model wrapped it in.
	if got := PartialOutput("use <output> and </output> here", "use <output> and </output> here"); got != "use <output> and </output> here" {
		t.Errorf("PartialOutput with the user's own tags = %q", got)
	}
	if got := PartialOutput("<output>use <output> and </output> here</output>", "use <output> and </output> here"); got != "use <output> and </output> here" {
		t.Errorf("PartialOutput with the model's wrapper around the user's tags = %q", got)
	}
}

// TestGlossaryInUser: past GlossaryAllUpTo terms the glossary (the terms
// the text mentions) opens the user message, so that the system prompt
// and the example stay the same whatever is said; up to it, the whole
// glossary is in the system prompt.
