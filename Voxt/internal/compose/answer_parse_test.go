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
	for _, tc := range []struct {
		out, in, want string
		custom        bool
	}{
		{"  Hello.  ", "hello", "Hello.", false},
		{"<output>\nHello.\n</output>", "hello", "Hello.", false},
		{"<output>\nHello.\n</output>\n\nI removed the filler “um”.", "hello", "Hello.", false},
		{"Here you go: <output>Hello.</output> Hope it helps.", "hello", "Hello.", false},
		{"<output>\n算计算量（FLOPs）—— 用 bf16 算和 int8 有区别吗？\n</output>\n\n清理时删掉了“就是说”。", "x", "算计算量（FLOPs）—— 用 bf16 算和 int8 有区别吗？", false},
		{"<output>\nHello.", "hello", "Hello.", false}, // cut short: the rest is the text
		{"<output></output>", "hello", "", false},
		{"<output>“你好。”</output>", "你好", "你好。", false},
		// The tags were in the input itself (dictated literally): the
		// answer is used as it is…
		{`use <output> and </output> here`, "use <output> and </output> here", `use <output> and </output> here`, false},
		{"<output>a</output><output>b</output>", "<output>a</output><output>b</output>", "<output>a</output><output>b</output>", false},
		// …but a contract-abiding model wrapped it in one more pair:
		// exactly that pair goes.
		{"<output>use <output> and </output> here</output>", "use <output> and </output> here", "use <output> and </output> here", false},
		// A tag-less answer to the built-in contract keeps only the
		// cleaned text's span: the contract allows nothing around it,
		// so the model's deliberation before or after goes (#4086).
		// The reply has the reported shape: deliberation in English
		// with quoted input fragments, blank lines, then the cleaned
		// text.
		{"Let me look at this again. \"初稿\" might be a mishearing - I want to double-check. The speaker could mean \"初稿\" as in a first draft, or it might be something else entirely. I will keep it as written since I cannot decide.\n\nAlso \"不不需要\" reads like a repeated word - \"不，不需要\" should become \"不需要\".\n\n我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不需要再等一个 sprint。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不不需要再等一个 sprint。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不需要再等一个 sprint。", false},
		{"We need to ship the release on Friday and tell the team tomorrow. I removed the filler \"uh\" and kept the rest of the sentence as it was, changing only its punctuation.",
			"we need to ship the release on friday, uh, and tell the team tomorrow",
			"We need to ship the release on Friday and tell the team tomorrow.", false},
		// …near the input's length or far past it, and even a finished
		// deliberation sentence before the cleaned text.
		{"Let me look at this again. The release is Friday.", "the release is friday. tell the quality team", "The release is Friday.", false},
		// A short dictation is covered the same way: the deliberation
		// before or after the cleaned text goes…
		{"Let me look at this again. Hello.", "hello", "Hello.", false},
		{"Hello. I fixed the typo.", "hello", "Hello.", false},
		// …down to an exact short cleaned answer (Hi., Yes., 你好。)…
		{"Let me look at this again. Hi.", "hi", "Hi.", false},
		{"Let me look at this again. Yes.", "yes", "Yes.", false},
		{"Let me look at this again. 你好。", "你好", "你好。", false},
		// …while a connector that is not the input (OK, 嗯) does not
		// anchor, so it cannot shield the deliberation after it…
		{"OK. Let me think once more. 我们周五要交方案初稿。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不不需要再等一个 sprint。",
			"我们周五要交方案初稿。", false},
		// …and one next to the cleaned text stays with it.
		{"嗯。我们周五要交方案初稿。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不不需要再等一个 sprint。",
			"嗯。我们周五要交方案初稿。", false},
		// …while a legitimate short answer keeps its added words when
		// nothing anchors the input (there is nothing to trim to).
		{"Hello world.", "hello", "Hello world.", false},
		// Near the input's own length the answer is trusted as it is…
		{"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不需要再等一个 sprint。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不不需要再等一个 sprint。",
			"我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不需要再等一个 sprint。", false},
		// …and so is a wholesale rewrite nothing like the input: no
		// sentence is the cleaned text, so there is nothing to trim to.
		{"我们的周五发布计划已经定下来了。这次改动比较大，涉及转录、改写和设置三个页面。下周先内部试用一周，没有问题再对外发布。",
			"the quick brown fox jumps over the lazy dog near the river bank",
			"我们的周五发布计划已经定下来了。这次改动比较大，涉及转录、改写和设置三个页面。下周先内部试用一周，没有问题再对外发布。", false},
		// With the user's own prompt (Output.Custom), the prompt owns
		// the output shape — a heading, an expansion, a translation —
		// and the answer is used as it is.
		{"Our launch is Friday. The team should prepare the release notes and confirm the build before sharing it with customers.",
			"our launch is friday",
			"Our launch is Friday. The team should prepare the release notes and confirm the build before sharing it with customers.", true},
		{"Release plan. Our launch is Friday. The team should prepare the release notes and confirm the build before sharing it with customers.",
			"our launch is friday",
			"Release plan. Our launch is Friday. The team should prepare the release notes and confirm the build before sharing it with customers.", true},
		{"我们的周五发布计划已经定下来了。这次改动比较大，涉及转录、改写和设置三个页面。下周先内部试用一周，没有问题再对外发布。",
			"the quick brown fox jumps over the lazy dog near the river bank",
			"我们的周五发布计划已经定下来了。这次改动比较大，涉及转录、改写和设置三个页面。下周先内部试用一周，没有问题再对外发布。", true},
	} {
		if got := CleanOutput(tc.out, tc.in, tc.custom); got != tc.want {
			t.Errorf("CleanOutput(%q, custom %v) = %q, want %q", tc.out, tc.custom, got, tc.want)
		}
	}
}

// TestPartialOutput: a streaming clean-up answer never shows the
// contract's <output> tags, whole or as far as they came.
func TestPartialOutput(t *testing.T) {
	full := "<output>\n大家好，周五发布。\n</output>\n\n我删掉了语气词。"
	for i := 1; i <= len(full); i++ {
		got := PartialOutput(full[:i], "周五发布", false)
		if strings.Contains(got, "<output>") || strings.Contains(got, "</") || strings.HasSuffix(got, "<") {
			t.Fatalf("at %d: a tag in %q", i, got)
		}
	}
	// Commentary before the opening tag goes too, as Clean drops it.
	prefixed := "Here you go: <output>Hello.</output>"
	for i := 1; i <= len(prefixed); i++ {
		got := PartialOutput(prefixed[:i], "hello", false)
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
		if got := PartialOutput(text, "hello", false); got != want {
			t.Errorf("PartialOutput(%q) = %q, want %q", text, got, want)
		}
	}
	// The tags the user said stay: CleanOutput keeps that text too,
	// less one outermost pair a contract-abiding model wrapped it in.
	if got := PartialOutput("use <output> and </output> here", "use <output> and </output> here", false); got != "use <output> and </output> here" {
		t.Errorf("PartialOutput with the user's own tags = %q", got)
	}
	if got := PartialOutput("<output>use <output> and </output> here</output>", "use <output> and </output> here", false); got != "use <output> and </output> here" {
		t.Errorf("PartialOutput with the model's wrapper around the user's tags = %q", got)
	}
	// A tag-less stream of the model's deliberation (#4086): a
	// finished deliberation sentence never shows, not even standing
	// alone; once the first sentence shows itself foreign, nothing
	// more of the deliberation streams. A sentence still being typed
	// passes through — it may be the cleaned text's start, and the
	// draft corrects when it completes.
	const input = "我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不不需要再等一个 sprint。"
	const cleaned = "我们周五要交方案初稿。就是虽然说时间有点紧，但我觉得不需要再等一个 sprint。"
	// The reply has the reported shape: deliberation in English with
	// quoted input fragments, blank lines, then the cleaned text.
	const tagless = "Let me look at this again. \"初稿\" might be a mishearing - I want to double-check. The speaker could mean \"初稿\" as in a first draft, or it might be something else entirely. I will keep it as written since I cannot decide.\n\nAlso \"不不需要\" reads like a repeated word - \"不，不需要\" should become \"不需要\".\n\n" + cleaned
	if got := PartialOutput("Let me look at this again.", input, false); got != "" {
		t.Errorf("a completed deliberation sentence = %q, want hidden", got)
	}
	// A short dictation is covered the same way: the finished
	// deliberation sentence hides, the cleaned text shows.
	if got := PartialOutput("Let me look at this again. ", "hello", false); got != "" {
		t.Errorf("a completed deliberation sentence for a short input = %q, want hidden", got)
	}
	if got := PartialOutput("Let me look at this again. Hello.", "hello", false); got != "Hello." {
		t.Errorf("the short dictation's cleaned text = %q, want %q", got, "Hello.")
	}
	if got := PartialOutput("Hello.", "hello", false); got != "Hello." {
		t.Errorf("a legitimate short answer = %q, want it whole", got)
	}
	// An exact short cleaned answer (Hi., Yes., 你好。) anchors and
	// shows, while its unfinished typing passes through and the
	// finished deliberation sentence before it hides.
	for _, tc := range []struct{ out, in, want string }{
		{"Let me look at this again. Hi.", "hi", "Hi."},
		{"Let me look at this again. Yes.", "yes", "Yes."},
		{"Let me look at this again. 你好。", "你好", "你好。"},
	} {
		if got := PartialOutput(tc.out, tc.in, false); got != tc.want {
			t.Errorf("PartialOutput(%q) = %q, want %q", tc.out, got, tc.want)
		}
		if got := CleanOutput(tc.out, tc.in, false); got != tc.want {
			t.Errorf("CleanOutput(%q) = %q, want %q", tc.out, got, tc.want)
		}
	}
	runes := []rune(tagless)
	for i := 1; i <= len(runes); i++ {
		got := PartialOutput(string(runes[:i]), input, false)
		if strings.Contains(got, "again.") || strings.Contains(got, "mishearing") || strings.Contains(got, "double-check") ||
			strings.Contains(got, "repeated word") || strings.Contains(got, "cannot decide") {
			t.Fatalf("at %d: the deliberation in the draft %q", i, got)
		}
		if anchor := strings.Index(got, "我们"); anchor >= 0 && !strings.HasPrefix(cleaned, got[anchor:]) {
			t.Fatalf("at %d: draft %q is not the cleaned text's start", i, got)
		}
	}
	if got := PartialOutput(tagless, input, false); got != cleaned {
		t.Errorf("PartialOutput(tagless) = %q, want %q", got, cleaned)
	}
	// A finished deliberation sentence before the cleaned text never
	// comes back when a later sentence anchors, near the input's
	// length too: the draft is the cleaned text's span from the
	// anchor on, as CleanOutput makes of the final text.
	const nearIn = "the release is friday. tell the quality team"
	if got := PartialOutput("Let me look at this again. ", nearIn, false); got != "" {
		t.Errorf("the finished deliberation sentence = %q, want hidden", got)
	}
	if got := PartialOutput("Let me look at this again. The rel", nearIn, false); got != "The rel" {
		t.Errorf("the cleaned text after the deliberation = %q, want its start shown", got)
	}
	if got := PartialOutput("Let me look at this again. The release is Friday.", nearIn, false); got != "The release is Friday." {
		t.Errorf("the near-length sequence = %q, want the cleaned text", got)
	}
	// Streaming and final agree: a finished sentence after the
	// cleaned text's span is not the cleaned text either — the
	// built-in contract allows nothing around it (#4086).
	const nearOut = "The release is Friday. Notify QA."
	if got := PartialOutput("The release is Friday. Notify Q", nearIn, false); got != "The release is Friday." {
		t.Errorf("the tail still coming = %q, want the cleaned text's span", got)
	}
	if got := PartialOutput(nearOut, nearIn, false); got != "The release is Friday." {
		t.Errorf("PartialOutput(%q) = %q, want the cleaned text's span", nearOut, got)
	}
	if got := CleanOutput(nearOut, nearIn, false); got != "The release is Friday." {
		t.Errorf("CleanOutput(%q) = %q, want the cleaned text's span", nearOut, got)
	}
	// With the user's own prompt (Output.Custom) the same answer
	// shows and stays whole, at every prefix: a custom prompt may
	// legitimately expand the dictation.
	for i := 1; i <= len(nearOut); i++ {
		if got := PartialOutput(nearOut[:i], nearIn, true); got != nearOut[:i] {
			t.Fatalf("custom at %d: PartialOutput = %q, want the text as it is", i, got)
		}
	}
	if got := CleanOutput(nearOut, nearIn, true); got != nearOut {
		t.Errorf("CleanOutput(%q, custom) = %q, want it whole", nearOut, got)
	}
	// A wholesale rewrite (a custom prompt's translation, say)
	// streams as it is at every prefix.
	const rewrite = "我们的周五发布计划已经定下来了。这次改动比较大，涉及转录、改写和设置三个页面。"
	for i, n := 0, len([]rune(rewrite)); i < n; i++ {
		if got := PartialOutput(string([]rune(rewrite)[:i+1]), nearIn, true); got != string([]rune(rewrite)[:i+1]) {
			t.Fatalf("rewrite at %d: PartialOutput = %q, want the text as it is", i+1, got)
		}
	}
}

// TestGlossaryInUser: past GlossaryAllUpTo terms the glossary (the terms
// the text mentions) opens the user message, so that the system prompt
// and the example stay the same whatever is said; up to it, the whole
// glossary is in the system prompt.
