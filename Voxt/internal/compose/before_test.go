package compose

import (
	"slices"
	"strings"
	"testing"
)

func TestSentenceEnds(t *testing.T) {
	for _, c := range []struct {
		text string
		want []int
	}{
		{"Hello there. How are", []int{12}},
		{"Hello there. How are you?", []int{12, 25}},
		{"版本 3.5 好了。再说", []int{len("版本 3.5 好了。")}},
		{"他说：“好。”然后走了", []int{len("他说：“好。”")}},
		{"Dr. Who", []int{3}},
		{"Line one\nline two", []int{8}},
		{"no end", nil},
	} {
		if got := SentenceEnds(c.text); !slices.Equal(got, c.want) {
			t.Errorf("SentenceEnds(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// note (luojiaxuan): TestBefore: a block after the first says what <before> is in the system
// prompt and sends the cleaned text before it ahead of its input, in both
// prompt sets and with the user's own prompt; a block without it is the
// request of a whole dictation, byte for byte.
func TestBefore(t *testing.T) {
	for _, main := range []string{"en", "zh"} {
		for _, prompt := range []string{"", "Clean up my dictation."} {
			in := Input{Task: Transcribe, Cleanup: true, MainLanguage: main, TaskPrompt: prompt, Text: "How are you?"}
			whole, err := Build(in)
			if err != nil {
				t.Fatal(err)
			}
			in.Before = "Hello there."
			block, err := Build(in)
			if err != nil {
				t.Fatal(err)
			}
			if want := BeforeMessage("Hello there.", "How are you?"); block.User != want {
				t.Errorf("%s %q: user %q, want %q", main, prompt, block.User, want)
			}
			if !slices.Contains(block.Layers, "before") || slices.Contains(whole.Layers, "before") || !strings.Contains(block.System, "<before>") {
				t.Errorf("%s %q: layers %v and %v", main, prompt, whole.Layers, block.Layers)
			}
			in.Before = ""
			if again, _ := Build(in); again.System != whole.System || again.User != UserMessage("How are you?") {
				t.Errorf("%s %q: a request without Before changed", main, prompt)
			}
		}
	}
	tr, err := Build(Input{Task: Translate, TargetLanguage: "en", Text: "你好", Before: "x"})
	if err != nil || tr.User != UserMessage("你好") || strings.Contains(tr.System, "<before>") {
		t.Errorf("Translate took Before: %q", tr.User)
	}
}
