package compose

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestAnswerDropsRewriteTags: a model that echoes the <selected_text> or
// <instruction> tags RewriteMessage wraps its input in never shows them
// on the answer card; the text between them stays.
func TestAnswerDropsRewriteTags(t *testing.T) {
	const selection, instruction = "The quick brown fox", "make it formal"
	for _, tc := range []struct{ out, want string }{
		{"OK: <selected_text>\nThe quick brown fox\n</selected_text>", "OK: The quick brown fox"},
		{"<instruction>\nmake it formal\n</instruction>\n\nDone.", "make it formal\n\nDone."},
		{"Fixed: <selected_text>The quick brown fox</selected_text>", "Fixed: The quick brown fox"},
		{"<selected_text>\nA quick brown fox.\n</selected_text>", "A quick brown fox."},
	} {
		checkAnswer(t, tc.out, tc.want, selection, instruction)
	}
}

// TestAnswerKeepsRequestedTags: tags that are part of the answer, such as
// a prompt template the user asked for, stay; so do tags the user's own
// selection has.
func TestAnswerKeepsRequestedTags(t *testing.T) {
	template := "Use this template:\n<instruction>\n{task}\n</instruction>\n<selected_text>\n{text}\n</selected_text>"
	checkAnswer(t, template, template, "", "Write a prompt template with instruction and selected_text tags")
	sentence := "Wrap the task in <instruction></instruction> tags, the input in <selected_text> tags."
	checkAnswer(t, sentence, sentence, "", "Rephrase it")
	own := "Put it in <instruction>\nthe task\n</instruction>, then answer."
	checkAnswer(t, own, own, own, "fix")
	checkAnswer(t, "<selected_text>"+own+"</selected_text>", own, own, "fix")
}

func checkAnswer(t *testing.T, out, want, selection, instruction string) {
	t.Helper()
	if got := Clean(out, selection+"\n"+instruction); got != want {
		t.Errorf("Clean(%q) = %q, want %q", out, got, want)
	}
	b, _ := json.Marshal(Answer{Title: "T", Content: out})
	if got, ok := ParseAnswer(string(b), selection, instruction); !ok || got.Content != want {
		t.Errorf("ParseAnswer(%s) = %q, want %q", b, got.Content, want)
	}
	if got := PartialAnswer(string(b), selection, instruction); got.Content != want {
		t.Errorf("PartialAnswer(%s) = %q, want %q", b, got.Content, want)
	}
	if got := PartialAnswer(out, selection, instruction); got.Content != want {
		t.Errorf("PartialAnswer(%q) = %q, want %q", out, got.Content, want)
	}
}

// TestAnswerDropsRewriteTagsChangelog keeps the user-visible fix in the release notes.
func TestAnswerDropsRewriteTagsChangelog(t *testing.T) {
	data, err := os.ReadFile("../../CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	sections := strings.SplitN(string(data), "\n## [", 3)
	if len(sections) < 2 {
		t.Fatal("CHANGELOG.md has no Unreleased section")
	}
	for _, entry := range []string{
		"Rewrite answers no longer show echoed input tags",
		"改写回答不再显示重复的输入标签",
	} {
		if !strings.Contains(sections[1], entry) {
			t.Errorf("Unreleased changelog missing %q", entry)
		}
	}
}
