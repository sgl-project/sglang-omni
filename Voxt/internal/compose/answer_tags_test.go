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
	for _, tc := range []struct{ out, want string }{
		{"OK: <selected_text>\nThe quick brown fox\n</selected_text>", "OK: The quick brown fox"},
		{"<instruction>\nmake it formal\n</instruction>\n\nDone.", "make it formal\n\nDone."},
		{"Fixed: <selected_text>a</selected_text>", "Fixed: a"},
	} {
		if got := Clean(tc.out, "x"); got != tc.want {
			t.Errorf("Clean(%q) = %q, want %q", tc.out, got, tc.want)
		}
		b, _ := json.Marshal(Answer{Title: "T", Content: tc.out})
		if got, ok := ParseAnswer(string(b)); !ok || got.Content != tc.want {
			t.Errorf("ParseAnswer(%s) = %q, want %q", b, got.Content, tc.want)
		}
		if got := PartialAnswer(string(b)); got.Content != tc.want {
			t.Errorf("PartialAnswer(%s) = %q, want %q", b, got.Content, tc.want)
		}
		if got := PartialAnswer(tc.out); got.Content != tc.want {
			t.Errorf("PartialAnswer(%q) = %q, want %q", tc.out, got.Content, tc.want)
		}
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
