package main

import (
	"html"
	"regexp"
	"strings"
	"testing"
)

func TestTranscriptCodeKeepsSource(t *testing.T) {
	strip := regexp.MustCompile(`<[^>]*>`)
	for _, tc := range []struct{ language, source string }{
		{"go", "// comment\nfunc main() { println(42) }\n"},
		{"python", "print(\"<script>alert('hi')</script>\")\n"},
		{"sh", "echo '$literal$'\n"},
		{"unknown", "<b>literal</b>\n"},
	} {
		t.Run(tc.language, func(t *testing.T) {
			got := transcriptMarkdown("```" + tc.language + "\n" + tc.source + "```")
			plain := html.UnescapeString(strip.ReplaceAllString(got, ""))
			if strings.TrimSpace(plain) != strings.TrimSpace(tc.source) {
				t.Errorf("code changed: %q", plain)
			}
			if tc.language != "unknown" && !strings.Contains(got, `class="token-`) {
				t.Errorf("missing code highlighting: %s", got)
			}
			if strings.Contains(got, "<script>") || strings.Contains(got, `class="math`) {
				t.Errorf("code interpreted as markup or math: %s", got)
			}
		})
	}
}
