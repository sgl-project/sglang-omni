package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// retiredTests are tests that went with Voxt's own language models. The
// history pages (llm-models.md, eval.md, behavior.md, manual-test.md) still
// name them as the record of what was measured.
var retiredTests = map[string]bool{
	"TestEvalLLM": true, "TestEvalGlossary": true, "TestEvalLong": true, "TestEvalRewrite": true,
	"TestEvalDictLLM": true, "TestEvalAppEnhancement": true, "TestEvalDictionary": true,
	"TestLLMMeasure": true, "TestLLMRealCancel": true, "TestLLMRealPromptCache": true,
	"TestLLMOpenerThroughLibrary": true,
}

// TestDocs: in Voxt's Markdown pages every relative link leads to a file,
// every Chinese page (x.zh-CN.md) has its English page (x.md), and every test
// a page names exists, or is the start of a test's name as `go test -run` reads it.
func TestDocs(t *testing.T) {
	root, _ := filepath.Glob("*.md")
	pages, _ := filepath.Glob("docs/*.md")
	pages = append(pages, root...)
	if len(pages) < 10 {
		t.Fatalf("found %d pages", len(pages))
	}

	var tests []string
	fn := regexp.MustCompile(`(?m)^func ((?:Test|Benchmark)\w*)\(`)
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || !strings.HasSuffix(path, "_test.go") {
			return err
		}
		b, err := os.ReadFile(path)
		for _, m := range fn.FindAllSubmatch(b, -1) {
			tests = append(tests, string(m[1]))
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}

	link := regexp.MustCompile(`\]\(([^)\s]+)\)`)
	name := regexp.MustCompile(`\b(?:Test|Benchmark)[A-Z]\w*`)
	for _, page := range pages {
		b, err := os.ReadFile(page)
		if err != nil {
			t.Fatal(err)
		}
		if en, ok := strings.CutSuffix(page, ".zh-CN.md"); ok {
			if _, err := os.Stat(en + ".md"); err != nil {
				t.Errorf("%s has no English page %s.md", page, en)
			}
		}
		for _, m := range link.FindAllSubmatch(b, -1) {
			target, _, _ := strings.Cut(string(m[1]), "#")
			if target == "" || strings.Contains(target, ":") {
				continue // a heading of this page, or a URL
			}
			if _, err := os.Stat(filepath.Join(filepath.Dir(page), target)); err != nil {
				t.Errorf("%s links to %s, which does not exist", page, m[1])
			}
		}
		seen := map[string]bool{}
		for _, n := range name.FindAllString(string(b), -1) {
			if seen[n] || retiredTests[n] {
				continue
			}
			seen[n] = true
			found := false
			for _, have := range tests {
				found = found || strings.HasPrefix(have, n)
			}
			if !found {
				t.Errorf("%s names %s, and no test's name starts with it", page, n)
			}
		}
	}
}
