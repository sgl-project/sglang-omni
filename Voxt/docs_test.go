package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestDocs(t *testing.T) {
	var docs []string
	err := filepath.WalkDir("docs", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".md") {
			docs = append(docs, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("relative links", func(t *testing.T) {
		inlineLink := regexp.MustCompile(`!?\[[^]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))`)
		referenceLink := regexp.MustCompile(`(?m)^\s*\[[^]]+\]:\s*(?:<([^>]+)>|(\S+))`)
		linkDestination := func(match []string) string {
			if match[1] != "" {
				return match[1]
			}
			return match[2]
		}
		code := regexp.MustCompile("(?s)```.*?```|`[^`\n]*`")
		for _, doc := range docs {
			content, err := os.ReadFile(doc)
			if err != nil {
				t.Fatal(err)
			}
			text := code.ReplaceAllString(string(content), "")
			matches := inlineLink.FindAllStringSubmatch(text, -1)
			matches = append(matches, referenceLink.FindAllStringSubmatch(text, -1)...)
			for _, match := range matches {
				destination := linkDestination(match)
				if destination == "" || strings.HasPrefix(destination, "#") || strings.HasPrefix(destination, "/") || strings.Contains(destination, "://") || strings.HasPrefix(destination, "mailto:") {
					continue
				}
				destination = strings.Split(destination, "#")[0]
				if destination == "" {
					continue
				}
				destination, err = url.PathUnescape(destination)
				if err != nil {
					t.Errorf("%s: %s: %v", doc, match[1], err)
					continue
				}
				if _, err := os.Stat(filepath.Join(filepath.Dir(doc), destination)); err != nil {
					t.Errorf("%s: %s: %v", doc, match[1], err)
				}
			}
		}
		for _, test := range []struct{ markdown, want string }{
			{`[missing]( missing.md)`, "missing.md"},
			{`[x](<a file.md>)`, "a file.md"},
			{`[x]: <a file.md>`, "a file.md"},
		} {
			match := inlineLink.FindStringSubmatch(test.markdown)
			if strings.HasPrefix(test.markdown, "[x]:") {
				match = referenceLink.FindStringSubmatch(test.markdown)
			}
			if match == nil || linkDestination(match) != test.want {
				t.Errorf("%s: got %v, want %s", test.markdown, match, test.want)
			}
		}
	})

	t.Run("language pairs", func(t *testing.T) {
		for _, base := range []string{"README", "docs/development", "docs/ui-development"} {
			for _, path := range []string{base + ".md", base + ".zh-CN.md"} {
				if _, err := os.Stat(path); err != nil {
					t.Errorf("%s: %v", path, err)
				}
			}
		}
	})

	t.Run("test names", func(t *testing.T) {
		declared := make(map[string]bool)
		fileSet := token.NewFileSet()
		err := filepath.WalkDir(".", func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() && path != "." && (entry.Name() == "build" || entry.Name() == "testdata" || entry.Name() == "third_party") {
				return filepath.SkipDir
			}
			if entry.IsDir() || !strings.HasSuffix(path, "_test.go") {
				return nil
			}
			parsed, err := parser.ParseFile(fileSet, path, nil, 0)
			if err != nil {
				return err
			}
			for _, declaration := range parsed.Decls {
				function, ok := declaration.(*ast.FuncDecl)
				if ok && function.Recv == nil && strings.HasPrefix(function.Name.Name, "Test") {
					declared[function.Name.Name] = true
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}

		quotedTest := regexp.MustCompile("`Test[A-Za-z0-9_]+`")
		for _, doc := range docs {
			content, err := os.ReadFile(doc)
			if err != nil {
				t.Fatal(err)
			}
			for _, quoted := range quotedTest.FindAllString(string(content), -1) {
				name := strings.Trim(quoted, "`")
				if !declared[name] {
					t.Errorf("%s: %s is not declared in a Go test file", doc, name)
				}
			}
		}
	})
}
