package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func lines(t *testing.T, files map[string]string) []string {
	t.Helper()
	dir := t.TempDir()
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	l, err := hashDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for i := range l {
		l[i] = strings.TrimPrefix(l[i], filepath.ToSlash(dir)+" ")
	}
	slices.Sort(l)
	return l
}

func TestDeclHash(t *testing.T) {
	one := lines(t, map[string]string{"a.go": `package p

import "fmt"

// A says a.
func A() { fmt.Println("a") }

// B says b.
func B() {
	// inner
	println("b")
}

type T struct{ X int }
`})
	if len(one) != 3 {
		t.Fatalf("%q", one)
	}
	split := lines(t, map[string]string{
		"a.go": "package p\n\nimport \"fmt\"\n\n// A says a.\nfunc A() { fmt.Println(\"a\") }\n\ntype T struct{ X int }\n",
		"b.go": "package p\n\n// B says b.\nfunc B() {\n\t// inner\n\tprintln(\"b\")\n}\n",
	})
	if !slices.Equal(one, split) {
		t.Errorf("a pure split changed:\n%q\n%q", one, split)
	}
	for name, files := range map[string]map[string]string{
		"string": {"a.go": "package p\n\nimport \"fmt\"\n\n// A says a.\nfunc A() { fmt.Println(\"A\") }\n\ntype T struct{ X int }\n",
			"b.go": "package p\n\n// B says b.\nfunc B() {\n\t// inner\n\tprintln(\"b\")\n}\n"},
		"inner comment": {"a.go": "package p\n\nimport \"fmt\"\n\n// A says a.\nfunc A() { fmt.Println(\"a\") }\n\ntype T struct{ X int }\n",
			"b.go": "package p\n\n// B says b.\nfunc B() {\n\t// other\n\tprintln(\"b\")\n}\n"},
		"constraint": {"a.go": "package p\n\nimport \"fmt\"\n\n// A says a.\nfunc A() { fmt.Println(\"a\") }\n\ntype T struct{ X int }\n",
			"b_darwin.go": "package p\n\n// B says b.\nfunc B() {\n\t// inner\n\tprintln(\"b\")\n}\n"},
	} {
		if got := lines(t, files); slices.Equal(one, got) {
			t.Errorf("%s: a change went unseen", name)
		}
	}
}

func TestDeclHashNoComments(t *testing.T) {
	defer func() { noComments = false }()
	orig := map[string]string{"a.go": "package p\n\n// A says a (a.go).\nfunc A() {\n\t// inner\n\tprintln(\"a\")\n}\n"}
	moved := map[string]string{"b.go": "package p\n\n// A says a (b.go).\nfunc A() {\n\t// inner, b.go\n\tprintln(\"a\")\n}\n"}
	changed := map[string]string{"b.go": "package p\n\n// A says a (b.go).\nfunc A() {\n\tprintln(\"A\")\n}\n"}
	if slices.Equal(lines(t, orig), lines(t, moved)) {
		t.Error("comments are hashed by default")
	}
	noComments = true
	if !slices.Equal(lines(t, orig), lines(t, moved)) {
		t.Error("-nocomments: a change of comments only shows")
	}
	if slices.Equal(lines(t, orig), lines(t, changed)) {
		t.Error("-nocomments: a change of code went unseen")
	}
}
