package archtest

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The most lines a Go file may have (docs/architecture.md, "Code structure
// rules"): about a topic's worth (a state and its actions, a page's panel,
// a dialog, a stage of a session). Table tests and fixtures run longer.
// codeAim is what a file of code should keep to: over it, TestFileSizes
// reports it (go test -v) without failing.
const (
	codeLimit = 600
	codeAim   = 400
	testLimit = 800
	// longFunc is the length from which TestLongFuncs reports a function.
	longFunc = 150
)

func limit(f goFile) int {
	if f.test() {
		return testLimit
	}
	return codeLimit
}

// ceiling is a line of allowlist.txt: a file over its limit today, and the
// most lines it may have.
type ceiling struct {
	lines int
	line  int // in allowlist.txt
}

func readAllowlist(t *testing.T) map[string]ceiling {
	t.Helper()
	f, err := os.Open("allowlist.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	list := map[string]ceiling{}
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		text, _, _ := strings.Cut(sc.Text(), "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		lines := 0
		if len(fields) == 2 {
			lines, err = strconv.Atoi(fields[1])
		}
		if len(fields) != 2 || err != nil || lines <= 0 {
			t.Fatalf("allowlist.txt:%d: want \"path lines  # batch\", got %q", n, sc.Text())
		}
		if c, ok := list[fields[0]]; ok {
			t.Fatalf("allowlist.txt:%d: %s is on line %d already", n, fields[0], c.line)
		}
		list[fields[0]] = ceiling{lines, n}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return list
}

// TestFileSizes keeps files under their limits: codeLimit lines for code,
// testLimit for tests. The files over them today are in allowlist.txt with
// their size as a ceiling, which a file may not grow past; a file split
// under its limit leaves the list.
func TestFileSizes(t *testing.T) {
	allow := readAllowlist(t)
	seen := map[string]bool{}
	var fails, hints []string
	for _, f := range goFiles(t) {
		c, listed := allow[f.path]
		seen[f.path] = listed
		if f.generated() {
			if listed {
				fails = append(fails, fmt.Sprintf("%s is generated, which has no limit: remove its line (allowlist.txt:%d).", f.path, c.line))
			}
			continue
		}
		n, max := f.lines(), limit(f)
		switch {
		case !listed && n > max:
			fails = append(fails, fmt.Sprintf("%s has %d lines, over the limit of %d: split it by topic into files of its feature "+
				"(<feature>_<part>.go). Only if that cannot wait, add \"%s %d  # why\" to internal/archtest/allowlist.txt, "+
				"and say why in the commit.", f.path, n, max, f.path, n))
		case listed && n <= max:
			fails = append(fails, fmt.Sprintf("%s has %d lines, within the limit of %d: remove its line from internal/archtest/allowlist.txt (line %d).",
				f.path, n, max, c.line))
		case listed && n > c.lines:
			fails = append(fails, fmt.Sprintf("%s has %d lines, over its ceiling of %d (allowlist.txt:%d): files over the limit of %d may not grow. "+
				"Move code out of it (a new file of its feature), or split it; raise the ceiling only for a fix that cannot wait, saying why in the commit.",
				f.path, n, c.lines, c.line, max))
		case listed && n < c.lines:
			hints = append(hints, fmt.Sprintf("%s has %d lines: its ceiling (allowlist.txt:%d) can come down from %d.", f.path, n, c.line, c.lines))
		case !listed && !f.test() && n > codeAim:
			hints = append(hints, fmt.Sprintf("%s has %d lines, over the aim of %d: split it when it next grows.", f.path, n, codeAim))
		}
	}
	for path, c := range allow {
		if _, ok := seen[path]; !ok {
			fails = append(fails, fmt.Sprintf("%s is gone: remove its line from internal/archtest/allowlist.txt (line %d).", path, c.line))
		}
	}
	slices.Sort(fails)
	slices.Sort(hints)
	for _, h := range hints {
		t.Log(h)
	}
	for _, f := range fails {
		t.Error(f)
	}
}

// TestLongFuncs reports, without failing, the functions over longFunc
// lines (go test -v ./internal/archtest): splitting them changes code, so
// it is not part of the batches that only move it.
func TestLongFuncs(t *testing.T) {
	type long struct {
		name  string
		lines int
	}
	var all []long
	for _, f := range goFiles(t) {
		if f.test() || f.generated() {
			continue
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, f.path, f.data, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range file.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			n := fset.Position(fn.End()).Line - fset.Position(fn.Pos()).Line + 1
			if n > longFunc {
				all = append(all, long{fmt.Sprintf("%s:%d %s", f.path, fset.Position(fn.Pos()).Line, fn.Name.Name), n})
			}
		}
	}
	slices.SortFunc(all, func(a, b long) int { return b.lines - a.lines })
	for _, l := range all {
		t.Logf("%4d lines  %s", l.lines, l.name)
	}
}
