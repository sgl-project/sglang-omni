package archtest

import (
	"bufio"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The layering of internal/ (docs/architecture.md §1 and "Code structure
// rules"). Package names here are paths under internal/, "a/**" is a and
// the packages under it, "main" is the app at the module root. Imports
// are read from every non-test file whatever its build constraints.
const (
	mygoUI   = "github.com/egoist/mygo/ui"
	i18nPath = "github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
)

// leaves import no internal package: the bottom layer.
var leaves = []string{"core", "settings", "i18n", "lang", "permission", "macx", "vad",
	"appbundle", "helper/proto", "llmrt/wire", "trayicon", "store", "icons", "models"}

// uiPackages may import MyGo's UI. The rest of internal/ is front-end
// free: the domain, and the pure logic taken out of the app, testable
// without a window (the overlay's layout and ink, the data folder).
var uiPackages = []string{"icons"}

// textPackages may import i18n: they make the words the UI shows. The
// domain does not know them (it returns codes and errors), so that its
// logic is the same in every language.
var textPackages = []string{"uitext"}

// mygoOther are the internal packages that use one of MyGo's other
// packages, and which.
var mygoOther = map[string]string{
	"store/sqlitestore": "github.com/egoist/mygo/plugins/sqlite",
	// mygo.Rectangle only: the display's work area and the overlay
	// window's frame, which its geometry computes.
	"overlay/layout": "github.com/egoist/mygo",
}

// actnowKit is the ActNow family's shared module (actnowkit, the family
// spec's reference implementation); actnowUsers are the internal packages
// that use it: VoxtNew's part in the family, its model library, the words
// of its errors, the logs (actnowkit/redact). Its conformance package is
// for tests only.
const actnowKit = "github.com/hehehai/actnowkit"

var actnowUsers = []string{"actnow", "models", "models/**", "local", "uitext", "logs"}

// forbidden are imports that break the layering, and why.
var forbidden = []struct {
	from, to []string
	why      string
}{
	{[]string{"tools/**"}, []string{"**"},
		"the commands in internal/tools stand alone (standard library only)"},
}

// testHelper reports whether pkg is a fake for tests, which only tests and
// other fakes may import: a package whose name, or that of a package it is
// in, ends in "test" (helper/helpertest and its command).
func testHelper(pkg string) bool {
	for _, part := range strings.Split(pkg, "/") {
		if strings.HasSuffix(part, "test") && part != "archtest" {
			return true
		}
	}
	return false
}

// match reports whether pkg matches pattern: a/** is a and what is under
// it, ** anything.
func match(pattern, pkg string) bool {
	if pattern == "**" {
		return true
	}
	if base, ok := strings.CutSuffix(pattern, "/**"); ok {
		return pkg == base || strings.HasPrefix(pkg, base+"/")
	}
	return pkg == pattern
}

func matchAny(patterns []string, pkg string) bool {
	return slices.ContainsFunc(patterns, func(p string) bool { return match(p, pkg) })
}

// edge is a package's import of another, both under internal/ (or "main").
type edge struct{ from, to string }

func (e edge) String() string { return e.from + " " + e.to }

// readEdges reads imports.txt: the imports of an internal package by
// another, a line each.
func readEdges(t *testing.T) map[edge]int {
	t.Helper()
	f, err := os.Open("imports.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	edges := map[edge]int{}
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		text, _, _ := strings.Cut(sc.Text(), "#")
		fields := strings.Fields(text)
		if len(fields) == 0 {
			continue
		}
		if len(fields) != 2 {
			t.Fatalf("imports.txt:%d: want \"from to\", got %q", n, sc.Text())
		}
		e := edge{fields[0], fields[1]}
		if m, ok := edges[e]; ok {
			t.Fatalf("imports.txt:%d: %s is on line %d already", n, e, m)
		}
		edges[e] = n
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return edges
}

// TestImports checks the imports of the module's packages against the
// layering above and imports.txt.
func TestImports(t *testing.T) {
	_, module := moduleRoot(t)
	internal := module + "/internal/"
	listed := readEdges(t)
	found := map[edge]string{} // the edge and a file that makes it
	broken := map[edge]bool{}  // edges a rule forbids: no need to list them
	fail := map[string]bool{}
	failf := func(format string, args ...any) { fail[fmt.Sprintf(format, args...)] = true }
	for _, f := range goFiles(t) {
		if f.test() {
			continue
		}
		var from string
		switch dir := f.dir(); {
		case dir == ".":
			from = "main"
		case strings.HasPrefix(dir, "internal/"):
			from = strings.TrimPrefix(dir, "internal/")
		default:
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), f.path, f.data, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, spec := range file.Imports {
			path, _ := strconv.Unquote(spec.Path.Value)
			if from == "archtest" {
				continue
			}
			if path == mygoUI || strings.HasPrefix(path, mygoUI+"/") {
				if from != "main" && !matchAny(uiPackages, from) {
					failf("%s imports %s (%s): only %s may use MyGo's UI. Move the UI code to the app (package main), "+
						"and pass this package what it needs.", from, path, f.path, strings.Join(uiPackages, ", "))
				}
				continue
			}
			if path == i18nPath && from != "main" && from != "i18n" && !matchAny(textPackages, from) {
				failf("%s imports i18n (%s): only %s may make the UI's words. Return a code or an error, "+
					"and let package main or internal/uitext say it.", from, f.path, strings.Join(textPackages, ", "))
			}
			if strings.HasPrefix(path, "github.com/egoist/mygo") && from != "main" {
				if want, ok := mygoOther[from]; !ok || want != path {
					failf("%s imports %s (%s): internal packages use MyGo's packages only as listed in mygoOther (internal/archtest/imports_test.go); "+
						"add it there if this one should, in its own commit.", from, path, f.path)
				}
				continue
			}
			if path == actnowKit || strings.HasPrefix(path, actnowKit+"/") {
				switch {
				case strings.HasPrefix(path, actnowKit+"/conformance") || strings.HasPrefix(path, actnowKit+"/cmd"):
					failf("%s imports %s (%s): actnowkit's conformance suite and commands are for tests and tools only.", from, path, f.path)
				case from != "main" && !matchAny(actnowUsers, from):
					failf("%s imports %s (%s): only %s use actnowkit (internal/archtest/imports_test.go: actnowUsers); "+
						"add it there if this one should, in its own commit.", from, path, f.path, strings.Join(actnowUsers, ", "))
				}
				continue
			}
			to, ok := strings.CutPrefix(path, internal)
			if !ok {
				continue
			}
			if testHelper(to) && !testHelper(from) {
				failf("%s imports the test fake %s (%s): fakes are for tests (_test.go files) and other fakes only.", from, to, f.path)
			}
			if from == "main" {
				continue
			}
			e := edge{from, to}
			if _, ok := found[e]; !ok {
				found[e] = f.path
			}
			for _, r := range forbidden {
				if matchAny(r.from, from) && matchAny(r.to, to) {
					broken[e] = true
					failf("%s imports %s (%s), which it may not: %s.", from, to, f.path, r.why)
				}
			}
			if slices.Contains(leaves, from) {
				broken[e] = true
				failf("%s imports %s (%s): %s is a leaf package, which imports no internal package (internal/archtest/imports_test.go: leaves). "+
					"Move what it needs into it, or pass it in.", from, to, f.path, from)
			}
			if match("pages/**", from) && match("pages/**", to) && pageOf(from) != pageOf(to) {
				broken[e] = true
				failf("%s imports %s (%s): a page reaches another only through its Host, which the app implements.", from, to, f.path)
			}
		}
	}
	for e, file := range found {
		if _, ok := listed[e]; !ok && !broken[e] {
			failf("%s imports %s (%s), which internal/archtest/imports.txt does not list: if this dependency is right "+
				"(docs/architecture.md, \"Code structure rules\"), add the line \"%s\" there so that review sees it; "+
				"otherwise pass the package what it needs instead.", e.from, e.to, file, e)
		}
	}
	for e, n := range listed {
		if _, ok := found[e]; !ok {
			failf("%s no longer imports %s: remove line %d of internal/archtest/imports.txt.", e.from, e.to, n)
		}
	}
	msgs := make([]string, 0, len(fail))
	for m := range fail {
		msgs = append(msgs, m)
	}
	slices.Sort(msgs)
	for _, m := range msgs {
		t.Error(m)
	}
}

// pageOf is the page package pkg is in: pages/<page>.
func pageOf(pkg string) string {
	parts := strings.SplitN(pkg, "/", 3)
	if len(parts) < 2 {
		return pkg
	}
	return parts[0] + "/" + parts[1]
}

// TestImportRules checks the rules' own logic.
func TestImportRules(t *testing.T) {
	for _, c := range []struct {
		pattern, pkg string
		want         bool
	}{
		{"pages/**", "pages", true}, {"pages/**", "pages/calls", true}, {"pages/**", "pagesx", false},
		{"asr", "asr/openai", false}, {"**", "core", true},
	} {
		if got := match(c.pattern, c.pkg); got != c.want {
			t.Errorf("match(%q, %q) = %v", c.pattern, c.pkg, got)
		}
	}
	for pkg, want := range map[string]bool{"helper/helpertest": true, "helper/helpertest/cmd/fakehelper": true, "asr/asrtest": true, "archtest": false, "core": false} {
		if testHelper(pkg) != want {
			t.Errorf("testHelper(%q) = %v", pkg, !want)
		}
	}
	if pageOf("pages/calls/x") != "pages/calls" {
		t.Error(pageOf("pages/calls/x"))
	}
}
