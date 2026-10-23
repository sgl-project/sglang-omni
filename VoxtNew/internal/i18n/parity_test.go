package i18n

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// moduleRoot is the repository root, two levels up from this package.
const moduleRoot = "../.."

// sourceKeys scans every non-test .go file of the module for calls to
// i18n.T, i18n.N and i18n.X whose message arguments are string literals,
// and returns the catalog keys they need, with a position for each.
// Calls with computed messages (i18n.T(name)) are returned in dynamic:
// their texts must be literal T calls somewhere else.
func sourceKeys(t *testing.T) (keys map[string]string, dynamic []string) {
	t.Helper()
	keys = map[string]string{}
	fset := token.NewFileSet()
	err := filepath.WalkDir(moduleRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			top := filepath.Dir(path) == moduleRoot
			if path != moduleRoot && (strings.HasPrefix(name, ".") || top && (name == "helper" || name == "third_party") || name == "build" || name == "testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		// The package's own name for i18n: "i18n" unless renamed; inside
		// package i18n itself calls are unqualified and not scanned.
		local := ""
		for _, imp := range f.Imports {
			if imp.Path.Value == `"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"` {
				local = "i18n"
				if imp.Name != nil {
					local = imp.Name.Name
				}
			}
		}
		if local == "" {
			return nil
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if id, ok := sel.X.(*ast.Ident); !ok || id.Name != local {
				return true
			}
			pos := fset.Position(call.Pos()).String()
			lit := func(i int) (string, bool) {
				if i >= len(call.Args) {
					return "", false
				}
				bl, ok := call.Args[i].(*ast.BasicLit)
				if !ok || bl.Kind != token.STRING {
					return "", false
				}
				s, err := strconv.Unquote(bl.Value)
				return s, err == nil
			}
			add := func(k string) { keys[k] = pos }
			switch sel.Sel.Name {
			case "T":
				if s, ok := lit(0); ok {
					add(s)
				} else {
					dynamic = append(dynamic, pos)
				}
			case "N":
				one, ok1 := lit(1)
				other, ok2 := lit(2)
				if ok1 && ok2 {
					add(one)
					add(other)
				} else {
					dynamic = append(dynamic, pos)
				}
			case "X":
				ctx, ok1 := lit(0)
				msg, ok2 := lit(1)
				if ok1 && ok2 {
					add(ctx + "\x04" + msg)
				} else {
					dynamic = append(dynamic, pos)
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return keys, dynamic
}

func readLocale(t *testing.T, tag string) map[string]string {
	t.Helper()
	data, err := localeFS.ReadFile("locales/" + tag + ".json")
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]string{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("%s: %v", tag, err)
	}
	return m
}

var verbRE = regexp.MustCompile(`%(?:\[\d+\])?[-+# 0]*\d*(?:\.\d+)?[a-zA-Z%]`)

// verbs lists the formatting verbs of s, without argument indexes, sorted:
// a translation may reorder its arguments with %[2]s but must use the
// same ones.
func verbs(s string) []string {
	var out []string
	for _, v := range verbRE.FindAllString(s, -1) {
		if v == "%%" {
			continue
		}
		if i := strings.Index(v, "]"); i > 0 {
			v = "%" + v[i+1:]
		}
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// TestKeyParity fails when a supported locale lacks a message the sources
// use, has one they no longer use, or formats it with other verbs.
func TestKeyParity(t *testing.T) {
	keys, dynamic := sourceKeys(t)
	for _, pos := range dynamic {
		t.Logf("computed message at %s: its texts must be literal T calls elsewhere", pos)
	}
	if len(keys) < 50 {
		t.Fatalf("found only %d messages: is the scan broken?", len(keys))
	}
	if en := readLocale(t, "en"); len(en) != 0 {
		t.Errorf("en.json must stay empty (English is the source): %d entries", len(en))
	}
	for _, tag := range Supported() {
		if tag == "en" {
			continue
		}
		m := readLocale(t, tag)
		var missing []string
		for k, pos := range keys {
			v, ok := m[k]
			switch {
			case !ok || v == "":
				missing = append(missing, strconv.Quote(k)+" ("+pos+")")
			case !slices.Equal(verbs(k), verbs(v)):
				t.Errorf("%s: %q → %q: verbs %v ≠ %v", tag, k, v, verbs(k), verbs(v))
			}
		}
		sort.Strings(missing)
		for _, k := range missing {
			t.Errorf("%s lacks %s", tag, k)
		}
		for k := range m {
			if _, ok := keys[k]; !ok {
				t.Errorf("%s has %q, which nothing uses", tag, k)
			}
		}
	}
}

func TestVerbs(t *testing.T) {
	if got := verbs("%[2]s of %[1]d, %.1f%% done"); !slices.Equal(got, []string{"%.1f", "%d", "%s"}) {
		t.Error(got)
	}
}
