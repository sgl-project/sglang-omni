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
// i18n.T, i18n.N, i18n.X, i18n.Platform, i18n.PlatformN, i18n.PlatformFor, i18n.PlatformNFor and an explicit
// translator's T whose message arguments are string literals,
// and returns the catalog keys they need, with a position for each.
// usage maps each key to the sorted module-relative paths of the files
// that use it. Calls with computed messages (i18n.T(name)) are returned
// in dynamic: their texts must be literal T calls somewhere else.
func sourceKeys(t *testing.T) (keys map[string]string, dynamic []string, usage map[string][]string) {
	t.Helper()
	keys = map[string]string{}
	files := map[string]map[string]bool{}
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
			if imp.Path.Value == `"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"` {
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
			if id, ok := sel.X.(*ast.Ident); !ok || id.Name != local && id.Name != "translator" {
				return true
			}
			pos := fset.Position(call.Pos()).String()
			rel := strings.TrimPrefix(filepath.ToSlash(path), moduleRoot+"/")
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
			add := func(k string) {
				keys[k] = pos
				if files[k] == nil {
					files[k] = map[string]bool{}
				}
				files[k][rel] = true
			}
			switch sel.Sel.Name {
			case "T":
				if s, ok := lit(0); ok {
					add(s)
				} else {
					dynamic = append(dynamic, pos)
				}
			case "Platform":
				mac, macOK := lit(0)
				windows, windowsOK := lit(1)
				if macOK && windowsOK {
					add(mac)
					add(windows)
				} else {
					dynamic = append(dynamic, pos)
				}
			case "PlatformFor":
				mac, macOK := lit(1)
				windows, windowsOK := lit(2)
				if macOK && windowsOK {
					add(mac)
					add(windows)
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
			case "PlatformN":
				macOne, macOneOK := lit(1)
				macOther, macOtherOK := lit(2)
				windowsOne, windowsOneOK := lit(3)
				windowsOther, windowsOtherOK := lit(4)
				if macOneOK && macOtherOK && windowsOneOK && windowsOtherOK {
					add(macOne)
					add(macOther)
					add(windowsOne)
					add(windowsOther)
				} else {
					dynamic = append(dynamic, pos)
				}
			case "PlatformNFor":
				macOne, macOneOK := lit(2)
				macOther, macOtherOK := lit(3)
				windowsOne, windowsOneOK := lit(4)
				windowsOther, windowsOtherOK := lit(5)
				if macOneOK && macOtherOK && windowsOneOK && windowsOtherOK {
					add(macOne)
					add(macOther)
					add(windowsOne)
					add(windowsOther)
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
	usage = map[string][]string{}
	for k, set := range files {
		for f := range set {
			usage[k] = append(usage[k], f)
		}
		sort.Strings(usage[k])
	}
	return keys, dynamic, usage
}

func readLocale(t *testing.T, tag string) map[string]string {
	t.Helper()
	m, err := readMessages(localeFS, tag)
	if err != nil {
		t.Fatal(err)
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
	keys, dynamic, _ := sourceKeys(t)
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

// catalogAreas are the areas with their own zh-Hans catalog file: the
// areas of the decoupling (docs/architecture.md), plus notes.
var catalogAreas = []string{
	"appenhance", "dictionary", "feature", "history", "home", "legacy",
	"meeting", "models", "notes", "onboarding", "overlay", "remote",
	"session", "settings", "shell", "updates",
}

// catalogAliases name a root file's area when its name prefix does not:
// meetings is the meeting area, update the updates one, try the feature
// area's try cards, tray the shell's, llm the remote (language models)
// area's debug page and wait dialog.
var catalogAliases = map[string]string{
	"meetings": "meeting",
	"update":   "updates",
	"try":      "feature",
	"tray":     "shell",
	"llm":      "remote",
}

// catalogFileOf names the zh-Hans catalog file (without .json) a key
// belongs in: the area of the files whose i18n calls use it, where a root
// file's area is its name up to the first "_" or "."; common when more
// than one area uses it or only shared code does (internal/, wiring, the
// UI kit).
func catalogFileOf(files []string) string {
	area := ""
	for _, f := range files {
		a := "common"
		if !strings.Contains(f, "/") {
			tok := f
			if i := strings.IndexAny(tok, "_."); i >= 0 {
				tok = tok[:i]
			}
			if alias, ok := catalogAliases[tok]; ok {
				tok = alias
			}
			if slices.Contains(catalogAreas, tok) {
				a = tok
			}
		}
		if area == "" {
			area = a
		} else if area != a {
			return "common"
		}
	}
	return area
}

// TestKeyPlacement: a key of a split locale lives in its area's catalog
// file (catalogFileOf) and in no other; the split exists to keep
// unrelated pull requests from conflicting on one zh-Hans.json.
// TestKeyParity covers a key missing from a locale; this covers one in
// the wrong file, and its error names the right one.
func TestKeyPlacement(t *testing.T) {
	_, _, usage := sourceKeys(t)
	for _, tag := range Supported() {
		entries, err := localeFS.ReadDir("locales/" + tag)
		if err != nil {
			continue // a single-file catalog: nothing to place
		}
		where := map[string]string{}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			name := e.Name()
			data, err := localeFS.ReadFile("locales/" + tag + "/" + name)
			if err != nil {
				t.Fatal(err)
			}
			m := map[string]string{}
			if err := json.Unmarshal(data, &m); err != nil {
				t.Fatalf("%s/%s: %v", tag, name, err)
			}
			for k := range m {
				if prev, dup := where[k]; dup {
					t.Errorf("%s: %q is in %s and %s", tag, k, prev, name)
					continue
				}
				where[k] = name
			}
		}
		for k, file := range where {
			used, ok := usage[k]
			if !ok {
				continue // unused: TestKeyParity reports it
			}
			if want := catalogFileOf(used) + ".json"; file != want {
				t.Errorf("%s: %q is in %s, belongs in %s (used in %s)",
					tag, k, file, want, strings.Join(used, ", "))
			}
		}
	}
}

// TestNoOldNames: the app is called Voxt and its text names no other
// brand: VoxtNew (its name before 2.0.0, still that of its data folders,
// processes and Swift packages), ActNow and actnow.dev are in no message (every
// message is a key of each translation: TestKeyParity) and no
// translation.
func TestNoOldNames(t *testing.T) {
	for _, tag := range Supported() {
		for k, v := range readLocale(t, tag) {
			for _, name := range []string{"VoxtNew", "ActNow", "actnow.dev"} {
				if strings.Contains(k, name) || strings.Contains(v, name) {
					t.Errorf("%s: %q names %s", tag, k, name)
				}
			}
		}
	}
}

// TestPlainChinese: the Chinese reads as written in Chinese, not
// translated word for word. Users found "及其", "以便", "无权", "将被" and
// a colon joining a request ("……：请……") stiff; each has a plain form
// ("和它们的", "方便", "没有……权限", "会", a comma).
func TestPlainChinese(t *testing.T) {
	for k, v := range readLocale(t, "zh-Hans") {
		for _, stiff := range []string{"及其", "以便", "无权", "将被", "：请"} {
			if strings.Contains(v, stiff) {
				t.Errorf("%q: %q has %q", k, v, stiff)
			}
		}
	}
}

// TestDockInChinese: macOS's Simplified Chinese calls the Dock 程序坞,
// and the strings use the same word.
func TestDockInChinese(t *testing.T) {
	for k, v := range readLocale(t, "zh-Hans") {
		if strings.Contains(v, "Dock") {
			t.Errorf("%q: %q says Dock, not 程序坞", k, v)
		}
	}
}

func TestVerbs(t *testing.T) {
	if got := verbs("%[2]s of %[1]d, %.1f%% done"); !slices.Equal(got, []string{"%.1f", "%d", "%s"}) {
		t.Error(got)
	}
}
