package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestBehaviours: testdata/behaviours.tsv lists what a user can do with Voxt
// and the check that covers each. Levels: L runs on every pull request with
// fake models and a fake helper, M runs on a Mac by hand, E is an evaluation
// with real models, manual is a page of steps. An L or E check is a test; an
// M check is a test or a script; a manual check is the page.
func TestBehaviours(t *testing.T) {
	tests := testNames(t)
	b, err := os.ReadFile("testdata/behaviours.tsv")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(string(b), "\n"), "\n")
	if lines[0] != "feature\tlevel\tcheck\tgap" {
		t.Fatalf("header = %q", lines[0])
	}
	features := map[string]bool{}
	for i, line := range lines[1:] {
		f := strings.Split(line, "\t")
		if len(f) != 4 || f[0] == "" || f[3] == "" {
			t.Errorf("line %d: want feature, level, check, gap: %q", i+2, line)
			continue
		}
		features[f[0]] = true
		_, statErr := os.Stat(f[2])
		switch f[1] {
		case "L", "E":
			if !tests[f[2]] {
				t.Errorf("line %d: %s: no test named %s", i+2, f[0], f[2])
			}
		case "M":
			if !tests[f[2]] && statErr != nil {
				t.Errorf("line %d: %s: %s is neither a test nor a file", i+2, f[0], f[2])
			}
		case "manual":
			if statErr != nil {
				t.Errorf("line %d: %s: no page %s", i+2, f[0], f[2])
			}
		default:
			t.Errorf("line %d: level %q, want L, M, E or manual", i+2, f[1])
		}
	}
	if len(features) < 32 {
		t.Errorf("%d features listed, want at least 32", len(features))
	}
}

// TestBehaviourGapsReflectCoverage keeps closed gaps out of the manifest.
func TestBehaviourGapsReflectCoverage(t *testing.T) {
	data, err := os.ReadFile("testdata/behaviours.tsv")
	if err != nil {
		t.Fatal(err)
	}
	covered := map[string]string{
		"The fake helper has no selection that is only whitespace":                "TestRewriteBlankSelectionReadsTheCopy",
		"No test plays a recording through a whole meeting":                       "TestMeetingPlaythrough",
		"No self-check that the catalogue and the interface list the same models": "TestModelsPageMatchesCatalogue",
		"The fake system dictation never ends with an empty result":               "TestSessionSystemDictation",
		"No check that the menu lists only installed models":                      "TestSavedOllamaSetupMenuIsTheServersModels",
		"No test walks the whole guide; fixtures only":                            "TestOnboardingWalk",
		"No test lists every address a default session contacts":                  "TestDefaultSessionContactsNoRemoteHost",
	}
	for i, line := range strings.Split(string(data), "\n") {
		fields := strings.Split(line, "\t")
		if len(fields) != 4 {
			continue
		}
		if check, ok := covered[fields[3]]; ok {
			t.Errorf("line %d: %s covers this gap; list it and clear the stale gap", i+1, check)
		}
	}
}

// testNames is every test function of the module's test files and of the
// helper's Swift tests (helper/Tests), which the helper's CI job runs.
func testNames(t *testing.T) map[string]bool {
	t.Helper()
	tests := map[string]bool{}
	goTest := regexp.MustCompile(`(?m)^func (Test\w*)\(`)
	swiftTest := regexp.MustCompile(`(?m)^\s*func (test\w*)\(\)`)
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		fn := goTest
		switch {
		case err != nil:
			return err
		case strings.HasSuffix(path, "_test.go"):
		case strings.HasPrefix(path, "helper/Tests/") && strings.HasSuffix(path, ".swift"):
			fn = swiftTest
		default:
			return nil
		}
		b, err := os.ReadFile(path)
		for _, m := range fn.FindAllSubmatch(b, -1) {
			tests[string(m[1])] = true
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return tests
}

// TestReports: testdata/reports.tsv lists problems that users reported and,
// for each, the check that shows the problem when its fix is reverted. A
// row keeps that check from being renamed or deleted unnoticed. Levels are
// those of TestBehaviours.
func TestReports(t *testing.T) {
	tests := testNames(t)
	b, err := os.ReadFile("testdata/reports.tsv")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(string(b), "\n"), "\n")
	if lines[0] != "report\tfix\tsymptom\tlevel\tcheck" {
		t.Fatalf("header = %q", lines[0])
	}
	number := regexp.MustCompile(`^#[0-9]+$`)
	seen := map[string]bool{}
	for i, line := range lines[1:] {
		f := strings.Split(line, "\t")
		if len(f) != 5 || !number.MatchString(f[0]) || !number.MatchString(f[1]) || f[2] == "" {
			t.Errorf("line %d: want issue, pull request, symptom, level, check: %q", i+2, line)
			continue
		}
		if seen[f[0]+"\t"+f[4]] {
			t.Errorf("line %d: %s with %s is listed twice", i+2, f[0], f[4])
		}
		seen[f[0]+"\t"+f[4]] = true
		_, statErr := os.Stat(f[4])
		switch f[3] {
		case "L", "E":
			if !tests[f[4]] {
				t.Errorf("line %d: %s: no test named %s", i+2, f[0], f[4])
			}
		case "M":
			if !tests[f[4]] && statErr != nil {
				t.Errorf("line %d: %s: %s is neither a test nor a file", i+2, f[0], f[4])
			}
		case "manual":
			if statErr != nil {
				t.Errorf("line %d: %s: no page %s", i+2, f[0], f[4])
			}
		default:
			t.Errorf("line %d: level %q, want L, M, E or manual", i+2, f[3])
		}
	}
}
