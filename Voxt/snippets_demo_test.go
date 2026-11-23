package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestSnippetsDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "retry", fixture: "en/snippets-error", steps: historySteps("Try again"), want: []string{"loadSnippets"}},
		{name: "import", fixture: "en/snippets-normal", steps: historySteps("Import snippets"), want: []string{"importSnippets"}},
		{name: "add", fixture: "en/snippets-normal", steps: historySteps("Add snippet"), check: func(t *testing.T, host *meetingDemoHost) {
			if !host.snipUI.editOpen || host.snipUI.edit.id != "" {
				t.Fatalf("dialog %+v", host.snipUI)
			}
		}},
		{name: "example", fixture: "en/snippets-empty", steps: historySteps("my work email"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.snipUI.edit.trigger != "my work email" {
				t.Fatalf("dialog %+v", host.snipUI)
			}
		}},
		{name: "open row", fixture: "en/snippets-normal", steps: historySteps("my email → name@example.com"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.snipUI.edit.id != "s-email" {
				t.Fatalf("dialog %+v", host.snipUI)
			}
		}},
		{name: "save", fixture: "en/snippets-edit", steps: historySteps("Save"), want: []string{`saveSnippet "s-sign" "sign off" "Best,\nAlex"`}},
		{name: "dialog delete", fixture: "en/snippets-edit", steps: historySteps("Delete"), want: []string{"deleteSnippet s-sign"}},
		{name: "clear search", fixture: "en/snippets-search-empty", steps: historySteps("Clear search"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.snipUI.query != "" {
				t.Fatalf("query %q", host.snipUI.query)
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				if err := tester.ClickControl(step.label); err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

// TestSnippetsDemoSwitch: another fixture does not keep the page's
// search or dialog.
func TestSnippetsDemoSwitch(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/snippets-edit")
	if !host.snipUI.editOpen || host.snipUI.edit.id != "s-sign" {
		t.Fatalf("edit fixture %+v", host.snipUI)
	}
	selectMeetingDemoFixture(t, host, "en/snippets-search")
	if host.snipUI.editOpen || host.snipUI.query != "zoom" {
		t.Fatalf("search fixture %+v", host.snipUI)
	}
	selectMeetingDemoFixture(t, host, "en/notes-normal")
	if host.snipUI.query != "" || host.snipUI.editOpen {
		t.Fatalf("notes fixture kept %+v", host.snipUI)
	}
}

// snippetsDemoSentinel is the text a Snippets fixture must show ("" for
// another).
func snippetsDemoSentinel(state meetingFixtureState) string {
	snippets, local := state.Snippets, state.SnippetsUI
	switch {
	case state.Kind != "snippets":
		return ""
	case local.Error != "":
		return local.Error
	case local.EditOpen && local.ID != "":
		return i18n.T("Edit snippet")
	case local.EditOpen:
		return i18n.T("Add snippet")
	case snippets.Error != "":
		return snippets.Error
	case !snippets.Ready:
		return i18n.T("Say a trigger and Voxt types the text you saved, as it is.")
	case snippets.Total == 0:
		return i18n.T("No snippets yet")
	case len(snippets.Rows) == 0:
		return i18n.T("No snippet matches “%s”", local.Query)
	}
	return snippets.Rows[0].Trigger
}
