package main

import (
	"errors"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// snippetsFixtureInput is a Snippets page fixture: the list as read, what
// went wrong ("read" or "save"), and the page's search, sort and dialog.
type snippetsFixtureInput struct {
	list    snippet.List
	loaded  bool
	err     string
	order   snippetOrder
	ui      snippetsFixtureUI
	problem string // the dialog's (snippet.Check)
	// importErr: the import dialog says the last file was not snippets.
	importErr bool
}

func snippetsFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	add := func(name string, mutate func(*snippetsFixtureInput)) {
		input := snippetsFixtureInput{list: snippetsFixtureList(), loaded: true}
		mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "snippets", snippets: input})
	}
	add("snippets-loading", func(input *snippetsFixtureInput) { input.list, input.loaded = snippet.List{}, false })
	add("snippets-error", func(input *snippetsFixtureInput) { input.list, input.loaded, input.err = snippet.List{}, false, "read" })
	add("snippets-empty", func(input *snippetsFixtureInput) { input.list = snippet.List{} })
	add("snippets-normal", func(input *snippetsFixtureInput) {})
	add("snippets-save-error", func(input *snippetsFixtureInput) { input.err = "save" })
	add("snippets-search", func(input *snippetsFixtureInput) { input.ui.Query = "zoom" })
	add("snippets-search-empty", func(input *snippetsFixtureInput) { input.ui.Query = "nothing" })
	add("snippets-by-trigger", func(input *snippetsFixtureInput) { input.order = snippetsByTrigger })
	add("snippets-long", func(input *snippetsFixtureInput) {
		input.list.Snippets = append(input.list.Snippets, snippet.Snippet{ID: "s-long",
			Trigger: "insert the long project status update for the weekly meeting",
			Text:    strings.Repeat("The release is on track; the last blockers are the installer and the docs. ", 6)})
	})
	add("snippets-add", func(input *snippetsFixtureInput) { input.ui.EditOpen = true })
	add("snippets-edit", func(input *snippetsFixtureInput) {
		s := input.list.Snippets[2]
		input.ui = snippetsFixtureUI{EditOpen: true, ID: s.ID, Trigger: s.Trigger, Text: s.Text}
	})
	add("snippets-edit-taken", func(input *snippetsFixtureInput) {
		input.ui = snippetsFixtureUI{EditOpen: true, Trigger: "sign off", Text: "Cheers,"}
		input.problem = snippet.ProblemTaken
	})
	add("snippets-import", func(input *snippetsFixtureInput) { input.ui.ImportOpen = true })
	add("snippets-import-error", func(input *snippetsFixtureInput) {
		input.ui.ImportOpen, input.ui.ImportExample, input.importErr = true, true, true
	})
	return scenarios
}

func snippetsFixtureList() snippet.List {
	return snippet.List{Snippets: []snippet.Snippet{
		{ID: "s-email", Trigger: "my email", Text: "name@example.com"},
		{ID: "s-link", Trigger: "my meeting link", Text: "https://zoom.us/j/1234567890"},
		{ID: "s-sign", Trigger: "sign off", Text: "Best,\nAlex"},
	}}
}

// buildSnippetsFixture is the page's view state and local state for
// input, its messages in translator's language.
func buildSnippetsFixture(translator *i18n.Translator, input snippetsFixtureInput) (*snippetsViewState, *snippetsFixtureUI) {
	err := ""
	switch input.err {
	case "read":
		err = translator.T("Could not read the snippets: %v", errors.New("fixture read failed"))
	case "save":
		err = translator.T("Could not save the snippets: %v", errors.New("fixture write failed"))
	}
	state := buildSnippetsViewState(translator, input.list, input.loaded, err, input.ui.Query, input.order)
	ui := input.ui
	if input.problem != "" {
		ui.Error = snippetProblemText(translator, input.problem)
	}
	if input.importErr {
		ui.ImportError = translator.T("Could not import %s: it is not a JSON list of snippets.", "terms.json")
	}
	return &state, &ui
}
