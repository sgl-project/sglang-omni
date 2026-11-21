package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// snippetsViewState is the Snippets page: its rows, the examples an empty
// page offers, and what went wrong reading or saving.
type snippetsViewState struct {
	Loading  bool                      `json:"loading"`
	Error    string                    `json:"error,omitempty"`
	Rows     []snippetRowViewState     `json:"rows"`
	Examples []snippetExampleViewState `json:"examples,omitempty"`
}

// snippetRowViewState is one snippet: its trigger, and its text on one
// line for the list (Preview) and whole for the dialog.
type snippetRowViewState struct {
	ID      string `json:"id"`
	Trigger string `json:"trigger"`
	Text    string `json:"text"`
	Preview string `json:"preview"`
}

// snippetExampleViewState is a trigger an empty page suggests, and what
// its text could be.
type snippetExampleViewState struct {
	Trigger string `json:"trigger"`
	Hint    string `json:"hint"`
}

func buildSnippetsViewState(translator *i18n.Translator, list snippet.List, loaded bool, err string) snippetsViewState {
	state := snippetsViewState{Loading: !loaded && err == "", Error: err, Rows: []snippetRowViewState{}}
	for _, s := range list.Snippets {
		state.Rows = append(state.Rows, snippetRowViewState{
			ID: s.ID, Trigger: s.Trigger, Text: s.Text, Preview: strings.Join(strings.Fields(s.Text), " "),
		})
	}
	if loaded && len(state.Rows) == 0 {
		state.Examples = []snippetExampleViewState{
			{translator.T("my email"), "name@example.com"},
			{translator.T("my address"), translator.T("Your street, city and postcode")},
			{translator.T("meeting link"), "https://zoom.us/j/…"},
		}
	}
	return state
}

// snippetProblemText says why a snippet cannot be saved (snippet.Check).
func snippetProblemText(translator *i18n.Translator, problem string) string {
	switch problem {
	case snippet.ProblemNoTrigger:
		return translator.T("Type what you will say.")
	case snippet.ProblemNoText:
		return translator.T("Type the text it stands for.")
	case snippet.ProblemLong:
		return translator.T("Too long: a trigger takes %d characters, a snippet %d.", snippet.MaxTrigger, snippet.MaxText)
	case snippet.ProblemTaken:
		return translator.T("Another snippet has this trigger.")
	}
	return problem
}
