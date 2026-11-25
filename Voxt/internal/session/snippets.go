package session

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// insertSnippet delivers text's snippet when text is only its trigger
// ("My email address."): as saved, not the language model's text, which
// would rewrite it (an early clean-up may have asked the model already).
// It reports whether it did.
func (r *run) insertSnippet(text string) bool {
	s, ok := snippet.Whole(text, r.snippets)
	if !ok {
		return false
	}
	r.logf("info", "the snippet %s was inserted for its trigger", r.logText(s.Trigger))
	r.finish(Done, nil, snippet.Alone(s.Text), r.transcribeOpts())
	return true
}

// expandSnippets replaces the triggers in a Transcribe's result with
// their snippets, after the language model, so that it cannot change
// them. text is what the model was given, unmarked: the model got its
// triggers marked (transcribeInput), and a mark it kept is its snippet
// even where it changed the words inside (snippet.Restore). Only the
// triggers said are replaced: one the model wrote itself ("my mail" made
// "my email") stays as written.
func (r *run) expandSnippets(text, result string) string {
	_, said := snippet.Mark(text, r.snippets)
	if len(said) == 0 {
		return result
	}
	expanded := snippet.Restore(result, said, said)
	if expanded != result {
		r.logf("info", "snippets replaced their triggers: %s", r.logText(expanded))
	}
	return expanded
}

// draftSnippets is a clean-up streaming in with its marked triggers as
// their snippets, so the card shows no marks; parts are what the model
// was given (a draft of text without marks is left as it is).
func (r *run) draftSnippets(draft string, parts []compose.Part) string {
	var said []snippet.Snippet
	for _, p := range parts {
		if strings.Contains(p.Text, snippet.Open) {
			_, s := snippet.Mark(snippet.Unmark(p.Text), r.snippets)
			said = append(said, s...)
		}
	}
	if len(said) == 0 {
		return draft
	}
	return snippet.Restore(draft, nil, said)
}
