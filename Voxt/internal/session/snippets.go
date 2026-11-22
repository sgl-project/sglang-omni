package session

import "github.com/sgl-project/sglang-omni/Voxt/internal/snippet"

// insertSnippet delivers text's snippet when text is only its trigger
// ("My email address."): as saved, without the language model, which
// would rewrite it. It reports whether it did.
func (r *run) insertSnippet(text string) bool {
	s, ok := snippet.Whole(text, r.snippets)
	if !ok {
		return false
	}
	r.logf("info", "the snippet %s was inserted for its trigger", r.logText(s.Trigger))
	r.finish(Done, nil, s.Text, r.transcribeOpts())
	return true
}

// expandSnippets replaces the triggers in a Transcribe's result with
// their snippets, after the language model, so that it cannot change
// them. text is what the model was given, unmarked: the model got its
// triggers marked (transcribeInput), and a mark it kept is its snippet
// even where it changed the words inside (snippet.Restore).
func (r *run) expandSnippets(text, result string) string {
	_, marked := snippet.Mark(text, r.snippets)
	expanded := snippet.Restore(result, marked, r.snippets)
	if expanded != result {
		r.logf("info", "snippets replaced their triggers: %s", r.logText(expanded))
	}
	return expanded
}

// draftSnippets is a clean-up streaming in with its marked triggers as
// their snippets, so the card shows no marks.
func (r *run) draftSnippets(draft string) string {
	if len(r.snippets) == 0 {
		return draft
	}
	return snippet.Restore(draft, nil, r.snippets)
}
