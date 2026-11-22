package session

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

var testSnippets = []snippet.Snippet{
	{ID: "1", Trigger: "my email address", Text: "ruiling@example.com"},
	{ID: "2", Trigger: "intro email", Text: "Hey, would love to find some time to chat."},
}

func withSnippets(c *Config, _ *settings.Settings) {
	c.Snippets = func(context.Context) ([]snippet.Snippet, error) { return testSnippets, nil }
}

// TestSnippets: a dictation that is only a trigger is its snippet as
// saved, without the language model; a trigger in a dictation is
// replaced after the clean-up, which therefore cannot change the snippet;
// the history keeps the speech as recognised.
func TestSnippets(t *testing.T) {
	for _, c := range []struct {
		name, said, cleaned, result string
		asked                       bool // the language model was asked
	}{
		{"only a trigger", "My email address.", "unused", "ruiling@example.com", false},
		{"in a sentence", "send it to my email address please", "Send it to my email address, please.", "Send it to ruiling@example.com, please.", true},
		{"none", "hello there", "Hello there.", "Hello there.", true},
		// The model kept the marks but changed the words inside them.
		{"changed inside", "send it to my email address please", "Send it to ⟦my e-mail⟧, please.", "Send it to ruiling@example.com, please.", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			model := &fakeLLM{text: c.cleaned}
			h := newHarness(t, &fakeFile{text: c.said}, model, withSnippets)
			h.speak(h.start(), 5)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != c.result || end.Raw != c.said {
				t.Fatalf("%s: result %q, raw %q", end.State, end.Result, end.Raw)
			}
			if asked := len(model.requests()) > 0; asked != c.asked {
				t.Errorf("the language model asked: %v, want %v", asked, c.asked)
			}
			if e := h.hist.all(); len(e) != 1 || e[0].Result != c.result || e[0].RawText != c.said {
				t.Errorf("history %+v", e)
			}
		})
	}
}

// TestSnippetsWithoutModel: without a language model the triggers are
// replaced in the text as recognised.
func TestSnippetsWithoutModel(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "here is the intro email for you"}, nil, withSnippets)
	h.speak(h.start(), 5)
	if end := h.rec.wait(t); end.Result != "here is the Hey, would love to find some time to chat. for you" {
		t.Fatalf("result %q", end.Result)
	}
}

// TestSnippetsMarkedForTheModel: the model gets the triggers marked and
// the rule to keep them; a dictation without one gets neither.
func TestSnippetsMarkedForTheModel(t *testing.T) {
	for _, c := range []struct {
		said string
		mark bool
	}{{"send it to my email address please", true}, {"hello there", false}} {
		model := &fakeLLM{text: "Done."}
		h := newHarness(t, &fakeFile{text: c.said}, model, withSnippets)
		h.speak(h.start(), 5)
		h.rec.wait(t)
		reqs := model.requests()
		if len(reqs) == 0 {
			t.Fatalf("%q: the model was not asked", c.said)
		}
		var all strings.Builder
		for _, m := range reqs[len(reqs)-1].Messages {
			all.WriteString(m.Content)
		}
		got := all.String()
		if marked := strings.Contains(got, "⟦my email address⟧"); marked != c.mark {
			t.Errorf("%q: trigger marked %v, want %v", c.said, marked, c.mark)
		}
		if rule := strings.Contains(got, "is a trigger for one of the user's saved snippets"); rule != c.mark {
			t.Errorf("%q: rule in the prompt %v, want %v", c.said, rule, c.mark)
		}
	}
}
