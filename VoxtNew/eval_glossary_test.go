package main

// TestEvalGlossary: does the clean-up write the dictionary's terms when
// the speech model got them wrong? Each dictionary clip's transcript as a
// speech model wrote it without the dictionary (eval.Clip.Heard) goes
// through Transcribe's clean-up with the whole eval dictionary as the
// glossary (and the alias the user noted for Auralink), as it was heard
// and with the dictionary's aliases replaced first (dict.Replace, as the
// session does); clips without terms check that no term is put in where
// none was said. Twice: with the eval dictionary (9 terms, the glossary
// in the system prompt) and with dictLarge (the glossary of the terms the
// text mentions; #24), timing the first token too (eval_test.go;
// glossary.md).

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dict"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

func TestEvalGlossary(t *testing.T) {
	e := evalSetup(t)
	rt := evalLLMRuntime(t, nil)
	ctx := context.Background()
	// The set's dictionary, and the same terms among dictLarge's (past
	// compose.GlossaryAllUpTo: only the terms the text mentions).
	glossaries := []struct {
		name  string
		words []string
	}{{"eval (9)", e.set.Dictionary}, {fmt.Sprintf("large (%d)", len(dictLarge())), dictLarge()}}
	var cases []eval.Clip
	for _, c := range e.set.Clips {
		if c.Heard != "" || c.Set == "zh-plain" || c.ID == "en-3" {
			cases = append(cases, c)
		}
	}
	tb := table{head: []string{"model", "dictionary", "input", "terms written", "terms put in unsaid", "median first token", "median total"}}
	answers := table{head: []string{"model", "dictionary", "input", "clip", "heard", "answer"}}
	for _, id := range e.llms(t, true) {
		b, err := llmrt.NewModel(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		client := &llm.Client{Backend: b}
		for _, g := range glossaries {
			var terms []compose.Term
			var dictionary []store.Term
			for _, d := range g.words {
				terms = append(terms, compose.Term{Term: d, Aliases: e.set.Aliases[d]})
				dictionary = append(dictionary, store.Term{Term: d, Aliases: e.set.Aliases[d], Enabled: true})
			}
			for _, aliases := range []bool{false, true} {
				variant := map[bool]string{false: "as heard", true: "aliases replaced"}[aliases]
				written, want, unsaid := 0, 0, 0
				var totals, firsts []time.Duration
				for _, c := range cases {
					heard := c.Heard
					if heard == "" {
						heard = c.Text
					}
					if aliases {
						heard = dict.Replace(heard, dictionary)
					}
					in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: c.Lang, Glossary: terms, Text: heard}
					req := chatRequest(t, in, true, llm.SkillTranscribeEnhance)
					req.Timeout = 2 * time.Minute
					var first time.Time
					req.OnDelta = func(string) {
						if first.IsZero() {
							first = time.Now()
						}
					}
					t0 := time.Now()
					resp, err := client.Chat(ctx, req)
					if err != nil {
						t.Fatalf("%s %s: %v", id, c.ID, err)
					}
					totals, firsts = append(totals, time.Since(t0)), append(firsts, first.Sub(t0))
					out := compose.Clean(resp.Text, heard)
					found := eval.TermsFound(c.Terms, out)
					written, want = written+len(found), want+len(c.Terms)
					unsaid += len(termsPutIn(g.words, out, c.Text))
					answers.add(id, g.name, variant, c.ID, mdText(heard), mdText(out))
				}
				tb.add(id, g.name, variant, fmt.Sprintf("%d of %d", written, want), unsaid, ms(median(firsts)), ms(median(totals)))
				if e.quick && id == llmrt.Balanced && g.name == glossaries[0].name && written*2 < want {
					t.Errorf("%s (%s) wrote %d of %d terms, under the quick floor of half", id, variant, written, want)
				}
			}
		}
		rt.Unload()
	}
	e.report(t, "glossary.md", "# The glossary on misheard terms\n\n"+tb.String()+"\n"+answers.String())
}
