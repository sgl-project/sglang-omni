package main

// TestEvalDictLLM: does the glossary make a language model put in terms
// nobody said? Every short clip's words (as said) go through Transcribe's
// clean-up and the Chinese ones through spoken Translate, with three
// dictionaries as the glossary (dictSmall, dictMedium, dictLarge: past
// compose.GlossaryAllUpTo only the terms the text mentions are sent); a
// term in the answer that the clip does not say counts as put in. The
// translation model (no glossary) is checked for keeping the dictionary
// clips' terms. Also the user's dictation of 10-19 whose transcript ended
// in its context echoed (behavior.md 3.6a): what each model makes of it
// (eval_test.go; dict-llm.md).

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// echoed is the user's transcript of 10-19 10:17 (its end), the
// dictionary's three terms echoed after the last sentence.
const echoed = "但是Grok Bot，嗯，对于一些信息资料处理，就日常打交道和信息资料处理的，我还挺推荐的。嗯，但是话说回来，呃，这个Bot的这个文本处理还是很拉的，嗯。Voxt，Omni，SGLang。"

func TestEvalDictLLM(t *testing.T) {
	e := evalSetup(t)
	rt := evalLLMRuntime(t, nil)
	ctx := context.Background()
	var clips, spoken []eval.Clip
	for _, c := range e.clips() {
		if c.Set == "long" {
			continue
		}
		clips = append(clips, c)
		if c.Lang == "zh" {
			spoken = append(spoken, c)
		}
	}
	dicts := []struct {
		name  string
		terms []string
	}{{"small (3)", dictSmall}, {"medium (30)", dictMedium}, {fmt.Sprintf("large (%d)", len(dictLarge())), dictLarge()}}
	tb := table{head: []string{"model", "task", "dictionary", "answers", "with a term put in", "terms written", "median first token"}}
	put := table{head: []string{"model", "task", "dictionary", "clip", "answer", "put in"}}
	echo := table{head: []string{"model", "answer to the echoed transcript"}}
	for _, id := range e.llms(t, false) {
		b, err := llmrt.NewModel(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		client := &llm.Client{Backend: b}
		chat := func(req llm.Request, input string) (string, time.Duration) {
			var first time.Time
			req.OnDelta = func(string) {
				if first.IsZero() {
					first = time.Now()
				}
			}
			req.Timeout = 2 * time.Minute
			t0 := time.Now()
			resp, err := client.Chat(ctx, req)
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}
			return compose.Clean(resp.Text, input), first.Sub(t0)
		}
		if id == llmrt.Translate {
			written, want := 0, 0
			for _, c := range clips {
				if len(c.Terms) == 0 || c.Lang != "zh" {
					continue
				}
				out, _ := chat(llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: c.Text, Target: "en"}}, c.Text)
				found := eval.TermsFound(c.Terms, out)
				written, want = written+len(found), want+len(c.Terms)
				put.add(id, "translate", "none", c.ID, mdText(out), "")
			}
			tb.add(id, "translate", "none (the model takes none)", "", "", fmt.Sprintf("%d of %d", written, want), "")
			rt.Unload()
			continue
		}
		for _, d := range dicts {
			glossary := make([]compose.Term, len(d.terms))
			for i, term := range d.terms {
				glossary[i] = compose.Term{Term: term}
			}
			for _, task := range []string{"transcribe", "translate"} {
				var firsts []time.Duration
				answers, withPut, written, want := 0, 0, 0, 0
				set := clips
				if task == "translate" {
					set = spoken
				}
				for _, c := range set {
					main, others := "zh", []string{"en"}
					if c.Lang == "en" {
						main, others = "en", []string{"zh"}
					}
					in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: main, OtherLanguages: others, Glossary: glossary, Text: c.Text}
					skill := llm.SkillTranscribeEnhance
					if task == "translate" {
						in.Task, in.TargetLanguage, skill = compose.Translate, "en", llm.SkillTranslateText
					}
					req := chatRequest(t, in, true, skill)
					if task == "translate" {
						req.Translate = &llm.TranslateHint{Text: c.Text, Target: "en"}
					}
					out, first := chat(req, c.Text)
					firsts = append(firsts, first)
					answers++
					if extra := termsPutIn(d.terms, out, c.Text); len(extra) > 0 {
						withPut++
						put.add(id, task, d.name, c.ID, mdText(out), strings.Join(extra, ", "))
					}
					found := eval.TermsFound(c.Terms, out)
					written, want = written+len(found), want+len(c.Terms)
				}
				tb.add(id, task, d.name, answers, withPut, fmt.Sprintf("%d of %d", written, want), ms(median(firsts)))
			}
		}
		in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: "zh", OtherLanguages: []string{"en"},
			Glossary: []compose.Term{{Term: "Voxt"}, {Term: "Omni"}, {Term: "SGLang"}}, Text: echoed}
		out, _ := chat(chatRequest(t, in, true, llm.SkillTranscribeEnhance), echoed)
		echo.add(id, mdText(out))
		rt.Unload()
	}
	e.report(t, "dict-llm.md", "# The glossary: terms put in\n\n"+tb.String()+"\nAnswers with a term put in (the translation model: every answer):\n\n"+
		put.String()+"\nThe user's transcript with its context echoed (dictionary Voxt, Omni, SGLang):\n\n"+echo.String())
}
