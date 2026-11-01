package main

// TestEvalLLM: the language models' clean-up on the eval set's short
// clips (eval_test.go; the long ones: TestEvalLong), from each clip's
// reference text (what was said, word for word): Transcribe's clean-up
// with compose's example turn and without it (and the English clips with
// Chinese as the main language), spoken Translate into
// English likewise, and Hy-MT2 (llm.translate) given the raw speech, the
// speech without its fillers (compose.StripFillers) or llm.balanced's
// clean-up. Every answer is checked (internal/eval: fillers,
// self-corrections, invented words, the example copied); the report is
// llm.md.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// llmCase is one request of the eval and its checked answer.
type llmCase struct {
	model, task, variant string
	clip                 eval.Clip
	out                  string
	verdict              eval.Verdict
	first, total         time.Duration
}

// evalLLMFloor is quick mode's least share of faultless clean-ups with
// the example (llm.balanced): a regression, not a target.
const evalLLMFloor = 0.6

func TestEvalLLM(t *testing.T) {
	e := evalSetup(t)
	rt := evalLLMRuntime(t, nil)
	ctx := context.Background()
	var clips, spoken []eval.Clip
	for _, c := range e.clips() {
		if c.Set == "long" {
			continue // TestEvalLong
		}
		clips = append(clips, c)
		if c.Set == "zh-filler" || c.Set == "zh-plain" {
			spoken = append(spoken, c)
		}
	}
	terms := evalTerms(e)
	var cases []llmCase
	cleaned := map[string]string{} // llm.balanced's clean-up of each clip, for Hy-MT2
	mem := table{head: []string{"model", "load", "footprint", "peak footprint", "MLX peak"}}
	for _, id := range e.llms(t, false) {
		b, err := llmrt.NewModel(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if err := b.Warm(ctx); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		load := time.Since(t0)
		client := &llm.Client{Backend: b}
		run := func(task, variant string, c eval.Clip, req llm.Request, check func(out string) eval.Verdict) llmCase {
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
				t.Fatalf("%s %s %s: %v", id, task, c.ID, err)
			}
			out := compose.Clean(resp.Text, c.Text)
			lc := llmCase{model: id, task: task, variant: variant, clip: c, out: out, verdict: check(out), first: first.Sub(t0), total: time.Since(t0)}
			cases = append(cases, lc)
			return lc
		}
		if id == llmrt.Translate {
			for _, c := range spoken {
				for _, v := range []struct{ name, text string }{{"raw", c.Text}, {"no fillers", compose.StripFillers(c.Text)}, {"balanced clean-up", cleaned[c.ID]}} {
					if v.text == "" {
						continue
					}
					req := llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: v.text, Target: "en"}}
					run("translate", v.name, c, req, func(out string) eval.Verdict {
						return eval.CheckTranslation(c.MustEn, c.MustNotEn, c.Text, out, "")
					})
				}
			}
			mem.add(evalMemory(ctx, rt, id, load)...)
			rt.Unload()
			continue
		}
		for _, c := range clips {
			main, others := "zh", []string{"en"}
			if c.Lang == "en" {
				main, others = "en", []string{"zh"}
			}
			in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: main, OtherLanguages: others, Glossary: terms, Text: c.Text}
			for _, example := range []bool{true, false} {
				lc := run("transcribe", exampleName(example), c, chatRequest(t, in, example, llm.SkillTranscribeEnhance), func(out string) eval.Verdict {
					_, answer, _ := compose.Example(in)
					return eval.CheckClean(c, c.Text, out, answer)
				})
				if id == llmrt.Balanced && example {
					cleaned[c.ID] = lc.out
				}
			}
			if c.Lang == "en" {
				// An English sentence from a user whose main language is
				// Chinese: the Chinese template set, which must not
				// translate it.
				zh := in
				zh.MainLanguage, zh.OtherLanguages = "zh", []string{"en"}
				run("transcribe", "example, main zh", c, chatRequest(t, zh, true, llm.SkillTranscribeEnhance), func(out string) eval.Verdict {
					_, answer, _ := compose.Example(zh)
					return eval.CheckClean(c, c.Text, out, answer)
				})
			}
		}
		for _, c := range spoken {
			in := compose.Input{Task: compose.Translate, Cleanup: true, MainLanguage: "zh", OtherLanguages: []string{"en"}, TargetLanguage: "en",
				Glossary: terms, Text: c.Text}
			for _, example := range []bool{true, false} {
				req := chatRequest(t, in, example, llm.SkillTranslateText)
				req.Translate = &llm.TranslateHint{Text: c.Text, Target: "en"}
				run("translate", exampleName(example), c, req, func(out string) eval.Verdict {
					_, answer, _ := compose.Example(in)
					return eval.CheckTranslation(c.MustEn, c.MustNotEn, c.Text, out, answer)
				})
			}
		}
		mem.add(evalMemory(ctx, rt, id, load)...)
		rt.Unload()
	}
	e.report(t, "llm.md", llmReport(cases, e.quick)+"\nMemory after the model's chats:\n\n"+mem.String())
	if e.quick {
		ok, n := 0, 0
		for _, c := range cases {
			if c.model == llmrt.Balanced && c.task == "transcribe" && c.variant == "example" {
				n++
				if c.verdict.OK() {
					ok++
				}
			}
		}
		if n > 0 && float64(ok)/float64(n) < evalLLMFloor { // n 0: VOXTNEW_EVAL_LLM left llm.balanced out
			t.Errorf("llm.balanced: %d of %d clean-ups without a fault, under the quick floor %.0f%%", ok, n, 100*evalLLMFloor)
		}
	}
}

// evalLLMRuntime is voxtnew-llm ($VOXTNEW_LLM, else ~/voxtnew-runtime's),
// stopped at the test's end; log gets its log lines (nil: none).
func evalLLMRuntime(t *testing.T, log func(core.LogEntry)) *llmrt.Runtime {
	bin := os.Getenv("VOXTNEW_LLM")
	if bin == "" {
		home, _ := os.UserHomeDir()
		bin = filepath.Join(home, "voxtnew-runtime", "llm", llmrt.BinName)
	}
	if log == nil {
		log = func(core.LogEntry) {}
	}
	rt := llmrt.New(llmrt.Config{Bin: bin, AppVersion: "eval", LogDir: t.TempDir(), IdleTimeout: -1, Log: log})
	t.Cleanup(rt.Stop)
	return rt
}

// evalMemory is a memory table's row: the process's footprint (the
// runtime's memory call) after model id's chats, and its load time.
func evalMemory(ctx context.Context, rt *llmrt.Runtime, id string, load time.Duration) []any {
	res, err := rt.Memory(ctx)
	if err != nil {
		return []any{id, ms(load), err, "", ""}
	}
	m := res.Memory
	return []any{id, ms(load), fmt.Sprintf("%d MB", m.FootprintBytes>>20), fmt.Sprintf("%d MB", m.PeakFootprintBytes>>20), fmt.Sprintf("%d MB", m.MLXPeakBytes>>20)}
}

func exampleName(on bool) string {
	if on {
		return "example"
	}
	return "no example"
}

// chatRequest is compose's request for in, without the example turn when
// example is off (the prompts before M3).
func chatRequest(t *testing.T, in compose.Input, example bool, skill llm.Skill) llm.Request {
	t.Helper()
	out, err := compose.Build(in)
	if err != nil {
		t.Fatal(err)
	}
	if !example {
		out.History = nil
	}
	return llm.Request{Skill: skill, Messages: evalMessages(out)}
}

func llmReport(cases []llmCase, quick bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Language models' clean-up%s\n\n", map[bool]string{true: " (quick)", false: ""}[quick])
	type key struct{ model, task, variant, clips string }
	var order []key
	groups := map[key][]llmCase{}
	for _, c := range cases {
		clips := "short"
		if c.clip.Set == "long" {
			clips = "long"
		}
		k := key{c.model, c.task, c.variant, clips}
		if len(groups[k]) == 0 {
			order = append(order, k)
		}
		groups[k] = append(groups[k], c)
	}
	tb := table{head: []string{"model", "task", "prompt", "clips", "n", "no fault", "fillers left", "corrections missed", "invented", "language changed", "copied the example", "error vs expected", "first token (median)", "total (median)"}}
	for _, k := range order {
		g := groups[k]
		var ok, fillers, kept, invented, language, copied int
		var sum eval.Score
		var firsts, totals []time.Duration
		for _, c := range g {
			v := c.verdict
			if v.OK() {
				ok++
			}
			if len(v.Fillers) > 0 {
				fillers++
			}
			if v.Corrections() > 0 {
				kept++
			}
			if len(v.Invented) > 1 {
				invented++
			}
			if v.Copied {
				copied++
			}
			if v.Language {
				language++
			}
			sum = sum.Add(v.Score)
			firsts, totals = append(firsts, c.first), append(totals, c.total)
		}
		errCell := "–"
		if k.task == "transcribe" {
			errCell = pct(sum.Rate())
		}
		tb.add(k.model, k.task, k.variant, k.clips, len(g), ok, fillers, kept, invented, language, copied, errCell, ms(median(firsts)), ms(median(totals)))
	}
	b.WriteString(tb.String())
	b.WriteString("\nAnswers with a fault:\n\n")
	ft := table{head: []string{"model", "task", "prompt", "clip", "answer", "faults"}}
	for _, c := range cases {
		if !c.verdict.OK() {
			ft.add(c.model, c.task, c.variant, c.clip.ID, mdText(c.out), mdText(c.verdict.Faults()))
		}
	}
	b.WriteString(ft.String() + "\nEvery answer:\n\n")
	at := table{head: []string{"model", "task", "prompt", "clip", "answer"}}
	for _, c := range cases {
		at.add(c.model, c.task, c.variant, c.clip.ID, mdText(c.out))
	}
	b.WriteString(at.String())
	return b.String()
}
