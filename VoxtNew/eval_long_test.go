package main

// TestEvalLong: long dictation (the set's long clips, 95–104 s, by their
// words) on the language models: Transcribe's clean-up as one request and
// in pieces (compose.BuildParts, as the session sends them, after warming
// the shared prefix as the session does while the user speaks; the
// prompt cache's hits are counted), spoken Translate into English as one
// request (Hy-MT2: the speech without its fillers). Then whole clean-ups
// of the long clips' first clips, 3 to 13 of them: where a whole
// clean-up starts to fail. The full eval only (eval_test.go); the report
// is long.md.

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// longCase is a long request of the eval, whole or in parts, checked.
type longCase struct {
	llmCase
	parts int
	// piece is the time to the first part's answer: when the overlay
	// shows the first cleaned piece.
	piece time.Duration
	// hits are the parts whose prompt the cache served.
	hits int
	// slowest is the longest part's request, firstMax its longest wait for
	// a first token and gap the longest wait between two tokens: what
	// the session's deadlines bound (behavior.md 5.5).
	slowest, firstMax, gap time.Duration
	tokens                 int
}

func TestEvalLong(t *testing.T) {
	e := evalSetup(t)
	if e.quick {
		t.Skip("long dictation: the full eval only")
	}
	var longs []eval.Clip
	for _, c := range e.clips() {
		if c.Set == "long" {
			longs = append(longs, c)
		}
	}
	if len(longs) == 0 {
		t.Skip("no long clip selected")
	}
	var hits atomic.Int32
	rt := evalLLMRuntime(t, func(l core.LogEntry) {
		if strings.Contains(l.Message, "prompt cache hit") && !strings.Contains(l.Message, " prefill:") {
			hits.Add(1)
		}
	})
	ctx := context.Background()
	terms := evalTerms(e)
	var cases []longCase
	for _, id := range e.llms(t, false) {
		b, err := llmrt.NewModel(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Warm(ctx); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		client := &llm.Client{Backend: b}
		// run sends in whole or in pieces, each part as the session does,
		// and checks the joined answer.
		run := func(task, variant string, c eval.Clip, in compose.Input, pieces bool, check func(out string) eval.Verdict) longCase {
			whole, err := compose.Build(in)
			if err != nil {
				t.Fatal(err)
			}
			parts := []compose.Part{{Output: whole, Piece: compose.Piece{Text: in.Text}}}
			if pieces {
				parts, _ = compose.BuildParts(in)
			}
			lc := longCase{llmCase: llmCase{model: id, task: task, variant: variant, clip: c}, parts: len(parts)}
			skill := llm.SkillTranscribeEnhance
			if task == "translate" {
				skill = llm.SkillTranslateText
			}
			warm := in
			warm.Text = ""
			if out, err := compose.Build(warm); err == nil {
				if err := client.Prefill(ctx, llm.Request{Skill: skill, Messages: evalMessages(out)}); err != nil {
					t.Logf("%s: warming up: %v", id, err)
				}
			}
			before := hits.Load()
			var answers []string
			t0 := time.Now()
			for _, p := range parts {
				req := llm.Request{Skill: skill, Messages: evalMessages(p.Output), Timeout: 3 * time.Minute}
				if task == "translate" {
					req.Translate = &llm.TranslateHint{Text: compose.StripFillers(p.Text), Target: in.TargetLanguage}
				}
				start := time.Now()
				last := start
				req.OnDelta = func(string) {
					now := time.Now()
					if lc.first == 0 {
						lc.first = now.Sub(t0)
					}
					if last == start {
						lc.firstMax = max(lc.firstMax, now.Sub(start))
					} else {
						lc.gap = max(lc.gap, now.Sub(last))
					}
					last = now
				}
				resp, err := client.Chat(ctx, req)
				if err != nil {
					t.Fatalf("%s %s %s: %v", id, task, c.ID, err)
				}
				lc.slowest = max(lc.slowest, time.Since(start))
				lc.tokens += resp.Usage.CompletionTokens
				answers = append(answers, compose.Clean(resp.Text, p.Text))
				if lc.piece == 0 {
					lc.piece = time.Since(t0)
				}
			}
			lc.total = time.Since(t0)
			lc.hits = int(hits.Load() - before)
			lc.out = compose.JoinParts(parts, answers)
			lc.verdict = check(lc.out)
			cases = append(cases, lc)
			return lc
		}
		for _, c := range longs {
			if id != llmrt.Translate {
				for _, pieces := range []bool{false, true} {
					in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: "zh", OtherLanguages: []string{"en"}, Glossary: terms, Text: c.Text}
					run("transcribe", map[bool]string{false: "whole", true: "pieces"}[pieces], c, in, pieces, func(out string) eval.Verdict {
						_, answer, _ := compose.Example(in)
						return eval.CheckClean(c, c.Text, out, answer)
					})
				}
			}
			in := compose.Input{Task: compose.Translate, Cleanup: true, MainLanguage: "zh", OtherLanguages: []string{"en"}, TargetLanguage: "en", Glossary: terms, Text: c.Text}
			run("translate", "whole", c, in, false, func(out string) eval.Verdict {
				_, answer, _ := compose.Example(in)
				return eval.CheckTranslation(c.MustEn, c.MustNotEn, c.Text, out, answer)
			})
		}
		if id != llmrt.Translate {
			// The threshold: whole clean-ups of the first n clips.
			for _, l := range e.set.Long {
				for n := 3; n < len(l.Of); n += 2 {
					part := l
					part.ID, part.Of = fmt.Sprintf("%s[:%d]", l.ID, n), l.Of[:n]
					c := e.set.Join(part)
					in := compose.Input{Task: compose.Transcribe, Cleanup: true, MainLanguage: "zh", OtherLanguages: []string{"en"}, Glossary: terms, Text: c.Text}
					check := func(out string) eval.Verdict {
						_, answer, _ := compose.Example(in)
						return eval.CheckClean(c, c.Text, out, answer)
					}
					run("transcribe", "first clips", c, in, false, check)
					if parts, _ := compose.BuildParts(in); len(parts) > 1 {
						run("transcribe", "first clips, pieces", c, in, true, check)
					}
				}
			}
		}
		rt.Unload()
	}
	e.report(t, "long.md", longReport(cases))
}

// evalTerms is the eval dictionary as the glossary.
func evalTerms(e *evalRun) []compose.Term {
	terms := make([]compose.Term, len(e.set.Dictionary))
	for i, d := range e.set.Dictionary {
		terms[i] = compose.Term{Term: d}
	}
	return terms
}

// evalMessages are a composed request's messages.
func evalMessages(out compose.Output) []llm.Message {
	var msgs []llm.Message
	for _, m := range out.Messages() {
		msgs = append(msgs, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
	}
	return msgs
}

func longReport(cases []longCase) string {
	var b strings.Builder
	b.WriteString("# Long dictation\n\n")
	tb := table{head: []string{"model", "task", "prompt", "clip", "characters", "parts", "cache hits", "faults", "error vs expected", "first token", "first part", "total", "tokens/s", "slowest part", "longest first token", "longest gap"}}
	for _, c := range cases {
		faults, errCell := c.verdict.Faults(), "–"
		if faults == "" {
			faults = "none"
		}
		if c.task == "transcribe" {
			errCell = pct(c.verdict.Score.Rate())
		}
		tb.add(c.model, c.task, c.variant, c.clip.ID, len([]rune(c.clip.Text)), c.parts, c.hits, mdText(faults), errCell, ms(c.first), ms(c.piece), ms(c.total),
			fmt.Sprintf("%.1f", float64(c.tokens)/c.total.Seconds()), ms(c.slowest), ms(c.firstMax), ms(c.gap))
	}
	b.WriteString(tb.String() + "\nEvery answer:\n\n")
	at := table{head: []string{"model", "task", "prompt", "clip", "answer"}}
	for _, c := range cases {
		at.add(c.model, c.task, c.variant, c.clip.ID, mdText(c.out))
	}
	b.WriteString(at.String())
	return b.String()
}
