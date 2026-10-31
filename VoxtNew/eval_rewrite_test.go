package main

// TestEvalRewrite: Rewrite on the general language models (eval_test.go;
// rewrite.md). A few selections with a spoken instruction (polite and
// shorter, grammar, a list, formal, shorter, friendlier) and two answers
// written from an instruction alone (the answer card's JSON), in English
// and Chinese, each sampled with two seeds (one in quick mode). An answer
// has a fault when it loses a fact the selection or instruction gave
// (must), keeps what the instruction asked to fix (mustNot), answers in
// another language, is not shorter when asked to be, is no list when
// asked for one, opens with a preamble ("Sure, here is…"), or, from an
// instruction alone, is not the JSON answer. Wording and tone are left to
// the reader: the report lists every answer.

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
)

// rewriteCase is one Rewrite request and what its answer must do.
type rewriteCase struct {
	id, lang          string // lang: the selection's (else the instruction's) language
	instruction, text string
	must, mustNot     []string // must: "a|b" is either
	shorter, list     bool
	answer            bool // written from the instruction: the JSON answer
}

var rewriteCases = []rewriteCase{
	{id: "en-polite", lang: "en", instruction: "make it more polite and a bit shorter",
		text: "send me the report by friday, i told you twice already and it's still not here, this is getting ridiculous",
		must: []string{"report", "friday"}, mustNot: []string{"ridiculous"}, shorter: true},
	{id: "en-grammar", lang: "en", instruction: "fix the grammar",
		text: "me and him goes to the store yesterday and buyed three apple, then we was going home.",
		must: []string{"went|go", "bought", "apples"}, mustNot: []string{"buyed", " goes ", "we was"}},
	{id: "en-list", lang: "en", instruction: "uh turn this into a bulleted list",
		text: "For the trip we need to book the hotel, rent a car, buy travel insurance and print the tickets.",
		must: []string{"hotel", "car", "insurance", "tickets"}, list: true},
	{id: "zh-formal", lang: "zh", instruction: "改得正式一点",
		text: "老板，明天的会我去不了了，家里有点事，材料我今晚发你邮箱。",
		must: []string{"明天|明日", "材料|资料", "邮箱|邮件"}},
	{id: "zh-shorter", lang: "zh", instruction: "嗯，帮我把这段，呃，改短一点",
		text: "关于下周的团建活动，我们初步考虑了几个方案，第一个是去郊区爬山，第二个是去市里的密室逃脱，第三个是在公司附近吃火锅，大家可以在周五之前把自己的意见告诉我，我会统计一下然后再做决定。",
		must: []string{"周五", "爬山", "密室", "火锅"}, shorter: true},
	{id: "zh-friendly", lang: "zh", instruction: "make it sound friendlier",
		text: "请于本周五前提交报销单，逾期不予处理。",
		must: []string{"周五", "报销"}},
	{id: "zh-answer", lang: "zh", instruction: "帮我写一条消息告诉团队周五的会议改到下周一上午十点",
		must: []string{"周一", "十点|10"}, answer: true},
	{id: "en-answer", lang: "en", instruction: "write a short thank-you note to Anna for fixing the login bug so quickly",
		must: []string{"Anna", "login"}, answer: true},
}

// evalRewriteFloor is quick mode's least share of faultless answers on
// llm.balanced: a regression, not a target.
const evalRewriteFloor = 0.6

type rewriteResult struct {
	model string
	c     rewriteCase
	seed  int64
	out   string
	fault []string
	first time.Duration
	total time.Duration
}

func TestEvalRewrite(t *testing.T) {
	e := evalSetup(t)
	rt := evalLLMRuntime(t, nil)
	ctx := context.Background()
	seeds := []int64{7, 8}
	if e.quick {
		seeds = seeds[:1]
	}
	var results []rewriteResult
	for _, id := range e.llms(t, true) {
		b, err := llmrt.NewModel(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.Warm(ctx); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		client := &llm.Client{Backend: b}
		for _, c := range rewriteCases {
			main := map[string]string{"zh": "zh", "en": "en"}[c.lang]
			in := compose.Input{Task: compose.Rewrite, Cleanup: true, MainLanguage: main, Instruction: c.instruction, Text: c.text, Answer: c.answer}
			for _, seed := range seeds {
				req := chatRequest(t, in, true, llm.SkillRewrite)
				req.Seed, req.Timeout = &seed, 2*time.Minute
				if c.answer {
					req.Format = llm.FormatJSON
				}
				r := rewriteResult{model: id, c: c, seed: seed}
				t0 := time.Now()
				req.OnDelta = func(string) {
					if r.first == 0 {
						r.first = time.Since(t0)
					}
				}
				resp, err := client.Chat(ctx, req)
				if err != nil {
					t.Fatalf("%s %s: %v", id, c.id, err)
				}
				r.total = time.Since(t0)
				r.out, r.fault = checkRewrite(c, resp.Text)
				results = append(results, r)
			}
		}
		rt.Unload()
	}
	e.report(t, "rewrite.md", rewriteReport(results, e.quick))
	if e.quick {
		ok, n := 0, 0
		for _, r := range results {
			if r.model == llmrt.Balanced {
				n++
				if len(r.fault) == 0 {
					ok++
				}
			}
		}
		if n > 0 && float64(ok)/float64(n) < evalRewriteFloor {
			t.Errorf("llm.balanced: %d of %d rewrites without a fault, under the quick floor %.0f%%", ok, n, 100*evalRewriteFloor)
		}
	}
}

// checkRewrite is the answer as the session takes it (the JSON answer's
// content) and its faults.
func checkRewrite(c rewriteCase, text string) (string, []string) {
	var faults []string
	out := compose.Clean(text, c.text)
	if c.answer {
		a, ok := compose.ParseAnswer(text)
		if !ok || a.Title == "" {
			faults = append(faults, "no JSON answer")
		} else {
			out = a.Content
		}
	}
	low := strings.ToLower(out)
	for _, m := range c.must {
		found := false
		for _, alt := range strings.Split(m, "|") {
			found = found || strings.Contains(low, strings.ToLower(alt))
		}
		if !found {
			faults = append(faults, "lost "+m)
		}
	}
	for _, m := range c.mustNot {
		if strings.Contains(" "+low+" ", strings.ToLower(m)) {
			faults = append(faults, "kept "+strings.TrimSpace(m))
		}
	}
	if share := hanShare(out); c.lang == "zh" && share < 0.5 || c.lang == "en" && share > 0.05 {
		faults = append(faults, fmt.Sprintf("language (%.0f%% Chinese)", 100*share))
	}
	if c.shorter && utf8.RuneCountInString(out) >= utf8.RuneCountInString(c.text) {
		faults = append(faults, "not shorter")
	}
	if c.list && listItems(out) < 3 {
		faults = append(faults, "no list")
	}
	for _, p := range []string{"sure", "here is", "here's", "certainly", "以下是", "好的", "当然"} {
		if strings.HasPrefix(low, p) {
			faults = append(faults, "preamble")
			break
		}
	}
	return out, faults
}

// hanShare is the share of text's letters that are Chinese characters.
func hanShare(text string) float64 {
	han, letters := 0, 0
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Han, r):
			han++
			letters++
		case unicode.IsLetter(r):
			letters++
		}
	}
	if letters == 0 {
		return 0
	}
	return float64(han) / float64(letters)
}

// listItems counts the lines that start like a list's items.
func listItems(text string) int {
	n := 0
	for _, l := range strings.Split(text, "\n") {
		l = strings.TrimSpace(l)
		if strings.HasPrefix(l, "- ") || strings.HasPrefix(l, "* ") || strings.HasPrefix(l, "• ") ||
			len(l) > 2 && l[0] >= '1' && l[0] <= '9' && (l[1] == '.' || l[1] == ')') {
			n++
		}
	}
	return n
}

func rewriteReport(results []rewriteResult, quick bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Rewrite%s\n\n", map[bool]string{true: " (quick)", false: ""}[quick])
	tb := table{head: []string{"model", "n", "no fault", "selections", "answers (JSON)", "first token (median)", "total (median)"}}
	var order []string
	groups := map[string][]rewriteResult{}
	for _, r := range results {
		if len(groups[r.model]) == 0 {
			order = append(order, r.model)
		}
		groups[r.model] = append(groups[r.model], r)
	}
	for _, id := range order {
		var ok, sel, selOK, ans, ansOK int
		var firsts, totals []time.Duration
		for _, r := range groups[id] {
			good := len(r.fault) == 0
			if good {
				ok++
			}
			if r.c.answer {
				ans++
				if good {
					ansOK++
				}
			} else {
				sel++
				if good {
					selOK++
				}
			}
			firsts, totals = append(firsts, r.first), append(totals, r.total)
		}
		tb.add(id, len(groups[id]), ok, fmt.Sprintf("%d of %d", selOK, sel), fmt.Sprintf("%d of %d", ansOK, ans), ms(median(firsts)), ms(median(totals)))
	}
	b.WriteString(tb.String() + "\nAnswers with a fault:\n\n")
	ft := table{head: []string{"model", "case", "seed", "answer", "faults"}}
	at := table{head: []string{"model", "case", "seed", "answer"}}
	for _, r := range results {
		if len(r.fault) > 0 {
			ft.add(r.model, r.c.id, r.seed, mdText(r.out), strings.Join(r.fault, "; "))
		}
		at.add(r.model, r.c.id, r.seed, mdText(r.out))
	}
	b.WriteString(ft.String() + "\nEvery answer:\n\n" + at.String())
	return b.String()
}
