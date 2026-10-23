package session

import (
	"context"
	"errors"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// draftEvery is how often a streaming answer is published at most.
const draftEvery = 40 * time.Millisecond

// answerSchema is Rewrite's JSON answer.
var answerSchema = func() *llm.Schema {
	s := llm.Object(map[string]*llm.Schema{"title": llm.String(), "content": llm.String()}, "title", "content")
	s.Name = "answer"
	return s
}()

// writeInput is Translate's or Rewrite's request to compose: raw and
// selection as write's.
func (r *run) writeInput(raw, selection string) compose.Input {
	in := compose.Input{
		MainLanguage: MainLanguage(r.settings, r.m.cfg.Locale()), OtherLanguages: r.settings.General.OtherLanguages,
		Glossary: r.glossary(), Scene: r.composeScene(), // the group decides whether it applies to the feature (appctx)
	}
	switch r.req.Kind {
	case store.KindTranslate:
		f := r.settings.Features.Translate
		in.Task, in.TaskPrompt, in.TargetLanguage = compose.Translate, f.Prompt, r.target
		if selection != "" {
			in.Text, in.Selected = selection, true
		} else {
			in.Text, in.Cleanup = raw, f.Cleanup
		}
	case store.KindRewrite:
		f := r.settings.Features.Rewrite
		in.Task, in.TaskPrompt, in.Cleanup, in.Instruction, in.Text = compose.Rewrite, f.Prompt, f.Cleanup, raw, selection
		in.Previous = r.req.Previous
		// Written from an instruction: an answer with a title; a follow-up
		// answers in plain text, as a conversation does (Voxt's).
		in.Answer = selection == "" && len(in.Previous) == 0
	}
	return in
}

// write is Translate's and Rewrite's language-model step: raw is what
// was said (Translate's speech, Rewrite's instruction; "" when Translate
// translates a selection), selection the selected text. The answer
// streams into the snapshot's Draft (and into the answer card when it
// will end there); a failure ends the session failed, in the card, to
// try again.
func (r *run) write(raw, selection string) {
	cfg := r.m.cfg
	kind := r.req.Kind
	if r.svc.LLM == nil {
		r.endWriteFailed(&Problem{Code: ProblemNoLLM, Err: errors.New("no language model is ready")}, false)
		return
	}
	in := r.writeInput(raw, selection)
	var o finishOpts
	switch kind {
	case store.KindTranslate:
		o.replace = selection != ""
		if r.settings.Features.Translate.ShowCard {
			o.card = deliver.ReasonCard
		}
	case store.KindRewrite:
		// Every answer goes to the answer card, which inserts it (or
		// replaces the selection) when asked, and takes follow-ups (Voxt's:
		// a Rewrite result always shows in the card).
		o.card = deliver.ReasonCard
	}
	if r.req.Card {
		o.card = deliver.ReasonCard
	}
	r.m.update(r, func(s *Snapshot) { s.State, s.Selection, s.Card = Composing, selection, o.card != "" && !r.req.Try })

	// Can it replace the selection? If not, the answer streams into the
	// card at once. Other answers go where the user is typing when they
	// are ready (finish): the user may still switch apps meanwhile.
	if o.card == "" && o.replace && !r.req.Try {
		ctx, cancel := context.WithTimeout(r.ctx, cfg.DeliverTime)
		res := deliver.Check(ctx, cfg.Text, deliver.Request{Text: "-", Target: r.req.Target, Self: cfg.Self, Replace: true})
		cancel()
		if res.Reason != deliver.ReasonNone && r.ctx.Err() == nil {
			r.logf("info", "the answer goes to the card: %s", res.Reason)
			o.card = res.Reason
			r.m.update(r, func(s *Snapshot) { s.Card, s.Delivery = true, res.Reason })
		}
	}

	source := in.Text
	if kind == store.KindRewrite {
		source = selection
	}
	llmStart := r.now()
	res, profile, err := r.ask(in)
	if err == nil && kind == store.KindTranslate && compose.Untranslated(source, res.Content, in.TargetLanguage) && r.ctx.Err() == nil {
		r.logf("warning", "the translation looks untranslated: asking again, strictly")
		r.m.update(r, func(s *Snapshot) { s.Retried, s.Draft = true, "" })
		in.Strict = true
		if again, p, err2 := r.ask(in); err2 == nil {
			res, profile = again, p
		} else if r.ctx.Err() == nil {
			r.logf("warning", "the strict retry failed, keeping the first answer: %v", err2)
		}
	}
	llmTime := r.now().Sub(llmStart)
	if r.cancelled.Load() {
		r.endCancelled(nil)
		return
	}
	r.m.update(r, func(s *Snapshot) { s.Profile, s.Timings.LLM = profile, llmTime })
	if err != nil {
		r.logf("warning", "language model %s (model %s) failed after %v: %v", r.svc.LLMInfo.ID, r.svc.LLMInfo.Model, llmTime.Round(time.Millisecond), err)
		r.endWriteFailed(&Problem{Code: ProblemLLM, Err: err}, true)
		return
	}
	r.logf("info", "wrote %s in %v", r.logText(res.Content), llmTime.Round(time.Millisecond))
	if res.Content == "" {
		if kind == store.KindTranslate && selection == "" {
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return
		}
		r.endWriteFailed(&Problem{Code: ProblemEmpty, Err: errors.New("the language model answered nothing")}, true)
		return
	}
	r.m.update(r, func(s *Snapshot) { s.Title, s.Draft = res.Title, res.Content })
	r.finish(Done, nil, res.Content, o)
}

// ask sends one request and streams its answer into the snapshot. It
// returns the answer (Content only, but for a JSON answer) and the
// composition's profile.
func (r *run) ask(in compose.Input) (compose.Answer, string, error) {
	cfg := r.m.cfg
	out, err := compose.Build(in)
	if err != nil {
		return compose.Answer{}, "", err
	}
	msgs := make([]llm.Message, 0, len(out.History)+2)
	for _, m := range out.Messages() {
		msgs = append(msgs, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
	}
	skill := llm.SkillTranslateText
	if in.Task == compose.Rewrite {
		skill = llm.SkillRewrite
	}
	var buf strings.Builder
	var last time.Time
	publish := func(force bool) {
		now := time.Now()
		if !force && now.Sub(last) < draftEvery {
			return
		}
		last = now
		var a compose.Answer
		if in.Answer {
			a = compose.PartialAnswer(buf.String())
		} else {
			a.Content = strings.TrimSpace(buf.String())
		}
		r.m.update(r, func(s *Snapshot) { s.Draft, s.Title = a.Content, a.Title })
	}
	req := llm.Request{
		Messages: msgs, Skill: skill,
		OnDelta: func(d string) {
			buf.WriteString(d)
			publish(false)
		},
	}
	if in.Answer {
		req.Format, req.Schema = llm.FormatJSON, answerSchema
	}
	if in.Task == compose.Translate {
		// For a model that builds its own prompt and takes no clean-up
		// rules (Hy-MT2): speech without its fillers (docs/eval.md).
		text := in.Text
		if !in.Selected {
			text = compose.StripFillers(text)
		}
		req.Translate = &llm.TranslateHint{Text: text, Target: in.TargetLanguage}
	}
	resp, err := r.llmChat(cfg.llmDeadlines(req, utf8.RuneCountInString(in.Text+in.Instruction), cfg.WriteTimeout))
	if err != nil {
		return compose.Answer{}, out.Profile(), err
	}
	if buf.Len() > 0 {
		publish(true)
	}
	if in.Answer {
		if a, ok := compose.ParseAnswer(resp.Text); ok {
			return a, out.Profile(), nil
		}
		r.logf("warning", "the answer is not the JSON asked for: taking it as text")
		return compose.Answer{Content: compose.Clean(strings.TrimSpace(strings.Trim(strings.TrimSpace(resp.Text), "`")), in.Instruction)}, out.Profile(), nil
	}
	input := in.Text
	return compose.Answer{Content: compose.Clean(resp.Text, input)}, out.Profile(), nil
}

// endWriteFailed ends a Translate or Rewrite session failed. With card,
// the answer card shows the error and offers to try again (not for a
// try); the entry is kept whatever the recording's length: it holds the
// text.
func (r *run) endWriteFailed(p *Problem, card bool) {
	r.cancel()
	r.logf("warning", "failed: %s: %v", p.Code, p.Err)
	total := r.now().Sub(r.m.Current().StartedAt)
	r.m.end(r, Failed, func(s *Snapshot) {
		s.Problem, s.Timings.Total = p, total
		s.Card = card && !r.req.Try
		s.Draft = ""
	}, func(s Snapshot) { r.keep(s, store.StatusFailed) })
}
