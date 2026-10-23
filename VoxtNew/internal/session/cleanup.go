package session

import (
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/dict"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
)

// spoken is the recognised text with the dictionary's aliases replaced
// by their terms (dict.Replace), when the dictionary is on: before the
// clean-up, so it holds without a language model too. The snapshot's
// Raw, and the history's, stay as recognised.
func (r *run) spoken(raw string) string {
	if !r.settings.Dictionary.Enabled {
		return raw
	}
	text := dict.Replace(raw, r.terms)
	if text != raw {
		r.logf("info", "the dictionary's aliases replaced: %s", r.logText(text))
		r.m.update(r, func(s *Snapshot) { s.Partial = text })
	}
	return text
}

// transcribe cleans up text with the language model (unless the feature
// has it off or no model is ready) and delivers it. A long dictation is
// cleaned up in pieces (compose.BuildParts); where the model fails, the
// text goes on as recognised, with a warning.
func (r *run) transcribe(text string) {
	result := text
	var problem *Problem
	if f := r.settings.Features.Transcribe; f.Enhance && r.svc.LLM != nil {
		r.m.update(r, func(s *Snapshot) { s.State = Composing })
		llmStart := r.now()
		cleaned, profile, err := r.cleanUp(text)
		llmTime := r.now().Sub(llmStart)
		if r.cancelled.Load() {
			r.endCancelled(nil)
			return
		}
		r.m.update(r, func(s *Snapshot) { s.Profile, s.Timings.LLM = profile, llmTime })
		result = cleaned
		switch {
		case err != nil:
			r.logf("warning", "language model %s (%s) failed after %v, the text goes on as recognised: %v", r.svc.LLMInfo.ID, r.svc.LLMInfo.Model, llmTime.Round(time.Millisecond), err)
			problem = &Problem{Code: ProblemLLM, Err: err, Degraded: true}
		case result == "":
			r.finish(Done, &Problem{Code: ProblemNoSpeech}, "", finishOpts{})
			return
		default:
			r.logf("info", "composed %s in %v", r.logText(result), llmTime.Round(time.Millisecond))
		}
	}
	r.finish(Done, problem, result, finishOpts{})
}

// cleanUp runs the language model over text, a request per part, each
// answer streaming into the snapshot's Draft after those before it. A
// part that fails keeps its text as recognised; err is the first
// failure (the text itself when the request could not be composed).
func (r *run) cleanUp(text string) (string, string, error) {
	parts, err := compose.BuildParts(r.transcribeInput(text))
	if err != nil {
		return text, "", err
	}
	if len(parts) > 1 {
		r.logf("info", "cleaning up %d characters in %d pieces", len([]rune(text)), len(parts))
	}
	var answers []string
	var failed error
	for i, p := range parts {
		draft := newDrafter(r, parts[:i+1], answers)
		req := llm.Request{Messages: llmMessages(p.Output), Skill: llm.SkillTranscribeEnhance, OnDelta: draft.add}
		resp, err := r.llmChat(r.m.cfg.llmDeadlines(req, utf8.RuneCountInString(p.Text), r.m.cfg.LLMTimeout))
		switch {
		case r.ctx.Err() != nil:
			return text, p.Profile(), r.ctx.Err()
		case err != nil:
			if len(parts) > 1 {
				err = fmt.Errorf("piece %d of %d: %w", i+1, len(parts), err)
				r.logf("warning", "%v: kept as recognised", err)
			}
			if failed == nil {
				failed = err
			}
			answers = append(answers, p.Text)
		default:
			answers = append(answers, compose.Clean(resp.Text, p.Text))
		}
		draft.done(answers)
	}
	return compose.JoinParts(parts, answers), parts[0].Profile(), failed
}

// llmMessages are a composed request's messages for the model.
func llmMessages(out compose.Output) []llm.Message {
	msgs := make([]llm.Message, 0, len(out.History)+2)
	for _, m := range out.Messages() {
		msgs = append(msgs, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
	}
	return msgs
}

// drafter streams the clean-up of a part into the snapshot's Draft,
// after the answers of the parts before it, every draftEvery at most.
type drafter struct {
	r       *run
	parts   []compose.Part
	answers []string
	buf     strings.Builder
	last    time.Time
}

func newDrafter(r *run, parts []compose.Part, answers []string) *drafter {
	return &drafter{r: r, parts: parts, answers: answers[:len(answers):len(answers)]}
}

func (d *drafter) add(delta string) {
	d.buf.WriteString(delta)
	if now := time.Now(); now.Sub(d.last) >= draftEvery {
		d.last = now
		d.publish()
	}
}

// publish puts the draft so far in the snapshot.
func (d *drafter) publish() {
	draft := compose.JoinParts(d.parts, append(d.answers, d.buf.String()))
	d.r.m.update(d.r, func(s *Snapshot) { s.Draft = draft })
}

// done puts the answers of the parts so far in the snapshot.
func (d *drafter) done(answers []string) {
	draft := compose.JoinParts(d.parts, answers)
	d.r.m.update(d.r, func(s *Snapshot) { s.Draft = draft })
}
