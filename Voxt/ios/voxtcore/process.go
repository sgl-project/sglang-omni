package voxtcore

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// Errors a step ends with, besides the language model's own (llm.Error).
var (
	ErrTask          = errors.New("voxtcore: unknown task")
	ErrNoText        = errors.New("voxtcore: nothing to process")
	ErrEmptyAnswer   = errors.New("the language model returned an empty answer")
	ErrNotTranslated = errors.New("the language model did not translate the text")
)

// LLMTimeout bounds one attempt of a request (the macOS app's default
// write timeout).
const LLMTimeout = 60 * time.Second

// answerSchema is Rewrite's JSON answer (internal/session's).
var answerSchema = func() *llm.Schema {
	s := llm.Object(map[string]*llm.Schema{"title": llm.String(), "content": llm.String()}, "title", "content")
	s.Name = "answer"
	return s
}()

// chatter is what a step needs of llm.Client; tests count its calls.
type chatter interface {
	Chat(ctx context.Context, req llm.Request) (llm.Response, error)
}

// Process runs req's language-model step over client (nil:
// http.DefaultClient).
func Process(ctx context.Context, req ProcessRequest, client *http.Client) (ProcessResult, error) {
	backend, err := openLLM(req.Provider, client)
	if err != nil && req.Task != string(compose.Transcribe) {
		return ProcessResult{}, err
	}
	start := time.Now()
	var res ProcessResult
	switch compose.Task(req.Task) {
	case compose.Transcribe:
		res, err = transcribe(ctx, backend, err, req)
	case compose.Translate:
		res, err = translate(ctx, backend, req)
	case compose.Rewrite:
		res, err = rewrite(ctx, backend, req)
	default:
		return ProcessResult{}, fmt.Errorf("%w: %q", ErrTask, req.Task)
	}
	res.LatencyMillis = time.Since(start).Milliseconds()
	return res, err
}

// openLLM is the client of p with retries and timeouts.
func openLLM(p Provider, client *http.Client) (chatter, error) {
	backend, err := openBackend(p, client)
	if err != nil {
		return nil, err
	}
	// As providers.Open: a second attempt when the first fails in a way
	// worth retrying (a timeout, a 5xx, a dropped connection).
	return &llm.Client{Backend: backend, Timeout: LLMTimeout, Retry: llm.RetryPolicy{MaxAttempts: 2}}, nil
}

func baseInput(req ProcessRequest) compose.Input {
	return compose.Input{
		MainLanguage: req.MainLanguage, OtherLanguages: req.OtherLanguages,
		TaskPrompt: req.Feature.Prompt, Dictionary: req.Notes, Glossary: terms(req.Terms),
	}
}

// transcribe cleans up the speech in parts, as internal/session's
// cleanParts does: a part the model fails on stays as recognised. A
// backend that could not be opened (openErr) keeps all of it so.
func transcribe(ctx context.Context, backend chatter, openErr error, req ProcessRequest) (ProcessResult, error) {
	text := strings.TrimSpace(req.Text)
	if text == "" {
		return ProcessResult{}, ErrNoText
	}
	list := snippets(req.Snippets)
	if s, ok := snippet.Whole(text, list); ok { // only a trigger: the snippet as saved
		return ProcessResult{Text: snippet.Alone(s.Text)}, nil
	}
	if !req.Feature.Cleanup {
		return ProcessResult{Text: expandSnippets(text, text, list)}, nil
	}
	if openErr != nil {
		return ProcessResult{Text: expandSnippets(text, text, list), Degraded: true, Warning: openErr.Error()}, nil
	}
	in := baseInput(req)
	marked, said := snippet.Mark(text, list)
	in.Task, in.Cleanup, in.Structure, in.Text, in.Snippets = compose.Transcribe, true, req.Feature.Structure, marked, len(said) > 0
	parts, err := compose.BuildParts(in)
	if err != nil {
		return ProcessResult{}, err
	}
	res := ProcessResult{Profile: parts[0].Profile()}
	answers := make([]string, 0, len(parts))
	for i, p := range parts {
		resp, err := backend.Chat(ctx, llm.Request{
			Messages: messages(p.Output), Skill: llm.SkillTranscribeEnhance, InputChars: utf8.RuneCountInString(text),
		})
		if ctx.Err() != nil {
			return ProcessResult{}, ctx.Err()
		}
		if err != nil {
			if !res.Degraded {
				res.Degraded, res.Warning = true, fmt.Sprintf("piece %d of %d kept as recognised: %v", i+1, len(parts), err)
			}
			answers = append(answers, p.Text) // marked: expandSnippets restores it
			continue
		}
		res.PromptTokens += resp.Usage.PromptTokens
		res.CompletionTokens += resp.Usage.CompletionTokens
		answers = append(answers, compose.CleanOutput(resp.Text, p.Text, p.Custom()))
	}
	res.Text = expandSnippets(text, compose.JoinParts(parts, answers), list)
	return res, nil
}

// expandSnippets replaces the triggers said in text with their snippets
// in result, after the model (internal/session's expandSnippets).
func expandSnippets(text, result string, list []snippet.Snippet) string {
	_, said := snippet.Mark(text, list)
	if len(said) == 0 {
		return result
	}
	return snippet.Restore(result, said, said)
}

// translate asks once and, when the answer looks untranslated, once
// more, strictly (internal/session's write).
func translate(ctx context.Context, backend chatter, req ProcessRequest) (ProcessResult, error) {
	in := baseInput(req)
	in.Task, in.TargetLanguage, in.BackLanguage = compose.Translate, req.TargetLanguage, req.MainLanguage
	in.Text = strings.TrimSpace(req.Text)
	if in.Text == "" {
		return ProcessResult{}, ErrNoText
	}
	if req.Typed {
		in.Selected = true
	} else {
		in.Cleanup = req.Feature.Cleanup
	}
	answer, res, err := ask(ctx, backend, in)
	if err != nil {
		return res, err
	}
	if compose.Untranslated(in.Text, answer.Content, in.TargetLanguage) && ctx.Err() == nil {
		in.Strict = true
		if again, strict, err := ask(ctx, backend, in); err == nil {
			strict.PromptTokens += res.PromptTokens
			strict.CompletionTokens += res.CompletionTokens
			answer, res = again, strict
		}
		res.Retried = true
		if compose.Untranslated(in.Text, answer.Content, in.TargetLanguage) && compose.LanguageMismatch(in.Text, in.TargetLanguage) {
			return res, ErrNotTranslated
		}
	}
	if answer.Content == "" {
		return res, ErrEmptyAnswer
	}
	res.Text = answer.Content
	return res, nil
}

// rewrite rewrites the text by the instruction, or writes from scratch
// with a title (an answer card's).
func rewrite(ctx context.Context, backend chatter, req ProcessRequest) (ProcessResult, error) {
	in := baseInput(req)
	in.Task, in.Cleanup, in.Instruction, in.Text = compose.Rewrite, req.Feature.Cleanup, strings.TrimSpace(req.Instruction), req.Text
	for _, t := range req.Previous {
		in.Previous = append(in.Previous, compose.Turn{Instruction: t.Instruction, Selection: t.Source, Answer: compose.Answer{Title: t.Title, Content: t.Answer}})
	}
	in.Answer = strings.TrimSpace(in.Text) == "" && len(in.Previous) == 0
	answer, res, err := ask(ctx, backend, in)
	if err != nil {
		return res, err
	}
	if answer.Content == "" {
		return res, ErrEmptyAnswer
	}
	res.Text, res.Title = answer.Content, answer.Title
	return res, nil
}

// ask sends one Translate or Rewrite request and cleans its answer
// (internal/session's ask, without the streaming).
func ask(ctx context.Context, backend chatter, in compose.Input) (compose.Answer, ProcessResult, error) {
	out, err := compose.Build(in)
	if err != nil {
		return compose.Answer{}, ProcessResult{}, err
	}
	skill := llm.SkillTranslateText
	if in.Task == compose.Rewrite {
		skill = llm.SkillRewrite
	}
	req := llm.Request{Messages: messages(out), Skill: skill, InputChars: utf8.RuneCountInString(in.Text + in.Typed + in.Instruction)}
	if in.Answer {
		req.Format, req.Schema = llm.FormatJSON, answerSchema
	}
	res := ProcessResult{Profile: out.Profile()}
	resp, err := backend.Chat(ctx, req)
	if err != nil {
		return compose.Answer{}, res, err
	}
	res.PromptTokens, res.CompletionTokens = resp.Usage.PromptTokens, resp.Usage.CompletionTokens
	inputs := []string{in.Text + "\n" + in.Typed + "\n" + in.Instruction}
	for _, turn := range in.Previous {
		inputs = append(inputs, turn.Instruction+"\n"+turn.Selection+"\n"+turn.Typed+"\n"+turn.Answer.Title+"\n"+turn.Answer.Content)
	}
	quote := cmp.Or(in.Text, in.Instruction)
	switch {
	case in.Answer:
		if a, ok := compose.ParseAnswer(resp.Text, inputs...); ok {
			return a, res, nil
		}
		text := strings.TrimSpace(strings.Trim(strings.TrimSpace(resp.Text), "`"))
		return compose.Answer{Content: compose.CleanRewrite(text, inputs, false, quote)}, res, nil
	case in.Task == compose.Rewrite:
		return compose.Answer{Content: compose.CleanRewrite(resp.Text, inputs, false, quote)}, res, nil
	default:
		return compose.Answer{Content: compose.CleanWithInputs(resp.Text, inputs, quote)}, res, nil
	}
}

func messages(out compose.Output) []llm.Message {
	msgs := make([]llm.Message, 0, len(out.History)+2)
	for _, m := range out.Messages() {
		msgs = append(msgs, llm.Message{Role: llm.Role(m.Role), Content: m.Content})
	}
	return msgs
}
