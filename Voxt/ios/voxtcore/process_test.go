package voxtcore

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

func TestTranscribeCleansUpWithTheComposedPrompt(t *testing.T) {
	f, p := serveChat(t, "Let's meet at three tomorrow.")
	res, err := Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: "um let's meet at uh three tomorrow", MainLanguage: "en",
		Feature: Feature{Cleanup: true}, Terms: []Term{{Term: "Voxt", Aliases: []string{"vox"}}}, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Let's meet at three tomorrow." || res.Degraded {
		t.Fatalf("result = %+v", res)
	}
	if !strings.HasPrefix(res.Profile, "transcribe:cleanup") || res.PromptTokens != 10 || res.CompletionTokens != 3 {
		t.Fatalf("profile %q, tokens %d/%d", res.Profile, res.PromptTokens, res.CompletionTokens)
	}
	calls := f.calls()
	if len(calls) != 1 {
		t.Fatalf("%d requests, want 1", len(calls))
	}
	// The speech only in the user message, the glossary in the system
	// prompt (a small dictionary's), as on macOS (internal/compose).
	if c := calls[0]; !strings.Contains(c.user(), "<input>") || !strings.Contains(c.user(), "three tomorrow") ||
		strings.Contains(c.system(), "three tomorrow") || !strings.Contains(c.system(), "Voxt") || c.Model != "test-model" {
		t.Fatalf("request: system %q user %q model %q", c.system(), c.user(), c.Model)
	}
}

func TestTranscribeWithoutCleanupExpandsSnippetsWithoutTheModel(t *testing.T) {
	f, p := serveChat(t, "unused")
	list := []Snippet{{Trigger: "my email address", Text: "me@example.com"}}
	res, err := Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: "send it to my email address please", MainLanguage: "en", Snippets: list, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Text, "me@example.com") || strings.Contains(res.Text, "my email address") {
		t.Fatalf("text = %q", res.Text)
	}
	whole, err := Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: "My email address.", MainLanguage: "en", Feature: Feature{Cleanup: true}, Snippets: list, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if whole.Text != "me@example.com" {
		t.Fatalf("a trigger alone = %q, want the snippet as saved", whole.Text)
	}
	if n := len(f.calls()); n != 0 {
		t.Fatalf("%d requests, want none", n)
	}
}

func TestTranscribeRestoresSnippetsAfterTheModel(t *testing.T) {
	list := []Snippet{{Trigger: "my email address", Text: "me@example.com"}}
	text := "send it to my email address"
	marked, said := snippet.Mark(text, snippets(list))
	if len(said) != 1 {
		t.Fatalf("Mark found %d triggers", len(said))
	}
	// The model keeps the marked trigger; the snippet replaces it after,
	// so the model never sees or rewrites the snippet's text.
	f, p := serveChat(t, "Send it to "+strings.TrimPrefix(marked, "send it to ")+".")
	res, err := Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: text, MainLanguage: "en", Feature: Feature{Cleanup: true}, Snippets: list, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Send it to me@example.com." {
		t.Fatalf("text = %q", res.Text)
	}
	if user := f.calls()[0].user(); !strings.Contains(user, marked) || strings.Contains(user, "me@example.com") {
		t.Fatalf("user message %q: want the marked trigger, not the snippet", user)
	}
}

func TestTranscribeKeepsTheSpeechWhenTheModelFails(t *testing.T) {
	f := &fakeChat{t: t, answers: []fakeAnswer{{status: http.StatusBadRequest}}}
	p := startChat(t, f)
	res, err := Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: "hello there", MainLanguage: "en", Feature: Feature{Cleanup: true}, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "hello there" || !res.Degraded || res.Warning == "" {
		t.Fatalf("result = %+v", res)
	}
	// A setup that cannot be opened keeps the speech too, and says why.
	res, err = Process(context.Background(), ProcessRequest{
		Task: "transcribe", Text: "hello there", MainLanguage: "en", Feature: Feature{Cleanup: true}, Provider: Provider{Preset: "openai"},
	}, nil)
	if err != nil || res.Text != "hello there" || !res.Degraded || !strings.Contains(res.Warning, "API key") {
		t.Fatalf("no key: %+v, %v", res, err)
	}
}

func TestTranslateAsksAgainStrictly(t *testing.T) {
	f, p := serveChat(t, "我们明天三点见。", "Let's meet at three tomorrow.")
	res, err := Process(context.Background(), ProcessRequest{
		Task: "translate", Text: "我们明天三点见", TargetLanguage: "en", MainLanguage: "zh", Feature: Feature{Cleanup: true}, Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Let's meet at three tomorrow." || !res.Retried || res.PromptTokens != 20 {
		t.Fatalf("result = %+v", res)
	}
	calls := f.calls()
	if len(calls) != 2 || calls[0].system() == calls[1].system() {
		t.Fatalf("%d requests; the second must carry the strict layer", len(calls))
	}
}

func TestTranslateFailsWhenStillUntranslated(t *testing.T) {
	_, p := serveChat(t, "我们明天三点见。")
	_, err := Process(context.Background(), ProcessRequest{
		Task: "translate", Text: "我们明天三点见", TargetLanguage: "en", MainLanguage: "zh", Typed: true, Provider: p,
	}, nil)
	if !errors.Is(err, ErrNotTranslated) {
		t.Fatalf("err = %v, want ErrNotTranslated", err)
	}
}

func TestRewriteFromScratchAsksForATitledAnswer(t *testing.T) {
	f, p := serveChat(t, `{"title":"Leave request","content":"I would like to take Friday off."}`)
	res, err := Process(context.Background(), ProcessRequest{
		Task: "rewrite", Instruction: "write a short leave request for friday", MainLanguage: "en", Provider: p,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Title != "Leave request" || res.Text != "I would like to take Friday off." {
		t.Fatalf("result = %+v", res)
	}
	// The custom preset takes the schema in the prompt (llm.JSONPromptMode).
	if c := f.calls()[0]; !strings.Contains(c.system()+c.user(), `"title"`) {
		t.Fatalf("the request does not ask for the titled answer: system %q user %q", c.system(), c.user())
	}
}

func TestRewriteOfTextIsPlain(t *testing.T) {
	f, p := serveChat(t, "Could you send me the report by Friday?")
	res, err := Process(context.Background(), ProcessRequest{
		Task: "rewrite", Text: "send report friday", Instruction: "make it polite", MainLanguage: "en", Provider: p,
		Previous: nil,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Could you send me the report by Friday?" || res.Title != "" {
		t.Fatalf("result = %+v", res)
	}
	if c := f.calls()[0]; c.ResponseFormat != nil || !strings.Contains(c.user(), "make it polite") {
		t.Fatalf("request: format %v user %q", c.ResponseFormat, c.user())
	}
	// A follow-up sends the earlier turn before this one.
	if _, err := Process(context.Background(), ProcessRequest{
		Task: "rewrite", Instruction: "shorter", MainLanguage: "en", Provider: p,
		Previous: []Turn{{Instruction: "make it polite", Source: "send report friday", Answer: "Could you send me the report by Friday?"}},
	}, nil); err != nil {
		t.Fatal(err)
	}
	if c := f.calls()[1]; len(c.Messages) < 4 || c.ResponseFormat != nil {
		t.Fatalf("follow-up: %d messages, format %v", len(c.Messages), c.ResponseFormat)
	}
}

func TestProcessRefusesWhatItCannotDo(t *testing.T) {
	_, p := serveChat(t, "x")
	for name, req := range map[string]ProcessRequest{
		"unknown task":    {Task: "summarize", Text: "x", Provider: p},
		"no speech":       {Task: "transcribe", Text: "  ", Provider: p},
		"no translation":  {Task: "translate", Text: "", TargetLanguage: "en", Provider: p},
		"unknown preset":  {Task: "translate", Text: "x", TargetLanguage: "en", Provider: Provider{Preset: "nope"}},
		"sign-in service": {Task: "rewrite", Text: "x", Instruction: "y", Provider: Provider{Preset: "codex"}},
	} {
		if _, err := Process(context.Background(), req, nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
