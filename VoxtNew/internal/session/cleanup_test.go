package session

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// TestAliases: with the dictionary on, its aliases become their terms
// before the clean-up, also without a language model or with the
// glossary and the hints off; the history keeps the text as recognised.
// The dictionary off leaves the text alone.
func TestAliases(t *testing.T) {
	for _, c := range []struct {
		name   string
		set    func(*settings.Settings)
		result string
	}{
		{"on", nil, "hello VoxtNew"},
		{"glossary and hints off", func(s *settings.Settings) { s.Dictionary.Glossary, s.Dictionary.ASRHints = false, false }, "hello VoxtNew"},
		{"dictionary off", func(s *settings.Settings) { s.Dictionary.Enabled = false }, "hello aura link"},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "hello aura link"}, nil, func(_ *Config, s *settings.Settings) {
				if c.set != nil {
					c.set(s)
				}
			})
			h.speak(h.start(), 5)
			end := h.rec.wait(t)
			e := h.hist.all()
			if end.Result != c.result || end.Raw != "hello aura link" || len(e) != 1 || e[0].RawText != "hello aura link" || e[0].Result != c.result {
				t.Errorf("result %q, raw %q, entries %+v", end.Result, end.Raw, e)
			}
		})
	}
}

// longZH is a dictation compose cleans up in two pieces.
var longZH = strings.Repeat("嗯，那个，我们下周二，呃不对，下周三下午三点开个会，讨论一下预算。", 16)

// TestLongDictation: a long dictation is cleaned up in pieces, every
// request with the same system prompt and example turn, the answers
// streaming into the draft in order; a piece that fails keeps its text as
// recognised, with a warning.
func TestLongDictation(t *testing.T) {
	pieces := compose.Pieces(longZH)
	if len(pieces) != 2 {
		t.Fatalf("%d pieces", len(pieces))
	}
	for _, c := range []struct {
		name    string
		llm     *fakeLLM
		result  string
		problem string
	}{
		{"pieces", &fakeLLM{texts: []string{"我们下周三开会。", "讨论预算。"}, chunk: 2, pause: 10 * time.Millisecond}, "我们下周三开会。讨论预算。", ""},
		{"a piece fails", &fakeLLM{texts: []string{"我们下周三开会。"}, err: errors.New("503"), failOn: 2}, "我们下周三开会。" + pieces[1].Text, ProblemLLM},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: longZH}, c.llm, nil)
			h.speak(h.start(), 5)
			end := h.rec.wait(t)
			if end.Result != c.result || end.Raw != longZH || end.Profile != "transcribe:cleanup+glossary" {
				t.Errorf("result %q, raw %q, profile %q", end.Result, end.Raw, end.Profile)
			}
			if (end.Problem == nil) != (c.problem == "") || end.Problem != nil && (end.Problem.Code != c.problem || !end.Problem.Degraded) {
				t.Errorf("problem %+v", end.Problem)
			}
			reqs := c.llm.requests()
			if len(reqs) != 2 {
				t.Fatalf("%d requests", len(reqs))
			}
			for i, req := range reqs {
				m := req.Messages
				if len(m) != 4 || m[0] != reqs[0].Messages[0] || m[1] != reqs[0].Messages[1] || m[2] != reqs[0].Messages[2] ||
					m[3].Content != compose.UserMessage(pieces[i].Text) || req.OnDelta == nil {
					t.Errorf("request %d: %+v", i, m)
				}
			}
			h.rec.waitFor(t, "the first piece's answer as the draft", func(s Snapshot) bool { return s.State == Composing && s.Draft == "我们下周三开会。" })
			if c.problem == "" {
				h.rec.waitFor(t, "the second piece streaming after it", func(s Snapshot) bool {
					return s.State == Composing && strings.HasPrefix(s.Draft, "我们下周三开会。讨") && s.Draft != end.Result
				})
			}
		})
	}
}

// TestLongTranslate: long spoken Translate stays one request (its
// target is one language; in pieces it was no better, docs/eval.md).
func TestLongTranslate(t *testing.T) {
	en := strings.Repeat("Um, let's ship the release on Friday, no wait, on Thursday morning. 嗯，我们下周三开会。", 7)
	model := &fakeLLM{text: "我们周四上午发布。"}
	h := newHarness(t, &fakeFile{text: en}, model, func(c *Config, s *settings.Settings) { c.Select = &fakeSelector{} })
	id, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 5)
	end := h.rec.wait(t)
	reqs := model.requests()
	if end.Result != "我们周四上午发布。" || len(reqs) != 1 || reqs[0].Messages[len(reqs[0].Messages)-1].Content != compose.UserMessage(strings.TrimSpace(en)) {
		t.Errorf("result %q, %d requests", end.Result, len(reqs))
	}
}
