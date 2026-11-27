package session

import (
	"context"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
)

// TestSnippetsPlaced: a snippet saved with spaces at its ends comes out
// without them, alone or in a sentence, and with a space between it and
// Chinese, as the clean-up writes Chinese with Latin text.
func TestSnippetsPlaced(t *testing.T) {
	spaced := func(c *Config, _ *settings.Settings) {
		c.Snippets = func(context.Context) ([]snippet.Snippet, error) {
			return []snippet.Snippet{{ID: "e", Trigger: "我的邮箱", Text: " zhao@example.com "}}, nil
		}
	}
	for _, c := range []struct{ name, said, cleaned, result string }{
		{"only a trigger", "我的邮箱", "unused", "zhao@example.com"},
		{"in a sentence", "请发到我的邮箱", "请发到⟦我的邮箱⟧，谢谢。", "请发到 zhao@example.com，谢谢。"},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: c.said}, &fakeLLM{text: c.cleaned}, spaced)
			h.speak(h.start(), 5)
			if end := h.rec.wait(t); end.State != Done || end.Result != c.result {
				t.Fatalf("%s: result %q, want %q", end.State, end.Result, c.result)
			}
		})
	}
}
