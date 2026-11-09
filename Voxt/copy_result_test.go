package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestCopyResult: with "Also copy the result to the clipboard" (Voxt
// 1.x's autoCopyWhenNoFocusedInput), a pasted result stays on the
// clipboard and one the answer card shows is copied; off, the paste
// restores the clipboard and nothing is copied.
func TestCopyResult(t *testing.T) {
	for _, c := range []struct {
		copy, card bool
	}{{true, false}, {true, true}, {false, false}, {false, true}} {
		srv, _ := mockOpenAI(t, "早上好。")
		useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
		ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
		a := newSessionApp(t, helpertest.Opt("selection", "Good morning.")+","+ev, func(s *settings.Settings) {
			s.General.CopyResult = c.copy
			s.Features.Translate.Target, s.Features.Translate.LLM, s.Features.ShowCard = "zh-Hans", llm.Custom, c.card
			s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
		}, local.NotInstalled{})
		var copied []string
		a.copyText = func(s string) error { copied = append(copied, s); return nil }
		a.pump("the end", func() bool { return a.sess.State.Ended() })
		pasted := a.pasted()
		switch {
		case c.card && (len(pasted) != 0 || c.copy != slices.Equal(copied, []string{"早上好。"}) || !c.copy && len(copied) != 0):
			t.Errorf("%+v: card result: pasted %+v, copied %q", c, pasted, copied)
		case !c.card && (len(pasted) != 1 || pasted[0].KeepClipboard != c.copy || len(copied) != 0):
			t.Errorf("%+v: pasted %+v, copied %q", c, pasted, copied)
		}
	}
}
