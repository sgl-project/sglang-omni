package session

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestLatinShortSourceCanBeTranslated(t *testing.T) {
	for _, test := range []struct {
		source, result string
	}{
		{"it", "petit"},
		{"in", "matin"},
		{"or", "alors"},
	} {
		t.Run(test.source, func(t *testing.T) {
			model := &fakeLLM{text: test.result}
			h := newHarness(t, &fakeFile{text: "x"}, model, func(config *Config, _ *settings.Settings) {
				config.Select = &fakeSelector{text: test.source}
			})
			_, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, TargetLanguage: "fr"})
			if err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.State != Done || end.Result != test.result || len(model.requests()) != 1 {
				t.Fatalf("end %+v, requests %d", end, len(model.requests()))
			}
			if pastes := h.text.pastes(); len(pastes) != 1 || pastes[0] != test.result {
				t.Fatalf("pastes %q", pastes)
			}
		})
	}
}
