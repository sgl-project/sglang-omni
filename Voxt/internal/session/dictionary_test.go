package session

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestPersonalDictionary: the personal dictionary (Dictionary.Notes) is
// in the language model's system prompt of Transcribe, Translate and
// Rewrite, after the task's rules and before the glossary, while the
// settings send it; speech recognition never gets it, with a streaming
// or a file model.
func TestPersonalDictionary(t *testing.T) {
	const notes = "| Standard | Mapping | When |\n|---|---|---|\n| 101 | `幺零幺` → 101 | Bay Area highways |"
	for _, tc := range []struct {
		name   string
		kind   Kind
		sel    string
		stream bool
		answer string
	}{
		{"transcribe, streaming", store.KindTranscribe, "", true, "Hello."},
		{"transcribe, file", store.KindTranscribe, "", false, "Hello."},
		{"translate speech", store.KindTranslate, "", false, "你好"},
		{"translate the selection", store.KindTranslate, "Good morning", false, "早上好"},
		{"rewrite the selection", store.KindRewrite, "ship friday?", true, "We ship on Friday."},
		{"write from scratch", store.KindRewrite, "", false, `{"title":"Friday","content":"We ship on Friday."}`},
	} {
		for _, mode := range []string{"on", "switched off", "dictionary off"} {
			t.Run(tc.name+", "+mode, func(t *testing.T) {
				var prov asr.Provider = &fakeFile{text: " hello aura link "}
				if tc.stream {
					prov = &fakeStreamer{final: "hello aura link"}
				}
				model := &fakeLLM{text: tc.answer}
				h := newHarness(t, prov, model, func(c *Config, s *settings.Settings) {
					c.Select = &fakeSelector{text: tc.sel}
					s.Features.Transcribe.Enhance = true
					s.Dictionary.SpeechContext = "Leo works on LLM serving."
					s.Dictionary.Notes = notes
					s.Dictionary.NotesOff = mode == "switched off"
					s.Dictionary.Enabled = mode != "dictionary off"
				})
				id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack})
				if err != nil {
					t.Fatal(err)
				}
				if tc.kind != store.KindTranslate || tc.sel == "" {
					h.speak(id, 5)
				}
				if end := h.rec.wait(t); end.State != Done {
					t.Fatalf("end %s (problem %+v)", end.State, end.Problem)
				}
				reqs := model.requests()
				if len(reqs) == 0 {
					t.Fatal("no language model request")
				}
				for _, r := range reqs {
					sys := r.Messages[0].Content
					in := strings.Contains(sys, "`幺零幺` → 101")
					if in != (mode == "on") {
						t.Errorf("the dictionary in the system prompt: %v\n%s", in, sys)
					}
					if in && (strings.Count(sys, "`幺零幺` → 101") != 1 || strings.Contains(sys, "dictionary>") || strings.Index(sys, "Bay Area highways") > strings.Index(sys, "- VoxtNew")) {
						t.Errorf("the dictionary once, before the glossary:\n%s", sys)
					}
					for _, m := range r.Messages[1:] {
						if strings.Contains(m.Content, "幺零幺") {
							t.Errorf("the dictionary in a %s message", m.Role)
						}
					}
				}
				// What the speech model got: hints and the prompt.
				var asrText string
				switch p := prov.(type) {
				case *fakeStreamer:
					asrText = strings.Join(p.hints, "\n") + "\n" + p.prompt
				case *fakeFile:
					asrText = p.prompt
				}
				if spoke := tc.kind != store.KindTranslate || tc.sel == ""; spoke != strings.Contains(asrText, "Leo works on LLM serving.") {
					t.Errorf("the speech model's prompt %q", asrText)
				}
				if strings.Contains(asrText, "幺零幺") || strings.Contains(asrText, "Standard") {
					t.Errorf("speech recognition got the dictionary: %q", asrText)
				}
			})
		}
	}
}
