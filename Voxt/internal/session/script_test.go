package session

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestDictationScript: Whisper mixes Traditional characters into
// Simplified dictation; the recognised text, the stream's partials and
// the history follow the script the user reads (SpeechScript), and stay
// as the model wrote them when nothing says.
func TestDictationScript(t *testing.T) {
	const heard = "幫我用請假模板发给王老师"
	for _, c := range []struct {
		name  string
		main  string
		reads []string
		want  string
	}{
		{"Simplified Mac", "zh", []string{"zh-Hans-US", "en-US"}, "帮我用请假模板发给王老师"},
		{"Taiwan Mac", "zh", []string{"zh-Hant-TW"}, "幫我用請假模板發給王老師"},
		{"English Mac", "zh", []string{"en-US"}, heard},
		{"English main language", "en", []string{"zh-Hans-CN"}, heard},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: heard}, nil, func(cfg *Config, s *settings.Settings) {
				s.General.MainLanguage = c.main
				cfg.Languages = func() []string { return c.reads }
			})
			h.speak(h.start(), 5)
			end := h.rec.wait(t)
			if end.Raw != c.want || end.Result != c.want {
				t.Errorf("raw %q, result %q, want %q", end.Raw, end.Result, c.want)
			}
			if e := h.hist.all(); len(e) != 1 || e[0].RawText != c.want {
				t.Errorf("history %+v", e)
			}
		})
	}
	t.Run("stream partial", func(t *testing.T) {
		h := newHarness(t, &fakeStreamer{partials: []string{"請假"}, final: heard}, nil, func(cfg *Config, s *settings.Settings) {
			s.General.MainLanguage = "zh"
			cfg.Languages = func() []string { return []string{"zh-Hans-CN"} }
		})
		h.speak(h.start(), 5)
		h.rec.waitFor(t, "the partial in Simplified", func(s Snapshot) bool { return s.Partial == "请假" })
		if end := h.rec.wait(t); end.Raw != "帮我用请假模板发给王老师" {
			t.Errorf("raw %q", end.Raw)
		}
	})
}
