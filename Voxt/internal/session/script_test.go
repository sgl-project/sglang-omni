package session

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestDictationScript: Whisper mixes Traditional characters into
// Simplified dictation; the recognised text, the stream's partials and
// the history follow the script the user reads (SpeechScript), and stay
// as the model wrote them when nothing says. Text in another language is
// not Chinese to convert.
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
	// Japanese said by a user whose main language is Chinese keeps its
	// kanji: the speech model names the language, or the text has kana.
	for _, c := range []struct {
		name, heard, said string
	}{
		{"Japanese by its kana", "東京駅で会いましょう", ""},
		{"Japanese the model named", "東京駅", "ja"},
		{"Japanese by name and kana", "資料を確認してください", "Japanese"},
	} {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: c.heard, said: c.said}, nil, func(cfg *Config, s *settings.Settings) {
				s.General.MainLanguage = "zh"
				cfg.Languages = func() []string { return []string{"zh-Hans-CN"} }
			})
			h.speak(h.start(), 5)
			if end := h.rec.wait(t); end.Raw != c.heard || end.Result != c.heard {
				t.Errorf("raw %q, result %q, want %q", end.Raw, end.Result, c.heard)
			}
		})
	}
	t.Run("Chinese the model named", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: heard, said: "chinese"}, nil, func(cfg *Config, s *settings.Settings) {
			s.General.MainLanguage = "zh"
			cfg.Languages = func() []string { return []string{"zh-Hans-CN"} }
		})
		h.speak(h.start(), 5)
		if end := h.rec.wait(t); end.Raw != "帮我用请假模板发给王老师" {
			t.Errorf("raw %q", end.Raw)
		}
	})
	t.Run("Japanese stream partial", func(t *testing.T) {
		const said = "東京駅で会いましょう"
		h := newHarness(t, &fakeStreamer{partials: []string{"東京駅で"}, final: said}, nil, func(cfg *Config, s *settings.Settings) {
			s.General.MainLanguage = "zh"
			cfg.Languages = func() []string { return []string{"zh-Hans-CN"} }
		})
		h.speak(h.start(), 5)
		h.rec.waitFor(t, "the partial as heard", func(s Snapshot) bool { return s.Partial == "東京駅で" })
		if end := h.rec.wait(t); end.Raw != said {
			t.Errorf("raw %q", end.Raw)
		}
	})
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
