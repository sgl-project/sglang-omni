package session

import (
	"errors"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestTranslateFallback: when the language model fails, Translate asks
// its fallback model once, which then answers the rest of the session
// (the strict retry too); the snapshot and the entry name the model that
// answered last.
func TestTranslateFallback(t *testing.T) {
	boom := errors.New("503 overloaded")
	for _, tc := range []struct {
		name             string
		main, fallback   *fakeLLM
		state            State
		result, model    string
		mainN, fallbackN int
	}{
		{"the model answers", &fakeLLM{text: "早上好"}, &fakeLLM{text: "早安"}, Done, "早上好", "deepseek", 1, 0},
		{"the model fails", &fakeLLM{err: boom}, &fakeLLM{text: "早安"}, Done, "早安", "ollama", 1, 1},
		{"the fallback is asked again strictly", &fakeLLM{err: boom}, &fakeLLM{texts: []string{"Good morning.", "早安"}}, Done, "早安", "ollama", 1, 2},
		{"both fail", &fakeLLM{err: boom}, &fakeLLM{err: boom}, Failed, "", "ollama", 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "x"}, tc.main, func(c *Config, _ *settings.Settings) { c.Select = &fakeSelector{text: "Good morning"} })
			h.fallback = tc.fallback
			if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack}); err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.State != tc.state || end.Result != tc.result || end.LLM.ID != tc.model {
				t.Errorf("end %s %q by %q, want %s %q by %q (problem %+v)", end.State, end.Result, end.LLM.ID, tc.state, tc.result, tc.model, end.Problem)
			}
			if n, m := len(tc.main.requests()), len(tc.fallback.requests()); n != tc.mainN || m != tc.fallbackN {
				t.Errorf("%d requests to the model, %d to the fallback; want %d, %d", n, m, tc.mainN, tc.fallbackN)
			}
			if es := h.hist.all(); len(es) != 1 || es[0].LLMProvider != tc.model || es[0].LLMModel != end.LLM.Model {
				t.Errorf("entries %+v", es)
			}
		})
	}
}

func TestTranslateTarget(t *testing.T) {
	s := settings.Defaults()
	for _, tc := range []struct {
		main, target, locale, override, want string
	}{
		{"", "", "en", "", "zh-Hans"},
		{"", "", "zh-Hans", "", "en"},
		{"zh", "", "en", "", "en"},
		{"en", "", "zh-Hans", "", "zh-Hans"},
		{"ja", "", "en", "", "en"},
		{"zh", "ja", "en", "", "ja"},
		{"zh", "ja", "en", "fr", "fr"},
	} {
		s.General.MainLanguage, s.Features.Translate.Target = tc.main, tc.target
		if got := TranslateTarget(s, tc.locale, tc.override); got != tc.want {
			t.Errorf("%+v: %q", tc, got)
		}
	}
}

// TestTranslateDirection: Translate goes between the user's main
// language and the target by the source (#3897): a selection or speech
// already in the target goes into the main language; the rest, and a
// language the user picked on the card, keep the target. Scripts cannot
// tell English from French, so Latin into Latin keeps the target.
func TestTranslateDirection(t *testing.T) {
	for _, tc := range []struct {
		name, main, target, override, sel, want string
	}{
		{"Chinese into English", "zh", "", "", "早上好，团队。", "en"},
		{"English into Chinese", "zh", "", "", "Good morning, team.", "zh-Hans"},
		{"mostly Chinese, English words", "zh", "", "", "我用 Voxt 写 code", "en"},
		{"mostly English, a Chinese name", "zh", "", "", "Ask 张伟 about the release", "zh-Hans"},
		{"English main, Chinese into English", "en", "", "", "早上好", "en"},
		{"English main, English into Chinese", "en", "", "", "Good morning", "zh-Hans"},
		{"Japanese target, Japanese text", "zh", "ja", "", "おはようございます", "zh-Hans"},
		{"French target, English text", "en", "fr", "", "Good morning", "fr"},
		{"picked on the card", "zh", "", "en", "Good morning", "en"},
		{"digits only", "zh", "", "", "2026-11-12 10:00", "en"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "x"}
			h := newHarness(t, &fakeFile{text: "x"}, model, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: tc.sel}
				s.General.MainLanguage, s.Features.Translate.Target = tc.main, tc.target
			})
			if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, TargetLanguage: tc.override}); err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.TargetLanguage != tc.want {
				t.Errorf("into %q, want %q", end.TargetLanguage, tc.want)
			}
			name := compose.LanguageName(tc.want, compose.TemplateSet(tc.main))
			if reqs := model.requests(); len(reqs) == 0 || !strings.Contains(reqs[0].Messages[0].Content, name) {
				t.Errorf("the prompt does not say %q: %+v", name, reqs)
			}
			if es := h.hist.all(); len(es) != 1 || es[0].TargetLang != tc.want {
				t.Errorf("entries %+v", es)
			}
		})
	}
}

// TestTranslateBack: the prompt tells the model to write in the main
// language when all of the text is already in the target, for languages
// that share a script, which TranslateInto cannot tell apart (#4024). A
// language picked on the card stays.
func TestTranslateBack(t *testing.T) {
	for _, tc := range []struct {
		name, main, override string
		want                 bool
	}{
		{"Spanish main, English text", "es", "", true},
		{"picked on the card", "es", "en", false},
		{"main is the target", "en", "zh-Hans", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "x"}
			h := newHarness(t, &fakeFile{text: "x"}, model, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: "Good morning, team."}
				s.General.MainLanguage = tc.main
			})
			if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, TargetLanguage: tc.override}); err != nil {
				t.Fatal(err)
			}
			h.rec.wait(t)
			reqs := model.requests()
			if len(reqs) == 0 {
				t.Fatal("no request")
			}
			if got := strings.Contains(reqs[0].Messages[0].Content, "translate it into Spanish instead"); got != tc.want {
				t.Errorf("switch line %v, want %v:\n%s", got, tc.want, reqs[0].Messages[0].Content)
			}
		})
	}
}
