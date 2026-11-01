package llmrt

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
)

func TestProfiles(t *testing.T) {
	ps := Profiles()
	if len(ps) != 5 || ps[0].ID != Balanced || ps[1].ID != Fast || ps[2].ID != Translate || ps[3].ID != Qwen3 || ps[4].ID != Gemma4 {
		t.Fatalf("profiles %+v", ps)
	}
	for _, p := range ps {
		if p.Repo == "" || (p.Format == wire.FormatGGUF) != (p.File != "") || p.Default.RepetitionPenalty == 0 {
			t.Errorf("profile %+v", p)
		}
		if p.TranslateOnly() != (p.ID == Translate) {
			t.Errorf("%s translate only: %v", p.ID, p.TranslateOnly())
		}
	}
	b, _ := Lookup(Balanced)
	for _, c := range []struct {
		skill llm.Skill
		chars int
		want  int
	}{
		{llm.SkillTranscribeEnhance, 0, 129}, {llm.SkillTranscribeEnhance, 100, 238}, {llm.SkillTranscribeEnhance, 5000, 1024},
		{llm.SkillTranslateText, 100, 295}, {llm.SkillTranslateText, 900, 1024},
		{llm.SkillRewrite, 10, 256}, {llm.SkillRewrite, 400, 732}, {llm.SkillRewrite, 2000, 1536},
	} {
		if got := b.MaxTokens(c.skill, c.chars); got != c.want {
			t.Errorf("MaxTokens(%s, %d) = %d, want %d", c.skill, c.chars, got, c.want)
		}
	}
	if b.PrefillStepSize(999) != 256 || b.PrefillStepSize(2000) != 512 || b.PrefillStepSize(5000) != 768 {
		t.Error("prefill steps")
	}
	h, _ := Lookup(Translate)
	if h.PrefillStepSize(10) != 0 || h.SamplingFor(llm.SkillTranslateText).TopP != 0.6 {
		t.Errorf("Hy-MT2 %+v", h)
	}
}

func TestHyMTPrompt(t *testing.T) {
	for _, c := range []struct{ text, target, want string }{
		{"今天天气真好。", "en", "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n今天天气真好。"},
		{"The weather is nice.", "zh-Hans", "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\nThe weather is nice."},
		{" Bonjour ", "ja", "Translate the following text into Japanese. Note that you should only output the translated result without any additional explanation:\n\nBonjour"},
		{"Hi", "xx-unknown", "Translate the following text into xx-unknown. Note that you should only output the translated result without any additional explanation:\n\nHi"},
	} {
		if got := HyMTPrompt(c.text, c.target); got != c.want {
			t.Errorf("HyMTPrompt(%q, %q) =\n%q\nwant\n%q", c.text, c.target, got, c.want)
		}
	}
}

// Hy-MT2 gets its own prompt, one user message and its card's sampling;
// it takes nothing but translations.
func TestTranslationModel(t *testing.T) {
	log := filepath.Join(t.TempDir(), "chats.jsonl")
	rt, _ := newRuntime(t, llmrttest.Opt("reply", "The weather is nice today.")+","+llmrttest.Opt("chat-log", log), nil)
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "Hy-MT2-1.8B-Q4_K_M.gguf"), []byte("GGUF"), 0o644)
	chat, err := Open(context.Background(), rt, Translate, dir)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := chat.Chat(context.Background(), llm.Request{
		Messages:  []llm.Message{{Role: llm.System, Content: "You are the translator of VoxtNew…"}, {Role: llm.User, Content: "<input>\n今天天气真好。\n</input>"}},
		Skill:     llm.SkillTranslateText,
		Translate: &llm.TranslateHint{Text: "今天天气真好。", Target: "en"},
	})
	if err != nil || resp.Text != "The weather is nice today." {
		t.Fatalf("%+v %v", resp, err)
	}
	c := chatLog(t, log)[0]
	if c.Model.Format != wire.FormatGGUF || !strings.HasSuffix(c.Model.Path, ".gguf") || len(c.Messages) != 1 || c.Messages[0].Role != "user" ||
		!strings.HasPrefix(c.Messages[0].Content, "将以下文本翻译为英语") || c.Sampling.Temperature != 0.7 || c.Template != nil || c.MaxTokens != 169 {
		t.Errorf("chat %+v", c)
	}
	_, err = chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "write"}}, Skill: llm.SkillRewrite})
	if llm.Code(err) != llm.CodeUnsupported {
		t.Errorf("rewrite on the translation model: %v", err)
	}
	// The GGUF file itself is a dir too.
	if _, err := NewModel(rt, Translate, filepath.Join(dir, "Hy-MT2-1.8B-Q4_K_M.gguf")); err != nil {
		t.Error(err)
	}
}

func TestDevHook(t *testing.T) {
	if _, err := ParseDev("llm.balanced"); err == nil {
		t.Error("id without dir")
	}
	if _, err := ParseDev("llm.huge=/x"); err == nil {
		t.Error("unknown id")
	}
	home, _ := os.UserHomeDir()
	ms, err := ParseDev(" llm.balanced=~/m/q , llm.translate=/m/hy.gguf,")
	if err != nil || len(ms) != 2 || ms[0].Dir != filepath.Join(home, "m/q") || ms[1].ID != Translate {
		t.Fatalf("%+v %v", ms, err)
	}
	rt, _ := newRuntime(t, "", nil)
	dir := mlxDir(t)
	s := DevServices(baseServices{}, rt, []DevModel{{ID: Fast, Dir: dir}})
	if l := s.Installed(local.LLM); len(l) != 2 || l[0].ID != Fast || l[1].ID != "other" {
		t.Errorf("installed %+v", l)
	}
	if l := s.Installed(local.ASR); len(l) != 1 || l[0].ID != "asr" {
		t.Errorf("asr %+v", l)
	}
	chat, m, err := s.LLM(context.Background(), "")
	if err != nil || m.ID != Fast || chat == nil {
		t.Fatalf("LLM(\"\") = %v %+v %v", chat, m, err)
	}
	if _, m, err := s.LLM(context.Background(), "other"); err == nil || m.ID != "other" {
		t.Errorf("base's model: %+v %v", m, err)
	}
	if _, _, err := s.ASR(context.Background(), ""); !errors.Is(err, local.ErrNotInstalled) {
		t.Errorf("asr %v", err)
	}
}

// baseServices has one model of each kind, which never open.
type baseServices struct{}

func (baseServices) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{{Kind: local.ASR, ID: "asr"}}
	}
	return []local.Model{{Kind: local.LLM, ID: "other"}}
}

func (baseServices) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
}

func (baseServices) LLM(_ context.Context, id string) (local.Chat, local.Model, error) {
	return nil, local.Model{Kind: local.LLM, ID: id}, errors.New("base")
}

// TestProfilesMatchTheCatalogue: every curated language model has a
// profile with its repository, runtime and (GGUF) file.
func TestProfilesMatchTheCatalogue(t *testing.T) {
	n := 0
	for _, m := range models.Catalog() {
		if m.Kind != models.LLM {
			continue
		}
		n++
		p, ok := Lookup(m.ID)
		if !ok {
			t.Errorf("no profile for %s", m.ID)
			continue
		}
		if p.Repo != m.Repo || p.Format != string(m.Runtime) || (p.Format == wire.FormatGGUF && p.File != m.Files[0].Path) {
			t.Errorf("%s: profile %+v, catalogue %s %s %s", m.ID, p, m.Repo, m.Runtime, m.Files[0].Path)
		}
		if p.TranslateOnly() != (len(m.Features) == 1 && m.Features[0] == "translate") {
			t.Errorf("%s: translate only %v, features %v", m.ID, p.TranslateOnly(), m.Features)
		}
	}
	if n != len(Profiles()) {
		t.Errorf("%d catalogue models, %d profiles", n, len(Profiles()))
	}
}
