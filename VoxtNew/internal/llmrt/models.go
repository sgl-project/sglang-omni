package llmrt

import (
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
)

// The curated language models' ids (settings.Feature.LLM holds one).
const (
	Balanced  = "llm.balanced"  // the default: clean-up, translation and rewriting
	Fast      = "llm.fast"      // smaller and quicker
	Translate = "llm.translate" // translation only, a model of its own
	// Other ActNow apps' models VoxtNew runs when they are on the Mac
	// (models.Model.Borrowed).
	Qwen3  = "llm.qwen3"  // AudioSmith's and Dicta's Qwen3-4B-Instruct-2507
	Gemma4 = "llm.gemma4" // Dicta's Gemma 4 E4B
)

// PromptStyle is how a model takes a request.
type PromptStyle int

const (
	// PromptChat sends compose's messages through the model's chat
	// template, as they are.
	PromptChat PromptStyle = iota
	// PromptHyMT is Hy-MT2's own translation prompt: one user message,
	// "Translate the following text into <language> …", built from
	// llm.Request.Translate (hymt.go); other requests are refused.
	PromptHyMT
)

// Profile is one curated model and how VoxtNew runs it: the one table of
// the models' generation settings. The catalogue (M1b: downloads, the
// names the UI shows) keys its entries by ID.
type Profile struct {
	ID     string
	Format string // wire.FormatMLX or wire.FormatGGUF
	Repo   string // the Hugging Face repository it comes from
	File   string // a GGUF model's file in Repo ("" for MLX: the whole snapshot)
	Prompt PromptStyle
	// Sampling per skill; Default for a skill not listed. They win over
	// llm.Request's Temperature and TopP: they are tuned per model.
	Sampling map[llm.Skill]wire.Sampling
	Default  wire.Sampling
	// Template are the chat template's variables (MLX).
	Template map[string]any
	// MaxTokensCap bounds the answer whatever the input's length.
	MaxTokensCap int
}

// skills and their token budgets: Voxt's CustomLLMRequestRuntime
// (enhancement 1.10 × the input's characters + 128, at least 128, at most
// 1024; translation 1.35 × + 160, 128…1024; rewriting 1.35 × + 192,
// 256…1536).
type budget struct {
	per        float64
	extra, min int
	max        int
}

var budgets = map[llm.Skill]budget{
	llm.SkillTranscribeEnhance: {1.10, 128, 128, 1024},
	llm.SkillTranslateText:     {1.35, 160, 128, 1024},
	llm.SkillRewrite:           {1.35, 192, 256, 1536},
}

// MaxTokens is the answer's budget for skill over an input of chars
// characters (capped by the profile).
func (p Profile) MaxTokens(skill llm.Skill, chars int) int {
	b, ok := budgets[skill]
	if !ok {
		b = budgets[llm.SkillRewrite]
	}
	n := min(max(int(float64(max(1, chars))*b.per)+b.extra, b.min), b.max)
	if p.MaxTokensCap > 0 {
		n = min(n, p.MaxTokensCap)
	}
	return n
}

// SamplingFor is the sampling of skill.
func (p Profile) SamplingFor(skill llm.Skill) wire.Sampling {
	if s, ok := p.Sampling[skill]; ok {
		return s
	}
	return p.Default
}

// PrefillStepSize is MLX's prompt chunk for a prompt of chars characters
// (Voxt's: 256 under 1000, 512 under 3000, else 768); 0 for GGUF.
func (p Profile) PrefillStepSize(chars int) int {
	switch {
	case p.Format != wire.FormatMLX:
		return 0
	case chars < 1000:
		return 256
	case chars < 3000:
		return 512
	}
	return 768
}

// Qwen3.5's sampling (Voxt's local runs: greedy, with Qwen's 1.05
// repetition penalty over the last 32 tokens), but rewriting, which may
// choose its words: Qwen's non-thinking recommendation (0.7, top-p 0.8,
// top-k 20).
var (
	qwenExact   = wire.Sampling{Temperature: 0, TopP: 1, RepetitionPenalty: 1.05, RepetitionContextSize: 32}
	qwenRewrite = wire.Sampling{Temperature: 0.7, TopP: 0.8, TopK: 20, RepetitionPenalty: 1.05, RepetitionContextSize: 32}
	qwenNoThink = map[string]any{"enable_thinking": false}
)

// profiles are the curated models, the recommended one first.
var profiles = []Profile{
	{
		ID: Balanced, Format: wire.FormatMLX, Repo: "mlx-community/Qwen3.5-4B-OptiQ-4bit",
		Sampling: map[llm.Skill]wire.Sampling{llm.SkillRewrite: qwenRewrite},
		Default:  qwenExact, Template: qwenNoThink,
	},
	{
		ID: Fast, Format: wire.FormatMLX, Repo: "mlx-community/Qwen3.5-2B-4bit",
		Sampling: map[llm.Skill]wire.Sampling{llm.SkillRewrite: qwenRewrite},
		Default:  qwenExact, Template: qwenNoThink,
	},
	{
		// Hy-MT2's card: temperature 0.7, top-p 0.6, top-k 20, repetition
		// penalty 1.05, no system prompt.
		ID: Translate, Format: wire.FormatGGUF, Repo: "tencent/Hy-MT2-1.8B-GGUF", File: "Hy-MT2-1.8B-Q4_K_M.gguf",
		Prompt:       PromptHyMT,
		Default:      wire.Sampling{Temperature: 0.7, TopP: 0.6, TopK: 20, RepetitionPenalty: 1.05, RepetitionContextSize: 64},
		MaxTokensCap: 2048,
	},
	{
		// Qwen3-4B-Instruct-2507 has no thinking mode (no template
		// variable); its card recommends 0.7, top-p 0.8, top-k 20
		// (rewriting), greedy for the exact skills as the other Qwens
		// (docs/llm-models.md, "Borrowed models": the eval).
		ID: Qwen3, Format: wire.FormatMLX, Repo: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
		Sampling: map[llm.Skill]wire.Sampling{llm.SkillRewrite: qwenRewrite},
		Default:  qwenExact,
	},
	{
		// Gemma 4's card: temperature 1.0, top-p 0.95, top-k 64
		// (rewriting); greedy for the exact skills, no repetition penalty
		// (Gemma's own sampling has none, and no answer looped). Its
		// template thinks only when enable_thinking is set: off.
		ID: Gemma4, Format: wire.FormatMLX, Repo: "mlx-community/gemma-4-e4b-it-4bit",
		Sampling: map[llm.Skill]wire.Sampling{llm.SkillRewrite: {Temperature: 1.0, TopP: 0.95, TopK: 64, RepetitionPenalty: 1}},
		Default:  wire.Sampling{Temperature: 0, TopP: 1, RepetitionPenalty: 1},
		Template: qwenNoThink,
	},
}

// Profiles lists the curated models, the recommended one first.
func Profiles() []Profile { return append([]Profile(nil), profiles...) }

// Lookup is the profile of id.
func Lookup(id string) (Profile, bool) {
	for _, p := range profiles {
		if p.ID == id {
			return p, true
		}
	}
	return Profile{}, false
}

// TranslateOnly reports whether the model only translates (Translate's
// model list; other features must not offer it).
func (p Profile) TranslateOnly() bool { return p.Prompt == PromptHyMT }
