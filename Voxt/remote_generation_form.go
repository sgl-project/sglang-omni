package main

// A remote language model's generation settings as typed in its dialog.
// Numbers stay as text and menus as their titles until Save parses them.

import (
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// genForm is the generation settings as typed: numbers as text, menus as
// their titles (settings parses them).
type genForm struct {
	think, effort, budget           string
	shortOff                        bool   // Off for short inputs (#3951)
	shortUnder                      string // its limit, in characters; "": shortUnderDefault
	maxTokens, temperature, topP    string
	topK, minP, seed                string
	presence, frequency, repetition string
	logprobs                        bool
	topLogprobs                     string
	format, schema, stop            string // stop: one per line
	verbosity, keepAlive            string
	usage                           bool // oMLX: usage in stream events (on by default)
	extraBody                       string
	advanced                        bool // the folding section, open
	// api is the API format's title (it is saved in settings.Provider, not
	// in Generation).
	api string
}

// The menus' choices: values and their titles.
func thinkTitle(mode string) string { return thinkTitleFor(i18n.CurrentTranslator(), mode) }

func thinkTitleFor(translator *i18n.Translator, mode string) string {
	switch mode {
	case llm.ThinkOff:
		return translator.T("Thinking off")
	case llm.ThinkOn:
		return translator.T("Thinking on")
	case llm.ThinkEffort:
		return translator.T("Effort")
	case llm.ThinkBudget:
		return translator.T("Budget")
	}
	return translator.T("Default")
}

func effortTitle(v string) string { return effortTitleFor(i18n.CurrentTranslator(), v) }

func effortTitleFor(translator *i18n.Translator, v string) string {
	switch v {
	case "none":
		return translator.T("None")
	case "minimal":
		return translator.T("Minimal")
	case "low":
		return translator.T("Low")
	case "medium":
		return translator.T("Medium")
	case "high":
		return translator.T("High")
	case "xhigh":
		return translator.T("Extra high")
	case "max":
		return translator.T("Max")
	}
	return v
}

func formatTitle(v string) string { return formatTitleFor(i18n.CurrentTranslator(), v) }

func formatTitleFor(translator *i18n.Translator, v string) string {
	switch v {
	case llm.ResponseJSON:
		return "JSON"
	case llm.ResponseJSONSchema:
		return "JSON Schema"
	}
	return translator.T("Plain text")
}

var (
	responseFormats = []string{"", llm.ResponseJSON, llm.ResponseJSONSchema}
	verbosities     = []string{"", "low", "medium", "high"}
	apiFormats      = []string{"", llm.APIResponses}
)

func apiFormatTitle(v string) string {
	if v == llm.APIResponses {
		return "Responses"
	}
	return "Chat Completions"
}

func verbosityTitle(v string) string { return verbosityTitleFor(i18n.CurrentTranslator(), v) }

func verbosityTitleFor(translator *i18n.Translator, v string) string {
	if v == "" {
		return translator.T("Default")
	}
	return effortTitleFor(translator, v)
}

// valueOf is the value among values whose title is title ("" for none).
func valueOf(title string, values []string, titleOf func(string) string) string {
	for _, v := range values {
		if titleOf(v) == title {
			return v
		}
	}
	return ""
}

func titles(values []string, titleOf func(string) string) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = titleOf(v)
	}
	return out
}

// shortUnderDefault is Off for short inputs' limit when its field is
// empty: about a sentence or two.
const shortUnderDefault = 200

// thinks says thinking mode m asks the model to think, so Off for short
// inputs changes something.
func thinks(m string) bool { return m == llm.ThinkOn || m == llm.ThinkEffort || m == llm.ThinkBudget }

// genFormOf is the form of saved settings g.
func genFormOf(g settings.Generation) genForm {
	f := genForm{
		think: thinkTitle(g.Thinking), effort: g.ThinkingEffort,
		maxTokens: numText(g.MaxOutputTokens), temperature: numText(g.Temperature), topP: numText(g.TopP),
		topK: numText(g.TopK), minP: numText(g.MinP), seed: numText(g.Seed),
		presence: numText(g.PresencePenalty), frequency: numText(g.FrequencyPenalty), repetition: numText(g.RepetitionPenalty),
		logprobs: g.Logprobs, topLogprobs: numText(g.TopLogprobs),
		format: formatTitle(g.ResponseFormat), schema: g.JSONSchema, stop: strings.Join(g.Stop, "\n"),
		verbosity: verbosityTitle(g.Verbosity), keepAlive: g.KeepAlive, usage: g.IncludeUsage == nil || *g.IncludeUsage,
		extraBody: g.ExtraBodyJSON,
	}
	if g.ThinkingBudget > 0 {
		f.budget = strconv.Itoa(g.ThinkingBudget)
	}
	if f.shortOff = g.ThinkingOffUnder > 0; f.shortOff {
		f.shortUnder = strconv.Itoa(g.ThinkingOffUnder)
	}
	return f
}

func numText[T int | int64 | float64](v *T) string {
	if v == nil {
		return ""
	}
	switch x := any(*v).(type) {
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case int:
		return strconv.Itoa(x)
	case int64:
		return strconv.FormatInt(x, 10)
	}
	return ""
}

// settings parses f; bad is the title of the first field that is not a
// number ("" when all are).
func (f *genForm) settings() (g settings.Generation, bad string) {
	float := func(s, title string) *float64 {
		if s = strings.TrimSpace(s); s == "" {
			return nil
		}
		v, err := strconv.ParseFloat(s, 64)
		if (err != nil || math.IsNaN(v) || math.IsInf(v, 0)) && bad == "" {
			bad = title
		}
		return &v
	}
	integer := func(s, title string) *int {
		if s = strings.TrimSpace(s); s == "" {
			return nil
		}
		v, err := strconv.Atoi(s)
		if err != nil && bad == "" {
			bad = title
		}
		return &v
	}
	g.Thinking = valueOf(f.think, []string{llm.ThinkDefault, llm.ThinkOff, llm.ThinkOn, llm.ThinkEffort, llm.ThinkBudget}, thinkTitle)
	switch g.Thinking {
	case llm.ThinkEffort:
		g.ThinkingEffort = strings.TrimSpace(f.effort)
	case llm.ThinkBudget:
		if b := integer(f.budget, i18n.T("Budget tokens")); b != nil {
			g.ThinkingBudget = *b
		}
	}
	if f.shortOff && thinks(g.Thinking) {
		g.ThinkingOffUnder = shortUnderDefault
		if n := integer(f.shortUnder, i18n.T("Fewer characters than")); n != nil && *n > 0 {
			g.ThinkingOffUnder = *n
		}
	}
	// 0 asks for no limit, as an empty field does (the request would ask
	// for 1 token).
	if g.MaxOutputTokens = integer(f.maxTokens, i18n.T("Max output tokens (optional)")); g.MaxOutputTokens != nil && *g.MaxOutputTokens < 1 {
		g.MaxOutputTokens = nil
	}
	g.Temperature = float(f.temperature, i18n.T("Temperature"))
	g.TopP = float(f.topP, "Top P")
	g.TopK = integer(f.topK, "Top K")
	g.MinP = float(f.minP, "Min P")
	if s := integer(f.seed, i18n.T("Seed")); s != nil {
		seed := int64(*s)
		g.Seed = &seed
	}
	g.PresencePenalty = float(f.presence, i18n.T("Presence penalty"))
	g.FrequencyPenalty = float(f.frequency, i18n.T("Frequency penalty"))
	g.RepetitionPenalty = float(f.repetition, i18n.T("Repetition penalty"))
	if g.Logprobs = f.logprobs; g.Logprobs {
		g.TopLogprobs = integer(f.topLogprobs, i18n.T("Top logprobs (optional)"))
	}
	g.ResponseFormat = valueOf(f.format, responseFormats, formatTitle)
	if g.ResponseFormat == llm.ResponseJSONSchema {
		g.JSONSchema = strings.TrimSpace(f.schema)
	}
	for _, s := range strings.Split(f.stop, "\n") {
		if s = strings.TrimSpace(s); s != "" {
			g.Stop = append(g.Stop, s)
		}
	}
	g.Verbosity = valueOf(f.verbosity, verbosities, verbosityTitle)
	g.KeepAlive = strings.TrimSpace(f.keepAlive)
	if !f.usage {
		off := false
		g.IncludeUsage = &off
	}
	g.ExtraBodyJSON = strings.TrimSpace(f.extraBody)
	return g, bad
}

// fit makes f's Think, and its effort, ones that caps offers, as the
// dialog shows them: opening the dialog must not count as a change.
func (f *genForm) fit(caps providers.GenCaps) {
	if len(caps.ThinkModes) == 0 {
		return
	}
	if modes := titles(caps.ThinkModes, thinkTitle); !slices.Contains(modes, f.think) {
		f.think = modes[0]
	}
	if valueOf(f.think, caps.ThinkModes, thinkTitle) == llm.ThinkEffort && len(caps.Efforts) > 0 && !slices.Contains(caps.Efforts, f.effort) {
		f.effort = caps.Efforts[0]
	}
}
