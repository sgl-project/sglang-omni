package main

// A remote language model's generation settings in its dialog
// (remote_dialog.go), field for field as Voxt 1.x had them: Generation
// settings, the service's own options, then the folding Advanced and
// Expert sections, each field only where the service takes it
// (providers.Capabilities).

import (
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// genForm is the generation settings as typed: numbers as text, menus as
// their titles (settings parses them).
type genForm struct {
	think, effort, budget           string
	maxTokens, temperature, topP    string
	topK, minP, seed                string
	presence, frequency, repetition string
	logprobs                        bool
	topLogprobs                     string
	format, schema, stop            string // stop: one per line
	verbosity, keepAlive            string
	usage                           bool // oMLX: usage in stream events (on by default)
	options, extraBody              string
	advanced, expert                bool // the folding sections, open
}

// The menus' choices: values and their titles.
func thinkTitle(mode string) string {
	switch mode {
	case llm.ThinkOff:
		return i18n.T("Thinking off")
	case llm.ThinkOn:
		return i18n.T("Thinking on")
	case llm.ThinkEffort:
		return i18n.T("Effort")
	case llm.ThinkBudget:
		return i18n.T("Budget")
	}
	return i18n.T("Default")
}

func effortTitle(v string) string {
	switch v {
	case "none":
		return i18n.T("None")
	case "minimal":
		return i18n.T("Minimal")
	case "low":
		return i18n.T("Low")
	case "medium":
		return i18n.T("Medium")
	case "high":
		return i18n.T("High")
	case "xhigh":
		return i18n.T("Extra high")
	case "max":
		return i18n.T("Max")
	}
	return v
}

func formatTitle(v string) string {
	switch v {
	case llm.ResponseJSON:
		return "JSON"
	case llm.ResponseJSONSchema:
		return "JSON Schema"
	}
	return i18n.T("Plain text")
}

var (
	responseFormats = []string{"", llm.ResponseJSON, llm.ResponseJSONSchema}
	verbosities     = []string{"", "low", "medium", "high"}
)

func verbosityTitle(v string) string {
	if v == "" {
		return i18n.T("Default")
	}
	return effortTitle(v)
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
		options: g.OptionsJSON, extraBody: g.ExtraBodyJSON,
	}
	if g.ThinkingBudget > 0 {
		f.budget = strconv.Itoa(g.ThinkingBudget)
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
	g.OptionsJSON, g.ExtraBodyJSON = strings.TrimSpace(f.options), strings.TrimSpace(f.extraBody)
	return g, bad
}

// generationFields draws d's generation settings, those its service
// takes for the model chosen.
func (a *app) generationFields(c *ui.Context, d *providerDialog) {
	t := c.Theme()
	f := &d.gen
	caps := providers.Capabilities(d.id, d.setup().Model)
	num := func(key, title, placeholder string, value *string) {
		ui.Column(c).Grow(1).Basis(0).MinWidth(0).Children(func() {
			fieldInfo(c, title, "", func() {
				ui.TextInput(c.Key(key), value).Placeholder(placeholder).Label(title)
			})
		})
	}
	pair := func(fn func()) { ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(fn) }

	sectionTitle(c, i18n.T("Generation settings"))
	if len(caps.ThinkModes) > 0 {
		modes := titles(caps.ThinkModes, thinkTitle)
		if !slices.Contains(modes, f.think) {
			f.think = modes[0] // the model changed: a mode it lacks
		}
		fieldInfo(c, i18n.T("Think"), "", func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Select(c.Key("gen-think"), &f.think, modes).Label(i18n.T("Think")).Width(px(t, 240))
				switch valueOf(f.think, caps.ThinkModes, thinkTitle) {
				case llm.ThinkEffort:
					if len(caps.Efforts) == 0 {
						ui.TextInput(c.Key("gen-effort"), &f.effort).Placeholder(i18n.T("e.g. low")).Label(i18n.T("Effort")).Width(px(t, 160))
						break
					}
					effort := effortTitle(f.effort)
					if !slices.Contains(caps.Efforts, f.effort) {
						effort = effortTitle(caps.Efforts[0])
					}
					if ui.Select(c.Key("gen-effort"), &effort, titles(caps.Efforts, effortTitle)).Label(i18n.T("Effort")).Width(px(t, 180)).Changed() || f.effort == "" || !slices.Contains(caps.Efforts, f.effort) {
						f.effort = valueOf(effort, caps.Efforts, effortTitle)
					}
				case llm.ThinkBudget:
					ui.TextInput(c.Key("gen-budget"), &f.budget).Placeholder(i18n.T("Budget tokens")).Label(i18n.T("Budget tokens")).Width(px(t, 160))
				}
			})
		})
	}
	if caps.MaxOutputTokens {
		num("gen-max", i18n.T("Max output tokens (optional)"), i18n.T("e.g. 4096"), &f.maxTokens)
	}
	pair(func() {
		if caps.Temperature {
			num("gen-temp", i18n.T("Temperature"), "0.2", &f.temperature)
		}
		if caps.TopP {
			num("gen-topp", "Top P", "0.9", &f.topP)
		}
	})
	if caps.Verbosity {
		fieldInfo(c, i18n.T("Verbosity"), "", func() {
			ui.Select(c.Key("gen-verbosity"), &f.verbosity, titles(verbosities, verbosityTitle)).Label(i18n.T("Verbosity")).Width(px(t, 180))
		})
	}
	if caps.KeepAlive {
		num("gen-keepalive", i18n.T("Keep the model loaded for"), i18n.T("e.g. 5m"), &f.keepAlive)
	}
	if caps.IncludeUsage {
		ui.Switch(c.Key("gen-usage"), &f.usage).Label(i18n.T("Usage in stream events"))
	}

	collapsible(c, i18n.T("Advanced"), &f.advanced, nil, func() {
		if caps.ResponseFormat {
			formats := responseFormats[:2]
			if caps.JSONSchema {
				formats = responseFormats
			}
			fieldInfo(c, i18n.T("Response format"), "", func() {
				ui.Select(c.Key("gen-format"), &f.format, titles(formats, formatTitle)).Label(i18n.T("Response format")).Width(px(t, 240))
			})
			if caps.JSONSchema && valueOf(f.format, formats, formatTitle) == llm.ResponseJSONSchema {
				ui.TextArea(c.Key("gen-schema"), &f.schema).Label("JSON Schema").Placeholder(`{"type": "object", …}`).MinHeight(px(t, 96))
			}
		}
		pair(func() {
			if caps.TopK {
				num("gen-topk", "Top K", "40", &f.topK)
			}
			if caps.MinP {
				num("gen-minp", "Min P", "0.05", &f.minP)
			}
		})
		if caps.Seed {
			num("gen-seed", i18n.T("Seed"), "42", &f.seed)
		}
		switch {
		case caps.FrequencyOnly:
			num("gen-frequency", i18n.T("Frequency penalty"), "0", &f.frequency)
		case caps.Penalties:
			pair(func() {
				num("gen-presence", i18n.T("Presence penalty"), "0", &f.presence)
				num("gen-frequency", i18n.T("Frequency penalty"), "0", &f.frequency)
			})
			num("gen-repetition", i18n.T("Repetition penalty"), "1.1", &f.repetition)
		}
		if caps.Logprobs {
			ui.Switch(c.Key("gen-logprobs"), &f.logprobs).Label("Logprobs")
			if f.logprobs {
				num("gen-toplogprobs", i18n.T("Top logprobs (optional)"), "5", &f.topLogprobs)
			}
		}
		if caps.Stop {
			fieldInfo(c, i18n.T("Stop sequences"), "", func() {
				ui.TextArea(c.Key("gen-stop"), &f.stop).Label(i18n.T("Stop sequences")).Placeholder(i18n.T("One stop sequence per line")).MinHeight(px(t, 56))
			})
		}
	})
	if caps.ExtraBody || caps.Options {
		collapsible(c, i18n.T("Expert"), &f.expert, nil, func() {
			if caps.ExtraBody {
				fieldInfo(c, i18n.T("Extra Body JSON"), i18n.T("Merged into the request body after Voxt's own fields, so its keys win."), func() {
					ui.TextArea(c.Key("gen-extra"), &f.extraBody).Label(i18n.T("Extra Body JSON")).Placeholder(`{"key": "value"}`).MinHeight(px(t, 72))
				})
			}
			if caps.Options {
				fieldInfo(c, "Options JSON", i18n.T("Merged into Ollama's options, so its keys win."), func() {
					ui.TextArea(c.Key("gen-options"), &f.options).Label("Options JSON").Placeholder(`{"num_ctx": 8192}`).MinHeight(px(t, 72))
				})
			}
		})
	}
}
