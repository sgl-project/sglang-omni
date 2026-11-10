package main

// A remote language model's generation settings in its dialog
// (remote_dialog.go), field for field as Voxt 1.x had them: Generation
// settings, the service's own options, then the folding Advanced section,
// each field only where the service takes it (providers.Capabilities).
// Unlike 1.x, every sampling field (temperature, top P, top K, min P, the
// penalties) is in Advanced, folded by default, and so are 1.x's Expert
// JSON fields (the extra body, Ollama's options).

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
	extraBody                       string
	advanced                        bool // the folding section, open
	// api is the API format's title (it is saved in settings.Provider, not
	// in Generation).
	api string
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
	apiFormats      = []string{"", llm.APIResponses}
)

func apiFormatTitle(v string) string {
	if v == llm.APIResponses {
		return "Responses"
	}
	return "Chat Completions"
}

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
		extraBody: g.ExtraBodyJSON,
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

// generationFields draws d's generation settings, those its service
// takes for the model chosen.
func (a *app) generationFields(c *ui.Context, d *providerDialog) {
	t := c.Theme()
	f := &d.gen
	caps := providers.Capabilities(d.id, d.setup().Model)
	num := func(key, title, placeholder string, value *string) {
		fieldInfo(c, title, "", func() {
			ui.TextInput(c.Key(key), value).Placeholder(placeholder).Label(title)
		})
	}
	// half is a num in a pair's row: the two share its width. Alone in the
	// dialog's column, Basis(0) would make a num 0 high and the rows after
	// it would cover its field.
	half := func(key, title, placeholder string, value *string) {
		ui.Column(c).Grow(1).Basis(0).MinWidth(0).Children(func() { num(key, title, placeholder, value) })
	}
	pair := func(fn func()) { ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Start).Children(fn) }

	sectionTitle(c, i18n.T("Generation settings"))
	f.fit(caps) // the model changed: a mode it lacks
	if len(caps.ThinkModes) > 0 {
		modes := titles(caps.ThinkModes, thinkTitle)
		fieldInfo(c, i18n.T("Thinking"), "", func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Wrap().Children(func() {
				ui.Select(c.Key("gen-think"), &f.think, modes).Label(i18n.T("Thinking")).Width(px(t, 240))
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
	if caps.Verbosity {
		fieldInfo(c, i18n.T("Verbosity"), "", func() {
			ui.Select(c.Key("gen-verbosity"), &f.verbosity, titles(verbosities, verbosityTitle)).Label(i18n.T("Verbosity")).Width(px(t, 180))
		})
	}
	if caps.KeepAlive {
		num("gen-keepalive", i18n.T("Keep the model loaded for"), i18n.T("e.g. 5m"), &f.keepAlive)
	}
	if caps.IncludeUsage {
		ui.Switch(c.Key("gen-usage"), &f.usage).Label(i18n.T("Include usage in streamed responses"))
	}

	// The sampling fields: empty sends nothing, and the service uses its
	// own default for the model.
	serviceDefault := i18n.T("Service default")
	collapsible(c, i18n.T("Advanced"), &f.advanced, nil, func() {
		if caps.APIFormat {
			fieldInfo(c, i18n.T("API format"), i18n.T("Chat Completions works with nearly every service. Responses is OpenAI's newer API, and a custom service must offer POST /responses to use it. Stop sequences and Logprobs are not sent with Responses."), func() {
				ui.Select(c.Key("gen-api"), &f.api, titles(apiFormats, apiFormatTitle)).Label(i18n.T("API format")).Width(px(t, 240))
			})
		}
		responses := valueOf(f.api, apiFormats, apiFormatTitle) == llm.APIResponses
		pair(func() {
			if caps.Temperature {
				half("gen-temp", i18n.T("Temperature"), serviceDefault, &f.temperature)
			}
			if caps.TopP {
				half("gen-topp", "Top P", serviceDefault, &f.topP)
			}
		})
		pair(func() {
			if caps.TopK {
				half("gen-topk", "Top K", serviceDefault, &f.topK)
			}
			if caps.MinP {
				half("gen-minp", "Min P", serviceDefault, &f.minP)
			}
		})
		switch {
		case caps.FrequencyOnly:
			num("gen-frequency", i18n.T("Frequency penalty"), serviceDefault, &f.frequency)
		case caps.Penalties:
			pair(func() {
				half("gen-presence", i18n.T("Presence penalty"), serviceDefault, &f.presence)
				half("gen-frequency", i18n.T("Frequency penalty"), serviceDefault, &f.frequency)
			})
			num("gen-repetition", i18n.T("Repetition penalty"), serviceDefault, &f.repetition)
		}
		if caps.Seed {
			num("gen-seed", i18n.T("Seed"), "42", &f.seed)
		}
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
		if caps.Logprobs && !responses {
			fieldInfo(c, "Logprobs", "", func() {
				ui.Switch(c.Key("gen-logprobs"), &f.logprobs).Label("Logprobs")
			})
			if f.logprobs {
				num("gen-toplogprobs", i18n.T("Top logprobs (optional)"), "5", &f.topLogprobs)
			}
		}
		if caps.Stop && !responses {
			fieldInfo(c, i18n.T("Stop sequences"), "", func() {
				ui.TextArea(c.Key("gen-stop"), &f.stop).Label(i18n.T("Stop sequences")).Placeholder(i18n.T("One stop sequence per line")).MinHeight(px(t, 56))
			})
		}
		if caps.ExtraBody {
			example := `{"user": "me"}`
			if caps.KeepAlive { // note (luojiaxuan): Ollama, whose model options go in "options"
				example = `{"options": {"num_ctx": 8192}}`
			}
			fieldInfo(c, i18n.T("Request parameters"), i18n.T("A JSON object merged into the request. Its keys override Voxt's, nested objects are merged, and null leaves a field out."), func() {
				ui.TextArea(c.Key("gen-extra"), &f.extraBody).Label(i18n.T("Request parameters")).Placeholder(example).MinHeight(px(t, 72))
			})
		}
	})
}
