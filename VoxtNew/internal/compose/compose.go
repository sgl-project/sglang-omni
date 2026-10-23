// Package compose is VoxtNew's prompt composer (plan §4.3): it builds the
// one language-model request of a session from layers, in a fixed order,
// instead of a long prompt per feature that repeats the same rules (Voxt's
// docs/TextEnhancementPromptLayeringPlan, which Voxt itself never landed):
//
//	system = Role            what VoxtNew is doing
//	       + Input policy    what the input is (speech, a selection, an instruction)
//	       + [Cleanup]       basic clean-up of speech recognition
//	       + [Structure]     lists and paragraphs (Transcribe only), or Form: keep it as said
//	       + Task            the feature's rules, then the user's addition to them
//	       + [Scene]         app enhancement: the app and the group's rules (M5)
//	       + [Glossary]      the dictionary's terms (a dictionary of up to GlossaryAllUpTo), or the rule for a <glossary> block
//	       + Output contract only the final text
//	[user, assistant]        the clean-up's example (speech: Example)
//	user   = [<glossary>…</glossary>]  a larger dictionary's terms that the text mentions
//	         <input>…</input>
//
// The input text is only ever in the user message, never in the system
// prompt (no {{RAW_TRANSCRIPTION}}): the system prompt stays the same for
// the same settings, and dictated text cannot pose as rules. For the same
// reason a glossary that depends on the text goes in the user message:
// the system prompt and the example, which the model's prompt cache
// keeps, stay the same from one dictation to the next.
//
// The rules are embedded templates in two sets, English and Simplified
// Chinese, chosen by the user's main language. Variables, in the
// templates and in the user's own prompts: {{USER_MAIN_LANGUAGE}},
// {{USER_OTHER_LANGUAGES}}, {{TARGET_LANGUAGE}}, {{APP_NAME}};
// "{{#if NAME}}…{{/if}}" keeps its text only when NAME is not empty.
//
// Translate builds on the same layers: speech gets the clean-up first (the
// clean-up layer's rules for translation), a selection is translated as
// it is (Input policy for a selection, no clean-up, its form kept); a
// Strict layer asks again when a first answer came back untranslated
// (Untranslated). Rewrite reads a spoken instruction (its own clean-up
// layer, for reading it, never applied to the selection) and either
// rewrites the selection or writes from scratch, then as a JSON answer
// {title, content} for the answer card (ParseAnswer, PartialAnswer);
// Previous turns make a follow-up.
//
// Imports: lang.
package compose

import (
	"embed"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
)

// Task is the feature a request is for.
type Task string

const (
	Transcribe Task = "transcribe"
	Translate  Task = "translate"
	Rewrite    Task = "rewrite"
)

// ErrTask: the task has no layers.
var ErrTask = errors.New("compose: unknown task")

// ErrInput: the input lacks what the task needs (Translate a target
// language, Rewrite an instruction).
var ErrInput = errors.New("compose: incomplete input")

// Template sets.
const (
	English = "en"
	Chinese = "zh-Hans"
)

// Term is a glossary entry: a term as it must be written and what speech
// recognition tends to make of it.
type Term struct {
	Term    string
	Aliases []string
}

// Scene is app enhancement's layer: the app in front and the matched
// group's prompt (M5).
type Scene struct {
	AppName string
	Prompt  string
}

// Input is everything a request is built from.
type Input struct {
	Task Task
	// Cleanup and Structure are the capability layers the feature has on
	// (settings.Transcribe.Cleanup, Structure; Structure only for
	// Transcribe).
	Cleanup   bool
	Structure bool
	// TaskPrompt is the user's addition to the task's rules
	// (settings.Feature.Prompt); "" for none.
	TaskPrompt string
	// Scene is the matched app enhancement group; nil for none.
	Scene *Scene
	// Glossary are the dictionary's enabled terms (Build keeps the ones
	// that fit: all of a small dictionary, of a large one those found in
	// Text).
	Glossary []Term
	// MainLanguage and OtherLanguages are lang codes ("zh", "en"):
	// settings.General's, with "" resolved by the caller. TargetLanguage
	// is a translation's (M4).
	MainLanguage   string
	OtherLanguages []string
	TargetLanguage string
	// Text is the input: the recognised speech (Transcribe, Translate),
	// the selection (Translate with Selected, Rewrite; "" when Rewrite
	// writes from scratch).
	Text string
	// Selected: Translate's Text is selected text, not speech (no clean-up,
	// its form kept).
	Selected bool
	// Instruction is Rewrite's spoken instruction.
	Instruction string
	// Strict asks Translate again, more firmly (Untranslated).
	Strict bool
	// Answer asks Rewrite for a JSON answer {title, content} (writing from
	// scratch, for the answer card).
	Answer bool
	// Previous are a Rewrite conversation's earlier turns, oldest first
	// (a follow-up from the answer card).
	Previous []Turn
}

// Turn is an earlier request of a conversation and its answer.
type Turn struct {
	Instruction, Selection string
	Answer                 Answer
}

// Message is a chat message of a composed request.
type Message struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// Output is a composed request.
type Output struct {
	System string
	User   string
	// History are the messages between System and User: a follow-up's
	// earlier turns, or the clean-up's example turn (Example).
	History []Message
	// Glossary is the <glossary> block of a dictionary of more than
	// GlossaryAllUpTo terms (the terms the text mentions), which opens
	// User; "" for none (a smaller dictionary's is in System).
	Glossary string
	// Language is the template set used (English, Chinese).
	Language string
	// Layers names the layers in the system prompt, in order; Profile
	// joins them for logs and history ("transcribe:cleanup+structure").
	Layers []string
}

// Profile is a short name of the composition.
func (o Output) Profile() string {
	if len(o.Layers) == 0 {
		return ""
	}
	task, rest := o.Layers[0], o.Layers[1:]
	var caps []string
	for _, l := range rest {
		switch l {
		case "selection", "cleanup", "structure", "strict", "custom", "scene", "glossary", "answer", "followup":
			caps = append(caps, l)
		}
	}
	if len(caps) == 0 {
		return task
	}
	return task + ":" + strings.Join(caps, "+")
}

//go:embed templates
var templates embed.FS

func template(set, name string) string {
	b, err := templates.ReadFile("templates/" + set + "/" + name + ".txt")
	if err != nil {
		panic(fmt.Sprintf("compose: no template %s/%s", set, name))
	}
	return strings.TrimRight(string(b), "\n")
}

// TemplateSet is the set of rules for a main language: Chinese for
// Chinese and Cantonese, English for every other language.
func TemplateSet(mainLanguage string) string {
	switch lang.Canonical(mainLanguage) {
	case "zh", "yue":
		return Chinese
	}
	return English
}

// Limits of the glossary. A dictionary of up to GlossaryAllUpTo terms
// goes in whole, in its order; a larger one only with the terms found in
// the text: unrelated terms would only cost tokens and distract the model.
const (
	MaxGlossaryTerms = 60
	MaxGlossaryChars = 2000
	GlossaryAllUpTo  = 30
)

// Messages is the request's messages, in order: System, History, User.
func (o Output) Messages() []Message {
	out := []Message{{Role: "system", Content: o.System}}
	out = append(out, o.History...)
	return append(out, Message{Role: "user", Content: o.User})
}

// Build composes the request of in.
func Build(in Input) (Output, error) {
	switch in.Task {
	case Transcribe, Translate, Rewrite:
	default:
		return Output{}, fmt.Errorf("%w: %s", ErrTask, in.Task)
	}
	if in.Task == Translate && strings.TrimSpace(in.TargetLanguage) == "" {
		return Output{}, fmt.Errorf("%w: no target language", ErrInput)
	}
	if in.Task == Rewrite && strings.TrimSpace(in.Instruction) == "" {
		return Output{}, fmt.Errorf("%w: no instruction", ErrInput)
	}
	set := TemplateSet(in.MainLanguage)
	vars := Variables(in, set)
	var parts, layers []string
	add := func(layer, text string) {
		text = strings.TrimSpace(text)
		if text == "" {
			return
		}
		parts = append(parts, text)
		layers = append(layers, layer)
	}
	// flags are the templates' own switches ("{{#if SELECTED}}").
	on := func(b bool) string {
		if b {
			return "1"
		}
		return ""
	}
	selected := in.Task == Rewrite && strings.TrimSpace(in.Text) != "" || in.Task == Translate && in.Selected
	flags := map[string]string{
		"FOR_TRANSCRIBE": on(in.Task == Transcribe),
		"FOR_TRANSLATE":  on(in.Task == Translate),
		"SELECTED":       on(selected),
	}
	render := func(name string, extra map[string]string) string {
		x := maps.Clone(flags)
		maps.Copy(x, extra)
		return Render(template(set, name), vars, x)
	}

	add(string(in.Task), render("role-"+string(in.Task), nil))
	switch in.Task {
	case Transcribe:
		add("input", render("input-speech", nil))
		if in.Cleanup {
			add("cleanup", render("cleanup", nil))
		}
		if in.Structure {
			add("structure", render("structure", nil))
		} else {
			add("form", render("form", nil))
		}
		add("task", render("task-transcribe", nil))
	case Translate:
		if in.Selected {
			add("selection", render("input-selection", nil))
		} else {
			add("input", render("input-speech", nil))
			if in.Cleanup {
				add("cleanup", render("cleanup", nil))
			}
			add("form", render("form", nil))
		}
		add("task", render("task-translate", nil))
		if in.Strict {
			add("strict", render("strict", nil))
		}
	case Rewrite:
		layer := "input"
		if selected {
			layer = "selection"
		}
		add(layer, render("input-rewrite", nil))
		if in.Cleanup {
			add("cleanup", render("cleanup-instruction", nil))
		}
		if selected {
			add("task", render("task-rewrite", nil))
		} else {
			add("task", render("task-generate", nil))
		}
		if len(in.Previous) > 0 {
			add("followup", render("followup", nil))
		}
	}
	if p := strings.TrimSpace(in.TaskPrompt); p != "" {
		add("custom", render("custom", map[string]string{"TASK_PROMPT": Render(p, vars, nil)}))
	}
	if s := in.Scene; s != nil && strings.TrimSpace(s.AppName) != "" {
		add("scene", render("scene", map[string]string{"SCENE_PROMPT": Render(strings.TrimSpace(s.Prompt), vars, nil)}))
	}
	out := Output{Language: set}
	g := Glossary(in.Glossary, in.Instruction+"\n"+in.Text, set)
	if len(distinctTerms(in.Glossary)) <= GlossaryAllUpTo {
		if g != "" {
			add("glossary", render("glossary", map[string]string{"GLOSSARY": g}))
		}
	} else {
		// The rule is the same for every dictation; the terms, which the
		// text picks, open the user message.
		add("glossary", render("glossary-user", nil))
		if g != "" {
			out.Glossary = wrap("glossary", g)
		}
	}
	switch {
	case in.Task == Rewrite && in.Answer:
		add("answer", render("output-answer", nil))
	case in.Task == Rewrite:
		add("output", render("output-rewrite", nil))
	default:
		add("output", render("output-"+string(in.Task), nil))
	}
	out.System, out.Layers = strings.Join(parts, "\n\n"), layers
	if in.Task == Rewrite {
		for _, t := range in.Previous {
			out.History = append(out.History,
				Message{Role: "user", Content: RewriteMessage(t.Instruction, t.Selection)},
				Message{Role: "assistant", Content: t.Answer.reply(in.Answer)})
		}
		out.User = out.withGlossary(RewriteMessage(in.Instruction, in.Text))
	} else {
		out.User = out.withGlossary(UserMessage(in.Text))
		if input, answer, ok := Example(in); ok {
			out.History = []Message{{Role: "user", Content: UserMessage(input)}, {Role: "assistant", Content: answer}}
		}
	}
	return out, nil
}

// withGlossary is a user message msg after the glossary block, if any.
func (o Output) withGlossary(msg string) string {
	if o.Glossary == "" {
		return msg
	}
	return o.Glossary + "\n\n" + msg
}

// RewriteMessage is Rewrite's user message: the instruction, then the
// selection when there is one.
func RewriteMessage(instruction, selection string) string {
	msg := wrap("instruction", instruction)
	if strings.TrimSpace(selection) != "" {
		msg += "\n\n" + wrap("selected_text", selection)
	}
	return msg
}

func wrap(tag, text string) string {
	text = strings.ReplaceAll(strings.TrimSpace(text), "</"+tag+">", "<\\/"+tag+">")
	return "<" + tag + ">\n" + text + "\n</" + tag + ">"
}

// UserMessage wraps the input for the user message.
func UserMessage(text string) string { return wrap("input", text) }

// Variables are the public variables of in, in template set set's
// language.
func Variables(in Input, set string) map[string]string {
	name := func(code string) string {
		if l, ok := lang.Lookup(code); ok {
			return l.NameIn(set)
		}
		return code
	}
	sep := ", "
	if set == Chinese {
		sep = "、"
	}
	var others []string
	for _, c := range in.OtherLanguages {
		if c = strings.TrimSpace(c); c != "" && lang.Canonical(c) != lang.Canonical(in.MainLanguage) {
			others = append(others, name(c))
		}
	}
	app := ""
	if in.Scene != nil {
		app = strings.TrimSpace(in.Scene.AppName)
	}
	main := name(in.MainLanguage)
	if strings.TrimSpace(in.MainLanguage) == "" {
		main = name("en")
	}
	return map[string]string{
		"USER_MAIN_LANGUAGE":   main,
		"USER_OTHER_LANGUAGES": strings.Join(others, sep),
		"TARGET_LANGUAGE":      LanguageName(in.TargetLanguage, set),
		"APP_NAME":             app,
	}
}

// VariableNames are the variables a user's prompt may use, for the
// feature pages.
var VariableNames = []string{"USER_MAIN_LANGUAGE", "USER_OTHER_LANGUAGES", "TARGET_LANGUAGE", "APP_NAME"}

var ifBlock = regexp.MustCompile(`(?s)\{\{#if ([A-Z_]+)\}\}(.*?)\{\{/if\}\}`)

// Render fills a template: "{{#if NAME}}…{{/if}}" blocks, then
// "{{NAME}}" from vars and extra, in one pass (a value is never expanded
// again). Unknown names stay as written.
func Render(tpl string, vars, extra map[string]string) string {
	get := func(k string) (string, bool) {
		if v, ok := extra[k]; ok {
			return v, true
		}
		v, ok := vars[k]
		return v, ok
	}
	tpl = ifBlock.ReplaceAllStringFunc(tpl, func(m string) string {
		sub := ifBlock.FindStringSubmatch(m)
		if v, _ := get(sub[1]); strings.TrimSpace(v) != "" {
			return sub[2]
		}
		return ""
	})
	var pairs []string
	for k, v := range vars {
		if _, over := extra[k]; !over {
			pairs = append(pairs, "{{"+k+"}}", v)
		}
	}
	for k, v := range extra {
		pairs = append(pairs, "{{"+k+"}}", v)
	}
	return strings.NewReplacer(pairs...).Replace(tpl)
}

// Glossary lists the terms that fit the limits, one per line ("- Term
// (heard as: alias, alias)"): a dictionary of at most GlossaryAllUpTo
// terms whole, in its order, whatever text says (the system prompt stays
// the same from one dictation to the next, which the model's prompt cache
// keeps); a larger one only with the terms found in text (their spelling
// or an alias, ignoring case and spaces). "" when there are none.
func Glossary(terms []Term, text, set string) string {
	norm := func(s string) string {
		return strings.Map(func(r rune) rune {
			if unicode.IsSpace(r) || r == '-' || r == '_' {
				return -1
			}
			return unicode.ToLower(r)
		}, s)
	}
	ntext := norm(text)
	found := func(t Term) bool {
		if ntext == "" {
			return false
		}
		for _, s := range append([]string{t.Term}, t.Aliases...) {
			if n := norm(s); n != "" && strings.Contains(ntext, n) {
				return true
			}
		}
		return false
	}
	all := distinctTerms(terms)
	var mentioned []Term
	for _, t := range all {
		if found(t) {
			mentioned = append(mentioned, t)
		}
	}
	heard := "heard as"
	sep := ", "
	if set == Chinese {
		heard, sep = "可能被听成", "、"
	}
	if len(all) > GlossaryAllUpTo {
		all = mentioned
	}
	var lines []string
	size := 0
	for _, t := range all {
		line := "- " + t.Term
		var aliases []string
		for _, a := range t.Aliases {
			if a = strings.TrimSpace(a); a != "" && !strings.EqualFold(a, t.Term) {
				aliases = append(aliases, a)
			}
		}
		if len(aliases) > 0 {
			if set == Chinese {
				line += "（" + heard + "：" + strings.Join(aliases, sep) + "）"
			} else {
				line += " (" + heard + ": " + strings.Join(aliases, sep) + ")"
			}
		}
		if len(lines) == MaxGlossaryTerms || size+len(line) > MaxGlossaryChars {
			break
		}
		lines = append(lines, line)
		size += len(line) + 1
	}
	return strings.Join(lines, "\n")
}

// distinctTerms are terms trimmed, each once (ignoring case), without
// empty ones.
func distinctTerms(terms []Term) []Term {
	var out []Term
	seen := map[string]bool{}
	for _, t := range terms {
		t.Term = strings.TrimSpace(t.Term)
		if t.Term == "" || seen[strings.ToLower(t.Term)] {
			continue
		}
		seen[strings.ToLower(t.Term)] = true
		out = append(out, t)
	}
	return out
}

// Clean tidies a model's answer to input: without the <input> tags a
// model echoed, or the quotes it wrapped the whole answer in (unless the
// input was quoted so itself).
func Clean(out, input string) string {
	s := strings.TrimSpace(out)
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(s, "<input>"), "</input>"))
	in := strings.TrimSpace(input)
	for _, q := range [][2]string{{`"`, `"`}, {"“", "”"}, {"「", "」"}} {
		wrapped := func(t string) bool {
			return len(t) > len(q[0])+len(q[1]) && strings.HasPrefix(t, q[0]) && strings.HasSuffix(t, q[1]) &&
				!strings.Contains(t[len(q[0]):len(t)-len(q[1])], q[0])
		}
		if wrapped(s) && !wrapped(in) {
			s = strings.TrimSpace(s[len(q[0]) : len(s)-len(q[1])])
		}
	}
	return s
}
