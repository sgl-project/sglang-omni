package compose

import (
	"maps"
	"strings"
)

// buildFullPrompt sends the user's template as the system prompt, with
// scene and glossary layers when they apply (FB3).
func buildFullPrompt(in Input, set string, vars, flags map[string]string, full string) (Output, error) {
	render := func(name string, extra map[string]string) string {
		x := maps.Clone(flags)
		maps.Copy(x, extra)
		return Render(template(set, name), vars, x)
	}
	var parts, layers []string
	add := func(layer, text string) {
		text = strings.TrimSpace(text)
		if text == "" {
			return
		}
		parts = append(parts, text)
		layers = append(layers, layer)
	}
	add("custom", strings.TrimSpace(Render(full, vars, flags)))
	dictionary := addDictionary(in, add)
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
		add("glossary", render("glossary-user", nil))
		if g != "" {
			out.Glossary = wrap("glossary", g)
		}
	}
	out.System, out.Layers, out.Dictionary = strings.Join(parts, "\n\n"), layers, dictionary
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
