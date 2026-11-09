package compose

import "strings"

// BuiltinLayer renders one built-in template fragment (for example
// output-rewrite) with the same flags as Build.
func BuiltinLayer(set, name string, in Input) string {
	vars := Variables(in, set)
	selected := in.Task == Rewrite && strings.TrimSpace(in.Text) != "" || in.Task == Translate && in.Selected
	on := func(b bool) string {
		if b {
			return "1"
		}
		return ""
	}
	flags := map[string]string{
		"FOR_TRANSCRIBE": on(in.Task == Transcribe),
		"FOR_TRANSLATE":  on(in.Task == Translate),
		"SELECTED":       on(selected),
	}
	return strings.TrimSpace(Render(template(set, name), vars, flags))
}

// rewriteOutputLayer is the output layer for Rewrite: built-in or user output rules.
func rewriteOutputLayer(in Input, vars, flags map[string]string, render func(string, map[string]string) string) string {
	outLayer := render("output-rewrite", nil)
	if s := strings.TrimSpace(in.OutputPrompt); s != "" {
		outLayer = strings.TrimSpace(Render(s, vars, flags))
	}
	return outLayer
}
