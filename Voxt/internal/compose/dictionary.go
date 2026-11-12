package compose

import "strings"

// addDictionary adds the personal dictionary's layer, if any: after the
// task's own layers and before the scene, glossary and input, so the
// system prompt's prefix stays the same from one request to the next
// (the model server's prompt cache keeps it). It returns the layer.
func addDictionary(in Input, add func(layer, text string), render func(name string, extra map[string]string) string) string {
	if strings.TrimSpace(in.Dictionary) == "" {
		return ""
	}
	layer := strings.TrimSpace(render("dictionary", map[string]string{"DICTIONARY": wrap("dictionary", in.Dictionary)}))
	add("dictionary", layer)
	return layer
}
