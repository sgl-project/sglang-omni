package compose

import "strings"

// addDictionary adds the personal dictionary's layer, if any: after the
// task's own layers and before the scene, glossary and input, so the
// system prompt's prefix stays the same from one request to the next
// (the model server's prompt cache keeps it). The layer is the user's
// notes as written, with no text or tags of Voxt's around them: every
// prompt the user sees can be edited (#3872). It returns the layer.
func addDictionary(in Input, add func(layer, text string)) string {
	if strings.TrimSpace(in.Dictionary) == "" {
		return ""
	}
	layer := strings.TrimSpace(in.Dictionary)
	add("dictionary", layer)
	return layer
}
