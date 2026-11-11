package compose

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// Example is the clean-up's example turn for in, when it has one: a short
// dictation (fillers, a self-correction) and its answer, sent as an
// earlier user and assistant message before the input. Small local models
// follow the clean-up rules much better after one example (docs/eval.md).
// Speech only (Transcribe, or Translate of speech) with Cleanup on; for
// Translate, when the template set has the answer in the target language
// (its own, or the other set's: English and Simplified Chinese).
func Example(in Input) (input, answer string, ok bool) {
	if !in.Cleanup || in.Task != Transcribe && (in.Task != Translate || in.Selected) {
		return "", "", false
	}
	set := TemplateSet(in.MainLanguage)
	name := "example-clean"
	if in.Task == Translate {
		target := ""
		switch strings.ToLower(strings.TrimSpace(in.TargetLanguage)) {
		case "zh-hans", "zh-cn", "zh-sg", "zh":
			target = Chinese
		default:
			if lang.Canonical(in.TargetLanguage) == "en" {
				target = English
			}
		}
		switch target {
		case "":
			return "", "", false
		case set:
		default:
			name = "example-translated"
		}
	}
	return template(set, "example-input"), template(set, name), true
}
