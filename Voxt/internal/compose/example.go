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
// Translate, when a template set translates into the target language
// (English and Simplified Chinese: each set's example-translated is in the
// other set's language).
func Example(in Input) (input, answer string, ok bool) {
	if !in.Cleanup || in.Task != Transcribe && (in.Task != Translate || in.Selected) {
		return "", "", false
	}
	set := TemplateSet(in.MainLanguage)
	if in.Task == Transcribe {
		return template(set, "example-input"), template(set, "example-clean"), true
	}
	target := ""
	switch strings.ToLower(strings.TrimSpace(in.TargetLanguage)) {
	case "zh-hans", "zh-cn", "zh-sg", "zh":
		target = Chinese
	default:
		if lang.Canonical(in.TargetLanguage) == "en" {
			target = English
		}
	}
	// The set whose example translates into the target: never a
	// same-language clean-up, which would teach the model to echo.
	switch target {
	case English:
		set = Chinese
	case Chinese:
		set = English
	default:
		return "", "", false
	}
	return template(set, "example-input"), template(set, "example-translated"), true
}
