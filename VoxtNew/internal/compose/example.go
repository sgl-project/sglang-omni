package compose

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
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

var (
	// Hesitations anywhere; "啊", "哦" (also sentence particles) and
	// "那个", "就是说" (also words) only at the start of a clause and
	// followed by a pause or a correction.
	cjkFiller   = regexp.MustCompile(`[嗯呃额唔][，,、。.…\s]*`)
	cjkOpener   = regexp.MustCompile(`(^|[，,。.！!？?；;、\s])(?:啊|哦|那个|就是说)(?:[，,、…\s]+|(不对|不是))`)
	latinFiller = regexp.MustCompile(`(?i)(^|[\s,.;!?，。])(?:um+|uh+|erm|er|hmm+)\b[,.…]*\s*`)
)

// StripFillers drops spoken fillers from a transcript ("嗯", "呃",
// "um", "uh", with the comma after them), for a model that translates the
// text as it is (Hy-MT2 takes no clean-up rules). Nothing else changes:
// self-corrections stay.
func StripFillers(text string) string {
	s := cjkFiller.ReplaceAllString(text, "")
	for prev := ""; prev != s; {
		prev, s = s, cjkOpener.ReplaceAllString(s, "$1$2")
	}
	s = latinFiller.ReplaceAllString(s, "$1")
	s = strings.TrimLeft(strings.Join(strings.Fields(s), " "), "，,、 ")
	// A text that began with "Um, so": "So".
	if r, n := utf8.DecodeRuneInString(s); unicode.IsLower(r) {
		if first, _ := utf8.DecodeRuneInString(strings.TrimSpace(text)); unicode.IsUpper(first) {
			s = string(unicode.ToUpper(r)) + s[n:]
		}
	}
	return s
}
