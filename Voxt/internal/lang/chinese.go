package lang

import (
	_ "embed"
	"strings"
	"sync"
	"unicode/utf8"
)

//go:embed STCharacters.txt
var simplifiedTraditionalCharacters string

//go:embed STPhrases.txt
var simplifiedTraditionalPhrases string

//go:embed TSCharacters.txt
var traditionalSimplifiedCharacters string

//go:embed TSPhrases.txt
var traditionalSimplifiedPhrases string

type chineseConversion struct {
	characters     map[rune]rune
	phrases        map[string]string
	maxPhraseRunes int
}

var simplifiedToTraditional = sync.OnceValue(func() *chineseConversion {
	return newChineseConversion(simplifiedTraditionalCharacters, simplifiedTraditionalPhrases)
})

var traditionalToSimplified = sync.OnceValue(func() *chineseConversion {
	return newChineseConversion(traditionalSimplifiedCharacters, traditionalSimplifiedPhrases)
})

// RenderChineseScript renders Chinese text in the script the language tag
// names. A tag without a Hans or Hant script leaves the text unchanged.
func RenderChineseScript(text, languageTag string) string {
	languageTag = strings.ToLower(strings.ReplaceAll(languageTag, "_", "-"))
	switch {
	case strings.Contains(languageTag, "hans"):
		return renderChineseConversion(text, "Traditional-Simplified", traditionalToSimplified())
	case strings.Contains(languageTag, "hant"):
		return renderChineseConversion(text, "Simplified-Traditional", simplifiedToTraditional())
	default:
		return text
	}
}

// ChineseScript is the script Chinese speech in the language main is
// written in, "zh-Hans" or "zh-Hant", for a user who reads the languages
// reads, most preferred first; "" when main is not Chinese or nothing
// says. main's own script decides ("zh-Hant"), else the first Chinese
// language read: its script, else its region's (TW, HK and MO
// Traditional), else its usual one (Mandarin Simplified, Cantonese
// Traditional). A bare "zh" main language with no Chinese read leaves the
// script to the speech model.
func ChineseScript(main string, reads []string) string {
	if c := Canonical(main); c != "zh" && c != "yue" {
		return ""
	}
	if s := chineseTagScript(main, false); s != "" {
		return s
	}
	for _, tag := range reads {
		if c := Canonical(tag); c == "zh" || c == "yue" {
			return chineseTagScript(tag, true)
		}
	}
	return ""
}

func chineseTagScript(tag string, usual bool) string {
	subtags := strings.Split(strings.ToLower(strings.ReplaceAll(strings.TrimSpace(tag), "_", "-")), "-")
	region := ""
	for _, subtag := range subtags[1:] {
		switch subtag {
		case "hans":
			return "zh-Hans"
		case "hant":
			return "zh-Hant"
		}
		if len(subtag) == 2 && region == "" {
			region = subtag
		}
	}
	switch region {
	case "tw", "hk", "mo":
		return "zh-Hant"
	case "cn", "sg", "my":
		return "zh-Hans"
	}
	if !usual {
		return ""
	}
	if language, ok := Lookup(tag); ok && language.Script == "Hant" {
		return "zh-Hant"
	}
	return "zh-Hans"
}

// CanonicalChinese folds Traditional Chinese text to Simplified Chinese for
// comparisons. Other text stays unchanged.
func CanonicalChinese(text string) string {
	converted, _ := CanonicalChineseWithOffsets(text)
	return converted
}

// CanonicalChineseWithOffsets also returns, for each converted rune, the
// exclusive rune index of its source text in text.
func CanonicalChineseWithOffsets(text string) (string, []int) {
	if converted, ok := convertChineseScript(text, "Traditional-Simplified"); ok && utf8.RuneCountInString(converted) == utf8.RuneCountInString(text) {
		sourceEnds := make([]int, utf8.RuneCountInString(converted))
		for index := range sourceEnds {
			sourceEnds[index] = index + 1
		}
		return converted, sourceEnds
	}
	return traditionalToSimplified().convert(text)
}

func renderChineseConversion(text, transformID string, fallback *chineseConversion) string {
	if converted, ok := convertChineseScript(text, transformID); ok {
		return converted
	}
	converted, _ := fallback.convert(text)
	return converted
}

func newChineseConversion(characterPairs, phrasePairs string) *chineseConversion {
	conversion := &chineseConversion{
		characters: map[rune]rune{},
		phrases:    chineseDictionary(phrasePairs),
	}
	for source, target := range chineseDictionary(characterPairs) {
		sourceRunes, targetRunes := []rune(source), []rune(target)
		if len(sourceRunes) == 1 && len(targetRunes) == 1 {
			conversion.characters[sourceRunes[0]] = targetRunes[0]
		}
	}
	for phrase := range conversion.phrases {
		conversion.maxPhraseRunes = max(conversion.maxPhraseRunes, utf8.RuneCountInString(phrase))
	}
	return conversion
}

func (conversion *chineseConversion) convert(text string) (string, []int) {
	sourceRunes := []rune(text)
	var converted strings.Builder
	converted.Grow(len(text))
	sourceEnds := make([]int, 0, len(sourceRunes))
	for start := 0; start < len(sourceRunes); {
		replacement := ""
		sourceLength := 1
		for length := min(conversion.maxPhraseRunes, len(sourceRunes)-start); length > 1; length-- {
			if phrase, ok := conversion.phrases[string(sourceRunes[start:start+length])]; ok {
				replacement = phrase
				sourceLength = length
				break
			}
		}
		if replacement == "" {
			if character, ok := conversion.characters[sourceRunes[start]]; ok {
				replacement = string(character)
			} else {
				replacement = string(sourceRunes[start])
			}
		}
		converted.WriteString(replacement)
		for range utf8.RuneCountInString(replacement) {
			sourceEnds = append(sourceEnds, start+sourceLength)
		}
		start += sourceLength
	}
	return converted.String(), sourceEnds
}

func chineseDictionary(pairs string) map[string]string {
	dictionary := map[string]string{}
	for line := range strings.Lines(pairs) {
		line = strings.TrimSpace(line)
		source, targets, ok := strings.Cut(line, "\t")
		if !ok {
			continue
		}
		target, _, _ := strings.Cut(targets, " ")
		if source != "" && target != "" {
			dictionary[source] = target
		}
	}
	return dictionary
}
