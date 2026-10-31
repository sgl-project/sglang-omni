package llmrt

import (
	"strings"
	"unicode"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/lang"
)

// Hy-MT2's default translation prompts (its model card, "Translation
// Task Instruction Examples"): one user message and no system prompt; the
// target language by its full name, in Chinese in the Chinese prompt and
// in English in the English one. Voxt sent the model its general
// translation prompt through the chat template instead; the card's own
// prompt keeps the 1.8B model from answering the instructions.
const (
	hyMTChinese = "将以下文本翻译为%s，注意只需要输出翻译后的结果，不要额外解释：\n\n"
	hyMTEnglish = "Translate the following text into %s. Note that you should only output the translated result without any additional explanation:\n\n"
)

// HyMTPrompt is Hy-MT2's prompt to translate text into target (a
// language code). The Chinese prompt serves translations into or out of
// Chinese, as the card's examples do.
func HyMTPrompt(text, target string) string {
	text = strings.TrimSpace(text)
	l, ok := lang.Lookup(target)
	name := target
	if ok {
		name = l.Name
	}
	if chineseCode(target) || mostlyHan(text) {
		if ok {
			name = l.NameIn("zh-Hans")
		}
		return strings.Replace(hyMTChinese, "%s", name, 1) + text
	}
	return strings.Replace(hyMTEnglish, "%s", name, 1) + text
}

// chineseCode reports whether code is a Chinese language.
func chineseCode(code string) bool {
	switch lang.Canonical(code) {
	case "zh", "yue", "lzh", "wuu", "hak", "nan":
		return true
	}
	return false
}

// mostlyHan reports whether at least half of s's letters are Han.
func mostlyHan(s string) bool {
	han, letters := 0, 0
	for _, r := range s {
		switch {
		case unicode.Is(unicode.Han, r):
			han++
			letters++
		case unicode.IsLetter(r):
			letters++
		}
	}
	return letters > 0 && han*2 >= letters
}
