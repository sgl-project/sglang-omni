package lang

import (
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestTable(t *testing.T) {
	if n := len(load().all); n != 652 {
		t.Errorf("%d languages, want 652 (646 OmniVoice + 6 Whisper-only)", n)
	}
	if n := len(omniVoiceCodes()); n != 646 {
		t.Errorf("OmniVoice codes: %d, want 646", n)
	}
	w := WhisperCodes()
	if len(w) != 100 || !slices.Contains(w, "jw") || !slices.Contains(w, "tl") || !slices.Contains(w, "yue") || slices.Contains(w, "jv") {
		t.Errorf("Whisper codes: %d %v", len(w), w)
	}
	seen := map[string]bool{}
	for _, l := range load().all {
		if seen[l.Code] || l.Code == "" || l.Name == "" || l.Script == "" || l.ISO639_3 == "" {
			t.Errorf("bad row %+v", l)
		}
		seen[l.Code] = true
		if !l.OmniVoice && !l.Whisper {
			t.Errorf("%s belongs to no model", l.Code)
		}
	}
	all := load().all
	for i := 1; i < len(all); i++ {
		if Fold(all[i-1].Name) > Fold(all[i].Name) {
			t.Errorf("All not sorted: %q before %q", all[i-1].Name, all[i].Name)
		}
	}
}

func TestLookup(t *testing.T) {
	cases := []struct {
		tag, code, name, native, script string
		rtl                             bool
	}{
		{"zh", "zh", "Chinese", "中文", "Hans", false},
		{"zh-Hans-CN", "zh", "Chinese", "中文", "Hans", false},
		{"cmn", "zh", "Chinese", "中文", "Hans", false},
		{"en_US", "en", "English", "English", "Latn", false},
		{"ENG", "en", "English", "English", "Latn", false},
		{"fre", "fr", "French", "Français", "Latn", false},
		{"ja", "ja", "Japanese", "日本語", "Jpan", false},
		{"ko-KR", "ko", "Korean", "한국어", "Kore", false},
		{"iw", "he", "Hebrew", "עברית", "Hebr", true},
		{"ar", "ar", "Arabic", "العربية", "Arab", true},
		{"arb", "arb", "Standard Arabic", "العربية", "Arab", true},
		{"jw", "jv", "Javanese", "Jawa", "Latn", false},
		{"tl", "fil", "Filipino", "Filipino", "Latn", false},
		{"yue", "yue", "Cantonese", "粵語", "Hant", false},
		{"eo", "eo", "Esperanto", "Esperanto", "Latn", false},
		{"aae", "aae", "Arbëreshë Albanian", "", "Latn", false},
	}
	for _, c := range cases {
		l, ok := Lookup(c.tag)
		if !ok || l.Code != c.code || l.Name != c.name || l.Native != c.native || l.Script != c.script || l.RTL != c.rtl {
			t.Errorf("Lookup(%q) = %+v ok %v", c.tag, l, ok)
		}
	}
	for _, bad := range []string{"", "xx", "auto", "-"} {
		if _, ok := Lookup(bad); ok {
			t.Errorf("Lookup(%q) succeeded", bad)
		}
	}
	zh, _ := Lookup("fr")
	if zh.NameIn("zh-Hans") != "法语" || zh.NameIn("en") != "French" {
		t.Errorf("NameIn: %q %q", zh.NameIn("zh-Hans"), zh.NameIn("en"))
	}
	if l, _ := Lookup("aae"); l.NameIn("zh-Hans") != "Arbëreshë Albanian" {
		t.Error("NameIn should fall back to English")
	}
	if l, _ := Lookup("arb"); l.Macro != "ar" || !l.OmniVoice || l.Whisper {
		t.Errorf("arb %+v", l)
	}
}

func TestModels(t *testing.T) {
	kokoro := []string{"en-US", "en-GB", "zh", "ja", "es", "fr", "hi", "it", "pt-BR"}
	cases := []struct {
		supported []string
		code      string
		want      string
		ok        bool
	}{
		{kokoro, "en", "en-US", true},
		{kokoro, "en-GB", "en-GB", true}, // the exact spelling first
		{kokoro, "en-AU", "en-US", true}, // then the base language
		{kokoro, "de", "", false},
		{WhisperCodes(), "jv", "jw", true},
		{WhisperCodes(), "arb", "ar", true},   // macrolanguage relation
		{omniVoiceCodes(), "ar", "arb", true}, // and back
		{omniVoiceCodes(), "ne", "npi", true},
		{[]string{"zh", "cmn"}, "cmn", "cmn", true},
		{[]string{"zh"}, "cmn", "zh", true},
		{nil, "en", "", false},
		{kokoro, "nonsense", "", false},
	}
	for _, c := range cases {
		got, ok := Match(c.supported, c.code)
		if got != c.want || ok != c.ok {
			t.Errorf("Match(%v…, %q) = %q,%v want %q,%v", c.supported[:min(3, len(c.supported))], c.code, got, ok, c.want, c.ok)
		}
	}
}

// The trailing-whitespace hook once stripped the trailing tab of every
// languages.tsv row (the whisper column is empty for most languages),
// and the table then panicked on the 10-column lines. The hook's exclude
// must keep covering the file. A diff that drops the exclusion turns
// this red.
func TestLanguagesTSVExcludedFromTrailingWhitespace(t *testing.T) {
	config, err := os.ReadFile("../../../.pre-commit-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	exclude := ""
	inHook := false
	for _, line := range strings.Split(string(config), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.Contains(trimmed, "- id: trailing-whitespace") {
			inHook = true
			continue
		}
		if inHook && strings.HasPrefix(trimmed, "- id:") {
			break
		}
		if inHook && strings.HasPrefix(trimmed, "exclude:") {
			exclude = strings.TrimSpace(strings.TrimPrefix(trimmed, "exclude:"))
			break
		}
	}
	if exclude == "" {
		t.Fatal("the trailing-whitespace hook has no exclude: it would strip the trailing tabs of languages.tsv")
	}
	// The full pattern uses Python lookaheads that Go's regexp cannot
	// compile; an alternative that compiles and matches is enough.
	for _, alternative := range strings.Split(strings.Trim(exclude, "()"), "|") {
		pattern, err := regexp.Compile(alternative)
		if err != nil {
			continue
		}
		if pattern.MatchString("Voxt/internal/lang/languages.tsv") {
			return
		}
	}
	t.Fatalf("trailing-whitespace exclude does not match Voxt/internal/lang/languages.tsv: %s", exclude)
}

// omniVoiceCodes is OmniVoice's codes, for the tests.
func omniVoiceCodes() []string {
	var out []string
	for _, l := range load().all {
		if l.OmniVoice {
			out = append(out, l.Code)
		}
	}
	return out
}
