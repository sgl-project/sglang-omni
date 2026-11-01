package lang

import (
	"slices"
	"testing"
)

func TestTable(t *testing.T) {
	if n := Len(); n != 652 {
		t.Errorf("Len = %d, want 652 (646 OmniVoice + 6 Whisper-only)", n)
	}
	if n := len(OmniVoiceCodes()); n != 646 {
		t.Errorf("OmniVoice codes: %d, want 646", n)
	}
	w := WhisperCodes()
	if len(w) != 100 || !slices.Contains(w, "jw") || !slices.Contains(w, "tl") || !slices.Contains(w, "yue") || slices.Contains(w, "jv") {
		t.Errorf("Whisper codes: %d %v", len(w), w)
	}
	seen := map[string]bool{}
	for _, l := range All() {
		if seen[l.Code] || l.Code == "" || l.Name == "" || l.Script == "" || l.ISO639_3 == "" {
			t.Errorf("bad row %+v", l)
		}
		seen[l.Code] = true
		if !l.OmniVoice && !l.Whisper {
			t.Errorf("%s belongs to no model", l.Code)
		}
	}
	all := All()
	for i := 1; i < len(all); i++ {
		if Fold(all[i-1].Name) > Fold(all[i].Name) {
			t.Errorf("All not sorted: %q before %q", all[i-1].Name, all[i].Name)
		}
	}
}

func TestLookup(t *testing.T) {
	cases := []struct {
		tag, code, name, native, script, flag string
		rtl                                   bool
	}{
		{"zh", "zh", "Chinese", "中文", "Hans", "🇨🇳", false},
		{"zh-Hans-CN", "zh", "Chinese", "中文", "Hans", "🇨🇳", false},
		{"cmn", "zh", "Chinese", "中文", "Hans", "🇨🇳", false},
		{"en_US", "en", "English", "English", "Latn", "🇺🇸", false},
		{"ENG", "en", "English", "English", "Latn", "🇺🇸", false},
		{"fre", "fr", "French", "Français", "Latn", "🇫🇷", false},
		{"ja", "ja", "Japanese", "日本語", "Jpan", "🇯🇵", false},
		{"ko-KR", "ko", "Korean", "한국어", "Kore", "🇰🇷", false},
		{"iw", "he", "Hebrew", "עברית", "Hebr", "🇮🇱", true},
		{"ar", "ar", "Arabic", "العربية", "Arab", "🇪🇬", true},
		{"arb", "arb", "Standard Arabic", "العربية", "Arab", "🇪🇬", true},
		{"jw", "jv", "Javanese", "Jawa", "Latn", "🇮🇩", false},
		{"tl", "fil", "Filipino", "Filipino", "Latn", "🇵🇭", false},
		{"yue", "yue", "Cantonese", "粵語", "Hant", "🇭🇰", false},
		{"eo", "eo", "Esperanto", "Esperanto", "Latn", "", false},
		{"aae", "aae", "Arbëreshë Albanian", "", "Latn", "🇮🇹", false},
	}
	for _, c := range cases {
		l, ok := Lookup(c.tag)
		if !ok || l.Code != c.code || l.Name != c.name || l.Native != c.native || l.Script != c.script || l.Flag() != c.flag || l.RTL != c.rtl {
			t.Errorf("Lookup(%q) = %+v flag %q ok %v", c.tag, l, l.Flag(), ok)
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

func codes(ls []Language) []string {
	var out []string
	for _, l := range ls {
		out = append(out, l.Code)
	}
	return out
}

func TestSearch(t *testing.T) {
	cases := []struct {
		q     string
		first []string // the first results, in order
		has   string   // and somewhere in the results
	}{
		{"fr", []string{"fr"}, ""},
		{"francais", []string{"fr"}, ""},
		{"FRANÇAIS", []string{"fr"}, ""},
		{"french", []string{"fr"}, ""},
		{"法语", []string{"fr"}, ""},
		{"中文", []string{"zh"}, ""},
		{"日本", []string{"ja"}, ""},
		{"cmn", []string{"zh"}, ""},
		{"arabic", []string{"ar"}, "arb"},
		{"arbereshe", []string{"aae"}, ""},
		{"  Nepali ", []string{"ne", "npi"}, ""},
		{"jw", []string{"jv"}, ""},
		{"عربية", nil, "ar"},
	}
	for _, c := range cases {
		got := codes(Search(c.q))
		if len(got) < len(c.first) || !slices.Equal(got[:len(c.first)], c.first) {
			t.Errorf("Search(%q) = %v…, want first %v", c.q, got[:min(5, len(got))], c.first)
		}
		if c.has != "" && !slices.Contains(got, c.has) {
			t.Errorf("Search(%q) lacks %s", c.q, c.has)
		}
	}
	if len(Search("")) != Len() {
		t.Error("empty query should return all")
	}
	if len(Search("zzzzqqq")) != 0 {
		t.Error("nonsense query matched")
	}
}

func TestModels(t *testing.T) {
	kokoro := []string{"en-US", "en-GB", "zh", "ja", "es", "fr", "hi", "it", "pt-BR"}
	opts := ForModel(kokoro)
	if len(opts) != 8 { // en-US and en-GB are both English
		t.Errorf("ForModel: %d options", len(opts))
	}
	for _, o := range opts {
		if o.Code == "en" && o.ModelCode != "en-US" {
			t.Errorf("first spelling should win: %+v", o)
		}
	}
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
		{OmniVoiceCodes(), "ar", "arb", true}, // and back
		{OmniVoiceCodes(), "ne", "npi", true},
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
	ann := Annotate(Search("arabic"), []string{"ar"})
	for _, o := range ann {
		if o.Code == "ar" && (!o.Supported || o.ModelCode != "ar") {
			t.Errorf("ar: %+v", o)
		}
		if o.Code == "arb" && (!o.Supported || o.ModelCode != "ar") {
			t.Errorf("arb: %+v", o)
		}
		if o.Code == "aao" && o.Supported {
			t.Errorf("aao should be unsupported: %+v", o)
		}
	}
}
