package lang

import "testing"

// TestChineseScript: the script dictation is written in, by the main
// language and the languages the user reads.
func TestChineseScript(t *testing.T) {
	for _, c := range []struct {
		main  string
		reads []string
		want  string
	}{
		{"zh", []string{"zh-Hans-US", "en-US"}, "zh-Hans"},
		{"zh", []string{"en-US", "zh-Hant-TW"}, "zh-Hant"},
		{"zh", []string{"zh-TW"}, "zh-Hant"},
		{"zh", []string{"zh-HK"}, "zh-Hant"},
		{"zh", []string{"zh-CN"}, "zh-Hans"},
		{"zh", []string{"en-US", "zh-Hans"}, "zh-Hans"}, // the app's language last
		{"zh", []string{"en-US", "en"}, ""},             // nothing says: the model's
		{"zh", nil, ""},
		{"zh-Hant", []string{"zh-Hans-CN"}, "zh-Hant"}, // the main language's own
		{"zh-Hans", []string{"zh-Hant-TW"}, "zh-Hans"},
		{"yue", []string{"zh-Hant-HK"}, "zh-Hant"}, // Hong Kong Cantonese
		{"yue", []string{"zh-Hans-CN"}, "zh-Hans"}, // Guangzhou Cantonese
		{"yue", []string{"yue"}, "zh-Hant"},
		{"yue", []string{"en-US"}, ""},
		{"en", []string{"zh-Hans-CN"}, ""},
		{"ja", []string{"zh-Hant-TW"}, ""},
		{"", []string{"zh-Hans-CN"}, ""},
	} {
		if got := ChineseScript(c.main, c.reads); got != c.want {
			t.Errorf("ChineseScript(%q, %q) = %q, want %q", c.main, c.reads, got, c.want)
		}
	}
}

// TestChineseScriptMixed: Whisper's mixed sentence comes out in one
// script, English words untouched.
func TestChineseScriptMixed(t *testing.T) {
	const mixed = "這個Pool Request我已经Rebase到Main了，測過了就可以Merge。"
	if got, want := RenderChineseScript(mixed, ChineseScript("zh", []string{"zh-Hans-US"})), "这个Pool Request我已经Rebase到Main了，测过了就可以Merge。"; got != want {
		t.Errorf("Simplified: %q, want %q", got, want)
	}
	if got, want := RenderChineseScript(mixed, ChineseScript("zh", []string{"zh-Hant-TW"})), "這個Pool Request我已經Rebase到Main了，測過了就可以Merge。"; got != want {
		t.Errorf("Traditional: %q, want %q", got, want)
	}
	if got := RenderChineseScript(mixed, ChineseScript("zh", []string{"en-US"})); got != mixed {
		t.Errorf("no Chinese read: %q, want it unchanged", got)
	}
}
