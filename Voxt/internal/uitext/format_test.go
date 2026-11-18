package uitext

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestMemoryBytes(t *testing.T) {
	for n, want := range map[int64]string{3_650_722_201: "3.4 GB", 1 << 30: "1.0 GB", 1_150 << 20: "1.1 GB", 512 << 20: "512 MB", 0: "0 MB"} {
		if got := MemoryBytes(n); got != want {
			t.Errorf("MemoryBytes(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestLangLabelExplicitLocale(t *testing.T) {
	previousLocale := i18n.Locale()
	i18n.SetLocale("en")
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	translator := i18n.NewTranslator("zh-Hans")
	if got := LangLabelFor(translator, "ja"); got != "日语 · 日本語" {
		t.Errorf("LangLabelFor = %q, want the explicit zh-Hans label", got)
	}
	if got := LangLabel("ja"); got != "Japanese · 日本語" {
		t.Errorf("LangLabel = %q, want the current-locale wrapper", got)
	}
	if got := LangLabelFor(translator, "unknown-language"); got != "unknown-language" {
		t.Errorf("LangLabelFor unknown language = %q", got)
	}
}
