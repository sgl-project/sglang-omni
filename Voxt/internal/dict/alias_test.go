package dict

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestReplace: aliases become their term: Latin ones as whole words
// without case, Chinese ones anywhere, the longest first; disabled
// terms, words that only contain an alias and the terms themselves stay.
func TestReplace(t *testing.T) {
	terms := []store.Term{
		{Term: "Auralink", Aliases: []string{"Or Link", "aura link", "奥拉林克", "奥拉"}, Enabled: true},
		{Term: "VoxtNew", Aliases: []string{"Vox Tunew", "Vox New", "Vox"}, Enabled: true},
		{Term: "SGLang", Aliases: []string{"S G Long"}, Enabled: true},
		{Term: "Kubernetes", Aliases: []string{"库伯"}, Enabled: false},
		{Term: "Pixcall", Aliases: []string{"pixcall", ""}, Enabled: true},
		{Term: "ActNow", Aliases: []string{"Or Link"}, Enabled: true},
	}
	for _, c := range []struct{ in, want string }{
		{"我们把 Or Link 的发布计划发给 team。", "我们把 Auralink 的发布计划发给 team。"},
		{"discuss the or  link launch", "discuss the Auralink launch"},
		{"OR LINK, then Vox Tunew.", "Auralink, then VoxtNew."},
		{"Vox New和Vox都是", "VoxtNew和VoxtNew都是"},
		{"Voxel and Voxt stay, Vox goes", "Voxel and Voxt stay, VoxtNew goes"},
		{"Or Linked and Door Link stay", "Or Linked and Door Link stay"},
		{"奥拉林克和奥拉的版本", "Auralink和Auralink的版本"},
		{"用 S G Long 部署，库伯也是", "用 SGLang 部署，库伯也是"},
		{"Auralink and VoxtNew and voxtnew", "Auralink and VoxtNew and voxtnew"},
		{"pixcall", "pixcall"},
		{"", ""},
	} {
		if got := Replace(c.in, terms); got != c.want {
			t.Errorf("%q: %q, want %q", c.in, got, c.want)
		}
		if again := Replace(Replace(c.in, terms), terms); again != c.want {
			t.Errorf("%q twice: %q", c.in, again)
		}
	}
	if got := Replace("Or Link", nil); got != "Or Link" {
		t.Errorf("no terms: %q", got)
	}
}

// TestHighlightSpans: the highlighter uses Replace's case, alias,
// whole-word, Chinese and longest-match rules, and includes terms without
// aliases. Byte ranges keep multibyte text intact for the UI.
func TestHighlightSpans(t *testing.T) {
	terms := []store.Term{
		{Term: "SGLang", Aliases: []string{"S G Long"}, Enabled: true},
		{Term: "Voxt", Enabled: true},
		{Term: "火山引擎", Aliases: []string{"火山 引擎"}, Enabled: true},
		{Term: "Go", Enabled: true},
		{Term: "Go Lang", Enabled: true},
		{Term: "Off", Enabled: false},
	}
	for _, tc := range []struct {
		name, text string
		want       []Span
	}{
		{"English terms and aliases ignore case", "sglang, S G LONG and VOXT", []Span{{0, 6}, {8, 16}, {21, 25}}},
		{"Chinese term and alias", "火山 引擎和火山引擎", []Span{{0, 13}, {16, 28}}},
		{"longest overlapping term", "go lang go", []Span{{0, 7}, {8, 10}}},
		{"word boundaries and disabled terms", "VoxtNew and Off stay", nil},
		{"empty", "", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := HighlightSpans(tc.text, terms); !slices.Equal(got, tc.want) {
				t.Errorf("HighlightSpans(%q) = %+v, want %+v", tc.text, got, tc.want)
			}
		})
	}
}
