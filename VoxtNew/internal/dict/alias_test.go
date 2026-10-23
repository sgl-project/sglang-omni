package dict

import (
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
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
