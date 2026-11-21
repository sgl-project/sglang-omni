package sounds

import (
	"os"
	"testing"
)

// TestRealCatalog checks a real catalogue when ACTNOW_SOUNDS_DIR names
// one (a checkout of the family's private sound library, or an app's
// copy): it loads, every file is there and matches its SHA-256, pairs and
// uses are known.
func TestRealCatalog(t *testing.T) {
	dir := os.Getenv("ACTNOW_SOUNDS_DIR")
	if dir == "" {
		t.Skip("ACTNOW_SOUNDS_DIR not set")
	}
	c, err := Load(os.DirFS(dir), ".")
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range c.Check() {
		t.Error(err)
	}
	uses := map[string]bool{}
	for _, u := range c.Uses {
		uses[u.ID] = true
	}
	for _, s := range c.Sounds {
		if s.Pair != "" {
			if _, ok := c.Sound(s.Pair); !ok {
				t.Errorf("%s: pair %s missing", s.ID, s.Pair)
			}
		}
		for _, u := range s.Uses {
			if !uses[u] {
				t.Errorf("%s: unknown use %s", s.ID, u)
			}
		}
		if s.Duration <= 0 || s.Name.Zh == "" {
			t.Errorf("%s: duration %v, zh %q", s.ID, s.Duration, s.Name.Zh)
		}
	}
	t.Logf("%d sounds in %d groups", len(c.Sounds), len(c.Groups))
}
