package appicons

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

func TestBook(t *testing.T) {
	var b Book
	b.Want(proto.AppRef{BundleID: "com.apple.Notes"})
	b.Want(proto.AppRef{BundleID: "com.apple.notes"}) // the same app
	b.Want(proto.AppRef{BundleID: ""})
	b.Want(proto.AppRef{BundleID: "com.apple.Safari", Path: "/Applications/Safari.app"})
	b.Want(proto.AppRef{BundleID: "com.tinyspeck.slackmacgap"})
	if b.Pending() != 3 || b.State("COM.APPLE.NOTES") != Wanted {
		t.Fatalf("pending %d %v", b.Pending(), b.State("com.apple.Notes"))
	}
	got := b.Take(2)
	if len(got) != 2 || got[0].BundleID != "com.apple.Notes" || got[1].Path != "/Applications/Safari.app" || b.State("com.apple.Safari") != Loading {
		t.Fatalf("take %+v", got)
	}
	b.Want(proto.AppRef{BundleID: "com.apple.Safari"}) // loading: not again
	if b.Pending() != 1 {
		t.Fatalf("pending %d", b.Pending())
	}
	b.Set("com.apple.Notes", Ready)
	b.Set("com.apple.Safari", Failed)
	b.Retry()
	if b.State("com.apple.Safari") != Unknown || b.State("com.apple.Notes") != Ready {
		t.Errorf("retry: %v %v", b.State("com.apple.Safari"), b.State("com.apple.Notes"))
	}
	if rest := b.Take(10); len(rest) != 1 || b.Pending() != 0 {
		t.Errorf("rest %+v", rest)
	}
}

func TestCache(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	c := Cache{Dir: filepath.Join(dir, "icons"), Now: func() time.Time { return now }}
	if p := c.Path("Com.Example/App 1", 64); filepath.Base(p) != "com.example_app_1-64.png" {
		t.Errorf("path %s", p)
	}
	if _, _, ok := c.Read("com.apple.Notes", 64); ok {
		t.Fatal("read before write")
	}
	if err := c.Write("com.apple.Notes", 64, []byte("png")); err != nil {
		t.Fatal(err)
	}
	if data, fresh, ok := c.Read("COM.APPLE.NOTES", 64); !ok || !fresh || string(data) != "png" {
		t.Errorf("read %q %v %v", data, fresh, ok)
	}
	if _, _, ok := c.Read("com.apple.Notes", 32); ok {
		t.Error("another size")
	}
	now = now.Add(TTL + time.Hour)
	if _, fresh, ok := c.Read("com.apple.Notes", 64); !ok || fresh {
		t.Errorf("stale: fresh %v ok %v", fresh, ok)
	}
	// No temporary files left behind.
	entries, _ := os.ReadDir(c.Dir)
	if len(entries) != 1 {
		t.Errorf("files %v", entries)
	}
	// A zero cache keeps nothing.
	if err := (Cache{}).Write("com.apple.Notes", 64, []byte("png")); err != nil {
		t.Error(err)
	}
	if _, _, ok := (Cache{}).Read("com.apple.Notes", 64); ok {
		t.Error("zero cache read")
	}
}

func TestFetch(t *testing.T) {
	now := time.Now()
	c := Cache{Dir: t.TempDir(), Now: func() time.Time { return now }}
	c.Write("com.apple.Notes", 64, []byte("notes-disk")) // fresh
	c.Write("com.old.App", 64, []byte("old-disk"))
	old := now.Add(-TTL - time.Hour)
	os.Chtimes(c.Path("com.old.App", 64), old, old) // stale
	refs := []proto.AppRef{{BundleID: "com.apple.Notes"}, {BundleID: "com.old.App"}, {BundleID: "com.tinyspeck.slackmacgap", Path: "/Applications/Slack.app"}, {BundleID: "com.example.gone"}}
	var asked []proto.AppRef
	ask := func(r []proto.AppRef) ([]proto.AppIcon, error) {
		asked = r
		return []proto.AppIcon{{BundleID: "com.old.app", PNG: []byte("old-new")}, {BundleID: "com.tinyspeck.slackmacgap", PNG: []byte("slack")}, {BundleID: "com.example.gone"}}, nil
	}
	got := Fetch(c, refs, 64, ask)
	if ids := func() (s []string) {
		for _, r := range asked {
			s = append(s, r.BundleID)
		}
		return
	}(); !slices.Equal(ids, []string{"com.old.App", "com.tinyspeck.slackmacgap", "com.example.gone"}) || asked[1].Path == "" {
		t.Errorf("asked %+v", asked)
	}
	want := []Result{
		{"com.apple.Notes", []byte("notes-disk"), Ready},
		{"com.old.App", []byte("old-new"), Ready},
		{"com.tinyspeck.slackmacgap", []byte("slack"), Ready},
		{"com.example.gone", nil, Missing},
	}
	for i, w := range want {
		if got[i].BundleID != w.BundleID || string(got[i].PNG) != string(w.PNG) || got[i].State != w.State {
			t.Errorf("%d: %+v, want %+v", i, got[i], w)
		}
	}
	if data, fresh, _ := c.Read("com.tinyspeck.slackmacgap", 64); string(data) != "slack" || !fresh {
		t.Error("not kept on disk")
	}
	// No helper: the stale icon, the others Failed.
	now = now.Add(2 * TTL)
	got = Fetch(c, refs[:2], 64, nil)
	if got[0].State != Ready || string(got[0].PNG) != "notes-disk" || got[1].State != Ready {
		t.Errorf("stale %+v", got)
	}
	got = Fetch(c, []proto.AppRef{{BundleID: "com.unknown"}}, 64, func([]proto.AppRef) ([]proto.AppIcon, error) { return nil, errors.New("helper gone") })
	if got[0].State != Failed {
		t.Errorf("failed %+v", got)
	}
}
