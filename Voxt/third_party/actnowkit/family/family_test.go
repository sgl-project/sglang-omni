package family

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

type sample struct {
	Schema string `json:"schema"`
	Rev    int    `json:"rev"`
	Name   string `json:"name,omitempty"`
	Inner  *inner `json:"inner,omitempty"`
	Extra  Extra  `json:"-"`
}

type inner struct {
	A     string `json:"a"`
	Extra Extra  `json:"-"`
}

func (s *sample) UnmarshalJSON(b []byte) error {
	type plain sample
	return UnmarshalKnown(b, (*plain)(s), &s.Extra)
}
func (s sample) MarshalJSON() ([]byte, error) {
	type plain sample
	return MarshalKnown(plain(s), s.Extra)
}
func (s *inner) UnmarshalJSON(b []byte) error {
	type plain inner
	return UnmarshalKnown(b, (*plain)(s), &s.Extra)
}
func (s inner) MarshalJSON() ([]byte, error) {
	type plain inner
	return MarshalKnown(plain(s), s.Extra)
}

func TestParseSchema(t *testing.T) {
	s, err := ParseSchema("actnow.models.claims/1")
	if err != nil || s.Kind != "models.claims" || s.Major != 1 {
		t.Fatalf("%+v %v", s, err)
	}
	if s.String() != "actnow.models.claims/1" {
		t.Error(s.String())
	}
	for _, bad := range []string{"", "actnow.x", "x.settings/1", "actnow./1", "actnow.x/0", "actnow.x/one"} {
		if _, err := ParseSchema(bad); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

func TestUnknownFieldsKept(t *testing.T) {
	in := `{"schema":"actnow.sample/1","rev":3,"future":{"x":[1,2]},"inner":{"a":"b","later":true},"zz":"last"}`
	var s sample
	if err := json.Unmarshal([]byte(in), &s); err != nil {
		t.Fatal(err)
	}
	if s.Rev != 3 || s.Inner.A != "b" || len(s.Extra) != 2 || len(s.Inner.Extra) != 1 {
		t.Fatalf("%+v %+v", s, s.Inner)
	}
	s.Rev++
	out, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["rev"].(float64) != 4 || m["zz"] != "last" || m["future"] == nil || m["inner"].(map[string]any)["later"] != true {
		t.Errorf("lost unknown members: %s", out)
	}
	// Known members are never duplicated from Extra.
	if strings.Count(string(out), `"rev"`) != 1 {
		t.Errorf("rev twice: %s", out)
	}
	// An empty object with extras.
	var e inner
	e.Extra = Extra{"k": json.RawMessage(`1`)}
	b, _ := MarshalKnown(struct{}{}, e.Extra)
	if string(b) != `{"k":1}` {
		t.Errorf("%s", b)
	}
}

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "x.json")
	var s sample
	if err := Load(p, "sample", 1, &s); !IsNotExist(err) {
		t.Errorf("missing: %v", err)
	}
	os.WriteFile(p, []byte("{nope"), 0o600)
	var ce *CorruptError
	if err := Load(p, "sample", 1, &s); !errors.As(err, &ce) {
		t.Errorf("corrupt: %v", err)
	}
	os.WriteFile(p, []byte(`{"rev":1}`), 0o600)
	if err := Load(p, "sample", 1, &s); !errors.As(err, &ce) {
		t.Errorf("no schema: %v", err)
	}
	os.WriteFile(p, []byte(`{"schema":"actnow.other/1"}`), 0o600)
	if err := Load(p, "sample", 1, &s); !errors.Is(err, ErrWrongKind) {
		t.Errorf("wrong kind: %v", err)
	}
	os.WriteFile(p, []byte(`{"schema":"actnow.sample/2","rev":9,"name":"n"}`), 0o600)
	s = sample{}
	if err := Load(p, "sample", 1, &s); !errors.Is(err, ErrNewerSchema) || s.Rev != 9 {
		t.Errorf("newer: %v %+v", err, s)
	}
	os.WriteFile(p, []byte(`{"schema":"actnow.sample/1","rev":"x"}`), 0o600)
	if err := Load(p, "sample", 1, &s); !errors.As(err, &ce) {
		t.Errorf("bad type: %v", err)
	}
	big := `{"schema":"actnow.sample/1","name":"` + strings.Repeat("a", MaxFileSize) + `"}`
	os.WriteFile(p, []byte(big), 0o600)
	if err := Load(p, "sample", 1, &s); !errors.As(err, &ce) {
		t.Errorf("too big: %v", err)
	}
	if err := Quarantine(p); err != nil {
		t.Fatal(err)
	}
	matches, _ := filepath.Glob(p + ".corrupt-*")
	if len(matches) != 1 {
		t.Errorf("quarantine: %v", matches)
	}
}

func TestWriteFileAtomicAndTemps(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "sub", "f.json")
	if err := WriteFileAtomic(p, []byte("one"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := WriteFileAtomic(p, []byte("two"), 0o600); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(p); string(b) != "two" {
		t.Errorf("%q", b)
	}
	st, _ := os.Stat(p)
	if runtime.GOOS == "windows" {
		t.Log("skipping the Unix permission-bit assertion on Windows")
	} else if st.Mode().Perm() != 0o600 {
		t.Errorf("perm %v", st.Mode())
	}
	stale := filepath.Join(dir, "sub", ".f.json.tmp-1-abc")
	fresh := filepath.Join(dir, "sub", ".f.json.tmp-2-def")
	os.WriteFile(stale, nil, 0o600)
	os.WriteFile(fresh, nil, 0o600)
	old := time.Now().Add(-2 * time.Hour)
	os.Chtimes(stale, old, old)
	if n := CleanTemps(filepath.Join(dir, "sub"), time.Hour); n != 1 {
		t.Errorf("cleaned %d", n)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Error("fresh temp removed")
	}
}

func TestEnsure(t *testing.T) {
	d := At(filepath.Join(t.TempDir(), "ActNow"))
	var wg sync.WaitGroup
	infos := make([]Info, 8)
	for i := range infos {
		wg.Go(func() {
			info, err := d.Ensure("dev.hehehai.app" + string(rune('a'+i)))
			if err != nil {
				t.Error(err)
			}
			infos[i] = info
		})
	}
	wg.Wait()
	for _, info := range infos {
		if info.CreatedBy != infos[0].CreatedBy || info.Schema != "actnow.family/1" || info.SpecVersion != 1 {
			t.Errorf("%+v vs %+v", info, infos[0])
		}
	}
	for _, sub := range []string{"apps", "models/claims", "runtime", "account", "locks"} {
		if st, err := os.Stat(filepath.Join(d.Root, sub)); err != nil || !st.IsDir() {
			t.Errorf("%s: %v", sub, err)
		}
	}
	// A newer spec version: read only, file untouched.
	os.WriteFile(d.FamilyFile(), []byte(`{"schema":"actnow.family/1","specVersion":2,"created":"2026-10-20T08:00:00Z","createdBy":"x","later":1}`), 0o600)
	info, err := d.Ensure("dev.hehehai.dicta")
	if err != nil || !info.ReadOnly() || info.Extra["later"] == nil {
		t.Errorf("%+v %v", info, err)
	}
	// Corrupt family.json is set aside.
	os.WriteFile(d.FamilyFile(), []byte(`garbage`), 0o600)
	info, err = d.Ensure("dev.hehehai.dicta")
	if err != nil || info.CreatedBy != "dev.hehehai.dicta" {
		t.Errorf("%+v %v", info, err)
	}
}

func TestLocks(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.lock")
	a, err := TryLock(p, Shared, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TryLock(p, Shared, 0o600)
	if err != nil {
		t.Fatal("two shared locks:", err)
	}
	if _, err := TryLock(p, Exclusive, 0o600); !errors.Is(err, ErrLocked) {
		t.Fatal("exclusive while shared:", err)
	}
	a.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := WaitLock(ctx, p, Exclusive, 0o600); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("wait should time out:", err)
	}
	go func() { time.Sleep(30 * time.Millisecond); b.Unlock() }()
	x, err := WaitLock(context.Background(), p, Exclusive, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	x.Unlock()
	x.Unlock() // twice is fine
	if _, err := os.Stat(p); err != nil {
		t.Error("lock file removed")
	}
}

func TestUpdateConcurrent(t *testing.T) {
	d := At(t.TempDir())
	p := filepath.Join(d.Root, "counter.json")
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			var s sample
			err := d.Update(context.Background(), "counter", p, "sample", 1, &s, func(bool) error {
				s.Schema = SchemaOf("sample", 1)
				s.Rev++
				return nil
			})
			if err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	var s sample
	if err := Load(p, "sample", 1, &s); err != nil || s.Rev != 20 {
		t.Fatalf("rev %d %v", s.Rev, err)
	}
	// Newer major: no write.
	os.WriteFile(p, []byte(`{"schema":"actnow.sample/2","rev":1}`), 0o600)
	s = sample{}
	err := d.Update(context.Background(), "counter", p, "sample", 1, &s, func(bool) error { s.Rev = 100; return nil })
	if !errors.Is(err, ErrNewerSchema) {
		t.Errorf("newer: %v", err)
	}
	if b, _ := os.ReadFile(p); !strings.Contains(string(b), `"rev":1`) {
		t.Errorf("overwritten: %s", b)
	}
	// Corrupt: quarantined, rewritten.
	os.WriteFile(p, []byte(`{`), 0o600)
	s = sample{}
	err = d.Update(context.Background(), "counter", p, "sample", 1, &s, func(exists bool) error {
		if exists {
			t.Error("corrupt file reported as existing")
		}
		s.Schema, s.Rev = SchemaOf("sample", 1), 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if m, _ := filepath.Glob(p + ".corrupt-*"); len(m) != 1 {
		t.Errorf("not quarantined: %v", m)
	}
}

func TestParseBundleID(t *testing.T) {
	cases := []struct {
		id      string
		app     string
		variant Variant
		err     error
	}{
		{"dev.hehehai.audiosmith", "audiosmith", Release, nil},
		{"dev.hehehai.dicta.dev", "dicta", Dev, nil},
		{"dev.hehehai.auralink.updtest", "auralink", Test, nil},
		{"dev.hehehai.auralink.betatest", "auralink", Test, nil},
		{"dev.hehehai.pik.m7", "pik", Test, nil},
		{"dev.hehehai.voxtnew.overlaycheck", "voxtnew", Test, nil},
		{"dev.hehehai.herenow", "herenow", Release, nil},
		{"dev.hehehai.herenow.dev", "herenow", Dev, nil},
		{"dev.hehehai.captioncraft", "captioncraft", Release, nil},
		{"dev.hehehai.captioncraft.dev", "captioncraft", Dev, nil},
		{"dev.hehehai.captioncraft.updtest", "captioncraft", Test, nil},
		{"dev.hehehai.captioncraft.engine", "captioncraft", "", ErrComponent},
		{"dev.hehehai.paralign", "paralign", Release, nil},
		{"dev.hehehai.paralign.dev", "paralign", Dev, nil},
		{"dev.hehehai.paralign.e2e", "paralign", Test, nil},
		{"dev.hehehai.paralign.engine", "paralign", "", ErrComponent},
		{"dev.hehehai.act", "act", Release, nil},
		{"dev.hehehai.act.dev", "act", Dev, nil},
		{"dev.hehehai.act.e2e", "act", Test, nil},
		{"dev.hehehai.act.engine", "act", "", ErrComponent},
		{"Dev.Hehehai.Dicta", "dicta", Release, nil},
		{"dev.hehehai.auralink.helper", "auralink", "", ErrComponent},
		{"dev.hehehai.dicta.engine", "dicta", "", ErrComponent},
		{"dev.hehehai.audiosmithx", "", "", ErrNotFamily},
		{"com.example.app", "", "", ErrNotFamily},
	}
	for _, c := range cases {
		a, v, err := ParseBundleID(c.id)
		if !errors.Is(err, c.err) || a.ID != c.app || v != c.variant {
			t.Errorf("%s: %s %s %v", c.id, a.ID, v, err)
		}
	}
	seen := map[string]bool{}
	for _, a := range Apps {
		if seen[a.BundleID] || !strings.HasPrefix(a.BundleID, BundlePrefix) || a.URLScheme == "" {
			t.Errorf("bad table row %+v", a)
		}
		seen[a.BundleID] = true
	}
	if a, _ := AppByID("voxtnew"); a.TeamID != TeamVoxtNew {
		t.Error("VoxtNew team")
	}
	if a, _ := AppByID("herenow"); a.TeamID != TeamActNow || a.URLScheme != "herenow" {
		t.Error("HereNow row")
	}
	if a, _ := AppByID("captioncraft"); a.TeamID != TeamActNow || a.URLScheme != "captioncraft" || a.ProductURL != "https://actnow.dev/captioncraft" {
		t.Error("CaptionCraft row")
	}
	if a, _ := AppByID("paralign"); a.Name != "Paralign" || a.TeamID != TeamActNow || a.URLScheme != "paralign" || a.ProductURL != "https://actnow.dev/paralign" {
		t.Error("Paralign row")
	}
	if a, _ := AppByID("act"); a.Name != "Act" || a.TeamID != TeamActNow || a.URLScheme != "act" || a.ProductURL != "https://actnow.dev/act" {
		t.Error("Act row")
	}
}

func TestNotify(t *testing.T) {
	d := At(t.TempDir())
	if name := d.NotifyName(TopicSettings); !strings.HasPrefix(name, NotifyPrefix) || name == NotifyPrefix+TopicSettings {
		t.Errorf("test dir uses the real name: %s", name)
	}
	if At(canonicalRoot()).NotifyName(TopicSettings) != "dev.actnow.family.settings" {
		t.Error("canonical name")
	}
	w, err := d.Watch(TopicSettings, TopicModels)
	if err != nil {
		t.Fatal(err)
	}
	other := At(t.TempDir())
	other.Post(TopicSettings) // another directory: not seen
	if err := d.Post(TopicModels); err != nil {
		t.Fatal(err)
	}
	select {
	case topic := <-w.C:
		if topic != TopicModels {
			t.Errorf("got %s", topic)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no notification")
	}
	d.Post(TopicApps) // not watched
	select {
	case topic := <-w.C:
		t.Errorf("unexpected %s", topic)
	case <-time.After(100 * time.Millisecond):
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, ok := <-w.C; ok {
		t.Error("C open after Close")
	}
	w.Close()
}

func TestTextFill(t *testing.T) {
	tx := Text{EN: "{app} uses {model}", ZH: "{app} 在用 {model}"}.Fill("app", "Dicta", "model", "Gemma")
	if tx.EN != "Dicta uses Gemma" || tx.ZH != "Dicta 在用 Gemma" {
		t.Error(tx)
	}
}

func TestConvert(t *testing.T) {
	p := filepath.Join(t.TempDir(), "x.lock")
	ctx := context.Background()
	l, err := TryLock(p, Exclusive, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Convert(ctx, Shared); err != nil || l.Mode != Shared {
		t.Fatal(err)
	}
	s, err := TryLock(p, Shared, 0o600)
	if err != nil {
		t.Fatal("shared beside a downgraded lock:", err)
	}
	// Back to exclusive waits for the other shared holder.
	c, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if err := l.Convert(c, Exclusive); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("converted past another holder:", err)
	}
	s.Unlock()
	if err := l.Convert(ctx, Exclusive); err != nil || l.Mode != Exclusive {
		t.Fatal(err)
	}
	if _, err := TryLock(p, Shared, 0o600); !errors.Is(err, ErrLocked) {
		t.Fatal("not exclusive")
	}
	l.Unlock()
	if err := l.Convert(ctx, Shared); err == nil {
		t.Error("convert after unlock")
	}
}
