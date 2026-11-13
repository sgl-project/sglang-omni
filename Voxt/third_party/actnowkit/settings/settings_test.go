package settings

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/hehehai/actnowkit/family"
)

func TestNormalize(t *testing.T) {
	ok := map[string][2]string{
		"lang":    {KeyLanguage, "zh-Hans"},
		"sys":     {KeyLanguage, ""},
		"auto":    {KeyProxy, ""},
		"direct":  {KeyProxy, "direct"},
		"http":    {KeyProxy, "http://127.0.0.1:7897"},
		"socks":   {KeyProxy, "socks5://127.0.0.1:7897"},
		"ep":      {KeyHFEndpoints, "mirror"},
		"urls":    {KeyHFEndpoints, "https://a.example/ https://b.example"},
		"root":    {KeyModelsRoot, "/Volumes/X/hub/"},
		"default": {KeyModelsRoot, ""},
	}
	for name, kv := range ok {
		if _, err := Normalize(kv[0], kv[1]); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	if v, _ := Normalize(KeyHFEndpoints, "https://a.example/ https://b.example"); v != "https://a.example https://b.example" {
		t.Error(v)
	}
	if v, _ := Normalize(KeyModelsRoot, "/Volumes/X/hub/"); v != "/Volumes/X/hub" {
		t.Error(v)
	}
	bad := [][2]string{{KeyLanguage, "zh_CN"}, {KeyProxy, "127.0.0.1:7897"}, {KeyProxy, "ftp://x"}, {KeyHFEndpoints, "x y"}, {KeyModelsRoot, "rel/path"}, {"other.key", "x"}}
	for _, kv := range bad {
		if _, err := Normalize(kv[0], kv[1]); err == nil {
			t.Errorf("%v accepted", kv)
		}
	}
	if e := Endpoints("auto"); len(e) != 2 || e[0] != OfficialEndpoint {
		t.Error(e)
	}
	if e := Endpoints("https://x/"); len(e) != 1 || e[0] != "https://x" {
		t.Error(e)
	}
}

func TestSetAndRead(t *testing.T) {
	d := family.At(t.TempDir())
	ctx := context.Background()
	s, ro, err := Read(d)
	if err != nil || ro || s.Rev != 0 {
		t.Fatal(s, ro, err)
	}
	if _, err := Set(ctx, d, "dev.hehehai.auralink", KeyProxy, "http://127.0.0.1:7897"); err != nil {
		t.Fatal(err)
	}
	s, _, _ = Read(d)
	if v, ok := s.Get(KeyProxy); !ok || v != "http://127.0.0.1:7897" || s.Rev != 1 || s.UpdatedBy != "dev.hehehai.auralink" || s.Schema != "actnow.settings/1" {
		t.Fatalf("%+v", s)
	}
	if _, ok := s.Get(KeyLanguage); ok {
		t.Error("language set")
	}
	// The same value again: no new rev.
	Set(ctx, d, "dev.hehehai.dicta", KeyProxy, "http://127.0.0.1:7897")
	if s, _, _ = Read(d); s.Rev != 1 {
		t.Errorf("rev %d", s.Rev)
	}
	// Unknown members (phase 2, newer apps) survive.
	raw, _ := os.ReadFile(d.SettingsFile())
	raw = []byte(strings.Replace(string(raw), `"network": {`, `"speech": {"mainLanguage": "zh"}, "network": {"future": true, `, 1))
	os.WriteFile(d.SettingsFile(), raw, 0o600)
	if _, err := Set(ctx, d, "dev.hehehai.dicta", KeyLanguage, "en"); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(d.SettingsFile())
	if !strings.Contains(string(raw), `"mainLanguage": "zh"`) || !strings.Contains(string(raw), `"future": true`) {
		t.Errorf("lost: %s", raw)
	}
	// Concurrent writers each bump rev once.
	var wg sync.WaitGroup
	for i := range 10 {
		wg.Go(func() {
			Update(ctx, d, "x", func(s *Settings) error { return s.SetValue(KeyHFEndpoints, []string{"auto", "mirror"}[i%2]) })
		})
	}
	wg.Wait()
	if s, _, _ = Read(d); s.Rev != 2+10 {
		t.Errorf("rev %d", s.Rev)
	}
	// A newer major: read only, never written.
	os.WriteFile(d.SettingsFile(), []byte(`{"schema":"actnow.settings/2","rev":50,"general":{"language":"fr"}}`), 0o600)
	s, ro, err = Read(d)
	if err != nil || !ro {
		t.Fatal(ro, err)
	}
	if v, _ := s.Get(KeyLanguage); v != "fr" {
		t.Error(v)
	}
	if _, err := Set(ctx, d, "x", KeyLanguage, "en"); !errors.Is(err, family.ErrNewerSchema) {
		t.Error(err)
	}
}

func TestEffectiveAndFollow(t *testing.T) {
	var fam Settings
	fam.SetValue(KeyProxy, "direct")
	if Effective(fam, KeyProxy, true, "http://x:1", "") != "direct" {
		t.Error("follow")
	}
	if Effective(fam, KeyProxy, false, "http://x:1", "") != "http://x:1" {
		t.Error("own")
	}
	if Effective(fam, KeyLanguage, true, "en", "zh-Hans") != "zh-Hans" {
		t.Error("default")
	}
	f := Follow{KeyProxy: false}
	if f.Of(KeyProxy) || !f.Of(KeyLanguage) {
		t.Error("flags")
	}
}

func TestMigrate(t *testing.T) {
	d := family.At(t.TempDir())
	ctx := context.Background()
	// AudioSmith first: a custom proxy and model folder, default language.
	out, err := Migrate(ctx, d, "dev.hehehai.audiosmith", map[string]Local{
		KeyLanguage:    {Value: "", IsDefault: true},
		KeyProxy:       {Value: "http://127.0.0.1:7897"},
		KeyHFEndpoints: {Value: "auto", IsDefault: true},
		KeyModelsRoot:  {Value: "/Volumes/Models/hub"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]Outcome{
		KeyLanguage:    {Follow: true},
		KeyProxy:       {Follow: true, Wrote: true},
		KeyHFEndpoints: {Follow: true},
		KeyModelsRoot:  {Follow: true, Wrote: true},
	}
	for _, o := range out {
		w := want[o.Key]
		if o.Follow != w.Follow || o.Wrote != w.Wrote || o.Conflict {
			t.Errorf("audiosmith %s: %+v", o.Key, o)
		}
	}
	// Dicta next: same proxy, another language it chose, models in the
	// default folder (in use), endpoints default.
	out, err = Migrate(ctx, d, "dev.hehehai.dicta", map[string]Local{
		KeyLanguage:    {Value: "zh-Hans"},
		KeyProxy:       {Value: "http://127.0.0.1:7897"},
		KeyHFEndpoints: {Value: "", IsDefault: true},
		KeyModelsRoot:  {Value: "", IsDefault: true, InUse: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	want = map[string]Outcome{
		KeyLanguage:    {Follow: true, Wrote: true},
		KeyProxy:       {Follow: true},
		KeyHFEndpoints: {Follow: true},
		KeyModelsRoot:  {Conflict: true},
	}
	for _, o := range out {
		w := want[o.Key]
		if o.Follow != w.Follow || o.Wrote != w.Wrote || o.Conflict != w.Conflict {
			t.Errorf("dicta %s: %+v", o.Key, o)
		}
	}
	// Auralink: another proxy the user set → keeps it; a fresh install
	// (default language) adopts the family's.
	out, _ = Migrate(ctx, d, "dev.hehehai.auralink", map[string]Local{
		KeyLanguage: {Value: "", IsDefault: true},
		KeyProxy:    {Value: "socks5://10.0.0.1:1080"},
	})
	for _, o := range out {
		switch o.Key {
		case KeyLanguage:
			if !o.Follow || o.Conflict {
				t.Errorf("auralink language %+v", o)
			}
		case KeyProxy:
			if o.Follow || !o.Conflict || o.Family != "http://127.0.0.1:7897" {
				t.Errorf("auralink proxy %+v", o)
			}
		}
	}
	s, _, _ := Read(d)
	if v, _ := s.Get(KeyProxy); v != "http://127.0.0.1:7897" {
		t.Error("family proxy changed")
	}
	if s.Rev != 2 {
		t.Errorf("rev %d (Auralink wrote nothing)", s.Rev)
	}
}
