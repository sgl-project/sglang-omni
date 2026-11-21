package sounds

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

func sum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

const catJSON = `{
 "schema": "actnow.sounds.catalog/1",
 "version": "test",
 "groups": [{"id": "tap", "name": {"en": "Tap", "zh": "轻触"}}, {"id": "notify", "name": {"en": "Notify", "zh": "通知"}}, {"id": "empty", "name": {"en": "Empty"}}],
 "tags": [{"id": "soft", "name": {"en": "Soft", "zh": "柔和"}, "featured": true}, {"id": "short", "name": {"en": "Short", "zh": "简短"}, "featured": true}],
 "uses": [{"id": "record.start", "name": {"en": "Start recording", "zh": "开始录音"}}],
 "sounds": [
  {"id": "pebble", "file": "sounds/pebble.wav", "name": {"en": "Pebble", "zh": "卵石"}, "group": "tap", "tags": ["soft", "short", "click"], "duration": 0.25, "audible": 0.2, "uses": ["record.start"], "replaces": ["Pop"], "sha256": "PEBBLE"},
  {"id": "tink", "file": "sounds/tink.wav", "name": {"en": "Tink", "zh": "叮"}, "group": "tap", "tags": ["short", "metal"], "duration": 0.1, "replaces": ["Tink"]},
  {"id": "chime", "file": "sounds/chime.wav", "name": {"en": "Chime", "zh": "风铃"}, "group": "notify", "tags": ["soft", "bell"], "duration": 1.4, "desc": {"en": "Three soft bells"}},
  {"id": "gone", "file": "sounds/gone.wav", "name": {"en": "Gone"}, "group": "notify", "duration": 1}
 ],
 "notice": "Sounds from somewhere."
}`

func testFS() fstest.MapFS {
	pebble := []byte("RIFFpebble")
	return fstest.MapFS{
		"res/sounds/catalog.json":      {Data: []byte(strings.Replace(catJSON, "PEBBLE", sum(pebble), 1))},
		"res/sounds/sounds/pebble.wav": {Data: pebble},
		"res/sounds/sounds/tink.wav":   {Data: []byte("RIFFtink")},
		"res/sounds/sounds/chime.wav":  {Data: []byte("RIFFchime")},
	}
}

func load(t *testing.T) *Catalog {
	t.Helper()
	c, err := Load(testFS(), "res/sounds")
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestLoad(t *testing.T) {
	c := load(t)
	if len(c.Sounds) != 4 || c.Version != "test" || c.Notice == "" {
		t.Fatalf("%+v", c)
	}
	s, ok := c.Sound("pebble")
	if !ok || s.Name.In("zh-Hans") != "卵石" || s.Name.In("en") != "Pebble" || s.Length() != 200*time.Millisecond {
		t.Errorf("pebble %+v", s)
	}
	if s, _ := c.Sound("tink"); s.Length() != 100*time.Millisecond {
		t.Errorf("tink length %v", s.Length())
	}
	if g, ok := c.Group("notify"); !ok || g.Name.In("zh") != "通知" {
		t.Errorf("group %+v", g)
	}
	if c.TagName("soft", "zh-CN") != "柔和" || c.TagName("metal", "zh") != "metal" || c.UseName("record.start", "en") != "Start recording" {
		t.Error("names")
	}
	if c.ForSystem("pop") != "pebble" || c.ForSystem("Tink") != "tink" || c.ForSystem("Basso") != "" {
		t.Error("ForSystem")
	}
	if _, err := Load(testFS(), "nowhere"); !errors.Is(err, ErrNoCatalog) {
		t.Errorf("missing: %v", err)
	}
	if _, err := Load(nil, ""); !errors.Is(err, ErrNoCatalog) {
		t.Errorf("nil fs: %v", err)
	}
	var nilCat *Catalog
	if _, ok := nilCat.Sound("pebble"); ok || nilCat.Filter(Filter{}) != nil || nilCat.ForSystem("Pop") != "" {
		t.Error("nil catalogue")
	}
}

func TestParseRejects(t *testing.T) {
	for name, bad := range map[string]string{
		"schema":   strings.Replace(catJSON, "catalog/1", "catalog/9", 1),
		"group":    strings.Replace(catJSON, `"group": "notify", "tags"`, `"group": "nope", "tags"`, 1),
		"repeat":   strings.Replace(catJSON, `"id": "tink"`, `"id": "pebble"`, 1),
		"reserved": strings.Replace(catJSON, `"id": "tink"`, `"id": "none"`, 1),
		"badid":    strings.Replace(catJSON, `"id": "tink"`, `"id": "Ti nk"`, 1),
		"escape":   strings.Replace(catJSON, `sounds/tink.wav`, `../tink.wav`, 1),
		"json":     catJSON[:40],
	} {
		if _, err := Parse([]byte(bad), testFS(), "res/sounds"); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestData(t *testing.T) {
	c := load(t)
	if b, err := c.Data("pebble"); err != nil || string(b) != "RIFFpebble" {
		t.Errorf("pebble %q %v", b, err)
	}
	if _, err := c.Data("nope"); !errors.Is(err, ErrUnknown) {
		t.Errorf("unknown %v", err)
	}
	errs := c.Check()
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "gone") {
		t.Errorf("check %v", errs)
	}
	fs := testFS()
	fs["res/sounds/sounds/pebble.wav"] = &fstest.MapFile{Data: []byte("tampered")}
	c2, _ := Load(fs, "res/sounds")
	if _, err := c2.Data("pebble"); err == nil || !strings.Contains(err.Error(), "SHA-256") {
		t.Errorf("tampered %v", err)
	}
}

func ids(ss []Sound) []string {
	var out []string
	for _, s := range ss {
		out = append(out, s.ID)
	}
	return out
}

func TestFilter(t *testing.T) {
	c := load(t)
	for _, tc := range []struct {
		f    Filter
		want []string
	}{
		{Filter{}, []string{"pebble", "tink", "chime", "gone"}},
		{Filter{Group: "tap"}, []string{"pebble", "tink"}},
		{Filter{Tags: []string{"soft"}}, []string{"pebble", "chime"}},
		{Filter{Tags: []string{"soft", "short"}}, []string{"pebble"}},
		{Filter{Query: "bell"}, []string{"chime"}},
		{Filter{Query: "柔和"}, []string{"pebble", "chime"}},
		{Filter{Query: "通知 three"}, []string{"chime"}},
		{Filter{Query: "TINK"}, []string{"tink"}},
		{Filter{Uses: []string{"record.start", "x"}}, []string{"pebble"}},
	} {
		if got := ids(c.Filter(tc.f)); !slices.Equal(got, tc.want) {
			t.Errorf("%+v: %v, want %v", tc.f, got, tc.want)
		}
	}
	tags := c.TagCounts(Filter{Tags: []string{"soft"}}, false)
	if len(tags) < 2 || tags[0].ID != "short" || tags[0].Count != 2 || tags[1].ID != "soft" || tags[1].Count != 2 || !tags[0].Featured {
		t.Errorf("tags %+v", tags)
	}
	if ft := c.TagCounts(Filter{Group: "notify"}, true); len(ft) != 1 || ft[0].ID != "soft" {
		t.Errorf("featured %+v", ft)
	}
	gc := c.GroupCounts(Filter{Group: "tap", Tags: []string{"soft"}})
	if len(gc) != 2 || gc[0].ID != "tap" || gc[0].Count != 1 || gc[1].Count != 1 {
		t.Errorf("groups %+v", gc)
	}
	var f Filter
	f.ToggleTag("soft")
	f.ToggleTag("short")
	f.ToggleTag("soft")
	if !slices.Equal(f.Tags, []string{"short"}) {
		t.Errorf("toggle %v", f.Tags)
	}
	ev := Event{ID: "record.start", Default: "tink", Uses: []string{"record.start"}}
	if got := ids(c.Suggested(ev)); !slices.Equal(got, []string{"tink", "pebble"}) {
		t.Errorf("suggested %v", got)
	}
}

func TestChoices(t *testing.T) {
	c := load(t)
	ev := Event{ID: "pick", Default: "pebble", Fallback: "Ping"}
	p := DefaultPrefs()
	if p.Choice(ev) != "pebble" || p.Target(c, ev) != (Target{KindCatalog, "pebble", 0.7}) {
		t.Errorf("default %+v", p.Target(c, ev))
	}
	p.Choose(ev, "chime")
	if p.Events["pick"] != "chime" || p.Target(c, ev).Name != "chime" {
		t.Error("choose")
	}
	p.Choose(ev, "pebble")
	if p.Events != nil {
		t.Errorf("default kept %v", p.Events)
	}
	p.Choose(ev, System("Glass"))
	if p.Target(c, ev) != (Target{KindSystem, "Glass", 0.7}) {
		t.Errorf("system %+v", p.Target(c, ev))
	}
	p.Choose(ev, None)
	if p.Target(c, ev).Kind != KindNone {
		t.Error("none")
	}
	p.Choose(ev, "gone-from-this-build")
	if p.Target(c, ev) != (Target{KindSystem, "Ping", 0.7}) {
		t.Errorf("missing sound falls back %+v", p.Target(c, ev))
	}
	if p.Target(nil, Event{ID: "x", Default: "pebble", Fallback: "Pop"}) != (Target{KindSystem, "Pop", 0.7}) {
		t.Error("no catalogue falls back")
	}
	if p.Target(nil, Event{ID: "x", Default: "pebble"}).Kind != KindNone {
		t.Error("no fallback: silence")
	}
	p.On = false
	if p.Target(c, Event{ID: "x", Default: "pebble"}).Kind != KindNone {
		t.Error("off")
	}
	q := Prefs{Volume: 140, Events: map[string]string{"a": "chime", "b": "../x", "c": "system:", "d": "system:Pop", "e": "Bad Id"}}
	q.Normalize()
	if q.Volume != 100 || len(q.Events) != 2 || q.Events["d"] != "system:Pop" {
		t.Errorf("normalize %+v", q)
	}
	if MigrateSystem(ev, "Ping", true) != "pebble" || MigrateSystem(ev, "Hero", false) != "system:Hero" || MigrateSystem(ev, "", false) != "pebble" {
		t.Error("MigrateSystem")
	}
	if MigrateClosest(c, "Pop") != "pebble" || MigrateClosest(c, "Hero") != "system:Hero" {
		t.Error("MigrateClosest")
	}
	if l := (Target{KindCatalog, "chime", 1}).Length(c); l != 1400*time.Millisecond {
		t.Errorf("length %v", l)
	}
	if l := (Target{KindSystem, "Pop", 1}).Length(c); l != 215*time.Millisecond {
		t.Errorf("system length %v", l)
	}
}

func TestPlayer(t *testing.T) {
	c := load(t)
	rec := &Recorder{}
	p := NewPlayer(c, rec)
	if err := p.Preload("pebble", "none", "system:Pop", "nope"); err != nil {
		t.Fatal(err)
	}
	ev := Event{ID: "pick", Default: "pebble", Fallback: "Ping"}
	prefs := Prefs{On: true, Volume: 50}
	if tg, err := p.PlayEvent(ev, prefs); err != nil || tg.Name != "pebble" {
		t.Fatal(tg, err)
	}
	p.PlayEvent(ev, prefs) // quick repeat: loaded once
	if _, err := p.Preview("chime", "", 30); err != nil {
		t.Fatal(err)
	}
	p.Preview(System("Glass"), "", 30)
	p.Preview(None, "", 30)
	p.StopPreview()
	want := []string{"pebble@0.5", "pebble@0.5", "chime@0.3", "stop chime", "system:Glass@0.3", "stop system:Glass"}
	if got := rec.Played(); !slices.Equal(got, want) {
		t.Errorf("played %v\nwant %v", got, want)
	}
	if got := rec.Loaded(); !slices.Equal(got, []string{"pebble", "chime"}) {
		t.Errorf("loaded %v", got)
	}
	if _, err := p.Preview("gone", "", 30); err == nil {
		t.Error("missing file: no error")
	}
	rec.Reset()
	np := NewPlayer(nil, rec)
	np.PlayEvent(ev, prefs)
	if got := rec.Played(); !slices.Equal(got, []string{"system:Ping@0.5"}) {
		t.Errorf("no catalogue %v", got)
	}
	rec.Err = errors.New("boom")
	if _, err := np.PlayEvent(ev, prefs); err == nil {
		t.Error("output error lost")
	}
}

func TestSubsetJSON(t *testing.T) {
	c := load(t)
	sub := c.Subset(func(s Sound) bool { return s.Group == "notify" && s.ID != "gone" })
	if len(sub.Sounds) != 1 || len(sub.Groups) != 1 || sub.Groups[0].ID != "notify" {
		t.Fatalf("%+v", sub)
	}
	b, err := sub.JSON()
	if err != nil {
		t.Fatal(err)
	}
	back, err := Parse(b, testFS(), "res/sounds")
	if err != nil || len(back.Sounds) != 1 || back.Sounds[0].Desc.En != "Three soft bells" || back.Notice == "" {
		t.Errorf("round trip %v %+v", err, back)
	}
}

func TestSilentAndSystemSounds(t *testing.T) {
	var o Output = Silent{}
	if err := o.Load("x", nil); err != nil {
		t.Error(err)
	}
	if v, err := o.Play("x", 1); err != nil || v == nil {
		t.Error(v, err)
	}
	if SystemSounds(t.TempDir()) != nil || !IsKnownSystem("Pop") || IsKnownSystem("pop") {
		t.Error("system sounds")
	}
}

func TestToneWAV(t *testing.T) {
	b := ToneWAV(440, 0.1, 0.5)
	if len(b) != 44+4410*2 || string(b[:4]) != "RIFF" || string(b[8:16]) != "WAVEfmt " || string(b[36:40]) != "data" {
		t.Errorf("header %q len %d", b[:44], len(b))
	}
}

func TestDecodeWAV(t *testing.T) {
	p, ok := DecodeWAV(ToneWAV(441, 0.1, 0.5))
	if !ok || p.Rate != 44100 || len(p.Left) != 4410 || len(p.Right) != 4410 || p.Left[100] != p.Right[100] {
		t.Fatalf("mono 16-bit: %v %d", ok, len(p.Left))
	}
	// Peak near 0.5 in the middle.
	var peak float32
	for _, v := range p.Left {
		peak = max(peak, v)
	}
	if peak < 0.49 || peak > 0.51 {
		t.Errorf("peak %v", peak)
	}
	// 22.05 kHz becomes 44.1 kHz.
	b := ToneWAV(441, 0.1, 0.5)
	binary.LittleEndian.PutUint32(b[24:28], 22050)
	if p, ok := DecodeWAV(b); !ok || len(p.Left) != 8820 {
		t.Errorf("resample %v %d", ok, len(p.Left))
	}
	for _, bad := range [][]byte{nil, []byte("RIFFxxxxWAVE"), []byte("OggS....")} {
		if _, ok := DecodeWAV(bad); ok {
			t.Errorf("%q decoded", bad)
		}
	}
	if dir := os.Getenv("ACTNOW_SOUNDS_DIR"); dir != "" {
		c, err := Load(os.DirFS(dir), ".")
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range c.Sounds {
			b, _ := c.Data(s.ID)
			p, ok := DecodeWAV(b)
			if !ok || math.Abs(float64(len(p.Left))/44100-s.Duration) > 0.01 {
				t.Errorf("%s: %v %d frames, %.3f s", s.ID, ok, len(p.Left), s.Duration)
			}
		}
	}
}

func TestPreviewingAndVolume(t *testing.T) {
	c := load(t)
	p := NewPlayer(c, &Recorder{})
	now := time.Unix(100, 0)
	p.SetClock(func() time.Time { return now })
	if ch, _ := p.Previewing(); ch != "" {
		t.Error("previewing before any preview")
	}
	p.Preview("chime", "", 50)
	if ch, left := p.Previewing(); ch != "chime" || left != 1400*time.Millisecond {
		t.Errorf("chime %q %v", ch, left)
	}
	now = now.Add(time.Second)
	if ch, left := p.Previewing(); ch != "chime" || left != 400*time.Millisecond {
		t.Errorf("later %q %v", ch, left)
	}
	now = now.Add(time.Second)
	if ch, _ := p.Previewing(); ch != "" {
		t.Errorf("ended %q", ch)
	}
	p.Preview(System("Glass"), "", 50) // not measured: UnknownLength
	if ch, left := p.Previewing(); ch != "system:Glass" || left != UnknownLength {
		t.Errorf("system %q %v", ch, left)
	}
	p.StopPreview()
	if ch, _ := p.Previewing(); ch != "" {
		t.Errorf("stopped %q", ch)
	}
	p.Preview(None, "", 50)
	if ch, _ := p.Previewing(); ch != "" {
		t.Errorf("none %q", ch)
	}
	for v, want := range map[int]int{0: 0, 10: 0, 13: 25, 70: 75, 62: 50, 100: 100, 140: 100} {
		if got := NearestVolumeStep(v); got != want {
			t.Errorf("NearestVolumeStep(%d) = %d, want %d", v, got, want)
		}
	}
	if VolumeLevel(0) != 0 || VolumeLevel(25) != 1 || VolumeLevel(50) != 1 || VolumeLevel(75) != 2 {
		t.Error("VolumeLevel")
	}
	if c.GroupName("notify", "zh") != "通知" || c.GroupName("nope", "en") != "nope" {
		t.Error("GroupName")
	}
}
