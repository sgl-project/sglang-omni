package registry

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/shortcuts"
)

// fakeSystem is LaunchServices and the process list for tests.
type fakeSystem struct {
	installed map[string][]string
	running   map[string][]Process
}

func (f fakeSystem) AppPaths(id string) []string       { return f.installed[id] }
func (f fakeSystem) Running(id string) []Process       { return f.running[id] }
func (f fakeSystem) BundleID(p string) (string, error) { return BundleIDAt(p) }

// makeApp writes a minimal .app with an XML Info.plist.
func makeApp(t *testing.T, dir, name, bundleID string) string {
	t.Helper()
	app := filepath.Join(dir, name+".app")
	os.MkdirAll(filepath.Join(app, "Contents"), 0o755)
	plist := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleDocumentTypes</key><array><dict><key>CFBundleIdentifier</key><string>nested.wrong</string></dict></array>
  <key>CFBundleIdentifier</key><string>` + bundleID + `</string>
  <key>CFBundleName</key><string>` + name + `</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
</dict></plist>`
	if err := os.WriteFile(filepath.Join(app, "Contents", "Info.plist"), []byte(plist), 0o644); err != nil {
		t.Fatal(err)
	}
	return app
}

func manifest(bid, path string) Manifest {
	a, _, _ := family.ParseBundleID(bid)
	return Manifest{BundleID: bid, Name: a.Name, Version: "1.0.0", Path: path, Capabilities: []string{CapModelsHF}, Models: &ModelsInfo{Root: "/hub"}}
}

func TestWriteReadList(t *testing.T) {
	d := family.At(t.TempDir())
	m := manifest("dev.hehehai.audiosmith", "/Applications/AudioSmith.app")
	m.Shortcuts = []shortcuts.Shortcut{{ID: "dictation", Title: "Dictation", Keys: "CmdOrCtrl+Shift+Space", Trigger: "long", Enabled: true, Mechanism: shortcuts.Carbon}}
	m.Links = map[string]string{LinkShortcuts: "audiosmith://settings/dictation"}
	if err := Write(d, m); err != nil {
		t.Fatal(err)
	}
	got, err := Read(d, m.BundleID)
	if err != nil {
		t.Fatal(err)
	}
	if got.App != "audiosmith" || got.Variant != family.Release || got.Schema != "actnow.app/1" || got.SpecVersion != 1 || got.PID != os.Getpid() {
		t.Errorf("%+v", got)
	}
	if got.Shortcuts[0].Keys != "Shift+Command+Space" || got.Shortcuts[0].Trigger != shortcuts.Hold {
		t.Errorf("shortcut not normalised: %+v", got.Shortcuts[0])
	}
	// A newer version added members: kept across our rewrite.
	raw, _ := os.ReadFile(d.AppFile(m.BundleID))
	raw = []byte(strings.Replace(string(raw), `"app":`, `"futureMember": [1], "app":`, 1))
	os.WriteFile(d.AppFile(m.BundleID), raw, 0o600)
	if err := Write(d, m); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(d.AppFile(m.BundleID))
	if !strings.Contains(string(raw), `"futureMember"`) {
		t.Error("unknown member lost")
	}
	// Validation.
	bad := manifest("dev.hehehai.dicta", "")
	if err := Write(d, bad); err == nil {
		t.Error("missing path accepted")
	}
	bad = manifest("dev.hehehai.dicta", "/x")
	bad.Models = nil
	if err := Write(d, bad); err == nil {
		t.Error("models.hf without root accepted")
	}
	// List skips files that lie about their bundle id, and corrupt ones.
	os.WriteFile(d.AppFile("dev.hehehai.pik"), raw, 0o600)
	os.WriteFile(d.AppFile("dev.hehehai.dicta"), []byte("{"), 0o600)
	ms, errs := List(d)
	if len(ms) != 1 || len(errs) != 2 {
		t.Errorf("%d manifests, errs %v", len(ms), errs)
	}
	if err := Remove(d, m.BundleID); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(d.AppFile(m.BundleID)); !os.IsNotExist(err) {
		t.Error("not removed")
	}
}

// TestLearningManifest: Paralign's manifest, as another app (CaptionCraft's
// "Study in Paralign") finds it by capability.
func TestLearningManifest(t *testing.T) {
	d := family.At(t.TempDir())
	m := manifest("dev.hehehai.paralign", "/Applications/Paralign.app")
	m.URLScheme = "paralign"
	m.Capabilities = []string{CapModelsHF, CapASRLocal, CapLLMLocal, CapTTSLocal, CapLearning}
	if err := Write(d, m); err != nil {
		t.Fatal(err)
	}
	ms, errs := List(d)
	if len(ms) != 1 || len(errs) != 0 {
		t.Fatalf("%d manifests, errs %v", len(ms), errs)
	}
	got := ms[0]
	if got.App != "paralign" || got.URLScheme != "paralign" || !got.Has(CapLearning) || got.Has(CapSubtitles) {
		t.Errorf("%+v", got)
	}
}

// TestLibraryManifest: Act's manifest, as another app ("Save to Act")
// finds it by capability.
func TestLibraryManifest(t *testing.T) {
	d := family.At(t.TempDir())
	m := manifest("dev.hehehai.act", "/Applications/Act.app")
	m.URLScheme = "act"
	m.Capabilities = []string{CapModelsHF, CapEmbedLocal, CapLibrary}
	if err := Write(d, m); err != nil {
		t.Fatal(err)
	}
	ms, errs := List(d)
	if len(ms) != 1 || len(errs) != 0 {
		t.Fatalf("%d manifests, errs %v", len(ms), errs)
	}
	got := ms[0]
	if got.App != "act" || got.URLScheme != "act" || !got.Has(CapLibrary) || !got.Has(CapEmbedLocal) || got.Has(CapVLMLocal) {
		t.Errorf("%+v", got)
	}
}

func TestDiscover(t *testing.T) {
	d := family.At(t.TempDir())
	apps := t.TempDir()
	smithPath := makeApp(t, apps, "AudioSmith", "dev.hehehai.audiosmith")
	moved := makeApp(t, apps, "Pik", "dev.hehehai.pik")
	// AudioSmith: registered, installed where its manifest says, running.
	Write(d, manifest("dev.hehehai.audiosmith", smithPath))
	// Dicta: registered, then deleted (path gone, LaunchServices forgot).
	Write(d, manifest("dev.hehehai.dicta", filepath.Join(apps, "Dicta.app")))
	// Dicta dev build: registered, path holds another app.
	Write(d, manifest("dev.hehehai.dicta.dev", smithPath))
	// Pik: registered at an old path, moved; LaunchServices knows the new one.
	Write(d, manifest("dev.hehehai.pik", "/Applications/Old/Pik.app"))
	sys := fakeSystem{
		installed: map[string][]string{"dev.hehehai.pik": {moved}, "dev.hehehai.auralink": {"/Applications/Auralink.app"}},
		running:   map[string][]Process{"dev.hehehai.audiosmith": {{PID: 42, Path: smithPath}}},
	}
	found, errs := Discover(d, sys)
	if len(errs) != 0 {
		t.Fatal(errs)
	}
	by := map[string]Found{}
	for _, f := range found {
		by[f.BundleID] = f
	}
	check := func(id string, installed, running, registered bool) {
		t.Helper()
		f, ok := by[id]
		if !ok {
			t.Errorf("%s not found", id)
			return
		}
		if f.Installed != installed || f.Running != running || (f.Manifest != nil) != registered {
			t.Errorf("%s: installed %v running %v registered %v", id, f.Installed, f.Running, f.Manifest != nil)
		}
	}
	check("dev.hehehai.audiosmith", true, true, true)
	check("dev.hehehai.dicta", false, false, true)
	check("dev.hehehai.dicta.dev", false, false, true)
	check("dev.hehehai.pik", true, false, true)
	check("dev.hehehai.auralink", true, false, false) // installed, never started
	if _, ok := by["dev.hehehai.voxtnew"]; ok {
		t.Error("VoxtNew is not installed")
	}
	if by["dev.hehehai.pik"].Path != moved {
		t.Error("moved path")
	}
	if v := Visible(found); len(v) != 4 {
		t.Errorf("visible %d", len(v))
	}
	st := States(found)
	if st["dev.hehehai.dicta"].Installed || !st["dev.hehehai.audiosmith"].Running || st["dev.hehehai.auralink"].Name != "Auralink" {
		t.Errorf("%+v", st)
	}
	if h := Holders(found); len(h) != 2 {
		t.Errorf("holders %d", len(h))
	}
}

func TestBundleOf(t *testing.T) {
	if got := bundleOf("/Applications/Dicta.app/Contents/MacOS/Dicta"); got != "/Applications/Dicta.app" {
		t.Error(got)
	}
	if got := bundleOf("/usr/local/bin/tool"); got != "" {
		t.Error(got)
	}
}

// bplist writes a binary property list of one dictionary of strings.
func bplist(kv [][2]string) []byte {
	var objs [][]byte
	str := func(s string) int {
		var b bytes.Buffer
		ascii := true
		for _, r := range s {
			if r > 127 {
				ascii = false
			}
		}
		if ascii {
			n := len(s)
			if n < 15 {
				b.WriteByte(0x50 | byte(n))
			} else {
				b.Write([]byte{0x5f, 0x10, byte(n)})
			}
			b.WriteString(s)
		} else {
			u := []rune(s)
			b.WriteByte(0x60 | byte(len(u)))
			for _, r := range u {
				binary.Write(&b, binary.BigEndian, uint16(r))
			}
		}
		objs = append(objs, b.Bytes())
		return len(objs) - 1
	}
	objs = append(objs, nil) // the dict, object 0
	var keys, vals []int
	for _, p := range kv {
		keys = append(keys, str(p[0]))
		vals = append(vals, str(p[1]))
	}
	var dict bytes.Buffer
	dict.WriteByte(0xd0 | byte(len(kv)))
	for _, k := range keys {
		dict.WriteByte(byte(k))
	}
	for _, v := range vals {
		dict.WriteByte(byte(v))
	}
	objs[0] = dict.Bytes()
	var out bytes.Buffer
	out.WriteString("bplist00")
	var offsets []int
	for _, o := range objs {
		offsets = append(offsets, out.Len())
		out.Write(o)
	}
	table := out.Len()
	for _, o := range offsets {
		out.WriteByte(byte(o))
	}
	tr := make([]byte, 32)
	tr[6], tr[7] = 1, 1
	binary.BigEndian.PutUint64(tr[8:], uint64(len(objs)))
	binary.BigEndian.PutUint64(tr[16:], 0)
	binary.BigEndian.PutUint64(tr[24:], uint64(table))
	out.Write(tr)
	return out.Bytes()
}

func TestBinaryPlist(t *testing.T) {
	data := bplist([][2]string{{"CFBundleName", "Dicta"}, {"CFBundleIdentifier", "dev.hehehai.dicta"}, {"NSHumanReadableCopyright", "© 2026 actnow"}})
	m, err := plistStrings(data)
	if err != nil {
		t.Fatal(err)
	}
	if m["CFBundleIdentifier"] != "dev.hehehai.dicta" || m["NSHumanReadableCopyright"] != "© 2026 actnow" {
		t.Errorf("%v", m)
	}
	for _, bad := range [][]byte{[]byte("bplist00"), append([]byte("bplist00"), make([]byte, 40)...)} {
		if _, err := plistStrings(bad); err == nil {
			t.Error("garbage parsed")
		}
	}
	if _, err := plistStrings([]byte("<plist><array/></plist>")); err == nil {
		t.Error("no dict accepted")
	}
}

// TestMacSystem asks the real LaunchServices about a system app:
//
//	ACTNOWKIT_MAC=1 go test -run TestMacSystem -v ./registry
func TestMacSystem(t *testing.T) {
	if os.Getenv("ACTNOWKIT_MAC") == "" || runtime.GOOS != "darwin" {
		t.Skip("set ACTNOWKIT_MAC=1 on a Mac")
	}
	paths := DefaultSystem.AppPaths("com.apple.calculator")
	if len(paths) == 0 {
		t.Fatal("LaunchServices does not know Calculator")
	}
	id, err := DefaultSystem.BundleID(paths[0])
	if err != nil || !strings.EqualFold(id, "com.apple.calculator") {
		t.Fatalf("%s: %q %v", paths[0], id, err)
	}
	t.Logf("Calculator at %v", paths)
	if p := DefaultSystem.AppPaths("dev.actnow.no-such-app"); len(p) != 0 {
		t.Errorf("unknown app found: %v", p)
	}
	procs := DefaultSystem.Running("com.apple.finder")
	t.Logf("Finder processes: %+v (empty when no GUI session is visible)", procs)
	for _, p := range procs {
		if p.PID <= 0 || !strings.HasSuffix(p.Path, ".app") {
			t.Errorf("bad process %+v", p)
		}
	}
	t.Logf("team id of the test binary: %q (ad hoc: empty)", TeamID())
	for _, a := range family.Apps {
		t.Logf("%s: installed at %v, running %v", a.BundleID, DefaultSystem.AppPaths(a.BundleID), DefaultSystem.Running(a.BundleID))
	}
}

// TestMacSystemGoroutines calls the AppKit-backed lookups from many
// goroutines at once, with the scheduler forced to move goroutines between
// threads, so a pool popped on another thread than it was pushed on (no
// LockOSThread in withPool) crashes here.
//
//	ACTNOWKIT_MAC=1 go test -run TestMacSystemGoroutines -race -count=3 ./registry
func TestMacSystemGoroutines(t *testing.T) {
	if os.Getenv("ACTNOWKIT_MAC") == "" || runtime.GOOS != "darwin" {
		t.Skip("set ACTNOWKIT_MAC=1 on a Mac")
	}
	dir := family.At(t.TempDir())
	if _, err := dir.Ensure("dev.actnow.test"); err != nil {
		t.Fatal(err)
	}
	const workers, rounds = 32, 40
	done := make(chan struct{})
	for w := 0; w < workers; w++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for i := 0; i < rounds; i++ {
				_ = DefaultSystem.AppPaths("com.apple.calculator")
				_ = DefaultSystem.Running("com.apple.finder")
				_ = TeamID()
				if _, errs := Discover(dir, DefaultSystem); len(errs) != 0 {
					t.Error(errs)
					return
				}
				runtime.Gosched()
			}
		}()
	}
	for w := 0; w < workers; w++ {
		<-done
	}
}
