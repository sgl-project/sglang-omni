package omni

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestModels(t *testing.T) {
	asrs := ASRModels()
	if len(asrs) != 2 || asrs[0].ID != ASRAccurate || asrs[1].ID != ASRFast {
		t.Fatalf("speech models %+v", asrs)
	}
	for _, s := range Models() {
		if s.Repo == "" || len(s.Revision) != 40 || (s.ASR == nil) == (s.VAD == nil) {
			t.Errorf("spec %+v", s)
		}
	}
	s, ok := Lookup(VADSilero)
	if !ok || s.Kind != kindSileroVAD || s.VAD.Chunk != 512 {
		t.Errorf("silero %+v", s)
	}
	// Copies: a caller cannot change the table.
	s, _ = Lookup(ASRFast)
	s.ASR.MaxTokens = 1
	s.Languages[0] = "xx"
	if fresh, _ := Lookup(ASRFast); fresh.ASR.MaxTokens != 8192 || fresh.Languages[0] != "zh" {
		t.Error("Lookup returned the table itself")
	}
	if args := strings.Join(again(ASRAccurate).launchArgs(true), " "); args != "--model-kind qwen3_asr --first-decode-ms 100 --decode-interval-ms 1000 --max-segment-seconds 30 --cache-limit-mb 512" {
		t.Errorf("launch args %q", args)
	}
	if args := strings.Join(again(ASRAccurate).launchArgs(false), " "); strings.Contains(args, "--cache-limit-mb") {
		t.Errorf("launch args for an older server %q", args)
	}
	if args := strings.Join(again(VADSilero).launchArgs(true), " "); args != "--model-kind silero_vad" {
		t.Errorf("launch args %q", args)
	}
}

func again(id string) Spec { s, _ := Lookup(id); return s }

func TestTokenBudget(t *testing.T) {
	q := qwenTuning()
	for _, c := range []struct {
		d    time.Duration
		want int
	}{{0, 256}, {5 * time.Second, 256}, {10 * time.Second, 344}, {100 * time.Second, 2864}, {time.Hour, 8192}, {-time.Second, 256}} {
		if got := q.tokenBudget(c.d); got != c.want {
			t.Errorf("budget(%v) = %d, want %d", c.d, got, c.want)
		}
	}
}

func TestLanguageHint(t *testing.T) {
	s := again(ASRFast)
	for in, want := range map[string]string{"zh": "zh", "zh-Hans": "zh", "en-US": "en", "yue": "yue", "": "", "  ": "", "sw": "", "xx": ""} {
		if got := s.languageHint(in); got != want {
			t.Errorf("hint(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestContextBias(t *testing.T) {
	q := qwenTuning()
	if got := q.contextBias([]string{" VoxtNew ", "voxtnew", "", "Aura  Link", "MLX"}, "").context; got != "VoxtNew\nAura Link\nMLX" {
		t.Errorf("bias %q", got)
	}
	if got := q.contextBias([]string{"MLX"}, "  A meeting about audio. ").context; got != "A meeting about audio.\nMLX" {
		t.Errorf("bias with a prompt %q", got)
	}
	if got := q.contextBias(nil, "").context; got != "" {
		t.Errorf("no bias %q", got)
	}
	// At most ContextTerms terms…
	many := make([]string, 600)
	for i := range many {
		many[i] = "t" + itoa(i)
	}
	if n := len(strings.Split(q.contextBias(many, "").context, "\n")); n != 500 {
		t.Errorf("%d terms", n)
	}
	// The fast model takes the 30 first (the most recently changed).
	fast, _ := Lookup(ASRFast)
	if b := fast.ASR.contextBias(many, ""); len(b.terms) != 30 || b.terms[29] != "t29" {
		t.Errorf("fast: %d terms", len(b.terms))
	}
	// …and ContextChars characters (runes, as Swift counts), a first long one kept.
	q.ContextChars = 10
	if got := q.contextBias([]string{"一二三四五", "六七八九", "十"}, "").context; got != "一二三四五\n六七八九" {
		t.Errorf("bounded %q", got)
	}
	if got := q.contextBias([]string{"abcdefghijklmnop", "x"}, "").context; got != "abcdefghijklmnop" {
		t.Errorf("long first %q", got)
	}
}

// tone is d of a 440 Hz tone at amp (0–1), PCM16 at 16 kHz.
func tone(d time.Duration, amp float64) []byte {
	n := int(d.Seconds() * sampleRate)
	b := make([]byte, n*2)
	for i := range n {
		v := amp * math.Sin(2*math.Pi*440*float64(i)/sampleRate)
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v*32767)))
	}
	return b
}

func silence(d time.Duration) []byte { return make([]byte, int(d.Seconds()*sampleRate)*2) }

func TestSplitChunks(t *testing.T) {
	q := qwenTuning()
	// Short: one chunk, padded to a second.
	cs := q.splitChunks(tone(300*time.Millisecond, 0.5))
	if len(cs) != 1 || len(cs[0].pcm) != sampleRate*2 || cs[0].offset != 0 {
		t.Fatalf("short: %d chunks, %d bytes", len(cs), len(cs[0].pcm))
	}
	// The curated tuning: 286 s (the eval's synthetic clip) is five or six
	// requests of 55–65 s (cut within 5 s of each minute), so the encoder
	// never sees minutes at once.
	cs = q.splitChunks(tone(286*time.Second, 0.5))
	if len(cs) < 5 || len(cs) > 6 {
		t.Errorf("286 s: %d chunks", len(cs))
	}
	for _, c := range cs {
		if len(c.pcm) > 65*sampleRate*2 {
			t.Errorf("a chunk of %d samples", len(c.pcm)/2)
		}
	}
	// Long (2 s chunks here): cut at the quiet stretch near each end, not
	// past it (pastEnd holds only while chunk + search ≤ 1200 s).
	q.ChunkSeconds, q.CutSearch = 2, 500*time.Millisecond
	pcm := append(append(append(tone(1700*time.Millisecond, 0.5), silence(200*time.Millisecond)...), tone(1500*time.Millisecond, 0.5)...), tone(200*time.Millisecond, 0.5)...)
	cs = q.splitChunks(pcm)
	if len(cs) != 2 {
		t.Fatalf("%d chunks", len(cs))
	}
	cut := cs[1].offset
	if cut < 1700*time.Millisecond || cut > 1900*time.Millisecond {
		t.Errorf("cut at %v, want in the silence (1.7–1.9 s)", cut)
	}
	if total := len(cs[0].pcm) + len(cs[1].pcm); total != len(pcm) {
		t.Errorf("chunks hold %d bytes of %d", total, len(pcm))
	}
	// The cut may pass the end when the limit allows.
	q.ChunkSeconds = 1.5
	if !(q.ChunkSeconds+q.CutSearch.Seconds() <= 1200) {
		t.Fatal("setup")
	}
	cs = q.splitChunks(pcm)
	if cs[1].offset < 1700*time.Millisecond || cs[1].offset > 1900*time.Millisecond {
		t.Errorf("past the end: cut at %v", cs[1].offset)
	}
}

func TestJoinParts(t *testing.T) {
	for _, c := range []struct {
		in   []string
		want string
	}{
		{[]string{"Hello", "world"}, "Hello world"},
		{[]string{"你好", "世界"}, "你好世界"},
		{[]string{"发布", "VoxtNew"}, "发布VoxtNew"},
		{[]string{" a ", "", "  ", "b."}, "a b."},
	} {
		if got := joinParts(c.in); got != c.want {
			t.Errorf("join(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestAssembler(t *testing.T) {
	var a assembler
	steps := []struct {
		index, seg int
		text       string
		final      bool
		want       string
		ok         bool
	}{
		{1, 0, "hel", false, "hel", true},
		{2, 0, "hello", false, "hello", true},
		{2, 0, "replayed", false, "", false},
		{3, 0, "hello there", true, "hello there", true},
		{4, 0, "late provisional", false, "", false},
		{5, 1, "how", false, "hello there how", true},
		{0, 1, "how are", false, "hello there how are", true}, // no index: always applied
	}
	for _, s := range steps {
		got, ok := a.apply(s.index, s.seg, s.text, s.final)
		if got != s.want || ok != s.ok {
			t.Errorf("apply(%d, %d, %q) = %q %v, want %q %v", s.index, s.seg, s.text, got, ok, s.want, s.ok)
		}
	}
}

func TestLocate(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "VoxtNew.app", "Contents")
	bin := filepath.Join(app, "Resources", "omni", ServerBinary)
	if err := os.MkdirAll(filepath.Dir(bin), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bin, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(app, "MacOS", "VoxtNew")
	t.Setenv(EnvRuntime, "")
	if got, err := Locate(exe); err != nil || got != bin {
		t.Errorf("bundle: %q %v", got, err)
	}
	if _, err := Locate(filepath.Join(dir, "elsewhere", "VoxtNew")); err == nil || !strings.Contains(err.Error(), EnvRuntime) {
		t.Errorf("none: %v", err)
	}
	t.Setenv(EnvRuntime, filepath.Join(app, "Resources", "omni"))
	if got, err := Locate(""); err != nil || got != bin {
		t.Errorf("env dir: %q %v", got, err)
	}
	t.Setenv(EnvRuntime, bin)
	if got, err := Locate(""); err != nil || got != bin {
		t.Errorf("env binary: %q %v", got, err)
	}
	t.Setenv(EnvRuntime, dir)
	if _, err := Locate(exe); err == nil {
		t.Error("a wrong env dir must not fall back to the bundle")
	}
}
