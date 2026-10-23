package omni

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
)

func TestMain(m *testing.M) {
	if omnitest.Enabled() {
		omnitest.Main()
	}
	os.Exit(m.Run())
}

// fake is a runtime over the fake server, with its log and model folders.
type fake struct {
	rt  *Runtime
	log string
	dir string // a model folder per id: dir/<id>
	mu  sync.Mutex
	out []string // the runtime's log lines
}

func newFake(t *testing.T, o omnitest.Opts, tune func(*Config)) *fake {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f := &fake{dir: t.TempDir(), log: filepath.Join(t.TempDir(), "fake.jsonl")}
	o.Log = f.log
	for _, s := range Models() {
		if err := os.MkdirAll(filepath.Join(f.dir, s.ID), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	cfg := Config{Binary: exe, Env: omnitest.Env(o), StartTimeout: 10 * time.Second,
		Log: func(level, msg string) { f.mu.Lock(); f.out = append(f.out, level+" "+msg); f.mu.Unlock() }}
	if tune != nil {
		tune(&cfg)
	}
	f.rt = NewRuntime(cfg)
	t.Cleanup(f.rt.Close)
	return f
}

func (f *fake) model(id string) string { return filepath.Join(f.dir, id) }

func (f *fake) records(event string) []omnitest.Record {
	return omnitest.Of(omnitest.Records(f.log), event)
}

func (f *fake) asr(t *testing.T, id string) *ASR {
	t.Helper()
	a, err := NewASR(f.rt, id, f.model(id))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func (f *fake) logged(sub string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.out, func(l string) bool { return strings.Contains(l, sub) })
}

// eventually waits for cond.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(10 * time.Second); time.Now().Before(end); time.Sleep(20 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out: %s", what)
}

func gone(pid int) bool { return syscall.Kill(pid, 0) != nil }

func TestNewASR(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	if _, err := NewASR(f.rt, VADSilero, f.model(VADSilero)); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("VAD as speech: %v", err)
	}
	if _, err := NewASR(f.rt, "whisper", f.model(ASRFast)); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("unknown: %v", err)
	}
	if _, err := NewASR(f.rt, ASRFast, filepath.Join(f.dir, "missing")); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("missing folder: %v", err)
	}
	a := f.asr(t, ASRFast)
	if c := a.Caps(); !c.Stream || !c.File || !c.Hints {
		t.Errorf("caps %+v", c)
	}
	if len(f.records("launch")) != 0 {
		t.Error("NewASR started a server")
	}
}

func TestFinal(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: " 发布计划 "}, nil)
	a := f.asr(t, ASRAccurate)
	pcm := tone(10*time.Second, 0.3)
	res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: 16000,
		Lang: "zh-Hans", Hints: []string{"VoxtNew", "MLX"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "发布计划" || res.Lang != "zh" || res.Model != ASRAccurate || res.Duration != 10*time.Second {
		t.Errorf("result %+v", res)
	}
	launches := f.records("launch")
	if len(launches) != 1 {
		t.Fatalf("%d launches", len(launches))
	}
	args := strings.Join(launches[0].Args, " ")
	for _, want := range []string{"--supervised", "--host 127.0.0.1 --port 0", "--model-directory " + f.model(ASRAccurate), "--model-kind qwen3_asr", "--first-decode-ms 100", "--decode-interval-ms 1000", "--cache-limit-mb 512"} {
		if !strings.Contains(args, want) {
			t.Errorf("launch %q lacks %q", args, want)
		}
	}
	finals := f.records("final")
	if len(finals) != 1 {
		t.Fatalf("%d finals", len(finals))
	}
	want := map[string]string{"model": "fake-qwen3_asr", "stream": "true", "response_format": "json", "language": "zh",
		"prompt": "VoxtNew\nMLX", "max_new_tokens": "344", "stop_at_end_of_text": "true", "stop_on_token_loop": "true",
		"include_generation_metadata": "true", "audio_layout": "voxt_swift"}
	for k, v := range want {
		if finals[0].Fields[k] != v {
			t.Errorf("field %s = %q, want %q", k, finals[0].Fields[k], v)
		}
	}
	if finals[0].Samples != 160000 {
		t.Errorf("%d samples sent", finals[0].Samples)
	}
	// A WAV file works too; another rate does not.
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: asr.WAV(pcm, 16000), Format: "wav"}); err != nil {
		t.Error(err)
	}
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: asr.WAV(pcm, 8000), Format: "wav"}); asr.KindOf(err) != asr.KindAudio {
		t.Errorf("8 kHz: %v", err)
	}
	if n := len(f.records("launch")); n != 1 {
		t.Errorf("the server was started %d times for three uses", n)
	}
	if _, ok := finals[0].Fields["language"]; !ok {
		t.Error("no language")
	}
}

// TestOlderServer: a runtime from before --cache-limit-mb (the user's
// installed bundle, an older release) still starts, without the flag,
// and the log says so; Silero's server never gets the flag.
func TestOlderServer(t *testing.T) {
	f := newFake(t, omnitest.Opts{Older: true}, nil)
	a := f.asr(t, ASRAccurate)
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm"}); err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch")
	if len(launches) != 1 || slices.Contains(launches[0].Args, "--cache-limit-mb") {
		t.Fatalf("launches %+v", launches)
	}
	if !f.logged("does not know --cache-limit-mb") {
		t.Error("the older runtime is not logged")
	}
}

// TestFinalChunks: a recording longer than a chunk is several requests
// sharing one budget, the first's language forced on the rest.
func TestFinalChunks(t *testing.T) {
	f := newFake(t, omnitest.Opts{Text: "片段"}, nil)
	a := f.asr(t, ASRFast)
	a.spec.ASR.ChunkSeconds, a.spec.ASR.CutSearch = 2, 100*time.Millisecond
	res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(5*time.Second, 0.3), Format: "pcm"})
	if err != nil {
		t.Fatal(err)
	}
	finals := f.records("final")
	if res.Text != "片段片段片段" || len(finals) != 3 {
		t.Fatalf("%q in %d requests", res.Text, len(finals))
	}
	for i, want := range []struct{ lang, tokens string }{{"", "256"}, {"Chinese", "246"}, {"Chinese", "236"}} {
		if got := finals[i].Fields; got["language"] != want.lang || got["max_new_tokens"] != want.tokens {
			t.Errorf("request %d: language %q budget %q, want %q %q", i, got["language"], got["max_new_tokens"], want.lang, want.tokens)
		}
	}
	if res.Lang != "Chinese" {
		t.Errorf("language %q", res.Lang)
	}
}

func TestFinalErrors(t *testing.T) {
	ctx := context.Background()
	pcm := tone(time.Second, 0.3)
	t.Run("error event", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeError}, nil)
		_, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm})
		var e *asr.Error
		if !errors.As(err, &e) || e.Kind != asr.KindServer || e.Code != "transcription_failed" {
			t.Errorf("%v", err)
		}
	})
	t.Run("start fails", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeFail}, nil)
		_, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm})
		if asr.KindOf(err) != asr.KindServer || !strings.Contains(err.Error(), "model load failed: fake") {
			t.Errorf("%v", err)
		}
	})
	t.Run("start hangs", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeHang}, func(c *Config) { c.StartTimeout = 300 * time.Millisecond })
		_, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm})
		if asr.KindOf(err) != asr.KindServer || !strings.Contains(err.Error(), "not ready within") {
			t.Errorf("%v", err)
		}
		pid := f.records("launch")[0].PID
		eventually(t, "the hung server killed", func() bool { return gone(pid) })
	})
	t.Run("crash", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeCrashFinal}, nil)
		a := f.asr(t, ASRFast)
		for range 2 {
			_, err := a.Transcribe(ctx, asr.FileRequest{Audio: pcm})
			if asr.KindOf(err) != asr.KindServer || !strings.Contains(err.Error(), "exited") {
				t.Errorf("%v", err)
			}
		}
		if n := len(f.records("launch")); n != 2 {
			t.Errorf("%d launches: a dead server is replaced at the next use", n)
		}
		if !f.logged("exited unexpectedly") {
			t.Error("the death was not logged")
		}
	})
	t.Run("stall", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeStall}, func(c *Config) { c.StallTimeout = 300 * time.Millisecond })
		_, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm})
		if asr.KindOf(err) != asr.KindServer || !strings.Contains(err.Error(), "stopped answering") {
			t.Errorf("%v", err)
		}
		pid := f.records("launch")[0].PID
		eventually(t, "the stalled server killed", func() bool { return gone(pid) })
	})
	t.Run("cancelled", func(t *testing.T) {
		f := newFake(t, omnitest.Opts{Delay: 5 * time.Second}, nil)
		cctx, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
		defer cancel()
		_, err := f.asr(t, ASRFast).Transcribe(cctx, asr.FileRequest{Audio: pcm})
		if k := asr.KindOf(err); k != asr.KindNetwork && k != asr.KindCancelled {
			t.Errorf("%v", err)
		}
	})
}

// TestIdleUnload: a server unused for IdleTimeout stops, and starts
// again at the next use.
func TestIdleUnload(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = 300 * time.Millisecond })
	ctx := context.Background()
	pcm := tone(time.Second, 0.3)
	fast := f.asr(t, ASRFast)
	if _, err := fast.Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
		t.Fatal(err)
	}
	if r := f.rt.Running(); len(r) != 1 || r[0] != ASRFast {
		t.Errorf("running %v", r)
	}
	if ps := f.rt.Processes(); len(ps) != 1 || ps[0].ID != ASRFast || ps[0].PID != f.records("launch")[0].PID {
		t.Errorf("processes %+v, launched %+v", ps, f.records("launch"))
	}
	eventually(t, "the idle unload", func() bool { return len(f.records("shutdown")) == 1 && len(f.rt.Running()) == 0 })
	if ps := f.rt.Processes(); len(ps) != 0 {
		t.Errorf("processes after the unload %+v", ps)
	}
	if !f.logged("unloaded after") {
		t.Error("the unload was not logged")
	}
	if _, err := fast.Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
		t.Fatal(err)
	}
	if n := len(f.records("launch")); n != 2 {
		t.Errorf("%d launches after an unload", n)
	}

}

// TestSetIdleTimeout: a change applies to the running servers at once:
// kept (never) → stopped after the new time, counted from the change;
// back to never, they stay; one in use counts from the end of its use.
func TestSetIdleTimeout(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = -1 })
	ctx := context.Background()
	pcm := tone(time.Second, 0.3)
	fast := f.asr(t, ASRFast)
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	use := func() {
		t.Helper()
		if _, err := v.Speech(ctx, pcm, 16000); err != nil {
			t.Fatal(err)
		}
		if _, err := fast.Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
			t.Fatal(err)
		}
	}
	use()
	time.Sleep(150 * time.Millisecond)
	if r := f.rt.Running(); len(r) != 2 {
		t.Fatalf("kept: running %v", r)
	}
	changed := time.Now()
	f.rt.SetIdleTimeout(250 * time.Millisecond)
	if d := f.rt.IdleTimeout(); d != 250*time.Millisecond {
		t.Errorf("IdleTimeout %v", d)
	}
	eventually(t, "the idle unload", func() bool { return len(f.rt.Running()) == 0 })
	if took := time.Since(changed); took < 250*time.Millisecond {
		t.Errorf("unloaded %v after the change", took)
	}
	if n := len(f.records("shutdown")); n != 2 {
		t.Errorf("%d shutdowns", n)
	}

	use()
	f.rt.SetIdleTimeout(-1)
	time.Sleep(500 * time.Millisecond)
	if r := f.rt.Running(); len(r) != 2 {
		t.Errorf("never: running %v", r)
	}
	f.rt.SetIdleTimeout(0)
	if d := f.rt.IdleTimeout(); d != DefaultIdleTimeout {
		t.Errorf("0 is %v", d)
	}
}

// TestSetIdleTimeoutInUse: a server in use when the time changes is
// stopped once unused for the new time after its use ends.
func TestSetIdleTimeoutInUse(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: 400 * time.Millisecond}, func(c *Config) { c.IdleTimeout = -1 })
	a := f.asr(t, ASRFast)
	done := make(chan error, 1)
	go func() {
		_, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3)})
		done <- err
	}()
	eventually(t, "the Final", func() bool { return len(f.records("final")) == 1 })
	f.rt.SetIdleTimeout(100 * time.Millisecond)
	if err := <-done; err != nil {
		t.Fatalf("the use in progress failed: %v", err)
	}
	ended := time.Now()
	eventually(t, "the idle unload", func() bool { return len(f.rt.Running()) == 0 })
	if took := time.Since(ended); took < 100*time.Millisecond {
		t.Errorf("unloaded %v after the use", took)
	}
}

// TestSwitch: Silero beside a speech model, then a switch to the other
// speech model: the first stops before the other starts; Silero stays.
// Close stops everything.
func TestSwitch(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	ctx := context.Background()
	pcm := tone(time.Second, 0.3)
	fast := f.asr(t, ASRFast)
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := v.Speech(ctx, pcm, 16000); err != nil {
		t.Fatal(err)
	}
	if _, err := fast.Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
		t.Fatal(err)
	}
	before := len(f.records("shutdown"))
	if _, err := f.asr(t, ASRAccurate).Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
		t.Fatal(err)
	}
	if n := len(f.records("shutdown")); n != before+1 {
		t.Errorf("%d shutdowns at the switch", n-before)
	}
	r := f.rt.Running()
	slices.Sort(r)
	if !slices.Equal(r, []string{ASRAccurate, VADSilero}) {
		t.Errorf("running %v", r)
	}
	pids := []int{}
	for _, l := range f.records("launch") {
		pids = append(pids, l.PID)
	}
	f.rt.Close()
	for _, pid := range pids {
		if !gone(pid) {
			t.Errorf("pid %d still runs after Close", pid)
		}
	}
	if _, err := fast.Transcribe(ctx, asr.FileRequest{Audio: pcm}); err == nil || !errors.Is(err, errClosed) {
		t.Errorf("a use after Close: %v", err)
	}
}

// TestConcurrentStart: uses that come while the server starts share one
// start.
func TestConcurrentStart(t *testing.T) {
	f := newFake(t, omnitest.Opts{Load: 300 * time.Millisecond}, nil)
	a := f.asr(t, ASRFast)
	a.Warm()
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3)}); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if n := len(f.records("launch")); n != 1 {
		t.Errorf("%d launches", n)
	}
}

// TestUnload: Unload stops a model's server, after the use in progress;
// the next use starts it again.
func TestUnload(t *testing.T) {
	f := newFake(t, omnitest.Opts{Delay: 400 * time.Millisecond}, nil)
	a := f.asr(t, ASRFast)
	done := make(chan error, 1)
	go func() {
		_, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3)})
		done <- err
	}()
	eventually(t, "the Final", func() bool { return len(f.records("final")) == 1 })
	f.rt.Unload(ASRFast)
	f.rt.Unload("nothing")
	if err := <-done; err != nil {
		t.Fatalf("the use in progress failed: %v", err)
	}
	eventually(t, "the shutdown", func() bool { return len(f.records("shutdown")) == 1 })
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3)}); err != nil {
		t.Fatal(err)
	}
	if n := len(f.records("launch")); n != 2 {
		t.Errorf("%d launches", n)
	}
}

func TestWithVAD(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	a := f.asr(t, ASRFast)
	if a.VAD() != nil {
		t.Error("a VAD without one")
	}
	v, err := NewVAD(f.rt, f.model(VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	if b := a.WithVAD(v); b.VAD() == nil || a.VAD() != nil {
		t.Error("WithVAD")
	}
}
