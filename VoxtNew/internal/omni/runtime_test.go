package omni

import (
	"cmp"
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
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
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
	// lines gets a token once a line was added (cap 1): waitLogged
	// looks again.
	lines chan struct{}
}

func newFake(t *testing.T, o omnitest.Opts, tune func(*Config)) *fake {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f := &fake{dir: t.TempDir(), log: filepath.Join(t.TempDir(), "fake.jsonl"), lines: make(chan struct{}, 1)}
	o.Log = f.log
	for _, s := range Models() {
		if err := os.MkdirAll(filepath.Join(f.dir, s.ID), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	cfg := Config{Binary: exe, Env: omnitest.Env(o), StartTimeout: 10 * time.Second, Log: f.add}
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

// add is the runtime's Config.Log.
func (f *fake) add(level, msg string) {
	f.mu.Lock()
	f.out = append(f.out, level+" "+msg)
	f.mu.Unlock()
	select {
	case f.lines <- struct{}{}:
	default:
	}
}

func (f *fake) logged(sub string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.ContainsFunc(f.out, func(l string) bool { return strings.Contains(l, sub) })
}

// waitLogged waits until the runtime logged a line with sub.
func (f *fake) waitLogged(t *testing.T, sub string) {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for !f.logged(sub) {
		select {
		case <-f.lines:
		case <-timeout:
			t.Fatalf("no log line with %q", sub)
		}
	}
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

// unloads makes c send the id of each model whose process exited
// (resident.Unloaded) on the channel it returns.
func unloads(c *Config) <-chan string {
	ch := make(chan string, 16)
	c.OnModel = func(ev resident.Event) {
		if ev.Kind == resident.Unloaded {
			ch <- ev.ID
		}
	}
	return ch
}

// waitUnloaded waits until model id's process exited.
func waitUnloaded(t *testing.T, unloaded <-chan string, id string) {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for got := ""; got != id; {
		select {
		case got = <-unloaded:
		case <-timeout:
			t.Fatalf("%s's process did not exit", id)
		}
	}
}

// fakeClock stands for the idle timers' clock: tests move it on by hand,
// and the timers due run in the caller, in order.
type fakeClock struct {
	mu     sync.Mutex
	now    time.Duration
	timers []*fakeTimer
}

type fakeTimer struct {
	at   time.Duration
	fn   func()
	done bool // stopped or run
}

// install makes rt's idle timers run on c.
func (c *fakeClock) install(rt *Runtime) *fakeClock {
	rt.afterFunc = func(d time.Duration, fn func()) func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		t := &fakeTimer{at: c.now + d, fn: fn}
		c.timers = append(c.timers, t)
		return func() {
			c.mu.Lock()
			t.done = true
			c.mu.Unlock()
		}
	}
	return c
}

// advance moves the clock on by d and runs the timers then due.
func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	c.now += d
	var due []*fakeTimer
	for _, t := range c.timers {
		if !t.done && t.at <= c.now {
			t.done = true
			due = append(due, t)
		}
	}
	c.mu.Unlock()
	slices.SortStableFunc(due, func(a, b *fakeTimer) int { return cmp.Compare(a.at, b.at) })
	for _, t := range due {
		t.fn()
	}
}

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

// TestLanguageHintLogged: the log says which language hint each request
// carries, and never the recognised text. A forced English hint
// translated Chinese dictation (#3398), and nothing in the log showed
// what hint was sent.
func TestLanguageHintLogged(t *testing.T) {
	for _, c := range []struct {
		name string
		lang string
		want string
	}{
		{"hint", "zh-Hans", `language hint "zh"`},
		{"detect", "", `language hint ""`},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFake(t, omnitest.Opts{Text: "识别出的文字"}, nil)
			a := f.asr(t, ASRFast)
			pcm := tone(time.Second, 0.3)
			if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", Lang: c.lang}); err != nil {
				t.Fatal(err)
			}
			st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: 16000, Lang: c.lang})
			if err != nil {
				t.Fatal(err)
			}
			if err := st.Send(pcm); err != nil {
				t.Fatal(err)
			}
			if err := st.Finish(); err != nil {
				t.Fatal(err)
			}
			drain(t, st)
			f.mu.Lock()
			logs := slices.Clone(f.out)
			f.mu.Unlock()
			for _, want := range []string{"Final request with " + c.want, "live session with " + c.want} {
				if !slices.ContainsFunc(logs, func(l string) bool { return strings.Contains(l, want) }) {
					t.Errorf("no log of %q; logs %q", want, logs)
				}
			}
			if slices.ContainsFunc(logs, func(l string) bool { return strings.Contains(l, "识别出的文字") }) {
				t.Errorf("the log holds the recognised text: %q", logs)
			}
		})
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
		var unloaded <-chan string
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeStall}, func(c *Config) {
			c.StallTimeout = 300 * time.Millisecond
			unloaded = unloads(c)
		})
		_, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm})
		if asr.KindOf(err) != asr.KindServer || !strings.Contains(err.Error(), "stopped answering") {
			t.Errorf("%v", err)
		}
		pid := f.records("launch")[0].PID
		waitUnloaded(t, unloaded, ASRFast)
		if !gone(pid) {
			t.Error("the stalled server still runs")
		}
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
// back to never, they stay.
func TestSetIdleTimeout(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = -1 })
	clock := new(fakeClock).install(f.rt)
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
	clock.advance(time.Hour)
	if r := f.rt.Running(); len(r) != 2 {
		t.Fatalf("kept: running %v", r)
	}
	f.rt.SetIdleTimeout(250 * time.Millisecond)
	if d := f.rt.IdleTimeout(); d != 250*time.Millisecond {
		t.Errorf("IdleTimeout %v", d)
	}
	clock.advance(249 * time.Millisecond)
	if r := f.rt.Running(); len(r) != 2 {
		t.Fatalf("stopped before the new time: running %v", r)
	}
	clock.advance(time.Millisecond) // the idle stop waits for the process to exit
	if r := f.rt.Running(); len(r) != 0 {
		t.Fatalf("not stopped at the new time: running %v", r)
	}
	if n := len(f.records("shutdown")); n != 2 {
		t.Errorf("%d shutdowns", n)
	}

	use()
	f.rt.SetIdleTimeout(-1)
	clock.advance(time.Hour)
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
	var unloaded <-chan string
	f := newFake(t, omnitest.Opts{}, func(c *Config) { unloaded = unloads(c) })
	ctx := context.Background()
	a := f.asr(t, ASRFast)
	l, err := f.rt.acquire(ctx, a.spec, a.dir) // the use in progress, as Transcribe holds it
	if err != nil {
		t.Fatal(err)
	}
	f.rt.Unload(ASRFast)
	f.rt.Unload("nothing")
	if _, _, err := a.final(ctx, l, tone(time.Second, 0.3), "", bias{}); err != nil {
		t.Fatalf("the use in progress failed: %v", err)
	}
	l.release()
	waitUnloaded(t, unloaded, ASRFast)
	if n := len(f.records("shutdown")); n != 1 {
		t.Errorf("%d shutdowns after the use", n)
	}
	if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: tone(time.Second, 0.3)}); err != nil {
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
