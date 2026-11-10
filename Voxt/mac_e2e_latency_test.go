package main

// The first transcription's start, timed on a Mac through the app itself
// (its shortcut handler, sessions, Library and Omni runtime) with the
// real speech models; only the helper is the fake one, playing the speech
// sample in real time (no microphone, window or key). VOXTNEW_LATENCY=1,
// VOXTNEW_OMNI_RUNTIME (a runtime, or Voxt.app's Contents/Resources/omni)
// and VOXTNEW_OMNI_MODELS ("id=folder;…": two speech models and
// vad.silero) turn it on. Each of VOXTNEW_LATENCY_ROUNDS (3) rounds opens
// an app and runs three live Transcribe sessions:
//
//	cold    the first session, VOXTNEW_LATENCY_SETTLE (3s) after the app was ready
//	warm    the same model again (the baseline)
//	switch  the first session after Transcribe's speech model changed, the same wait after the change
//
// MEASURE lines give each session's stages from the shortcut, then each
// scenario's median and range (scripts/mac-latency.sh runs it).
// VOXTNEW_LATENCY_PCM names another recording to play and
// VOXTNEW_LATENCY_SPEAK (2s) how long a session records after its first
// live text (long sessions: memory, the live text's pace, the Final):
//
//	VOXTNEW_LATENCY=1 VOXTNEW_OMNI_RUNTIME=… VOXTNEW_OMNI_MODELS="asr.accurate=…;asr.fast=…;vad.silero=…" \
//	  go test -count=1 -run TestMacE2EFirstLatency -v -timeout 30m .

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// latencyStages are a session's times from the shortcut. ready is when
// the speech model's server was ready: negative when it was before the
// shortcut (warmed), zero when it ran already before this scenario.
type latencyStages struct {
	heard, ready, load, partial, final time.Duration
}

// readyLine is the runtime's line about a server that became ready.
var readyLine = regexp.MustCompile(`omni: (\S+) ready on \S+ after (\S+) \(pid`)

// latencyLog keeps when each model's server became ready and how long
// its start took.
type latencyLog struct {
	mu    sync.Mutex
	ready map[string][]readyAt
}

type readyAt struct {
	at   time.Time
	load time.Duration
}

func (l *latencyLog) add(msg string) {
	m := readyLine.FindStringSubmatch(msg)
	if m == nil {
		return
	}
	load, _ := time.ParseDuration(m[2])
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.ready == nil {
		l.ready = map[string][]readyAt{}
	}
	l.ready[m[1]] = append(l.ready[m[1]], readyAt{time.Now(), load})
}

// since is the newest start of model id's server after t.
func (l *latencyLog) since(id string, t time.Time) (readyAt, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	r := l.ready[id]
	if len(r) == 0 || r[len(r)-1].at.Before(t) {
		return readyAt{}, false
	}
	return r[len(r)-1], true
}

// latencyHub is a model library with the models of VOXTNEW_OMNI_MODELS
// installed: links to the folders' files, laid out as the Hugging Face
// cache. It returns the speech models' ids, in order.
func latencyHub(t *testing.T) (hub string, ids []string) {
	t.Helper()
	hub = filepath.Join(t.TempDir(), "hub")
	for _, item := range strings.Split(os.Getenv("VOXTNEW_OMNI_MODELS"), ";") {
		id, dir, ok := strings.Cut(strings.TrimSpace(item), "=")
		m, known := models.Lookup(id)
		if !ok || !known {
			continue
		}
		repo := filepath.Join(hub, "models--"+strings.ReplaceAll(m.Repo, "/", "--"))
		snap := filepath.Join(repo, "snapshots", m.Revision)
		if err := os.MkdirAll(filepath.Join(repo, "blobs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(snap, 0o755); err != nil {
			t.Fatal(err)
		}
		for _, f := range m.Files {
			src, err := filepath.EvalSymlinks(filepath.Join(dir, f.Path))
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}
			if err := os.Symlink(src, filepath.Join(repo, "blobs", f.SHA256)); err != nil && !os.IsExist(err) {
				t.Fatal(err)
			}
			if err := os.Symlink("../../blobs/"+f.SHA256, filepath.Join(snap, f.Path)); err != nil {
				t.Fatal(err)
			}
		}
		if m.Kind == models.ASR {
			ids = append(ids, id)
		}
	}
	return hub, ids
}

func TestMacE2EFirstLatency(t *testing.T) {
	if os.Getenv("VOXTNEW_LATENCY") != "1" || runtime.GOOS != "darwin" {
		t.Skip("set VOXTNEW_LATENCY=1 on a Mac (scripts/mac-latency.sh)")
	}
	hub, ids := latencyHub(t)
	if len(ids) < 2 {
		t.Skip("set VOXTNEW_OMNI_MODELS to two speech models (and vad.silero)")
	}
	rounds, settle := 3, 3*time.Second
	if n, err := strconv.Atoi(os.Getenv("VOXTNEW_LATENCY_ROUNDS")); err == nil && n > 0 {
		rounds = n
	}
	if d, err := time.ParseDuration(os.Getenv("VOXTNEW_LATENCY_SETTLE")); err == nil {
		settle = d
	}
	lg := &latencyLog{}
	old := omniNewRuntime
	omniNewRuntime = func(cfg omni.Config) *omni.Runtime {
		log := cfg.Log
		cfg.Log = func(level, msg string) {
			lg.add(msg)
			t.Logf("%s %s", level, msg)
			log(level, msg)
		}
		return old(cfg)
	}
	defer func() { omniNewRuntime = old }()

	got := map[string][]latencyStages{}
	for round := 1; round <= rounds; round++ {
		t.Run(fmt.Sprintf("round%d", round), func(t *testing.T) {
			opened := time.Now()
			a := newSessionApp(t, "frame-ms=100,"+helpertest.Opt("pcm", latencyAudio()), func(s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
				s.Features.Transcribe.ASR = ids[0]
				s.General.MainLanguage = "zh"
			}, realLocal{hub: hub})
			a.pumpFor(30*time.Second, func() bool {
				_, ok := a.featureModel(local.ASR, "transcribe")
				return ok && a.helperReady()
			})
			a.pumpFor(settle, func() bool { return false })
			got["cold"] = append(got["cold"], latencySession(t, a, lg, opened, "cold", ids[0]))
			got["warm"] = append(got["warm"], latencySession(t, a, lg, time.Now(), "warm", ids[0]))
			changed := time.Now()
			a.updateSettings(func(s *settings.Settings) { setFeatureModel(s, local.ASR, "transcribe", ids[1]) })
			a.pumpFor(settle, func() bool { return false })
			got["switch"] = append(got["switch"], latencySession(t, a, lg, changed, "switch", ids[1]))
		})
	}
	for _, name := range []string{"cold", "warm", "switch"} {
		t.Logf("MEASURE latency %-6s n=%d settle %v:  first audio %s  server ready %s (its start %s)  first partial %s  final after Stop %s", name, len(got[name]), settle,
			spread(got[name], func(s latencyStages) time.Duration { return s.heard }),
			spread(got[name], func(s latencyStages) time.Duration { return s.ready }),
			spread(got[name], func(s latencyStages) time.Duration { return s.load }),
			spread(got[name], func(s latencyStages) time.Duration { return s.partial }),
			spread(got[name], func(s latencyStages) time.Duration { return s.final }))
	}
}

// pumpFor runs posted functions until cond holds, at most d.
func (s *sessionApp) pumpFor(d time.Duration, cond func() bool) bool {
	deadline := time.After(d)
	for !cond() {
		select {
		case fn := <-s.queue:
			fn()
		case <-deadline:
			return false
		}
	}
	return true
}

// spread is "median (min–max)" of a stage, "-" when no session had it.
func spread(all []latencyStages, get func(latencyStages) time.Duration) string {
	var ds []time.Duration
	for _, s := range all {
		if d := get(s); d != 0 {
			ds = append(ds, d)
		}
	}
	if len(ds) == 0 {
		return "-"
	}
	slices.Sort(ds)
	ms := func(d time.Duration) string { return d.Round(time.Millisecond).String() }
	return fmt.Sprintf("%s (%s–%s)", ms(ds[len(ds)/2]), ms(ds[0]), ms(ds[len(ds)-1]))
}

// latencySession presses Transcribe's shortcut, stops two seconds after
// the first live text and returns the session's stages; from is when the
// scenario began (the app opened, the model changed): a server ready
// since then counts as this scenario's.
func latencySession(t *testing.T, a *sessionApp, lg *latencyLog, from time.Time, name, id string) latencyStages {
	t.Helper()
	var st latencyStages
	last := a.sess.ID
	t0 := time.Now()
	a.toggleTranscribe()
	if !a.pumpFor(time.Minute, func() bool {
		if a.sess.ID == last {
			return false
		}
		if st.heard == 0 && !a.sess.HeardAt.IsZero() {
			st.heard = time.Since(t0)
		}
		return a.sess.State.Ended() || (a.sess.State == session.Recording && a.sess.Partial != "")
	}) || a.sess.State.Ended() {
		t.Fatalf("%s: no live text (session %s %+v)", name, a.sess.State, a.sess.Problem)
	}
	st.partial = time.Since(t0)
	// The live text's updates while the user speaks: their count and the longest wait between two.
	speak, updates, gap, seen, at := 2*time.Second, 0, time.Duration(0), a.sess.Partial, time.Now()
	if d, err := time.ParseDuration(os.Getenv("VOXTNEW_LATENCY_SPEAK")); err == nil {
		speak = d
	}
	a.pumpFor(speak, func() bool {
		if a.sess.Partial != seen {
			seen, updates, gap, at = a.sess.Partial, updates+1, max(gap, time.Since(at)), time.Now()
		}
		return a.sess.State.Ended()
	})
	stopped := time.Now()
	a.stopSession()
	if !a.pumpFor(time.Minute, func() bool { return a.sess.State.Ended() }) || a.sess.State != session.Done || a.sess.ASR.ID != id {
		t.Fatalf("%s: end %s %+v on %s", name, a.sess.State, a.sess.Problem, a.sess.ASR.ID)
	}
	st.final = time.Since(stopped)
	if r, ok := lg.since(id, from); ok {
		st.ready, st.load = r.at.Sub(t0), r.load
	}
	t.Logf("MEASURE latency %s %s: first audio %v, server ready %v (its start %v), first partial %v, %d live updates in %v (longest wait %v), final %v after Stop: %d characters %q", name, id,
		st.heard.Round(time.Millisecond), st.ready.Round(time.Millisecond), st.load.Round(time.Millisecond),
		st.partial.Round(time.Millisecond), updates, speak, gap.Round(time.Millisecond), st.final.Round(time.Millisecond), len([]rune(a.sess.Raw)), firstRunes(a.sess.Raw, 40))
	return st
}

// latencyAudio is the recording the fake helper plays:
// VOXTNEW_LATENCY_PCM, else the speech sample.
func latencyAudio() string {
	if p := os.Getenv("VOXTNEW_LATENCY_PCM"); p != "" {
		return p
	}
	return speechSample()
}

func firstRunes(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}
