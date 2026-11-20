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
// live text (long sessions: memory, the live text's pace, the Final).
// VOXTNEW_LATENCY_OUT appends one JSON object per session with separate
// ASR Final, helper-paste and whole-session completion times and a text hash.
// VOXTNEW_LATENCY_ATTEMPT and VOXTNEW_LATENCY_ARM label a run when several
// independent processes each use VOXTNEW_LATENCY_ROUNDS=1 for an A/B comparison.
// The native preview timing event supplies its actual min_preview_samples:
//
//	VOXTNEW_LATENCY=1 VOXTNEW_OMNI_RUNTIME=… VOXTNEW_OMNI_MODELS="asr.accurate=…;asr.fast=…;vad.silero=…" \
//	  go test -count=1 -run TestMacE2EFirstLatency -v -timeout 30m .

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
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
	appReady, heard, ready, load, partial, final, insert, done time.Duration
	minPreviewSamples                                          int
}

type latencyResult struct {
	MainSHA            string  `json:"main_sha"`
	Scenario           string  `json:"scenario"`
	Round              int     `json:"round"`
	Attempt            int     `json:"attempt"`
	Arm                string  `json:"arm"`
	MinPreviewSamples  int     `json:"min_preview_samples"`
	Model              string  `json:"model"`
	ASRCold            bool    `json:"asr_cold"`
	VADCold            bool    `json:"vad_cold"`
	AppReadyMS         float64 `json:"app_ready_ms"`
	FirstAudioMS       float64 `json:"first_audio_ms"`
	ServerReadyMS      float64 `json:"server_ready_ms"`
	ModelLoadMS        float64 `json:"model_load_ms"`
	FirstPartialMS     float64 `json:"first_partial_ms"`
	StopToFinalMS      float64 `json:"stop_to_final_ms"`
	StopToInsertMS     float64 `json:"stop_to_insert_ms"`
	StopToDoneMS       float64 `json:"stop_to_done_ms"`
	LiveUpdates        int     `json:"live_updates"`
	LongestUpdateGapMS float64 `json:"longest_update_gap_ms"`
	Text               string  `json:"text"`
	TextSHA256         string  `json:"text_sha256"`
}

// readyLine is the runtime's line about a server that became ready.
var readyLine = regexp.MustCompile(`omni: (\S+) ready on \S+ after (\S+) \(pid`)

var minPreviewLine = regexp.MustCompile(`"min_preview_samples"\s*:\s*(\d+)`)

// latencyLog keeps when each model's server became ready and how long
// its start took.
type latencyLog struct {
	mu                sync.Mutex
	ready             map[string][]readyAt
	minPreviewSamples int
}

type readyAt struct {
	at   time.Time
	load time.Duration
}

func (l *latencyLog) add(msg string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if m := minPreviewLine.FindStringSubmatch(msg); m != nil {
		l.minPreviewSamples, _ = strconv.Atoi(m[1])
	}
	m := readyLine.FindStringSubmatch(msg)
	if m == nil {
		return
	}
	load, _ := time.ParseDuration(m[2])
	if l.ready == nil {
		l.ready = map[string][]readyAt{}
	}
	l.ready[m[1]] = append(l.ready[m[1]], readyAt{time.Now(), load})
}

func (l *latencyLog) minPreview() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.minPreviewSamples
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
	attempt, _ := strconv.Atoi(os.Getenv("VOXTNEW_LATENCY_ATTEMPT"))
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
		if attempt > 0 {
			round = attempt
		}
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
			appReady := time.Now()
			a.pumpFor(settle, func() bool { return false })
			got["cold"] = append(got["cold"], latencySession(t, a, lg, opened, appReady, "cold", ids[0], round))
			got["warm"] = append(got["warm"], latencySession(t, a, lg, time.Now(), time.Time{}, "warm", ids[0], round))
			changed := time.Now()
			a.updateSettings(func(s *settings.Settings) { setFeatureModel(s, local.ASR, "transcribe", ids[1]) })
			a.pumpFor(settle, func() bool { return false })
			got["switch"] = append(got["switch"], latencySession(t, a, lg, changed, time.Time{}, "switch", ids[1], round))
		})
	}
	for _, name := range []string{"cold", "warm", "switch"} {
		t.Logf("MEASURE latency %-6s n=%d settle %v:  first audio %s  server ready %s (its start %s)  first partial %s  Final after Stop %s  insert after Stop %s  done after Stop %s", name, len(got[name]), settle,
			spread(got[name], func(s latencyStages) time.Duration { return s.heard }),
			spread(got[name], func(s latencyStages) time.Duration { return s.ready }),
			spread(got[name], func(s latencyStages) time.Duration { return s.load }),
			spread(got[name], func(s latencyStages) time.Duration { return s.partial }),
			spread(got[name], func(s latencyStages) time.Duration { return s.final }),
			spread(got[name], func(s latencyStages) time.Duration { return s.insert }),
			spread(got[name], func(s latencyStages) time.Duration { return s.done }))
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
func latencySession(t *testing.T, a *sessionApp, lg *latencyLog, from, readyAt time.Time, name, id string, round int) latencyStages {
	t.Helper()
	var st latencyStages
	if !readyAt.IsZero() {
		st.appReady = readyAt.Sub(from)
	}
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
	pastesBefore := len(a.pasted())
	stopped := time.Now()
	a.stopSession()
	if !a.pumpFor(time.Minute, func() bool {
		if st.final == 0 && a.sess.Raw != "" {
			st.final = time.Since(stopped)
		}
		if st.insert == 0 && len(a.pasted()) > pastesBefore {
			st.insert = time.Since(stopped)
		}
		return a.sess.State.Ended()
	}) || a.sess.State != session.Done || a.sess.ASR.ID != id {
		t.Fatalf("%s: end %s %+v on %s", name, a.sess.State, a.sess.Problem, a.sess.ASR.ID)
	}
	st.done = time.Since(stopped)
	if st.final == 0 {
		t.Fatalf("%s: no ASR Final was observed before delivery", name)
	}
	if st.insert == 0 {
		t.Fatalf("%s: no helper paste was observed after the ASR Final", name)
	}
	if r, ok := lg.since(id, from); ok {
		st.ready, st.load = r.at.Sub(t0), r.load
	}
	t.Logf("MEASURE latency %s %s: first audio %v, server ready %v (its start %v), first partial %v, %d live updates in %v (longest wait %v), Final %v / insert %v / done %v after Stop: %d characters %q", name, id,
		st.heard.Round(time.Millisecond), st.ready.Round(time.Millisecond), st.load.Round(time.Millisecond),
		st.partial.Round(time.Millisecond), updates, speak, gap.Round(time.Millisecond), st.final.Round(time.Millisecond), st.insert.Round(time.Millisecond), st.done.Round(time.Millisecond), len([]rune(a.sess.Raw)), firstRunes(a.sess.Raw, 40))
	st.minPreviewSamples = lg.minPreview()
	writeLatencyResult(t, name, id, round, st, updates, gap, a.sess.Raw)
	return st
}

func writeLatencyResult(t *testing.T, name, id string, round int, st latencyStages, updates int, gap time.Duration, text string) {
	t.Helper()
	path := os.Getenv("VOXTNEW_LATENCY_OUT")
	if path == "" {
		return
	}
	milliseconds := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
	result := latencyResult{
		MainSHA: latencyMainSHA(), Scenario: name, Round: round, Attempt: round,
		Arm: os.Getenv("VOXTNEW_LATENCY_ARM"), MinPreviewSamples: st.minPreviewSamples, Model: id,
		ASRCold: name != "warm", VADCold: name == "cold",
		AppReadyMS:   milliseconds(st.appReady),
		FirstAudioMS: milliseconds(st.heard), ServerReadyMS: milliseconds(st.ready), ModelLoadMS: milliseconds(st.load),
		FirstPartialMS: milliseconds(st.partial), StopToFinalMS: milliseconds(st.final),
		StopToInsertMS: milliseconds(st.insert), StopToDoneMS: milliseconds(st.done),
		LiveUpdates: updates, LongestUpdateGapMS: milliseconds(gap), Text: text,
		TextSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(text))),
	}
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(fh).Encode(result); err != nil {
		fh.Close()
		t.Fatal(err)
	}
	if err := fh.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLatencyResultJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "latency.jsonl")
	t.Setenv("VOXTNEW_LATENCY_OUT", path)
	t.Setenv("VOXTNEW_LATENCY_ARM", "A")
	t.Setenv("VOXTNEW_MAIN_SHA", "49f60636411e24ce4f7441878c9333d68e649a83")
	stages := latencyStages{appReady: 1500 * time.Millisecond, final: 1200 * time.Millisecond, insert: 1250 * time.Millisecond, done: 1300 * time.Millisecond, minPreviewSamples: 20000}
	writeLatencyResult(t, "cold", "asr.test", 2, stages, 4, 275*time.Millisecond, "fixed text")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got latencyResult
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.AppReadyMS != 1500 || got.StopToFinalMS != 1200 || got.StopToInsertMS != 1250 || got.StopToDoneMS != 1300 {
		t.Fatalf("app readiness or the three endpoints collapsed: %+v", got)
	}
	if !got.ASRCold || !got.VADCold || got.Round != 2 || got.LiveUpdates != 4 || got.LongestUpdateGapMS != 275 {
		t.Fatalf("scenario metadata: %+v", got)
	}
	if got.MainSHA != "49f60636411e24ce4f7441878c9333d68e649a83" || got.Attempt != 2 || got.Arm != "A" || got.MinPreviewSamples != 20000 {
		t.Fatalf("run identity: %+v", got)
	}
	if got.TextSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte("fixed text"))) {
		t.Fatalf("text hash: %+v", got)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Errorf("latency output mode = %o, want 600", info.Mode().Perm())
	}
}

func TestMacLatencyScriptRequiresMainSHA(t *testing.T) {
	script, err := os.ReadFile("scripts/mac-latency.sh")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(script), `VOXTNEW_MAIN_SHA="${VOXTNEW_MAIN_SHA:?`) {
		t.Fatal("the latency runner does not require an explicit source SHA")
	}
}

func TestLatencyLogReadsNativePreviewConfig(t *testing.T) {
	var log latencyLog
	log.add(`omni: native {"event":"moss_preview_timing","min_preview_samples":12000}`)
	if got := log.minPreview(); got != 12000 {
		t.Fatalf("min_preview_samples = %d, want 12000", got)
	}
}

func latencyMainSHA() string {
	// The runner passes the source SHA explicitly: test binaries built
	// with -buildvcs=false or without VCS stamps would else report
	// "unknown" and break the measurement provenance gate.
	if sha := os.Getenv("VOXTNEW_MAIN_SHA"); sha != "" {
		return sha
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			return setting.Value
		}
	}
	return "unknown"
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
