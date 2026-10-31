package main

import (
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/hehehai/actnowkit/sounds"
)

const testSpeakers = "BuiltInSpeakerDevice"

// muteRig is the test app with the mute's goroutine kept off the app:
// what it posts waits in a queue the test runs (settle), and the cues
// say whether the speakers were muted as each played.
type muteRig struct {
	*app
	fs    *fakeSupervisor
	ft    *fakeTimers
	mu    sync.Mutex
	queue []func()
	cues  []string
}

func newMuteRig(t *testing.T) *muteRig {
	a, fs, _ := newTestApp(t)
	r := &muteRig{app: a, fs: fs}
	r.ft = new(fakeTimers).install(a)
	a.postQueue = func(fn func()) { r.mu.Lock(); r.queue = append(r.queue, fn); r.mu.Unlock() }
	a.playSound = func(s sounds.Target) {
		state := "audible"
		if fs.out.Sys.Muted(testSpeakers) {
			state = "muted"
		}
		r.mu.Lock()
		r.cues = append(r.cues, s.Name+":"+state)
		r.mu.Unlock()
	}
	return r
}

// settle runs what the mute posted until nothing more comes.
func (r *muteRig) settle() {
	for range 10 {
		r.mute.Wait()
		r.mu.Lock()
		q := r.queue
		r.queue = nil
		r.mu.Unlock()
		if len(q) == 0 {
			return
		}
		for _, fn := range q {
			fn()
		}
	}
}

func (r *muteRig) feed(id string, st session.State, try ...bool) {
	r.sessionChanged(session.Snapshot{Rev: r.sess.Rev + 1, ID: id, Kind: store.KindTranscribe, State: st, Target: slackApp, Try: len(try) > 0})
	r.settle()
}

func (r *muteRig) muted() bool { r.settle(); return r.fs.out.Sys.Muted(testSpeakers) }

func (r *muteRig) takeCues() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	c := r.cues
	r.cues = nil
	return c
}

// TestRecordingMutesOutput: the output is muted once the start cue
// played and back before the stop cue; a cancel, a failure and a new
// session give it back too; off in Settings → Audio, nothing is touched.
func TestRecordingMutesOutput(t *testing.T) {
	r := newMuteRig(t)
	if !r.prefs.Audio.MuteOn() {
		t.Fatal("off by default")
	}
	r.feed("s1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted while recording")
	}
	r.feed("s1", session.Recognizing)
	if r.muted() {
		t.Fatal("still muted after the recording")
	}
	if got := r.takeCues(); !slices.Equal(got, []string{"tally:audible", "wrap:audible"}) {
		t.Fatalf("cues %v: the start cue before the mute, the stop cue after it", got)
	}
	for _, end := range []session.State{session.Cancelled, session.Failed} {
		r.feed("s-"+string(end), session.Recording)
		if !r.muted() {
			t.Fatalf("%s: not muted", end)
		}
		r.feed("s-"+string(end), end)
		if r.muted() {
			t.Fatalf("%s: still muted", end)
		}
	}
	// A new session while one records (a missed end): the old one's goes.
	r.feed("s4", session.Recording)
	r.feed("s5", session.Recording)
	r.feed("s5", session.Done)
	if r.muted() {
		t.Fatal("muted after two sessions")
	}
	// Translate reads first, then records.
	r.feed("t1", session.Reading)
	if r.muted() {
		t.Fatal("muted while reading the selection")
	}
	r.feed("t1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted once Translate records")
	}
	r.feed("t1", session.Composing)
	if r.muted() {
		t.Fatal("muted while composing")
	}
	// Off: the output is left alone; turned off while recording, it
	// comes back at once.
	r.updateSettings(func(s *settings.Settings) { off := false; s.Audio.MuteOutput = &off })
	r.feed("s6", session.Recording)
	if r.muted() {
		t.Fatal("muted while off")
	}
	r.feed("s6", session.Done)
	r.updateSettings(func(s *settings.Settings) { s.Audio.MuteOutput = nil })
	r.feed("s7", session.Recording)
	if !r.muted() {
		t.Fatal("not muted once on again")
	}
	r.settingsForm.muteOutput = false
	r.updateSettings(func(s *settings.Settings) { off := false; s.Audio.MuteOutput = &off })
	r.muteSettingChanged(false)
	if r.muted() {
		t.Fatal("still muted after turning it off")
	}
	r.feed("s7", session.Done)
	if w := r.fs.out.Sys.WriteLog(); len(w) != 14 || w[len(w)-1] != testSpeakers+":false" {
		t.Errorf("writes %v", w)
	}
}

// TestRecordingMuteWithoutSounds: no cue, so the mute comes at once; a
// session that stopped before the start cue ended is not muted after.
func TestRecordingMuteWithoutSounds(t *testing.T) {
	r := newMuteRig(t)
	r.updateSettings(func(s *settings.Settings) { s.Audio.Sounds = false })
	r.feed("s1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted")
	}
	r.feed("s1", session.Recognizing)
	if r.muted() || len(r.takeCues()) != 0 {
		t.Fatal("muted, or a cue")
	}
	// A start cue that does not end mutes once its audible part played
	// (startCueWait: Tally's 0.35 s, not its file's 0.45 s); one that ends
	// after the recording stopped mutes nothing (its engage is late).
	r.updateSettings(func(s *settings.Settings) { s.Audio.Sounds = true })
	block := make(chan struct{})
	defer close(block)
	r.playSound = func(sounds.Target) { <-block }
	t0 := time.Now()
	r.sessionChanged(session.Snapshot{Rev: r.sess.Rev + 1, ID: "s2", Kind: store.KindTranscribe, State: session.Recording, Target: slackApp})
	if d := time.Since(t0); d < r.startCueWait(r.cueTarget(startCue)) || d >= startCueMax {
		t.Fatalf("the engage did not wait for the cue's audible part: %v", d)
	}
	r.sessionChanged(session.Snapshot{Rev: r.sess.Rev + 1, ID: "s2", Kind: store.KindTranscribe, State: session.Cancelled, Target: slackApp})
	if r.muted() {
		t.Fatal("muted after the session ended")
	}
	r.sessionChanged(session.Snapshot{Rev: r.sess.Rev + 1, ID: "s3", Kind: store.KindTranscribe, State: session.Recording, Target: slackApp})
	if !r.muted() {
		t.Fatal("a cue that never ends kept the output on")
	}
}

// TestRecordingCannotMute: an output without a mute it can set leaves
// the recording as it is, and the pill says so on its text line for 3 s
// (not for a try, whose page shows no pill).
func TestRecordingCannotMute(t *testing.T) {
	r := newMuteRig(t)
	r.fs.out.Sys.SetDefault(helpertest.Display)
	r.feed("s1", session.Recording)
	want := "This output device cannot be muted. Recording continues."
	if got := r.pillSession().Partial; got != want || r.ov.mode != ovRecording || r.ov.h != layout.LiveH {
		t.Fatalf("pill %q, mode %s, height %d", got, r.ov.mode, r.ov.h)
	}
	r.ft.advance(muteNoteTime)
	if r.pillSession().Partial != "" || r.ov.h != layout.PillH || r.ov.w != layout.PillW {
		t.Fatalf("the note stays: %q %dx%d", r.pillSession().Partial, r.ov.w, r.ov.h)
	}
	r.feed("s1", session.Recognizing)
	// Stopped while it shows: gone with the recording.
	r.feed("s2", session.Recording)
	r.feed("s2", session.Recognizing)
	if r.muteNote.text != "" || r.pillSession().Partial != "" {
		t.Fatalf("the note outlived the recording: %q", r.muteNote.text)
	}
	r.feed("s3", session.Recording, true)
	if r.muteNote.text != "" {
		t.Error("a try's note")
	}
	if w := r.fs.out.Sys.WriteLog(); len(w) != 0 {
		t.Errorf("writes %v", w)
	}
}

// TestRecordingMuteHelper: a helper that becomes ready again undoes what
// its predecessor muted; quitting gives the output back.
func TestRecordingMuteHelper(t *testing.T) {
	r := newMuteRig(t)
	r.feed("s1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted")
	}
	r.closeMute() // quitting (app.quit)
	if r.muted() {
		t.Fatal("still muted after quitting")
	}
	// An older helper: nothing asked.
	r2 := newMuteRig(t)
	st := r2.fs.Status()
	hello := *st.Hello
	hello.Methods = slices.DeleteFunc(slices.Clone(hello.Methods), func(m string) bool { return m == "output.mute" })
	st.Hello = &hello
	r2.fs.set(helper.Status{State: helper.Starting})
	r2.fs.set(st)
	r2.settle()
	r2.feed("s1", session.Recording)
	if r2.muted() {
		t.Fatal("an older helper was asked to mute")
	}
}

// TestStartCueWait: the mute waits for the start cue to be heard (its
// audible part and the margin), never longer than startCueMax, which a
// sound of unknown length waits.
func TestStartCueWait(t *testing.T) {
	a, _, _ := newTestApp(t)
	for _, tc := range []struct {
		t    sounds.Target
		want time.Duration
	}{
		{sounds.Target{Kind: sounds.KindCatalog, Name: "tally"}, 500 * time.Millisecond},
		{sounds.Target{Kind: sounds.KindCatalog, Name: "chime"}, startCueMax}, // 1.44 s: capped
		{sounds.Target{Kind: sounds.KindSystem, Name: "Pop"}, 365 * time.Millisecond},
		{sounds.Target{Kind: sounds.KindSystem, Name: "Submarine"}, startCueMax}, // 0.87 s + the margin: capped
		{sounds.Target{Kind: sounds.KindSystem, Name: "Blow"}, startCueMax},      // unknown length
	} {
		if d := a.startCueWait(tc.t); d != tc.want {
			t.Errorf("%s: %v, want %v", tc.t.Name, d, tc.want)
		}
	}
}

// TestRecordingMutePreset: a preset whose start and stop are one sound
// (Hero, as Voxt's presets) plays it before the mute and after it, as the
// default pair.
func TestRecordingMutePreset(t *testing.T) {
	r := newMuteRig(t)
	r.updateSettings(func(s *settings.Settings) { s.Audio.StartSound, s.Audio.StopSound = "system:Hero", "system:Hero" })
	r.feed("s1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted")
	}
	r.feed("s1", session.Recognizing)
	if r.muted() {
		t.Fatal("still muted")
	}
	if got := r.takeCues(); !slices.Equal(got, []string{"Hero:audible", "Hero:audible"}) {
		t.Fatalf("cues %v", got)
	}
}

// The output journal lives in Application Support, not Caches (family
// audit A9); one an earlier VoxtNew left in the cache directory moves.
func TestOutputJournalPlace(t *testing.T) {
	home, cache := t.TempDir(), t.TempDir()
	old := filepath.Join(cache, "output-mute")
	os.MkdirAll(old, 0o755)
	os.WriteFile(filepath.Join(old, "claim.json"), []byte("{}"), 0o644)
	dir := outputJournal(home, cache)
	if dir != filepath.Join(home, "output-mute") {
		t.Fatalf("journal %s", dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "claim.json")); err != nil {
		t.Errorf("not moved: %v", err)
	}
	if outputJournal("", cache) != "" {
		t.Error("tests keep none")
	}
}
