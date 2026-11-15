package main

import (
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// soundsRig is Settings → Audio with the previews recorded ("tally" when
// one starts, "-tally" when it is stopped; system sounds by name) and the
// clock of a.later in the test's hands.
type soundsRig struct {
	*app
	tt    *ui.Tester
	ft    *fakeTimers
	plays []string
}

func newSoundsRig(t *testing.T) *soundsRig {
	a, _, tt := newTestApp(t)
	r := &soundsRig{app: a, tt: tt}
	r.ft = new(fakeTimers).install(a)
	a.startSound = func(s sounds.Target) func() {
		r.plays = append(r.plays, s.Name)
		return func() { r.plays = append(r.plays, "-"+s.Name) }
	}
	tt.SetSize(windowWidth, 1400)
	a.navigate("/settings/audio")
	frames(tt)
	if title, ok := tt.Find("Sounds"); ok { // the section at the top of the dialog (600 high), all of it in view
		tt.Scroll(windowWidth/2, 700, 0, title.Y-450)
		frames(tt)
	}
	return r
}

func (r *soundsRig) takePlays() []string {
	p := r.plays
	r.plays = nil
	return p
}

// TestSoundSource: Sound offers Built-in, macOS and None. Built-in lists
// the pairs this build has, Soft (Pebble → Nib) chosen by default; macOS
// has a line per cue with its sound and Try, Pop and Tink until the user
// picks; None turns the sounds off and hides the rest; Built-in again gives
// the default pair back.
func TestSoundSource(t *testing.T) {
	r := newSoundsRig(t)
	for _, s := range []string{"Sound", "Built-in", "macOS", "None", "Soft", "Pebble → Nib", "Beep", "Tally → Wrap", "Volume"} {
		if !has(r.tt, s) {
			t.Fatalf("no %q: %q", s, firstTexts(r.tt))
		}
	}
	if has(r.tt, "Switch") || has(r.tt, "Try the start sound") {
		t.Errorf("a pair this build lacks, or Try: %q", firstTexts(r.tt))
	}
	click(t, r.tt, "macOS")
	if s := r.settings.Get().Audio; !s.Sounds || s.StartSound != "system:Pop" || s.StopSound != "system:Tink" {
		t.Fatalf("macOS: %+v", s)
	}
	for _, s := range []string{"When recording starts", "When recording stops", "Pop", "Tink", "Try the start sound"} {
		if !has(r.tt, s) {
			t.Errorf("macOS: no %q", s)
		}
	}
	if p := r.takePlays(); !slices.Equal(p, []string{"Pop"}) {
		t.Errorf("macOS played %q", p)
	}
	click(t, r.tt, "None")
	if r.settings.Get().Audio.Sounds || has(r.tt, "When recording starts") || has(r.tt, "Volume") {
		t.Fatalf("None: %+v %q", r.settings.Get().Audio, firstTexts(r.tt))
	}
	click(t, r.tt, "Built-in")
	if s := r.settings.Get().Audio; !s.Sounds || s.StartSound != "" || s.StopSound != "" || !has(r.tt, "Pebble → Nib") {
		t.Errorf("Built-in again: %+v", s)
	}
}

// TestSoundPairs: a click on a pair keeps its two sounds and plays them one
// after the other; ▶ plays a pair without choosing it and stops it while it
// plays; a choice that is no pair shows first, as Your choice.
func TestSoundPairs(t *testing.T) {
	r := newSoundsRig(t)
	click(t, r.tt, "Beep")
	if s := r.settings.Get().Audio; s.StartSound != "tally" || s.StopSound != "wrap" {
		t.Fatalf("Beep: %+v", s)
	}
	if p := r.takePlays(); !slices.Equal(p, []string{"tally"}) {
		t.Fatalf("plays %q", p)
	}
	r.ft.advance(350*time.Millisecond + pairGap) // Tally's audible part, then the gap
	if p := r.takePlays(); !slices.Equal(p, []string{"-tally", "wrap"}) {
		t.Fatalf("then %q", p)
	}
	click(t, r.tt, "Play Soft")
	if s := r.settings.Get().Audio; s.StartSound != "tally" {
		t.Fatalf("▶ chose %q", s.StartSound)
	}
	if ch, left := r.previewPlaying(); ch != "pebble" || left <= 0 || !has(r.tt, "Stop Soft") {
		t.Fatalf("playing %q %v", ch, left)
	}
	click(t, r.tt, "Stop Soft")
	if ch, _ := r.previewPlaying(); ch != "" || !has(r.tt, "Play Soft") {
		t.Errorf("stopped %q", ch)
	}
	r.chooseSounds("chime", "tink")
	frames(r.tt)
	if !has(r.tt, "Your choice") || !has(r.tt, "Chime → Tink") {
		t.Errorf("no Your choice: %q", firstTexts(r.tt))
	}
}

// TestSoundsVolume: Volume offers the steps, the one nearest to a volume
// set before shown; choosing one sets the cues' volume.
func TestSoundsVolume(t *testing.T) {
	r := newSoundsRig(t)
	r.updateSettings(func(s *settings.Settings) { s.Audio.SoundVolume = 40 })
	if v := r.cueTarget(startCue).Volume; v != 0.4 {
		t.Fatalf("volume %v", v)
	}
	if err := r.tt.ClickControl("Volume"); err != nil { // the select, not the row's name
		t.Fatal(err)
	}
	frames(r.tt)
	for _, s := range []string{"100%", "75%", "50%", "25%"} {
		if !has(r.tt, s) {
			t.Errorf("no %q in the volumes: %q", s, firstTexts(r.tt))
		}
	}
	click(t, r.tt, "25%")
	if v := r.cueTarget(startCue).Volume; v != 0.25 {
		t.Errorf("25%%: %v", v)
	}
	// Muted before (the picker of 10-20 had Mute): None, and Built-in unmutes.
	r.updateSettings(func(s *settings.Settings) { s.Audio.SetSoundVolume(0) })
	if r.soundSource() != soundNone {
		t.Fatalf("muted: %v", r.soundSource())
	}
	r.setSoundSource(soundBuiltIn)
	if v := r.settings.Get().Audio.SoundVolumeOrDefault(); v != settings.DefaultSoundVolume {
		t.Errorf("unmuted to %d", v)
	}
}

// TestSoundsTryAccent: Try shows the accent while the sound is heard (at
// least previewMin), and a second Try keeps it on until its own end.
func TestSoundsTryAccent(t *testing.T) {
	r := newSoundsRig(t)
	r.trySound(startCue)
	if r.sounds.previewing != recordStartEvent.ID || !slices.Equal(r.takePlays(), []string{"pebble"}) {
		t.Fatal("Try did not play Pebble")
	}
	r.ft.advance(previewMin - 1)
	if r.sounds.previewing == "" {
		t.Fatal("the accent went before previewMin")
	}
	r.trySound(startCue)
	r.ft.advance(1)
	if r.sounds.previewing == "" {
		t.Fatal("the first preview's timer ended the second's accent")
	}
	r.ft.advance(previewMin)
	frames(r.tt)
	if r.sounds.previewing != "" {
		t.Fatal("the accent stayed")
	}
	// A long sound keeps it longer: Chime's 1.44 s and the margin.
	r.chooseSound(recordStartEvent, "chime")
	r.trySound(startCue)
	r.ft.advance(previewMin)
	if r.sounds.previewing == "" {
		t.Fatal("Chime's accent ended at previewMin")
	}
	r.ft.advance(1440*time.Millisecond + cueMargin - previewMin)
	if r.sounds.previewing != "" {
		t.Fatal("Chime's accent stayed")
	}
	// macOS's Try buttons play the cue's sound.
	click(t, r.tt, "macOS")
	r.takePlays()
	click(t, r.tt, "Try the stop sound")
	if p := r.takePlays(); !slices.Equal(p, []string{"-Pop", "Tink"}) {
		t.Errorf("Try: %q", p)
	}
}

// TestPreviewPlayerOrder: a preview that reaches the player after a newer
// one plays nothing, and each one stops the one before.
func TestPreviewPlayerOrder(t *testing.T) {
	var plays []string
	start := func(s sounds.Target) func() {
		plays = append(plays, s.Name)
		return func() { plays = append(plays, "-"+s.Name) }
	}
	sys := func(n string) sounds.Target { return sounds.Target{Kind: sounds.KindSystem, Name: n} }
	var p previewPlayer
	p.play(2, sys("Ping"), start)
	p.play(1, sys("Pop"), start) // late
	p.play(3, sys("Frog"), start)
	p.play(4, sys("Hero"), nil) // could not start: the one before stops anyway
	p.play(5, sys("Basso"), start)
	p.play(6, sounds.Target{}, start) // None: stops the last, plays nothing
	if want := []string{"Ping", "-Ping", "Frog", "-Frog", "Basso", "-Basso"}; !slices.Equal(plays, want) {
		t.Fatalf("plays %q, want %q", plays, want)
	}
}

// TestSoundsWhileRecording: a preview during a recording leaves its
// session and its muted output alone.
func TestSoundsWhileRecording(t *testing.T) {
	r := newMuteRig(t)
	var plays []string
	r.startSound = func(s sounds.Target) func() { plays = append(plays, s.Name); return nil }
	r.feed("s1", session.Recording)
	if !r.muted() {
		t.Fatal("not muted")
	}
	r.previewSound(recordStartEvent, sounds.System("Ping"), "")
	if !r.muted() || r.sess.State != session.Recording || r.muting != "s1" || !slices.Equal(plays, []string{"Ping"}) {
		t.Fatalf("muted %v, state %s, muting %q, plays %q", r.muted(), r.sess.State, r.muting, plays)
	}
	r.feed("s1", session.Recognizing)
	if got := r.takeCues(); !slices.Equal(got, []string{"pebble:audible", "nib:audible"}) {
		t.Fatalf("cues %v", got)
	}
}
