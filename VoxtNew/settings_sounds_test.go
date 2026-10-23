package main

import (
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/hehehai/actnowkit/sounds"
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
	tt.SetSize(windowWidth, 1400) // the whole section, without scrolling
	a.navigate("/settings/audio")
	frames(tt)
	return r
}

func (r *soundsRig) takePlays() []string {
	p := r.plays
	r.plays = nil
	return p
}

// center is the middle of the text s on screen.
func (r *soundsRig) center(t *testing.T, s string) (float32, float32) {
	t.Helper()
	rc, ok := r.tt.Find(s)
	if !ok {
		t.Fatalf("no %q on screen: %q", s, firstTexts(r.tt))
	}
	return rc.X + rc.W/2, rc.Y + rc.H/2
}

// TestSoundsRow: while Play sounds is on, a line for each cue shows its
// sound (the family's Tally and Wrap by default) and Try; off, they go.
func TestSoundsRow(t *testing.T) {
	r := newSoundsRig(t)
	for _, s := range []string{"When recording starts", "When recording stops", "Tally", "Wrap", "Try the start sound", "Try the stop sound"} {
		if !has(r.tt, s) {
			t.Fatalf("no %q: %q", s, firstTexts(r.tt))
		}
	}
	// The switch is above Try (the title's text, first, has its name).
	try, _ := r.tt.Find("Try the start sound")
	_, y := r.center(t, "Play sounds")
	sw := try.X + try.W/2 - 4
	r.tt.ClickAt(sw, y)
	frames(r.tt)
	if r.prefs.Audio.Sounds || has(r.tt, "When recording starts") || has(r.tt, "Try the start sound") {
		rc, _ := r.tt.Find("Play sounds")
		t.Fatalf("the cues show while the sounds are off: on %v, title at %+v", r.prefs.Audio.Sounds, rc)
	}
	r.tt.ClickAt(sw, y)
	frames(r.tt)
	if !r.prefs.Audio.Sounds || !has(r.tt, "Tally") {
		t.Fatal("the cues did not come back")
	}
	if p := r.takePlays(); len(p) != 0 {
		t.Errorf("the switch played %q", p)
	}
}

// TestSoundsChoose: the picker's panel shows the suggested sounds (None
// first) on one line each, with their type; its type menu has All,
// Suggested, every group and macOS's sounds; choosing one keeps it and
// plays it at once, ▶ plays one without choosing it, Try plays the choice.
func TestSoundsChoose(t *testing.T) {
	r := newSoundsRig(t)
	click(t, r.tt, "Start sound")
	for _, s := range []string{"Search sounds", "Type", "Suggested", "Volume", "None", "Tally", "Pebble", "Media", "Tap"} {
		if !has(r.tt, s) {
			t.Errorf("the panel has no %q", s)
		}
	}
	for _, s := range []string{"0.3 s", "0.2 s", "All"} { // no durations, tags or chips
		if has(r.tt, s) {
			t.Errorf("the panel shows %q", s)
		}
	}
	click(t, r.tt, "Type")
	if got := r.tt.Menu(); !slices.Equal(got, []string{"All", "Suggested", "-", "Tap", "Media", "Notify", "-", "macOS"}) {
		t.Errorf("type menu %q", got)
	}
	r.tt.CloseMenu()
	if has(r.tt, "Chime") {
		t.Error("Chime is suggested for the start of a recording")
	}
	click(t, r.tt, "Pebble")
	if got := r.settings.Get().Audio.StartSound; got != "pebble" {
		t.Fatalf("start %q", got)
	}
	click(t, r.tt, "Play Tally")
	if got := r.settings.Get().Audio.StartSound; got != "pebble" {
		t.Fatalf("▶ chose %q", got)
	}
	if p := r.takePlays(); !slices.Equal(p, []string{"pebble", "-pebble", "tally"}) {
		t.Fatalf("plays %q", p)
	}
	// A group, then a system sound found by searching.
	chooseIn(t, r.tt, "Type", "Notify")
	if !has(r.tt, "Chime") || has(r.tt, "Play Pebble") {
		tx := r.tt.Texts()
		t.Errorf("Notify: %q", tx[max(0, len(tx)-40):])
	}
	chooseIn(t, r.tt, "Type", "macOS")
	click(t, r.tt, "Search sounds")
	r.tt.Type("sub")
	frames(r.tt)
	if has(r.tt, "Basso") {
		t.Error("the search kept Basso")
	}
	click(t, r.tt, "Submarine")
	if got := r.settings.Get().Audio.StartSound; got != "system:Submarine" {
		t.Fatalf("system %q", got)
	}
	r.tt.Key(0, ui.KeyEscape)
	frames(r.tt)
	r.takePlays()
	// None: the start cue is silent; the stop cue is still Wrap.
	r.chooseSound(recordStartEvent, sounds.None)
	frames(r.tt)
	if r.cueTarget(startCue).Kind != sounds.KindNone || r.cueTarget(stopCue).Name != "wrap" {
		t.Fatalf("none: %+v %+v", r.cueTarget(startCue), r.cueTarget(stopCue))
	}
	click(t, r.tt, "Try the stop sound")
	if p := r.takePlays(); !slices.Equal(p, []string{"-Submarine", "wrap"}) { // the last preview stops
		t.Fatalf("Try: %q", p)
	}
	// Back to the default: nothing stored.
	r.chooseSound(recordStartEvent, "tally")
	if s := r.settings.Get().Audio; s.StartSound != "" || s.StopSound != "" {
		t.Errorf("defaults stored: %+v", s)
	}
}

// TestSoundsVolume: the panel's volume button lists the steps, loudest
// first (the current one checked: the nearest to a volume set before);
// choosing one sets the cues' volume, Mute silences them.
func TestSoundsVolume(t *testing.T) {
	r := newSoundsRig(t)
	r.updateSettings(func(s *settings.Settings) { s.Audio.SoundVolume = 40 })
	if v := r.cueTarget(startCue).Volume; v != 0.4 {
		t.Fatalf("volume %v", v)
	}
	click(t, r.tt, "Stop sound")
	click(t, r.tt, "Volume")
	if got := r.tt.Menu(); !slices.Equal(got, []string{"100%", "75%", "50%", "25%", "Mute"}) {
		t.Errorf("volume menu %q", got)
	}
	if err := r.tt.ChooseMenuItem("25%"); err != nil {
		t.Fatal(err)
	}
	frames(r.tt)
	if v := r.cueTarget(startCue).Volume; v != 0.25 {
		t.Errorf("25%%: %v", v)
	}
	chooseIn(t, r.tt, "Volume", "Mute")
	if a := r.settings.Get().Audio; a.SoundVolume != -1 || a.SoundVolumeOrDefault() != 0 || r.cueTarget(startCue).Volume != 0 {
		t.Errorf("mute: %+v", a)
	}
	if !has(r.tt, "Search sounds") {
		t.Error("the menus closed the panel")
	}
}

// TestSoundsPlaying: ▶ is in the accent while its sound plays, and a click
// on it then stops it.
func TestSoundsPlaying(t *testing.T) {
	r := newSoundsRig(t)
	click(t, r.tt, "Start sound")
	click(t, r.tt, "Play Pebble")
	if ch, left := r.previewPlaying(); ch != "pebble" || left <= 0 || !has(r.tt, "Stop Pebble") {
		t.Fatalf("playing %q %v", ch, left)
	}
	click(t, r.tt, "Stop Pebble")
	if ch, _ := r.previewPlaying(); ch != "" || !has(r.tt, "Play Pebble") {
		t.Errorf("stopped %q", ch)
	}
	if p := r.takePlays(); !slices.Equal(p, []string{"pebble", "-pebble"}) {
		t.Errorf("plays %q", p)
	}
}

// chooseIn opens the menu of the button labelled button and chooses item.
func chooseIn(t *testing.T, tt *ui.Tester, button, item string) {
	t.Helper()
	click(t, tt, button)
	if err := tt.ChooseMenuItem(item); err != nil {
		t.Fatalf("%s → %s: %v (menu %q)", button, item, err, tt.Menu())
	}
	frames(tt)
}

// TestSoundsTryAccent: Try shows the accent while the sound is heard (at
// least previewMin), and a second Try keeps it on until its own end.
func TestSoundsTryAccent(t *testing.T) {
	r := newSoundsRig(t)
	click(t, r.tt, "Try the start sound")
	if r.sounds.previewing != recordStartEvent.ID || !slices.Equal(r.takePlays(), []string{"tally"}) {
		t.Fatal("Try did not play Tally")
	}
	r.ft.advance(previewMin - 1)
	if r.sounds.previewing == "" {
		t.Fatal("the accent went before previewMin")
	}
	click(t, r.tt, "Try the start sound")
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
	click(t, r.tt, "Try the start sound")
	r.ft.advance(previewMin)
	if r.sounds.previewing == "" {
		t.Fatal("Chime's accent ended at previewMin")
	}
	r.ft.advance(1440*time.Millisecond + cueMargin - previewMin)
	if r.sounds.previewing != "" {
		t.Fatal("Chime's accent stayed")
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
	if got := r.takeCues(); !slices.Equal(got, []string{"tally:audible", "wrap:audible"}) {
		t.Fatalf("cues %v", got)
	}
}
