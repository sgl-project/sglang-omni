package main

import (
	"encoding/json"
	"slices"
	"sync"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestInfoTipClick: a click on an ⓘ shows its tip, with no wait for the
// pointer to rest on it, and Escape puts it away. Before, it showed only
// on hover and a click did nothing.
func TestInfoTipClick(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/audio")
	const tip = "Voxt uses the first connected microphone in the list, or the system default."
	shown := func() int { // the ⓘ's own label counts once
		frames(tt)
		n := 0
		for _, s := range tt.Texts() {
			if s == tip {
				n++
			}
		}
		return n
	}
	closed := shown()
	if err := tt.ClickControl(tip); err != nil {
		t.Fatal(err)
	}
	if shown() != closed+1 {
		t.Fatalf("a click did not show the tip: %q", firstTexts(tt))
	}
	tt.Key(0, ui.KeyEscape)
	if shown() != closed {
		t.Error("Escape left the tip open")
	}
}

// TestRecogniseAgainSwitch: Settings → Audio's "Recognise the whole
// recording again" is on by default; off, the live text is the result
// (Audio.LiveResult, #3644 FB7).
func TestRecogniseAgainSwitch(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 2400)
	a.navigate("/settings/audio")
	frames(tt)
	if r, ok := tt.Find("Recognise the whole recording again"); ok {
		tt.Scroll(windowWidth/2, 1200, 0, r.Y-600)
		frames(tt)
	}
	for _, want := range []bool{true, false} {
		if err := tt.ClickControl("Recognise the whole recording again"); err != nil {
			t.Fatal(err)
		}
		frames(tt)
		if got := a.settings.Get().Audio.LiveResult; got != want {
			t.Fatalf("after a click: live result %v, want %v", got, want)
		}
	}
}

func TestAudioSettingsDevices(t *testing.T) {
	a, fs, tt := newTestApp(t)
	a.navigate("/settings/audio")
	if !slices.Contains(fs.methods(), proto.MethodCaptureDevices) {
		t.Fatalf("the section did not list the inputs: %v", fs.methods())
	}
	if !has(tt, "No microphone chosen, so Voxt records from the system's default input.") || !has(tt, "MacBook Air Microphone") {
		t.Fatalf("empty list: %q", firstTexts(tt))
	}
	if !has(tt, "Skip silence") || !has(tt, "Optimise sound") {
		t.Errorf("no recognition switches: %q", firstTexts(tt))
	}
	click(t, tt, "Add AirPods Pro")
	click(t, tt, "Add MacBook Air Microphone")
	got := a.settings.Get().Audio
	if !slices.Equal(got.Devices, []string{"AirPodsPro:input", "BuiltInMicrophoneDevice"}) || got.DeviceNames["AirPodsPro:input"] != "AirPods Pro" {
		t.Fatalf("added: %+v", got)
	}
	if !has(tt, "In use") || !has(tt, "No other microphone is connected.") {
		t.Errorf("after adding: %q", firstTexts(tt))
	}
	click(t, tt, "Move MacBook Air Microphone up")
	if d := a.settings.Get().Audio.Devices; !slices.Equal(d, []string{"BuiltInMicrophoneDevice", "AirPodsPro:input"}) {
		t.Fatalf("moved: %q", d)
	}
	click(t, tt, "Remove MacBook Air Microphone")
	if d := a.settings.Get().Audio.Devices; !slices.Equal(d, []string{"AirPodsPro:input"}) {
		t.Fatalf("removed: %q", d)
	}
	// A device that is gone keeps its name, marked.
	fs.mu.Lock()
	fs.handlers[proto.MethodCaptureDevices] = func(json.RawMessage) (any, error) {
		return proto.DevicesResult{Devices: []proto.CaptureDevice{{ID: "BuiltInMicrophoneDevice", Name: "MacBook Air Microphone", Default: true}}}, nil
	}
	fs.mu.Unlock()
	click(t, tt, "Refresh microphones")
	if !has(tt, "AirPods Pro") || !has(tt, "Not connected") || !has(tt, "In use · System default") {
		t.Errorf("unplugged: %q", firstTexts(tt))
	}
	// Without the helper: a notice, no list.
	fs.set(helper.Status{State: helper.Stopped})
	if !has(tt, "The helper is not running, so microphones cannot be listed.") {
		t.Errorf("no helper: %q", firstTexts(tt))
	}
}

func TestDeviceInUse(t *testing.T) {
	devs := []proto.CaptureDevice{{ID: "mic"}, {ID: "usb"}, {ID: "def", Default: true}}
	for _, tc := range []struct {
		list []string
		want string
	}{
		{nil, "def"},
		{[]string{"gone", "usb"}, "usb"},
		{[]string{"gone"}, "def"},
		{[]string{"mic", "usb"}, "mic"},
	} {
		if got := deviceInUse(tc.list, devs); got != tc.want {
			t.Errorf("%q: %q, want %q", tc.list, got, tc.want)
		}
	}
	if got := deviceInUse([]string{"x"}, nil); got != "" {
		t.Errorf("no devices: %q", got)
	}
}

func TestRecordingCues(t *testing.T) {
	snap := func(id string, st session.State) session.Snapshot { return session.Snapshot{ID: id, State: st} }
	for _, tc := range []struct {
		name   string
		was, s session.Snapshot
		want   cue
	}{
		{"start", snap("", session.Idle), snap("a", session.Recording), startCue},
		{"after reading", snap("a", session.Reading), snap("a", session.Recording), startCue},
		{"a new session", snap("a", session.Recording), snap("b", session.Recording), startCue},
		{"still recording", snap("a", session.Recording), snap("a", session.Recording), noCue},
		{"stop", snap("a", session.Recording), snap("a", session.Recognizing), stopCue},
		{"stop to compose", snap("a", session.Recording), snap("a", session.Composing), stopCue},
		{"cancelled", snap("a", session.Recording), snap("a", session.Cancelled), noCue},
		{"failed", snap("a", session.Recording), snap("a", session.Failed), noCue},
		{"reading a selection", snap("a", session.Reading), snap("a", session.Composing), noCue},
		{"recognized", snap("a", session.Recognizing), snap("a", session.Composing), noCue},
	} {
		if got := recordingCue(tc.was, tc.s); got != tc.want {
			t.Errorf("%s: %v", tc.name, got)
		}
	}
	// The cues' sounds: the family's defaults, a chosen one, a system
	// sound, none; a family sound this build lacks plays Voxt's.
	a, _, _ := newTestApp(t)
	for _, tc := range []struct {
		start, stop string
		want        [2]string
	}{
		{"", "", [2]string{"tally", "wrap"}},
		{"pebble", "system:Tink", [2]string{"pebble", "Tink"}},
		{"none", "ghost", [2]string{"", "Tink"}},
	} {
		a.updateSettings(func(s *settings.Settings) { s.Audio.StartSound, s.Audio.StopSound = tc.start, tc.stop })
		if s, e := a.cueTarget(startCue), a.cueTarget(stopCue); s.Name != tc.want[0] || e.Name != tc.want[1] || e.Volume != 0.7 {
			t.Errorf("%q %q: %+v %+v", tc.start, tc.stop, s, e)
		}
	}
}

func TestSessionSounds(t *testing.T) {
	a, _, _ := newTestApp(t)
	var mu sync.Mutex
	var cues []string
	// The stop cue plays once the output is back (mute_link.go), from
	// the mute's goroutine: a.mute.Wait() before reading.
	a.playSound = func(s sounds.Target) { mu.Lock(); cues = append(cues, s.Name); mu.Unlock() }
	rev := a.sess.Rev
	feed := func(id string, st session.State) {
		rev++
		a.sessionChanged(session.Snapshot{Rev: rev, ID: id, Kind: store.KindTranscribe, State: st, Try: true})
	}
	feed("s1", session.Recording)
	feed("s1", session.Recognizing)
	feed("s1", session.Done)
	a.mute.Wait() // the stop cue comes from the mute's goroutine: before the next start
	feed("s2", session.Recording)
	feed("s2", session.Cancelled)
	a.mute.Wait()
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"tally", "wrap", "tally"}; !slices.Equal(cues, want) {
		t.Fatalf("cues %q, want %q", cues, want)
	}
	// Other sounds: a system sound to start, a family sound to stop.
	a.updateSettings(func(s *settings.Settings) { s.Audio.StartSound, s.Audio.StopSound = "system:Frog", "chime" })
	cues = nil
	mu.Unlock()
	feed("s4", session.Recording)
	feed("s4", session.Done)
	a.mute.Wait()
	mu.Lock()
	if want := []string{"Frog", "chime"}; !slices.Equal(cues, want) {
		t.Fatalf("frog: cues %q", cues)
	}
	// Off in Settings → Audio: silence.
	a.navigate("/settings/audio")
	a.updateSettings(func(s *settings.Settings) { s.Audio.Sounds = false })
	cues = nil
	mu.Unlock()
	feed("s3", session.Recording)
	feed("s3", session.Recognizing)
	a.mute.Wait()
	mu.Lock()
	if len(cues) != 0 {
		t.Errorf("sounds off: %q", cues)
	}
}
