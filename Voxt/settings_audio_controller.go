package main

import (
	"maps"
	"slices"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// audioState is the audio section's state, and the player of the
// recording's cues.
type audioState struct {
	audio struct {
		devices []proto.CaptureDevice // as capture.devices last said
		read    bool                  // devices were read at least once
		loading bool
		err     string // why the last read failed ("" for none)
	}
	// playSound plays a cue and returns when it ended (playCueWait, on a
	// goroutine); tests record the cues instead.
	playSound func(sounds.Target)
}

func (a *app) openAudio() {
	a.openSounds() // settings_sounds_controller.go: the family's sounds, the player
	a.playSound = a.playCueWait
}

// loadDevices asks the helper for the audio inputs.
func (a *app) loadDevices() {
	if !a.helperReady() || a.audio.loading {
		return
	}
	a.audio.loading = true
	helperCall(a, proto.MethodCaptureDevices, nil, 5*time.Second, func(r proto.DevicesResult, err error) {
		a.audio.loading = false
		if err != nil {
			a.audio.err = errText(err)
			return
		}
		a.audio.devices, a.audio.read, a.audio.err = r.Devices, true, ""
		a.rememberDeviceNames(r.Devices)
		a.buildTrayMenu() // tray.go: the Microphone submenu
	})
}

// rememberDeviceNames keeps the names of the listed devices that are
// connected, to show them while they are not.
func (a *app) rememberDeviceNames(devs []proto.CaptureDevice) {
	names := map[string]string{}
	for _, d := range devs {
		if slices.Contains(a.prefs.Audio.Devices, d.ID) && d.Name != "" && a.prefs.Audio.DeviceNames[d.ID] != d.Name {
			names[d.ID] = d.Name
		}
	}
	if len(names) == 0 {
		return
	}
	a.updateSettings(func(s *settings.Settings) {
		m := maps.Clone(s.Audio.DeviceNames)
		if m == nil {
			m = map[string]string{}
		}
		maps.Copy(m, names)
		s.Audio.DeviceNames = m
	})
}

// deviceInUse is the input a recording would use now: the first of list
// that is connected, else the system's default input ("" when unknown).
func deviceInUse(list []string, devs []proto.CaptureDevice) string {
	for _, id := range list {
		if slices.ContainsFunc(devs, func(d proto.CaptureDevice) bool { return d.ID == id }) {
			return id
		}
	}
	for _, d := range devs {
		if d.Default {
			return d.ID
		}
	}
	return ""
}

// addDevice puts a connected input at the end of the list.
func (a *app) addDevice(id string) {
	index := slices.IndexFunc(a.audio.devices, func(device proto.CaptureDevice) bool { return device.ID == id })
	if index < 0 {
		return
	}
	device := a.audio.devices[index]
	a.updateSettings(func(s *settings.Settings) {
		if slices.Contains(s.Audio.Devices, device.ID) {
			return
		}
		s.Audio.Devices = append(slices.Clone(s.Audio.Devices), device.ID)
		s.Audio.DeviceNames = withDeviceName(s.Audio.DeviceNames, device)
	})
}

// useDevice puts a connected input first in the list, so the next
// recording uses it (the menu bar's Microphone submenu).
func (a *app) useDevice(d proto.CaptureDevice) {
	a.updateSettings(func(s *settings.Settings) {
		rest := slices.DeleteFunc(slices.Clone(s.Audio.Devices), func(id string) bool { return id == d.ID })
		s.Audio.Devices = append([]string{d.ID}, rest...)
		s.Audio.DeviceNames = withDeviceName(s.Audio.DeviceNames, d)
	})
}

// withDeviceName is names with d's name, to show it while d is not
// connected.
func withDeviceName(names map[string]string, d proto.CaptureDevice) map[string]string {
	m := maps.Clone(names)
	if m == nil {
		m = map[string]string{}
	}
	m[d.ID] = d.Name
	return m
}

// moveDevice moves the list's input id by delta places.
func (a *app) moveDevice(id string, delta int) {
	a.updateSettings(func(s *settings.Settings) {
		list := slices.Clone(s.Audio.Devices)
		i := slices.Index(list, id)
		j := i + delta
		if i < 0 || j < 0 || i >= len(list) || j >= len(list) {
			return
		}
		list[i], list[j] = list[j], list[i]
		s.Audio.Devices = list
	})
}

// removeDevice takes the list's input id off it.
func (a *app) removeDevice(id string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Audio.Devices = slices.DeleteFunc(slices.Clone(s.Audio.Devices), func(x string) bool { return x == id })
	})
}

func (a *app) setSkipSilence(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.VAD.Off = !on })
}

func (a *app) setCleanSound(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.Clean.Off = !on })
}

func (a *app) setLiveResult(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.LiveResult = on })
}

func (a *app) setUploadMB(megabytes float64) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.UploadMB = int(megabytes) })
}

func (a *app) setMaxMinutes(minutes float64) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.MaxSeconds = int(minutes * 60) })
}

func (a *app) setMuteOutput(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.MuteOutput = &on })
	a.muteSettingChanged(on) // mute_link.go: off during a recording gives the output back now
}

func (a *app) audioSettingsModel(now time.Time) settingsAudioViewState {
	return buildSettingsAudioViewState(i18n.CurrentTranslator(), settingsAudioStateInput{
		Audio: a.prefs.Audio, Devices: a.audio.devices, HelperReady: a.helperReady(),
		DevicesRead: a.audio.read, DevicesLoading: a.audio.loading, DevicesError: a.audio.err,
		Sounds: a.soundsSettingsModel(now),
	})
}

// cue is a sound of a recording: its start, or its stop.
type cue int

const (
	noCue cue = iota
	startCue
	stopCue
)

// recordingCue is the cue a change of the session makes: one when it
// starts recording, another when the recording stops and the text is on
// its way. A cancelled or failed recording stops silently.
func recordingCue(was, s session.Snapshot) cue {
	switch {
	case s.State == session.Recording && (was.State != session.Recording || was.ID != s.ID):
		return startCue
	case was.State == session.Recording && was.ID == s.ID && s.State != session.Recording &&
		s.State != session.Cancelled && s.State != session.Failed:
		return stopCue
	}
	return noCue
}
