package main

// Settings → Audio (/settings/audio): the microphones in the user's
// order of preference (settings.Audio.Devices: the first one connected
// records, else the system's default input), whether the silence of a
// recording is left out before the speech model gets it (Audio.VAD.Off,
// internal/vad) and whether its sound is cleaned up (Audio.Clean.Off,
// internal/dsp), and the sounds a recording makes when it starts and
// stops (their presets and previews: settings_sounds.go). The helper
// lists the inputs
// (capture.devices) when the section opens, when the helper becomes
// ready while it shows, and with Refresh. The system's output is muted
// while recording (Audio.MuteOutput, mute_link.go), unless turned off.

import (
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// audioState is the audio section's view state, and the player of the
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
	a.openSounds() // settings_sounds.go: the family's sounds, the player
	a.playSound = a.playCueWait
}

// audioShown reads the inputs when the section shows; helperChanged
// calls it too, as a helper that just became ready can list them.
func (a *app) audioShown() {
	if a.settingsOpen && a.setRouter.Path() == "/audio" {
		a.loadDevices()
	}
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
func (a *app) addDevice(d proto.CaptureDevice) {
	a.updateSettings(func(s *settings.Settings) {
		if slices.Contains(s.Audio.Devices, d.ID) {
			return
		}
		s.Audio.Devices = append(slices.Clone(s.Audio.Devices), d.ID)
		m := maps.Clone(s.Audio.DeviceNames)
		if m == nil {
			m = map[string]string{}
		}
		m[d.ID] = d.Name
		s.Audio.DeviceNames = m
	})
}

// moveDevice moves the list's i-th input by delta places.
func (a *app) moveDevice(i, delta int) {
	a.updateSettings(func(s *settings.Settings) {
		list := slices.Clone(s.Audio.Devices)
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

// audioSettings is the Audio section of the settings dialog.
func (a *app) audioSettings(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	devs := a.audio.devices
	list := a.prefs.Audio.Devices
	inUse := deviceInUse(list, devs)
	byID := func(id string) (proto.CaptureDevice, bool) {
		i := slices.IndexFunc(devs, func(d proto.CaptureDevice) bool { return d.ID == id })
		if i < 0 {
			return proto.CaptureDevice{}, false
		}
		return devs[i], true
	}

	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		sectionTitleText(c, i18n.T("Microphones"))
		infoTip(c.Key("audio-mics-info"), i18n.T("VoxtNew records from the first microphone of the list that is connected. When none of them is, or the list is empty, it records from the system's default input."))
		ui.Spacer(c)
		if iconButton(c.Key("audio-refresh"), icons.Refresh, i18n.T("Refresh the microphones")).Disabled(!a.helperReady() || a.audio.loading).Clicked() {
			a.loadDevices()
		}
	})
	switch {
	case !a.helperReady():
		notice(c, "audio-no-helper", noticeWarn, i18n.T("The helper is not running: the microphones cannot be listed."), "", func() {
			if ui.Button(c.Key("audio-restart"), i18n.T("Restart the helper")).Clicked() {
				a.restartHelper()
			}
		}, nil)
	case a.audio.err != "":
		notice(c, "audio-err", noticeError, i18n.T("Could not list the microphones: %s", a.audio.err), "", nil, nil)
	}

	// The list, in order.
	card(c.Key("audio-list")).Gap(px(t, sp1)).Padding(px(t, sp2)).Children(func() {
		if len(list) == 0 {
			ui.Row(c).Padding(px(t, sp1), px(t, sp2)).Children(func() {
				caption(c, i18n.T("No microphone chosen: VoxtNew records from the system's default input."))
			})
		}
		for i, id := range list {
			d, present := byID(id)
			name := d.Name
			if !present {
				name = a.prefs.Audio.DeviceNames[id]
			}
			if name == "" {
				name = id
			}
			var notes []string
			switch {
			case !present && a.audio.read:
				notes = append(notes, i18n.T("Not connected"))
			case id == inUse:
				notes = append(notes, i18n.T("Records now"))
			}
			if d.Default {
				notes = append(notes, i18n.T("System default"))
			}
			ui.Row(c.Key("audio-dev-"+id)).Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, sp1), px(t, sp1), px(t, sp1), px(t, sp2)).Children(func() {
				ui.Text(c, itoa(i+1)).FontSize(px(t, fsCaption)).TextColor(k.Text3).Width(px(t, 14)).Shrink(0)
				deviceName(c, name, strings.Join(notes, " · "), id == inUse, present || !a.audio.read)
				if iconButton(c.Key("up"), icons.AltArrowUp, i18n.T("Move %s up", name)).Disabled(i == 0).Clicked() {
					a.moveDevice(i, -1)
				}
				if iconButton(c.Key("down"), icons.AltArrowDown, i18n.T("Move %s down", name)).Disabled(i == len(list)-1).Clicked() {
					a.moveDevice(i, 1)
				}
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", name)).Clicked() {
					a.removeDevice(id)
				}
			})
		}
	})

	// The other inputs connected now.
	var others []proto.CaptureDevice
	for _, d := range devs {
		if !slices.Contains(list, d.ID) {
			others = append(others, d)
		}
	}
	if a.audio.read {
		sectionTitle(c, i18n.T("Other microphones"))
		card(c.Key("audio-others")).Gap(px(t, sp1)).Padding(px(t, sp2)).Children(func() {
			if len(others) == 0 {
				ui.Row(c).Padding(px(t, sp1), px(t, sp2)).Children(func() {
					caption(c, i18n.T("No other microphone is connected."))
				})
			}
			for _, d := range others {
				var notes []string
				if d.ID == inUse {
					notes = append(notes, i18n.T("Records now"))
				}
				if d.Default {
					notes = append(notes, i18n.T("System default"))
				}
				ui.Row(c.Key("audio-other-"+d.ID)).Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, sp1), px(t, sp1), px(t, sp1), px(t, sp2)).Children(func() {
					deviceName(c, d.Name, strings.Join(notes, " · "), d.ID == inUse, true)
					if ghostIconButton(c.Key("add"), icons.Add, i18n.T("Add")).Label(i18n.T("Add %s", d.Name)).Shrink(0).Clicked() {
						a.addDevice(d)
					}
				})
			}
		})
	}

	sectionTitle(c, i18n.T("Recognition"))
	settingRow(c, "audio-silence", i18n.T("Skip silence"), i18n.T("Leaves out the silence before and after you speak and shortens long pauses before the speech model gets the recording, and sends nothing when no one spoke. The history keeps the whole recording."), func() {
		if ui.Switch(c.Key("audio-silence-switch"), &a.settingsForm.skipSilence).Label(i18n.T("Skip silence")).Changed() {
			off := !a.settingsForm.skipSilence
			a.updateSettings(func(s *settings.Settings) { s.Audio.VAD.Off = off })
		}
	})
	settingRow(c, "audio-clean", i18n.T("Optimize sound"), i18n.T("Filters out low hum (air conditioning, fans, mains) and evens out the volume before the speech model gets the recording: a quiet voice is raised, a very loud one lowered a little, and nothing clips. The history keeps the original recording."), func() {
		if ui.Switch(c.Key("audio-clean-switch"), &a.settingsForm.cleanSound).Label(i18n.T("Optimize sound")).Changed() {
			off := !a.settingsForm.cleanSound
			a.updateSettings(func(s *settings.Settings) { s.Audio.Clean.Off = off })
		}
	})

	sectionTitle(c, i18n.T("Sounds"))
	a.soundsRow(c) // settings_sounds.go
	settingRow(c, "audio-mute", i18n.T("Mute system audio while recording"), i18n.T("Mutes every sound of the current output device while you speak, so that videos, music or calls are not recorded: from just after the start sound until the recording stops, then they come back. A device you muted yourself stays muted. Some devices cannot be muted by software. No permission is needed: nothing of the system's sound is recorded."), func() {
		if ui.Switch(c.Key("audio-mute-switch"), &a.settingsForm.muteOutput).Label(i18n.T("Mute system audio while recording")).Changed() {
			on := a.settingsForm.muteOutput
			a.updateSettings(func(s *settings.Settings) { s.Audio.MuteOutput = &on })
			a.muteSettingChanged(on) // mute_link.go: off during a recording gives the output back now
		}
	})
}

// deviceName is an input's name and, under it, notes on its state; a
// check marks the one that records now, and a grey name one that is not
// connected.
func deviceName(c *ui.Context, name, notes string, inUse, present bool) {
	t := c.Theme()
	k := tok(t)
	ic, col := icons.MicrophoneLarge, k.Text3
	if inUse {
		ic, col = icons.CheckCircle, k.Accent
	}
	ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(col).Shrink(0)
	ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
		tx := truncText(c, name)
		if !present {
			tx.TextColor(k.Text2)
		}
		if notes != "" {
			ui.Text(c, notes).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
		}
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
