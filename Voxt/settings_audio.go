package main

// Settings → Audio (/settings/audio): the microphones in the user's
// order of preference (settings.Audio.Devices: the first one connected
// records, else the system's default input), whether the silence of a
// recording is left out before the speech model gets it (Audio.VAD.Off,
// internal/vad) and whether its sound is cleaned up (Audio.Clean.Off,
// internal/dsp), and the sounds a recording makes when it starts and
// stops (their presets and previews: settings_sounds.go). The helper
// lists the inputs (capture.devices) when the section opens, when the
// helper becomes ready, after each session and with Refresh; the menu
// bar's Microphone submenu shows the last list (tray.go). The system's
// output is muted while recording (Audio.MuteOutput, mute_link.go),
// unless turned off.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

type settingsAudioLocalUI struct {
	skipSilence bool
	liveResult  bool
	cleanSound  bool
	muteOutput  bool
	uploadMB    float64
	maxMinutes  float64
}

// audioSettings is the Audio section of the settings dialog.
func audioSettings(c *ui.Context, state settingsAudioViewState, actions settingsAudioActions, local *settingsAudioLocalUI) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Margin(px(t, sp2), 0, 0, 0).Children(func() {
		sectionTitleText(c, i18n.T("Microphones"))
		infoTip(c.Key("audio-mics-info"), i18n.T("Voxt uses the first connected microphone in the list, or the system default."))
		ui.Spacer(c)
		if iconButton(c.Key("audio-refresh"), icons.Refresh, i18n.T("Refresh microphones")).Disabled(!state.HelperReady || state.DevicesLoading).Clicked() {
			actions.loadDevices()
		}
	})
	switch {
	case !state.HelperReady:
		notice(c, "audio-no-helper", noticeWarn, i18n.T("The helper is not running, so microphones cannot be listed."), "", func() {
			if ui.Button(c.Key("audio-restart"), i18n.T("Restart helper")).Clicked() {
				actions.restartHelper()
			}
		}, nil)
	case state.DevicesError != "":
		notice(c, "audio-err", noticeError, i18n.T("Could not list the microphones: %s", state.DevicesError), "", nil, nil)
	}

	// The list, in order.
	card(c.Key("audio-list")).Gap(px(t, sp1)).Padding(px(t, sp2)).Children(func() {
		if len(state.Devices) == 0 {
			ui.Row(c).Padding(px(t, sp1), px(t, sp2)).Children(func() {
				caption(c, i18n.T("No microphone chosen, so Voxt records from the system's default input."))
			})
		}
		for index, device := range state.Devices {
			ui.Row(c.Key("audio-dev-"+device.ID)).Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, sp1), px(t, sp1), px(t, sp1), px(t, sp2)).Children(func() {
				ui.Text(c, strconv.Itoa(index+1)).FontSize(px(t, fsCaption)).TextColor(k.Text3).Width(px(t, 14)).Shrink(0)
				deviceName(c, device.Name, device.Notes, device.InUse, device.Present)
				if iconButton(c.Key("up"), icons.AltArrowUp, i18n.T("Move %s up", device.Name)).Disabled(index == 0).Clicked() {
					actions.moveDevice(device.ID, -1)
				}
				if iconButton(c.Key("down"), icons.AltArrowDown, i18n.T("Move %s down", device.Name)).Disabled(index == len(state.Devices)-1).Clicked() {
					actions.moveDevice(device.ID, 1)
				}
				if iconButton(c.Key("remove"), icons.Close, i18n.T("Remove %s", device.Name)).Clicked() {
					actions.removeDevice(device.ID)
				}
			})
		}
	})

	// The other inputs connected now.
	if state.DevicesRead {
		sectionTitle(c, i18n.T("Other microphones"))
		card(c.Key("audio-others")).Gap(px(t, sp1)).Padding(px(t, sp2)).Children(func() {
			if len(state.OtherDevices) == 0 {
				ui.Row(c).Padding(px(t, sp1), px(t, sp2)).Children(func() {
					caption(c, i18n.T("No other microphone is connected."))
				})
			}
			for _, device := range state.OtherDevices {
				ui.Row(c.Key("audio-other-"+device.ID)).Gap(px(t, sp2)).AlignItems(ui.Center).Padding(px(t, sp1), px(t, sp1), px(t, sp1), px(t, sp2)).Children(func() {
					deviceName(c, device.Name, device.Notes, device.InUse, device.Present)
					if ghostIconButton(c.Key("add"), icons.Add, i18n.T("Add")).Label(i18n.T("Add %s", device.Name)).Shrink(0).Clicked() {
						actions.addDevice(device.ID)
					}
				})
			}
		})
	}

	sectionTitle(c, i18n.T("Recognition"))
	settingRow(c, "audio-silence", i18n.T("Skip silence"), i18n.T("Trims silence and long pauses before recognition. History keeps the whole recording."), func() {
		if ui.Switch(c.Key("audio-silence-switch"), &local.skipSilence).Label(i18n.T("Skip silence")).Changed() {
			actions.setSkipSilence(local.skipSilence)
		}
	})
	settingRow(c, "audio-clean", i18n.T("Optimise sound"), i18n.T("Filters out low hum and evens out the volume before recognition. History keeps the original recording."), func() {
		if ui.Switch(c.Key("audio-clean-switch"), &local.cleanSound).Label(i18n.T("Optimise sound")).Changed() {
			actions.setCleanSound(local.cleanSound)
		}
	})
	settingRow(c, "audio-final", i18n.T("Recognise the whole recording again"), i18n.T("More accurate than the live text, but takes a moment after you stop. Only for speech models that show text as you speak."), func() {
		again := !local.liveResult
		if ui.Switch(c.Key("audio-final-switch"), &again).Label(i18n.T("Recognise the whole recording again")).Changed() {
			local.liveResult = !again
			actions.setLiveResult(local.liveResult)
		}
	})
	settingRow(c, "audio-upload", i18n.T("Upload size limit"), i18n.T("For speech services that take the whole recording (OpenAI, Doubao): a longer recording goes in pieces of at most this size. 0 uses the service's own limit: 25 MB for OpenAI, 100 MB for Doubao."), func() {
		if numberInput(c.Key("audio-upload-mb"), &local.uploadMB, 0, 100, 1, numOpts{Label: i18n.T("Upload size limit in MB")}) {
			actions.setUploadMB(local.uploadMB)
		}
		caption(c, i18n.T("MB"))
	})
	settingRow(c, "audio-length", i18n.T("Recording length limit"), i18n.T("Stops a recording at this length and recognises it; the recording window warns in the last minute. 0 means no limit."), func() {
		if numberInput(c.Key("audio-length-min"), &local.maxMinutes, 0, 240, 5, numOpts{Label: i18n.T("Recording length limit in minutes")}) {
			actions.setMaxMinutes(local.maxMinutes)
		}
		caption(c, i18n.T("min"))
	})

	sectionTitle(c, i18n.T("Sounds"))
	soundsRow(c, state.Sounds, actions)
	settingRow(c, "audio-mute", i18n.T("Mute system audio while recording"), i18n.T("Keeps videos, music and calls out of the recording. A device you muted yourself stays muted."), func() {
		if ui.Switch(c.Key("audio-mute-switch"), &local.muteOutput).Label(i18n.T("Mute system audio while recording")).Changed() {
			actions.setMuteOutput(local.muteOutput)
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
