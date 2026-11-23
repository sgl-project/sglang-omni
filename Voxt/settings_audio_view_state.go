package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type settingsAudioDeviceViewState struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Notes   string `json:"notes"`
	InUse   bool   `json:"in_use"`
	Present bool   `json:"present"`
}

type settingsAudioViewState struct {
	HelperReady    bool                           `json:"helper_ready"`
	DevicesRead    bool                           `json:"devices_read"`
	DevicesLoading bool                           `json:"devices_loading"`
	DevicesError   string                         `json:"devices_error"`
	Devices        []settingsAudioDeviceViewState `json:"devices"`
	OtherDevices   []settingsAudioDeviceViewState `json:"other_devices"`
	SkipSilence    bool                           `json:"skip_silence"`
	CleanSound     bool                           `json:"clean_sound"`
	LiveResult     bool                           `json:"live_result"`
	UploadMB       float64                        `json:"upload_mb"`
	MaxMinutes     float64                        `json:"max_minutes"`
	MuteOutput     bool                           `json:"mute_output"`
	Sounds         settingsSoundsViewState        `json:"sounds"`
}

type settingsAudioStateInput struct {
	Audio          settings.Audio
	Devices        []proto.CaptureDevice
	HelperReady    bool
	DevicesRead    bool
	DevicesLoading bool
	DevicesError   string
	Sounds         settingsSoundsViewState
}

func buildSettingsAudioViewState(translator *i18n.Translator, input settingsAudioStateInput) settingsAudioViewState {
	state := settingsAudioViewState{
		HelperReady: input.HelperReady, DevicesRead: input.DevicesRead, DevicesLoading: input.DevicesLoading, DevicesError: input.DevicesError,
		SkipSilence: !input.Audio.VAD.Off, CleanSound: !input.Audio.Clean.Off, LiveResult: input.Audio.LiveResult,
		UploadMB: float64(input.Audio.UploadMB), MaxMinutes: float64(input.Audio.MaxSeconds) / 60, MuteOutput: input.Audio.MuteOn(), Sounds: input.Sounds,
	}
	inUse := deviceInUse(input.Audio.Devices, input.Devices)
	for _, id := range input.Audio.Devices {
		deviceIndex := slices.IndexFunc(input.Devices, func(device proto.CaptureDevice) bool { return device.ID == id })
		present := deviceIndex >= 0
		var device proto.CaptureDevice
		if present {
			device = input.Devices[deviceIndex]
		}
		name := device.Name
		if !present {
			name = input.Audio.DeviceNames[id]
		}
		if name == "" {
			name = id
		}
		var notes []string
		switch {
		case !present && input.DevicesRead:
			notes = append(notes, translator.T("Not connected"))
		case id == inUse:
			notes = append(notes, translator.T("In use"))
		}
		if device.Default {
			notes = append(notes, translator.T("System default"))
		}
		state.Devices = append(state.Devices, settingsAudioDeviceViewState{
			ID: id, Name: name, Notes: strings.Join(notes, " · "), InUse: id == inUse, Present: present || !input.DevicesRead,
		})
	}
	for _, device := range input.Devices {
		if slices.Contains(input.Audio.Devices, device.ID) {
			continue
		}
		var notes []string
		if device.ID == inUse {
			notes = append(notes, translator.T("In use"))
		}
		if device.Default {
			notes = append(notes, translator.T("System default"))
		}
		state.OtherDevices = append(state.OtherDevices, settingsAudioDeviceViewState{
			ID: device.ID, Name: device.Name, Notes: strings.Join(notes, " · "), InUse: device.ID == inUse, Present: true,
		})
	}
	return state
}
