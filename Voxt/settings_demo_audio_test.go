package main

import "github.com/egoist/mygo/ui"

func settingsDemoAudioCases() []settingsDemoCase {
	return []settingsDemoCase{
		{name: "refresh microphones", fixture: "en/settings-audio-normal", steps: historySteps("Refresh microphones"), want: []string{"loadDevices"}},
		{name: "restart helper", fixture: "en/settings-audio-no-helper", steps: historySteps("Restart helper"), want: []string{"restartHelper"}},
		{name: "add microphone", fixture: "en/settings-audio-normal", steps: historySteps("Add Fixture Other Microphone"), want: []string{"addDevice fixture-mic-other"}},
		{name: "move microphones", fixture: "en/settings-audio-normal", steps: historySteps("Move Fixture Primary Microphone down", "Move Fixture Secondary Microphone up"), want: []string{
			"moveDevice fixture-mic-primary 1", "moveDevice fixture-mic-secondary -1",
		}},
		{name: "remove microphones", fixture: "en/settings-audio-normal", steps: historySteps("Remove Fixture Primary Microphone", "Remove Fixture Secondary Microphone"), want: []string{
			"removeDevice fixture-mic-primary", "removeDevice fixture-mic-secondary",
		}},
		{name: "skip silence", fixture: "en/settings-audio-normal", scroll: "Skip silence", steps: historySteps("Skip silence"), want: []string{"setSkipSilence false"}},
		{name: "clean sound", fixture: "en/settings-audio-normal", scroll: "Optimise sound", steps: historySteps("Optimise sound"), want: []string{"setCleanSound false"}},
		{name: "recognise again", fixture: "en/settings-audio-normal", scroll: "Recognise the whole recording again", steps: historySteps("Recognise the whole recording again"), want: []string{"setLiveResult false"}},
		{name: "upload limit", fixture: "en/settings-audio-normal", scroll: "Upload size limit in MB", steps: []historyDemoStep{{label: "Upload size limit in MB", click: true}, {key: ui.KeyUp}}, want: []string{"setUploadMB 13"}},
		{name: "recording length limit", fixture: "en/settings-audio-normal", scroll: "Recording length limit in minutes", steps: []historyDemoStep{{label: "Recording length limit in minutes", click: true}, {key: ui.KeyUp}}, want: []string{"setMaxMinutes 35"}},
		{name: "mute output", fixture: "en/settings-audio-normal", scroll: "Mute system audio while recording", steps: historySteps("Mute system audio while recording"), want: []string{"setMuteOutput false"}},
		{name: "system sound source", fixture: "en/settings-audio-normal", scroll: "Sound", steps: historySteps("macOS"), want: []string{"chooseSoundSource system"}},
		{name: "no sound source", fixture: "en/settings-audio-normal", scroll: "Sound", steps: historySteps("None"), want: []string{"chooseSoundSource none"}},
		{name: "built in sound source", fixture: "en/settings-audio-source-system", scroll: "Sound", steps: historySteps("Built-in"), want: []string{"chooseSoundSource built-in"}},
		{name: "choose sound pair", fixture: "en/settings-audio-normal", scroll: "Beep", steps: historySteps("Beep"), want: []string{"chooseSoundPair tally wrap"}},
		{name: "preview sound pair", fixture: "en/settings-audio-normal", scroll: "Play Beep", steps: historySteps("Play Beep"), want: []string{"previewSoundPair tally wrap"}},
		{name: "stop sound preview", fixture: "en/settings-audio-source-built-in-custom-playing", scroll: "Stop Your choice", steps: historySteps("Stop Your choice"), want: []string{"stopPreview"}},
		{name: "system sound choices", fixture: "en/settings-audio-source-system", scroll: "When recording starts", steps: []historyDemoStep{
			{label: "Start sound"}, {key: ui.KeyDown}, {key: ui.KeyEnter}, {label: "Stop sound"}, {key: ui.KeyUp}, {key: ui.KeyEnter},
		}, want: []string{
			"chooseSystemSound record.start system:Fixture Tink", "chooseSystemSound record.stop system:Fixture Pop",
		}},
		{name: "try system sounds", fixture: "en/settings-audio-source-system", scroll: "When recording starts", steps: historySteps("Try the start sound", "Try the stop sound"), want: []string{
			"tryCueSound record.start", "tryCueSound record.stop",
		}},
		{name: "sound volume", fixture: "en/settings-audio-normal", scroll: "Volume", steps: settingsDemoSelect("Volume", ui.KeyDown), want: []string{"setSoundVolume 50"}},
	}
}
