package main

import (
	"fmt"
	"slices"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

const (
	soundBuiltInID = "built-in"
	soundSystemID  = "system"
	soundNoneID    = "none"
)

type settingsSoundPairViewState struct {
	Start     string        `json:"start"`
	Stop      string        `json:"stop"`
	Name      string        `json:"name"`
	Sounds    string        `json:"sounds"`
	PlayLabel string        `json:"play_label"`
	Chosen    bool          `json:"chosen"`
	Playing   bool          `json:"playing"`
	Remaining time.Duration `json:"remaining"`
}

type settingsSystemCueViewState struct {
	EventID     string   `json:"event_id"`
	Label       string   `json:"label"`
	SelectLabel string   `json:"select_label"`
	Options     []string `json:"options"`
	Values      []string `json:"values"`
	Selected    string   `json:"selected"`
	TryLabel    string   `json:"try_label"`
	TryAccent   bool     `json:"try_accent"`
}

type settingsSoundsViewState struct {
	SourceOptions   []string                     `json:"source_options"`
	SourceValues    []string                     `json:"source_values"`
	SourceIndex     int                          `json:"source_index"`
	SourceID        string                       `json:"source_id"`
	Pairs           []settingsSoundPairViewState `json:"pairs"`
	SystemCues      []settingsSystemCueViewState `json:"system_cues"`
	VolumeOptions   []string                     `json:"volume_options"`
	VolumeValues    []int                        `json:"volume_values"`
	VolumeSelection string                       `json:"volume_selection"`
}

type settingsSoundsStateInput struct {
	Prefs        sounds.Prefs
	Source       soundSource
	Catalog      *sounds.Catalog
	StartTarget  sounds.Target
	StopTarget   sounds.Target
	SystemSounds []string
	Previewing   string
	Playing      string
	Remaining    time.Duration
}

func buildSettingsSoundsViewState(translator *i18n.Translator, input settingsSoundsStateInput) settingsSoundsViewState {
	state := settingsSoundsViewState{SourceID: soundSourceID(input.Source)}
	if input.Catalog != nil {
		state.SourceOptions = append(state.SourceOptions, translator.X("sound source", "Built-in"))
		state.SourceValues = append(state.SourceValues, soundBuiltInID)
	}
	state.SourceOptions = append(state.SourceOptions, translator.T("macOS"), translator.T("None"))
	state.SourceValues = append(state.SourceValues, soundSystemID, soundNoneID)
	state.SourceIndex = max(0, slices.Index(state.SourceValues, state.SourceID))

	if input.Catalog != nil {
		current := soundPair{"", input.Prefs.Choice(recordStartEvent), input.Prefs.Choice(recordStopEvent)}
		var pairs []soundPair
		for _, pair := range soundPairs {
			if _, ok := input.Catalog.Sound(pair.start); !ok {
				continue
			}
			if _, ok := input.Catalog.Sound(pair.stop); ok {
				pairs = append(pairs, pair)
			}
		}
		if !slices.ContainsFunc(pairs, func(pair soundPair) bool {
			return pair.start == current.start && pair.stop == current.stop
		}) {
			pairs = append([]soundPair{current}, pairs...)
		}
		for _, pair := range pairs {
			name := pairNameWithTranslator(translator, pair.key)
			chosen := pair.start == current.start && pair.stop == current.stop
			playing := input.Remaining > 0 && (input.Playing == pair.start || input.Playing == pair.stop)
			playLabel := translator.T("Play %s", name)
			remaining := time.Duration(0)
			if playing {
				playLabel = translator.T("Stop %s", name)
				remaining = input.Remaining
			}
			state.Pairs = append(state.Pairs, settingsSoundPairViewState{
				Start: pair.start, Stop: pair.stop, Name: name,
				Sounds:    translator.T("%s → %s", soundLabelWithTranslator(translator, input.Catalog, pair.start), soundLabelWithTranslator(translator, input.Catalog, pair.stop)),
				PlayLabel: playLabel, Chosen: chosen, Playing: playing, Remaining: remaining,
			})
		}
	}

	for _, c := range []cue{startCue, stopCue} {
		event := cueEvent(c)
		target := input.StartTarget
		label, selectLabel, tryLabel := translator.T("When recording starts"), translator.T("Start sound"), translator.T("Try the start sound")
		if c == stopCue {
			target = input.StopTarget
			label, selectLabel, tryLabel = translator.T("When recording stops"), translator.T("Stop sound"), translator.T("Try the stop sound")
		}
		cueState := settingsSystemCueViewState{
			EventID: event.ID, Label: label, SelectLabel: selectLabel, TryLabel: tryLabel, TryAccent: input.Previewing == event.ID,
			Options: []string{translator.T("None")}, Values: []string{sounds.None},
		}
		for _, name := range input.SystemSounds {
			cueState.Values = append(cueState.Values, sounds.System(name))
			cueState.Options = append(cueState.Options, name)
		}
		current := sounds.None
		if target.Kind == sounds.KindSystem {
			current = sounds.System(target.Name) // a family sound this build lacks plays its fallback
		}
		if !slices.Contains(cueState.Values, current) {
			cueState.Values = append(cueState.Values, current)
			cueState.Options = append(cueState.Options, sounds.SystemName(current))
		}
		cueState.Selected = cueState.Options[max(0, slices.Index(cueState.Values, current))]
		state.SystemCues = append(state.SystemCues, cueState)
	}

	state.VolumeValues = slices.Clone(soundVolumes)
	for _, volume := range state.VolumeValues {
		state.VolumeOptions = append(state.VolumeOptions, fmt.Sprintf("%d%%", volume))
	}
	selection := max(soundVolumes[len(soundVolumes)-1], sounds.NearestVolumeStep(input.Prefs.Volume))
	state.VolumeSelection = fmt.Sprintf("%d%%", selection)
	return state
}

func soundSourceID(source soundSource) string {
	switch source {
	case soundBuiltIn:
		return soundBuiltInID
	case soundSystem:
		return soundSystemID
	}
	return soundNoneID
}
