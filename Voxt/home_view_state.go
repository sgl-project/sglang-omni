package main

import (
	"strings"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type homeSloganViewState struct {
	Before string `json:"before"`
	Keys   string `json:"keys"`
	After  string `json:"after"`
	Model  string `json:"model,omitempty"`
}

type homeUsage struct {
	asrMS     int64
	asrCount  int
	lastModel string
}

type homeViewState struct {
	KeepHistory bool                `json:"keep_history"`
	Slogan      homeSloganViewState `json:"slogan"`
	// Activity is the activity calendar once its log was read.
	Activity *homeActivityViewState `json:"activity,omitempty"`
}

type homeViewStateInput struct {
	KeepHistory  bool
	Features     settings.Features
	Now          time.Time
	Activity     activity.Log
	ActivityRead bool
	ActivityPage int
	Usage        homeUsage
}

func buildHomeViewState(translator *i18n.Translator, input homeViewStateInput) homeViewState {
	state := homeViewState{
		KeepHistory: input.KeepHistory,
		Slogan:      buildHomeSloganViewState(translator, input.Features, input.Usage.lastModel),
	}
	if input.ActivityRead {
		activity := buildHomeActivityViewState(translator, input.Activity, input.ActivityPage, input.Now, input.Usage)
		state.Activity = &activity
	}
	return state
}

func buildHomeSloganViewState(translator *i18n.Translator, features settings.Features, lastModel string) homeSloganViewState {
	var first *settings.Hotkey
	if hotkeys := hotkey.HotkeysOf(features, hotkey.Transcribe); len(hotkeys) > 0 {
		first = &hotkeys[0]
	}
	before, keys, after := sloganHintFor(translator, first)
	state := homeSloganViewState{Before: before, Keys: keys, After: after}
	if lastModel != "" {
		state.Model = uitext.ModelNameFor(translator, lastModel)
	}
	return state
}

// sloganMark stands for the keys in a translated hint, which is cut
// around it to put a key cap there.
const sloganMark = "\x00"

// sloganHintFor is the slogan's second half, how to start Transcribe with
// its first shortcut (h, nil for none), cut around the keys: "Press ",
// "fn", " to start". keys is "" without a shortcut: then before is the
// whole hint.
func sloganHintFor(translator *i18n.Translator, h *settings.Hotkey) (before, keys, after string) {
	if h == nil {
		return translator.T("Add a shortcut to start"), "", ""
	}
	var hint string
	switch h.Trigger {
	case settings.TriggerDouble:
		hint = translator.T("Double-press %s to start", sloganMark)
	case settings.TriggerLong:
		hint = translator.T("Hold %s and speak", sloganMark)
	default:
		hint = translator.T("Press %s to start", sloganMark)
	}
	before, after, _ = strings.Cut(hint, sloganMark)
	return before, uitext.HotkeyTextFor(translator, h.Keys), after
}

// countChars counts the characters of s that are not spaces.
func countChars(s string) int {
	n := 0
	for _, r := range s {
		if !unicode.IsSpace(r) {
			n++
		}
	}
	return n
}

// groupDigits writes n with thousands separators: 12,345.
func groupDigits(n int) string {
	s := itoa(n)
	if n < 0 || len(s) <= 3 {
		return s
	}
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	return b.String()
}
