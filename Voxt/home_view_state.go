package main

import (
	"strings"
	"time"
	"unicode"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type homeSloganViewState struct {
	Before string `json:"before"`
	Keys   string `json:"keys"`
	After  string `json:"after"`
}

type homeKindCountViewState struct {
	Kind  string `json:"kind"`
	Title string `json:"title"`
	Count string `json:"count"`
}

type homeTodayViewState struct {
	Count           string                   `json:"count"`
	CountLabel      string                   `json:"count_label"`
	Characters      string                   `json:"characters"`
	CharactersLabel string                   `json:"characters_label"`
	Kinds           []homeKindCountViewState `json:"kinds"`
}

type homeRecentViewState struct {
	Kind      string              `json:"kind"`
	KindTitle string              `json:"kind_title"`
	Row       historyRowViewState `json:"row"`
}

type homeViewState struct {
	Read         bool                     `json:"read"`
	Loading      bool                     `json:"loading"`
	Error        string                   `json:"error"`
	KeepHistory  bool                     `json:"keep_history"`
	Slogan       homeSloganViewState      `json:"slogan"`
	Missing      []homeReadinessViewState `json:"missing"`
	Ready        []homeReadinessViewState `json:"ready"`
	ShowTry      bool                     `json:"show_try"`
	TryAvailable bool                     `json:"try_available"`
	TryRecording bool                     `json:"try_recording"`
	Today        homeTodayViewState       `json:"today"`
	Recent       []homeRecentViewState    `json:"recent"`
}

type homeViewStateInput struct {
	Read         bool
	Loading      bool
	Today        dayStats
	Recent       []store.Entry
	Error        string
	KeepHistory  bool
	Features     settings.Features
	Readiness    []homeReadinessViewState
	ShowTry      bool
	TryAvailable bool
	TryRecording bool
	Retry        entryRetry
	Now          time.Time
}

// dayStats are a day's finished results: how many, and their characters
// (without spaces), by feature too.
type dayStats struct {
	count, chars int
	byKind       map[store.Kind]int
}

func buildHomeViewState(translator *i18n.Translator, input homeViewStateInput) homeViewState {
	missing, ready := homeReadinessGroups(input.Readiness)
	state := homeViewState{
		Read: input.Read, Loading: input.Loading, Error: input.Error, KeepHistory: input.KeepHistory,
		Slogan: buildHomeSloganViewState(translator, input.Features), Missing: missing, Ready: ready,
		ShowTry: input.ShowTry, TryAvailable: input.TryAvailable, TryRecording: input.TryRecording,
		Today:  buildHomeTodayViewState(translator, input.Today, input.Features),
		Recent: make([]homeRecentViewState, len(input.Recent)),
	}
	for index, entry := range input.Recent {
		state.Recent[index] = homeRecentViewState{
			Kind: string(entry.Kind), KindTitle: historyKindName(translator, entry.Kind),
			Row: buildHistoryRowViewState(translator, entry, "", "", false, input.Retry, input.Now),
		}
	}
	return state
}

func homeReadinessGroups(list []homeReadinessViewState) (missing, ready []homeReadinessViewState) {
	for _, item := range list {
		if item.OK {
			ready = append(ready, item)
		} else {
			missing = append(missing, item)
		}
	}
	return missing, ready
}

func buildHomeSloganViewState(translator *i18n.Translator, features settings.Features) homeSloganViewState {
	var first *settings.Hotkey
	if hotkeys := hotkey.HotkeysOf(features, hotkey.Transcribe); len(hotkeys) > 0 {
		first = &hotkeys[0]
	}
	before, keys, after := sloganHintFor(translator, first)
	return homeSloganViewState{Before: before, Keys: keys, After: after}
}

func buildHomeTodayViewState(translator *i18n.Translator, stats dayStats, features settings.Features) homeTodayViewState {
	state := homeTodayViewState{
		Count: itoa(stats.count), CountLabel: translator.N(stats.count, "result", "results"),
		Characters: groupDigits(stats.chars), CharactersLabel: translator.N(stats.chars, "character", "characters"),
	}
	for _, kind := range store.Kinds {
		if features.IsHidden(string(kind)) {
			continue
		}
		state.Kinds = append(state.Kinds, homeKindCountViewState{
			Kind: string(kind), Title: historyKindName(translator, kind), Count: itoa(stats.byKind[kind]),
		})
	}
	return state
}

// sloganMark stands for the keys in a translated hint, which is cut
// around it to put a key cap there.
const sloganMark = "\x00"

func sloganHint(hotkey *settings.Hotkey) (before, keys, after string) {
	return sloganHintFor(i18n.CurrentTranslator(), hotkey)
}

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
