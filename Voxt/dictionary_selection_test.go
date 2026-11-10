package main

import (
	"context"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestDictionaryCandidate: Voxt 1.x's rule of a selection that is a term
// (SystemSelectionTextSupport.dictionaryCandidateTerm): trimmed, at most
// 5 words and 10 CJK characters in a row, with a letter or a digit; then
// one line, as every term.
func TestDictionaryCandidate(t *testing.T) {
	for _, c := range []struct{ selected, want string }{
		{"  Ghostty\n", "Ghostty"},
		{"Visual Studio Code Insiders Edition", "Visual Studio Code Insiders Edition"}, // 5 words
		{"one two three four five six", ""},
		{"通义千问大模型平台百炼", ""},                // 11 Han in a row
		{"通义千问大模型平台百", "通义千问大模型平台百"},       // 10
		{"通义千问 大模型平台百炼云", "通义千问 大模型平台百炼云"}, // the runs are 4 and 8
		{"ネットワークエンジニア", ""},                // 11 katakana, ー among them
		{"서울특별시 강남구", "서울특별시 강남구"},
		{"GPT-4o\nmini", "GPT-4o mini"},
		{"…", ""},
		{"", ""},
		{strings.Repeat("a", 81), ""}, // dict.Check: at most 80 characters
	} {
		if got := dictionaryCandidate(c.selected); got != c.want {
			t.Errorf("%q: %q, want %q", c.selected, got, c.want)
		}
	}
}

// TestSelectionToDictionary: the transcribe shortcut with "Ghostty"
// selected adds it to the dictionary instead of recording, and the
// overlay says so; pressed again, the overlay says it is there already.
func TestSelectionToDictionary(t *testing.T) {
	set, keys, _ := fileSetup(t, "hello", &fakeChat{text: "Hello."})
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(600, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "Ghostty")+","+ev, set, keys)
	recorded := false
	notice := func(text string) func() bool {
		return func() bool {
			recorded = recorded || a.sess.State == session.Recording
			return a.notice != nil && a.notice.text == text
		}
	}
	a.pump("added", notice("Added to the dictionary: Ghostty"))
	a.pump("there already", notice("Already in the dictionary: Ghostty"))
	terms, err := a.lib.store.Dictionary().List(context.Background(), "")
	if err != nil || len(terms) != 1 || terms[0].Term != "Ghostty" || !terms[0].Enabled {
		t.Fatalf("terms %+v %v", terms, err)
	}
	if recorded || a.sess.ID != "" || len(a.history()) != 0 {
		t.Errorf("recorded: session %+v", a.sess)
	}
}

// TestSelectionNotATermRecords: the transcribe shortcut records as usual
// when the selection is a whole sentence, when only ⌘C would read it
// (never pressed for this), when nothing is selected, or when the
// dictionary is off; the dictionary stays empty.
func TestSelectionNotATermRecords(t *testing.T) {
	for _, c := range []struct {
		name, opts string
		off        bool
	}{
		{"a sentence", helpertest.Opt("selection", "Let's meet at the station after lunch tomorrow."), false},
		{"only by copy", helpertest.Opt("copy-selection", "Ghostty"), false},
		{"none", "", false},
		{"dictionary off", helpertest.Opt("selection", "Ghostty"), true},
	} {
		t.Run(c.name, func(t *testing.T) {
			set, keys, _ := fileSetup(t, "hello", &fakeChat{text: "Hello."})
			ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger))
			a := newSessionApp(t, strings.TrimPrefix(c.opts+","+ev, ","), func(s *settings.Settings) {
				set(s)
				s.Dictionary.Enabled = !c.off
			}, keys)
			a.pump("the recording", func() bool { return a.sess.State == session.Recording })
			if terms, _ := a.lib.store.Dictionary().List(context.Background(), ""); len(terms) != 0 || a.sess.Kind != store.KindTranscribe {
				t.Errorf("terms %+v, session %s", terms, a.sess.Kind)
			}
		})
	}
}
