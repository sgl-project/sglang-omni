package main

// Selected short text to the dictionary, as Voxt 1.x's
// addSelectedShortTextToDictionaryIfPossible: the transcribe shortcut,
// pressed with no session going while a few words are selected in the app
// in front, adds them to the dictionary instead of recording, and the
// overlay says so. What counts is Voxt 1.x's rule
// (SystemSelectionTextSupport.dictionaryCandidateTerm), then dict.Check.
// The user chose the words: they go in as a term at once. The selection
// is read with Accessibility only, never ⌘C, and for at most
// selectionWait: the shortcut must start recording at once when nothing
// is selected. With the dictionary off, the shortcut records.

import (
	"context"
	"errors"
	"log"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Voxt 1.x's limits of a selection taken as a term.
const (
	maxCandidateWords  = 5  // separated by white space
	maxCandidateCJKRun = 10 // CJK characters in a row
)

// selectionWait bounds the read of the selection before the transcribe
// shortcut records.
const selectionWait = 300 * time.Millisecond

// selectionSkipped are the apps whose selection Voxt 1.x never took as a
// term (SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow).
var selectionSkipped = map[string]bool{"md.obsidian": true}

// cjkRanges are the characters Voxt 1.x counted as CJK
// (DictionaryMatchingSupport): Han, kana and Hangul.
var cjkRanges = [][2]rune{
	{0x4E00, 0x9FFF}, {0x3400, 0x4DBF}, {0x20000, 0x2A6DF}, {0x2A700, 0x2B73F}, {0x2B740, 0x2B81F}, {0x2B820, 0x2CEAF},
	{0x3040, 0x309F}, {0x30A0, 0x30FF}, {0x31F0, 0x31FF}, {0xFF66, 0xFF9F},
	{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xA960, 0xA97F}, {0xAC00, 0xD7AF}, {0xD7B0, 0xD7FF},
}

// dictionaryCandidate is the term selected text makes, "" when it is
// none: empty, no letter or digit, more than 5 words, more than 10 CJK
// characters in a row, or what dict.Check refuses.
func dictionaryCandidate(selected string) string {
	text := strings.TrimSpace(selected)
	if !strings.ContainsFunc(text, func(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }) ||
		len(strings.Fields(text)) > maxCandidateWords || longestCJKRun(text) > maxCandidateCJKRun {
		return ""
	}
	term := dict.Clean(store.Term{Term: text})
	if dict.Check(term) != nil {
		return ""
	}
	return term.Term
}

// longestCJKRun is the most CJK characters in a row in text.
func longestCJKRun(text string) int {
	longest, run := 0, 0
	for _, r := range text {
		run++
		if !isCJK(r) {
			run = 0
		}
		longest = max(longest, run)
	}
	return longest
}

func isCJK(r rune) bool {
	for _, span := range cjkRanges {
		if r >= span[0] && r <= span[1] {
			return true
		}
	}
	return false
}

// selectionPress is the transcribe shortcut while its selection is read:
// a long press released, or the cancel shortcut, keeps it from recording.
type selectionPress struct {
	req                 session.StartRequest
	released, cancelled bool
}

// pressDuringSelection takes a shortcut pressed while the selection is
// read; it reports whether there was such a read.
func (a *app) pressDuringSelection(e proto.HotkeyEventParams) bool {
	p := a.selPress
	if p == nil {
		return false
	}
	switch {
	case e.Binding == proto.HotkeyCancel:
		p.cancelled = true
	case e.Phase == proto.PhaseUp && e.Binding == p.req.HoldBinding:
		p.released = true
	}
	return true
}

// transcribeOrAddSelection starts the transcribe session req, unless the
// app in front has a few words selected: they go to the dictionary.
func (a *app) transcribeOrAddSelection(req session.StartRequest) {
	if !a.prefs.Dictionary.Enabled || selectionSkipped[req.Target.BundleID] {
		a.startSession(req)
		return
	}
	a.selPress = &selectionPress{req: req}
	helperCall(a, proto.MethodTextSelection, proto.SelectionParams{}, selectionWait, func(res proto.SelectionResult, err error) {
		p := a.selPress
		a.selPress = nil
		term := ""
		if err == nil {
			term = dictionaryCandidate(res.Text)
		}
		switch {
		case term != "":
			a.addSelectedTerm(term)
		case p.released, p.cancelled: // over before it could record
		default:
			a.startSession(p.req)
		}
	})
}

// addSelectedTerm adds term to the dictionary, enabled, unless a term of
// the same spelling (ignoring case) is there, and the overlay says which.
func (a *app) addSelectedTerm(term string) {
	repo := a.lib.store.Dictionary()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		have, err := repo.List(ctx, term)
		existing := ""
		for _, t := range have {
			if strings.EqualFold(t.Term, term) {
				existing = t.Term
			}
		}
		if err == nil && existing == "" {
			now := time.Now()
			err = repo.Put(ctx, store.Term{ID: core.NewID(now), Term: term, Enabled: true, CreatedAt: now, UpdatedAt: now})
			if errors.Is(err, core.ErrExists) {
				existing, err = term, nil
			}
		}
		a.post(func() {
			switch {
			case err != nil:
				log.Printf("dictionary: the selection (%d characters): %v", utf8.RuneCountInString(term), err)
				a.showNotice(sessionNotice{text: i18n.T("Could not add it to the dictionary.")})
			case existing != "":
				a.showNotice(sessionNotice{text: i18n.T("Already in the dictionary: %s", existing), ok: true})
			default:
				log.Printf("dictionary: the selection added (%d characters)", utf8.RuneCountInString(term))
				a.showNotice(sessionNotice{text: i18n.T("Added to the dictionary: %s", term), ok: true})
				a.loadDict() // dictionary_page.go
			}
		})
	})
}
