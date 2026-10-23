// Package eval is VoxtNew's quality evaluation of the curated models: the
// eval set of synthesised dictations (testdata/eval: clips.json, a WAV per
// clip), with their reference text and the expected cleaned text, and the
// scores: error rates of speech recognition (characters for Chinese, words
// for English, both in mixed speech), the dictionary's terms recognised,
// and checks of a language model's clean-up (fillers removed,
// self-corrections applied, nothing invented, the prompt's example not
// copied). The runs themselves are tests on a Mac (eval_*_test.go in
// package main, VOXTNEW_EVAL=1); the numbers are in docs/eval.md.
//
// Imports: asr.
package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
)

// Clip is one dictation of the set.
type Clip struct {
	ID string `json:"id"`
	// Set is the clip's group: zh-filler, zh-plain, en, mix, dict, long.
	Set string `json:"set"`
	// Lang is the language mostly spoken ("zh", "en").
	Lang string `json:"lang"`
	// Voice, Seed and Speak are the synthesis's (internal/tools/evalsynth).
	Voice string `json:"voice,omitempty"`
	Seed  int    `json:"seed,omitempty"`
	Speak string `json:"speak,omitempty"`
	// Text is what is said, word for word (fillers included); Clean what a
	// clean-up should make of it.
	Text  string `json:"text"`
	Clean string `json:"clean"`
	// Must are phrases the cleaned text keeps ("a|b": either), MustNot
	// phrases it drops (fillers, what a self-correction replaced). Both
	// compare without case.
	Must    []string `json:"must,omitempty"`
	MustNot []string `json:"mustNot,omitempty"`
	// MustEn and MustNotEn are the same of an English translation (a
	// Chinese clip spoken for Translate).
	MustEn    []string `json:"mustEn,omitempty"`
	MustNotEn []string `json:"mustNotEn,omitempty"`
	// Terms are the dictionary's terms said in the clip; Heard is how a
	// speech model wrote the clip without the dictionary (the glossary's
	// test: the clean-up should write the terms right).
	Terms []string `json:"terms,omitempty"`
	Heard string   `json:"heard,omitempty"`
	// PCM is the audio: 16-bit mono at 16 kHz (Load with audio).
	PCM []byte `json:"-"`
}

// Duration is the clip's audio's.
func (c Clip) Duration() time.Duration { return asr.Duration(c.PCM, asr.SampleRate) }

// Long is a long dictation made of clips, joined with Gap seconds of
// silence (the set keeps short files only).
type Long struct {
	ID   string   `json:"id"`
	Set  string   `json:"set"`
	Lang string   `json:"lang"`
	Gap  float64  `json:"gap"`
	Of   []string `json:"of"`
}

// Set is the eval set.
type Set struct {
	// Dictionary is the user's dictionary in the runs: every term of the
	// set (the recognition's context bias, the clean-up's glossary).
	Dictionary []string `json:"dictionary"`
	// MoreTerms are the dict2 clips' terms: in the dictionary evals'
	// dictionaries (eval_dict_test.go), not in Dictionary, which the other
	// evals send.
	MoreTerms []string `json:"moreTerms"`
	// Aliases are what the user noted the speech model writes for a term.
	Aliases map[string][]string `json:"aliases"`
	Clips   []Clip              `json:"clips"`
	Long    []Long              `json:"long"`
}

// Load reads the set in dir (clips.json), with the clips' audio when
// audio is set; Long's clips are built from theirs.
func Load(dir string, audio bool) (*Set, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "clips.json"))
	if err != nil {
		return nil, err
	}
	var s Set
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("clips.json: %w", err)
	}
	seen := map[string]bool{}
	for i, c := range s.Clips {
		if c.ID == "" || seen[c.ID] || strings.TrimSpace(c.Text) == "" || strings.TrimSpace(c.Clean) == "" {
			return nil, fmt.Errorf("clips.json: clip %d (%q): an id of its own, a text and a clean text", i, c.ID)
		}
		seen[c.ID] = true
		if !audio {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, c.ID+".wav"))
		if err != nil {
			return nil, err
		}
		pcm, rate, ch, err := asr.ReadWAV(b)
		if err != nil || rate != asr.SampleRate || ch != 1 {
			return nil, fmt.Errorf("%s.wav: want 16 kHz mono PCM16 (%d Hz × %d, %v)", c.ID, rate, ch, err)
		}
		s.Clips[i].PCM = pcm
	}
	for _, l := range s.Long {
		for _, id := range l.Of {
			if !seen[id] {
				return nil, fmt.Errorf("clips.json: %s joins %q, not a clip", l.ID, id)
			}
		}
	}
	return &s, nil
}

// Clip finds a clip by its id.
func (s *Set) Clip(id string) (Clip, bool) {
	i := slices.IndexFunc(s.Clips, func(c Clip) bool { return c.ID == id })
	if i < 0 {
		return Clip{}, false
	}
	return s.Clips[i], true
}

// Join is l as one clip: the texts joined (a space between two Latin
// ends), the audio joined with l.Gap of silence, the checks and terms
// together (for an English translation, the English clips' checks too).
func (s *Set) Join(l Long) Clip {
	out := Clip{ID: l.ID, Set: l.Set, Lang: l.Lang}
	gap := make([]byte, 2*int(l.Gap*asr.SampleRate))
	var texts, cleans []string
	for i, id := range l.Of {
		c, _ := s.Clip(id)
		texts, cleans = append(texts, c.Text), append(cleans, c.Clean)
		if i > 0 {
			out.PCM = append(out.PCM, gap...)
		}
		out.PCM = append(out.PCM, c.PCM...)
		out.Must = append(out.Must, c.Must...)
		out.MustNot = append(out.MustNot, c.MustNot...)
		// An English translation: the Chinese clips' checks, the English
		// clips' own (their phrases stay English).
		if c.Lang == "en" {
			out.MustEn, out.MustNotEn = append(out.MustEn, c.Must...), append(out.MustNotEn, c.MustNot...)
		} else {
			out.MustEn, out.MustNotEn = append(out.MustEn, c.MustEn...), append(out.MustNotEn, c.MustNotEn...)
		}
		for _, t := range c.Terms {
			if !slices.Contains(out.Terms, t) {
				out.Terms = append(out.Terms, t)
			}
		}
	}
	out.Text, out.Clean = joinText(texts), joinText(cleans)
	return out
}

func joinText(parts []string) string {
	var b strings.Builder
	for _, p := range parts {
		if s := b.String(); s != "" && s[len(s)-1] < 0x80 {
			b.WriteByte(' ')
		}
		b.WriteString(p)
	}
	return b.String()
}

// All is every clip, then every long one (with audio when the set has it).
func (s *Set) All() []Clip {
	out := slices.Clone(s.Clips)
	for _, l := range s.Long {
		out = append(out, s.Join(l))
	}
	return out
}
