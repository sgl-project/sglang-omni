package settings

import (
	"encoding/json"
	"slices"
	"strings"
)

// Features: the three features and the shared paste-last and cancel
// shortcuts.
type Features struct {
	Transcribe Transcribe `json:"transcribe"`
	Translate  Translate  `json:"translate"`
	Rewrite    Rewrite    `json:"rewrite"`
	// PasteLast pastes the latest result again (⌃⌥V by default).
	PasteLast Hotkey `json:"pasteLast"`
	// Cancel cancels the session while it records or the answer card
	// shows (Escape by default; empty keys: off). Only a tap, and only
	// through the helper's event tap: it never takes the keys while
	// nothing records.
	Cancel Hotkey `json:"cancel"`
	// Note records a note (Voxt 1.x's noteHotkeyBindings): a Transcribe
	// session whose text goes to the Notes page instead of where the
	// cursor is. Only a tap; empty keys: off.
	Note Hotkey `json:"note"`
	// Sessions switch the session (#4005): the first three go to session
	// 1, 2 and 3 when it exists (empty keys, the default: off), the
	// fourth to the next session in the list, around. Rewrite, Translate
	// and Transcribe run in the current session, each session keeping
	// its own answer card and conversation. Only taps.
	Sessions [4]Hotkey `json:"sessions,omitzero"`
	// SessionsOn turns the session shortcuts on. Off by default: sessions
	// are an experiment until people are seen to go back and forth.
	SessionsOn bool `json:"sessionsOn,omitempty"`
	// ReturnFinishes stops a Transcribe, Translate or Rewrite recording
	// when Return is pressed (Settings → Shortcuts); the key is consumed,
	// not sent after the text is inserted. Not a note's. Off: Return goes
	// to the app as usual. Until 2.1.3 it was Transcribe's alone
	// (features.transcribe.returnFinishes, migrateReturnFinishes).
	ReturnFinishes bool `json:"returnFinishes"`
	// StrictShortcuts: while recording, only the shortcuts of the feature
	// that started the session stop it; another feature's do nothing.
	// Off: any feature's shortcut stops (Voxt 1.x's common stop, #3823).
	StrictShortcuts bool    `json:"strictShortcuts,omitempty"`
	Meeting         Meeting `json:"meeting,omitzero"`
	// EndWord ends a recording when it is the last thing said (Voxt 1.x's
	// voiceEndCommandText), and is left out of the text; "" is off. Only
	// a speech model that streams hears it (internal/session).
	EndWord string `json:"endWord,omitempty"`
	// Hidden are the features left out of the sidebar, Home and the
	// command palette (Settings → General, as Voxt 1.x's home could):
	// HideTranslate, HideRewrite, HideAppEnhancement, HideNotes,
	// HideMeetings. A hidden feature's shortcuts and scene rules are off;
	// its settings and saved data stay.
	Hidden []string `json:"hidden,omitempty"`
	// CardSizes are the answer card's sizes by feature (store.Kind:
	// transcribe, translate, rewrite), 0 small, 1
	// medium, 2 large, 3 fill (layout.CardSizes); each feature its own
	// (#3644 FB41).
	CardSizes map[string]int `json:"cardSizes,omitempty"`
}

// The features Features.Hidden can hold, each its page's path without the
// slash: Translate's and Rewrite's are also internal/hotkey's ids.
const (
	HideTranslate      = "translate"
	HideRewrite        = "rewrite"
	HideAppEnhancement = "app-enhancement"
	HideNotes          = "notes"
	HideMeetings       = "meetings"
)

// IsHidden reports whether feature id is hidden (Features.Hidden).
func (f Features) IsHidden(id string) bool { return slices.Contains(f.Hidden, id) }

// CardSizeSteps is how many sizes the answer card has.
const CardSizeSteps = 4

// CardSize is the answer card's size for feature kind, 0 to
// CardSizeSteps-1 (Features.CardSizes).
func (f Features) CardSize(kind string) int { return min(max(f.CardSizes[kind], 0), CardSizeSteps-1) }

// ShowsCard reports whether feature kind ("transcribe", "translate",
// "rewrite") shows its results in the answer card: each feature's own
// switch.
func (f Features) ShowsCard(kind string) bool {
	switch kind {
	case "transcribe":
		return f.Transcribe.ShowCard
	case "translate":
		return f.Translate.ShowCard
	case "rewrite":
		return f.Rewrite.ShowCard
	}
	return false
}

// Feature is what every feature has: its shortcuts, the models it uses
// (internal/local's Model.ID: a local model's, or LLM a remote one's,
// Provider.ID; "" = the first one installed, the recommended one) and the
// user's addition to its task prompt, which the composer merges with the
// built-in rules.
type Feature struct {
	Hotkeys []Hotkey `json:"hotkeys,omitempty"`
	ASR     string   `json:"asr,omitempty"`
	LLM     string   `json:"llm,omitempty"`
	Prompt  string   `json:"prompt,omitempty"`
}

// Transcribe: speech → text, inserted where the cursor is.
type Transcribe struct {
	Feature
	// Enhance runs the LLM over the transcript; Cleanup and Structure
	// are the composer's capability layers it uses.
	Enhance   bool `json:"enhance"`
	Cleanup   bool `json:"cleanup"`
	Structure bool `json:"structure,omitempty"`
	// ShowCard shows the text in the answer card instead of typing it.
	ShowCard bool `json:"showCard,omitempty"`
	// note (luojiaxuan): CleanAfterStop sends nothing to the language model while the user
	// speaks: the whole clean-up runs after Stop. Off by default: the
	// clean-up starts on what the speech model has written so far (the
	// session's live and early clean-ups), the Transcribe page's "Clean up
	// while you speak".
	CleanAfterStop bool `json:"cleanAfterStop,omitempty"`
	// note (luojiaxuan): RecleanSentences is how many sentences of the text
	// the live clean-up cleaned go back to the model with the text left at
	// Stop, so that what was said last can correct them; 0: only the text
	// left (DefaultRecleanSentences, no page).
	RecleanSentences int `json:"recleanSentences"`
}

// note (luojiaxuan): DefaultRecleanSentences is Transcribe.RecleanSentences's
// default: the fewest sentences with which both language models of an
// offline AASR-Bench run (150 dictations, recleanSentences 0, 1, 2) scored
// no lower than a clean-up after Stop alone (behavior.md 5.31b).
const DefaultRecleanSentences = 2

// Translate: speech or selected text → a translation.
type Translate struct {
	Feature
	Cleanup bool `json:"cleanup"`
	// Target is the language to translate into (a lang code, "zh-Hans",
	// "zh-Hant"); "" = English, or Chinese when the main language is
	// English (session.TranslateTarget).
	Target string `json:"target,omitempty"`
	// Fallback is the language model asked once more when LLM fails or
	// does not open; "" = none.
	Fallback string `json:"fallback,omitempty"`
	// ShowCard shows the translation in the answer card instead of
	// typing it (or replacing the selection).
	ShowCard bool `json:"showCard,omitempty"`
}

// Rewrite: a spoken instruction rewrites the selection, or writes new
// text when nothing is selected.
type Rewrite struct {
	Feature
	// Cleanup reads past fillers and self-corrections in the spoken
	// instruction (never applied to the selection).
	Cleanup bool `json:"cleanup"`
	// ShowCard sends every answer to the answer card, which takes
	// follow-ups, instead of replacing the selection or typing it: on by
	// default, as in Voxt 1.x (no omitempty: off is kept). The field
	// cardChars of 0.1.0 is dropped; so is inputBox (never released):
	// the input box is the only Rewrite now.
	ShowCard bool `json:"showCard"`
}

// AppEnhancement: scene rules by app and URL. The groups are the
// document "appctx:rules"; these are the switches around them.
type AppEnhancement struct {
	Enabled bool `json:"enabled"`
	// Browsers are browsers the user added to the built-in ones, whose
	// tab URL VoxtNew tries to read.
	Browsers []Browser `json:"browsers,omitempty"`
}

// Browser is a browser by bundle id.
type Browser struct {
	BundleID string `json:"bundleID"`
	Name     string `json:"name,omitempty"`
}

// migrateReturnFinishes keeps the Return switch Transcribe had before it
// covered every feature (features.transcribe.returnFinishes): a document
// without features.returnFinishes takes it.
func migrateReturnFinishes(data []byte, f *Features) {
	var doc struct {
		Features struct {
			ReturnFinishes *bool `json:"returnFinishes"`
			Transcribe     struct {
				ReturnFinishes *bool `json:"returnFinishes"`
			} `json:"transcribe"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) == nil && doc.Features.ReturnFinishes == nil && doc.Features.Transcribe.ReturnFinishes != nil {
		f.ReturnFinishes = *doc.Features.Transcribe.ReturnFinishes
	}
}

// DefaultSessionNext is the next-session shortcut's default (#4005): the
// one session shortcut bound by default, two keys at most (the owner's
// rule), free in the default set and unused by macOS. The shortcuts for
// session 1, 2 and 3 are off until the user binds them.
const DefaultSessionNext = "Option+Tab"

// migrateSessionKeys gives a document from before the session shortcuts
// their defaults, unless another shortcut can match the same press
// (chordOverlaps: "LeftOption+Tab" takes "Option+Tab" too): that one
// then stays off. Such a document has no sessions key; one that has it
// is the user's.
func migrateSessionKeys(data []byte, f *Features) {
	var doc struct {
		Features struct {
			Sessions json.RawMessage `json:"sessions"`
		} `json:"features"`
	}
	if json.Unmarshal(data, &doc) != nil || doc.Features.Sessions != nil {
		return
	}
	defaults := Defaults().Features.Sessions
	for i, d := range defaults {
		if d.Keys != "" && shortcutTaken(*f, d.Keys) {
			f.Sessions[i] = Hotkey{}
		}
	}
}

// shortcutTaken reports whether a shortcut in f can match the same press
// as keys, as hotkey.Check counts a clash.
func shortcutTaken(f Features, keys string) bool {
	all := append([]Hotkey{f.PasteLast, f.Cancel, f.Note}, f.Transcribe.Hotkeys...)
	all = append(append(all, f.Translate.Hotkeys...), f.Rewrite.Hotkeys...)
	for _, h := range all {
		if chordOverlaps(h.Keys, keys) {
			return true
		}
	}
	return false
}

// chordModifiers maps a modifier's spelling to its family and side (""
// for either), as internal/hotkey's grammar writes them.
var chordModifiers = map[string][2]string{
	"Control": {"control", ""}, "LeftControl": {"control", "left"}, "RightControl": {"control", "right"},
	"Option": {"option", ""}, "LeftOption": {"option", "left"}, "RightOption": {"option", "right"},
	"Shift": {"shift", ""}, "LeftShift": {"shift", "left"}, "RightShift": {"shift", "right"},
	"Command": {"command", ""}, "LeftCommand": {"command", "left"}, "RightCommand": {"command", "right"},
	"Fn": {"fn", ""},
}

// chordOverlaps reports whether two shortcut spellings can match the same
// press, as hotkey.Chord.Overlaps judges a clash: the same key (or both
// modifier-only) and, per modifier family, sides that meet (either side
// or the same). It cannot use internal/hotkey, which imports this
// package.
func chordOverlaps(a, b string) bool {
	ka, ma := splitChord(a)
	kb, mb := splitChord(b)
	if ma == nil || mb == nil || ka != kb || len(ma) != len(mb) {
		return false
	}
	for fam, sa := range ma {
		sb, ok := mb[fam]
		if !ok || sa != sb && sa != "" && sb != "" {
			return false
		}
	}
	return true
}

// splitChord splits a spelling ("Control+LeftOption+1") into its key
// ("1"; "" for a modifier-only chord) and its modifier families with
// their sides. mods is nil for a spelling this grammar does not know.
func splitChord(keys string) (key string, mods map[string]string) {
	parts := strings.Split(keys, "+")
	mods = map[string]string{}
	for i, part := range parts {
		if m, ok := chordModifiers[part]; ok {
			if _, dup := mods[m[0]]; dup {
				return "", nil
			}
			mods[m[0]] = m[1]
			continue
		}
		if i != len(parts)-1 || part == "" {
			return "", nil
		}
		key = part
	}
	return key, mods
}
