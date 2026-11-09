package settings

import "slices"

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
	Note    Hotkey  `json:"note"`
	Meeting Meeting `json:"meeting,omitzero"`
	// EndWord ends a recording when it is the last thing said (Voxt 1.x's
	// voiceEndCommandText), and is left out of the text; "" is off. Only
	// a speech model that streams hears it (internal/session).
	EndWord string `json:"endWord,omitempty"`
	// ShowCard shows every feature's result in the answer card instead of
	// inserting it (or replacing the selection): one switch for all.
	ShowCard bool `json:"showCard,omitempty"`
	// Hidden are the features left out of the sidebar, Home and the
	// command palette (Settings → General, as Voxt 1.x's home could):
	// HideTranslate, HideRewrite, HideAppEnhancement. A hidden feature's
	// shortcuts and scene rules are off; its settings stay.
	Hidden []string `json:"hidden,omitempty"`
	// CardSizes are the answer card's sizes by feature (store.Kind:
	// "transcribe", "translate", "rewrite"): 0 small (the default), 1
	// medium, 2 large (layout.CardSizes); each feature its own (#3644 FB41).
	CardSizes map[string]int `json:"cardSizes,omitempty"`
}

// The features Features.Hidden can hold: Translate's and Rewrite's are
// internal/hotkey's ids.
const (
	HideTranslate      = "translate"
	HideRewrite        = "rewrite"
	HideAppEnhancement = "app-enhancement"
)

// IsHidden reports whether feature id is hidden (Features.Hidden).
func (f Features) IsHidden(id string) bool { return slices.Contains(f.Hidden, id) }

// CardSizeSteps is how many sizes the answer card has.
const CardSizeSteps = 3

// CardSize is the answer card's size for feature kind, 0 to
// CardSizeSteps-1 (Features.CardSizes).
func (f Features) CardSize(kind string) int { return min(max(f.CardSizes[kind], 0), CardSizeSteps-1) }

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
}

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
}

// Rewrite: a spoken instruction rewrites the selection, or writes new
// text when nothing is selected.
type Rewrite struct {
	Feature
	// Cleanup reads past fillers and self-corrections in the spoken
	// instruction (never applied to the selection). Every answer shows in
	// the answer card (Voxt's; the field cardChars of 0.1.0 is dropped).
	Cleanup bool `json:"cleanup"`
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
