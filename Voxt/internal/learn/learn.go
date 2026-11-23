// Package learn proposes dictionary terms from what the user does with
// Voxt's text. It never adds a term by itself: it puts suggestions in the
// Book, which the Dictionary page shows; the user accepts or dismisses
// them.
//
// Imports: compose, core, deliver, dict, helper/proto, llm, local, store.
package learn

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Source is where a suggestion came from.
type Source string

const (
	FromEdit    Source = "edit"
	FromHistory Source = "history"
)

// Suggestion is a term waiting for the user. It keeps no app, time or
// history entry: only what the row shows.
type Suggestion struct {
	Term   string   `json:"term"`            // as the user wrote it
	Heard  []string `json:"heard,omitempty"` // what Voxt wrote instead: shown, never added as aliases
	Source Source   `json:"source"`
}

// Key is the folded Term, the suggestion's identity.
func (s Suggestion) Key() string { return fold(s.Term) }

// AsTerm is the term "Add" saves: Term only, enabled, no aliases (Voxt 1.x's).
func (s Suggestion) AsTerm() store.Term { return store.Term{Term: s.Term, Enabled: true} }

// fold is dict.Merge's comparison: spelling, ignoring case.
func fold(term string) string { return strings.ToLower(strings.TrimSpace(term)) }
