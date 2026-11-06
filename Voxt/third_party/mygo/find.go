package mygo

import (
	"fmt"

	"github.com/egoist/mygo/internal/platform"
)

// FindOptions configures Page.FindInPage.
type FindOptions struct {
	// MatchCase finds only text with the same capitalization.
	MatchCase bool
	// Backward goes to the previous match rather than the next one.
	Backward bool
	// FindNext moves on from the current match of the same text; without
	// it the search starts over at the first match (the last, Backward).
	FindNext bool
}

// FindResult reports the matches of Page.FindInPage.
type FindResult struct {
	// Matches is how many there are on the page, Active which of them is
	// the current one, from 1; both are 0 without matches.
	Matches int `json:"matches"`
	Active  int `json:"active"`
}

// FindInPage highlights the occurrences of text in the page, scrolls to
// the active one and reports how many there are, like the find bar of a
// browser: call it as the user types, and with FindNext to go to the next
// match.
//
//	res, err := win.Page().FindInPage("mygo", mygo.FindOptions{FindNext: true})
//
// Text split by markup, such as "my<b>go</b>", is not found.
func (p *Page) FindInPage(text string, opts FindOptions) (FindResult, error) {
	res, err := EvalAs[FindResult](p, findScript(text, opts))
	if err != nil {
		return FindResult{}, fmt.Errorf("mygo: finding in the page: %w", err)
	}
	return res, nil
}

// StopFindInPage removes the highlights of FindInPage.
func (p *Page) StopFindInPage() {
	p.w.page(func(n platform.Window) { n.Eval("window.__mygo && window.__mygo.stopFind()") })
}
