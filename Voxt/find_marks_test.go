package main

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestFinderAdvance: Return steps forward and Shift+Return back, both
// wrapping; Shift+Return first goes to the last match; a list that shrank
// under the current match starts again; no matches, no move.
func TestFinderAdvance(t *testing.T) {
	var f finder
	var got []int
	for _, step := range []int{1, 1, 1, 1, -1, -1} {
		f.step = step
		f.advance(3)
		got = append(got, f.cur)
	}
	if want := []int{1, 2, 3, 1, 3, 2}; !slices.Equal(got, want) {
		t.Errorf("steps %v, want %v", got, want)
	}
	f = finder{step: -1}
	if !f.advance(5) || f.cur != 5 {
		t.Errorf("Shift+Return first: %d", f.cur)
	}
	f = finder{cur: 5, step: 1}
	if !f.advance(2) || f.cur != 1 {
		t.Errorf("after the list shrank: %d", f.cur)
	}
	f = finder{step: 2} // two Returns in one frame
	if f.advance(3); f.cur != 2 {
		t.Errorf("two Returns: %d", f.cur)
	}
	f = finder{step: -2}
	if f.advance(3); f.cur != 2 {
		t.Errorf("two Shift+Returns: %d", f.cur)
	}
	f = finder{step: 1}
	if f.advance(0) || f.cur != 0 || f.step != 0 {
		t.Errorf("no matches: %+v", f)
	}
}

// TestMarkSpans: the matches are cut out of the text, ignoring case, in
// Chinese mixed with English too, the current one marked apart; an empty
// text is one empty span.
func TestMarkSpans(t *testing.T) {
	red := ui.Hex("#eb5230")
	var texts []string
	var current string
	for _, s := range markSpans(red, ui.Span{}, "豆包 Doubao 和 doubao", "DOUBAO", 1) {
		texts = append(texts, s.Text)
		if s.Background == red {
			current = s.Text
		}
	}
	if s := markSpans(red, ui.Span{}, "", "x", -1); len(s) != 1 || s[0].Text != "" {
		t.Errorf("empty text: %+v", s)
	}
	if want := []string{"豆包 ", "Doubao", " 和 ", "doubao"}; !slices.Equal(texts, want) {
		t.Errorf("spans %q, want %q", texts, want)
	}
	if current != "doubao" {
		t.Errorf("current %q", current)
	}
}

// TestDictionaryFind (#3869): the dictionary's search counts the matches
// in the terms, aliases and notes, marks them in red, and Return and
// Shift+Return step through them, the list scrolling to the current one.
func TestDictionaryFind(t *testing.T) {
	a, tt := dictTester(t)
	for i := range 120 {
		tm := store.Term{ID: fmt.Sprintf("z%03d", i), Term: fmt.Sprintf("Zebra %03d", i), Enabled: true}
		if i == 0 {
			tm.Aliases, tm.Note = []string{"zebra crossing"}, "a ZEBRA"
		}
		if err := a.lib.store.Dictionary().Put(context.Background(), tm); err != nil {
			t.Fatal(err)
		}
	}
	a.loadDict()
	frames(tt)
	click(t, tt, "Search dictionary")
	tt.Type("zebra")
	frames(tt)
	if !has(tt, "122 matches") {
		t.Fatalf("no count: %q", firstTexts(tt))
	}
	img, red := tt.Image(), tok(ui.LightTheme()).Danger
	seen := 0
	for y := range img.Bounds().Dy() {
		for x := range img.Bounds().Dx() {
			if near(img.RGBAAt(x, y), red) {
				seen++
			}
		}
	}
	if seen < 20 {
		t.Errorf("the matches are not marked in red: %d red pixels", seen)
	}
	tt.Key(ui.Shift, ui.KeyEnter)
	frames(tt)
	if !has(tt, "122 of 122") {
		t.Errorf("Shift+Return first: no “122 of 122” in %q", firstTexts(tt))
	}
	if first, last := a.dv.listUI.Visible(); last != 119 {
		t.Errorf("the last match's row is not in view: rows %d..%d", first, last)
	}
	for _, want := range []string{"1 of 122", "2 of 122", "3 of 122", "4 of 122"} {
		tt.Key(0, ui.KeyEnter)
		frames(tt)
		if !has(tt, want) {
			t.Errorf("after Return: no %q (at %d)", want, a.dv.find.cur)
		}
	}
	if first, _ := a.dv.listUI.Visible(); first != 0 {
		t.Errorf("the first match's row is not in view: from row %d", first)
	}
}

// TestHistoryFind (#3869): a History row shows the line holding the
// search, its match marked and counted; Return goes to it.
func TestHistoryFind(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedHistory(t, a)
	a.navigate("/history")
	frames(tt)
	click(t, tt, "Search history")
	tt.Type("agenda") // on the second line of the result
	frames(tt)
	if !has(tt, "I'll send the new agenda tonight.") || !has(tt, "1 match") {
		t.Fatalf("no matching line or count: %q", firstTexts(tt))
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if !has(tt, "1 of 1") {
		t.Errorf("after Return: %q", firstTexts(tt))
	}
}
