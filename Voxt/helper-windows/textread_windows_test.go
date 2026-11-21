//go:build windows

package main

import (
	"strings"
	"testing"
	"unicode/utf16"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestCutReadRangeClampsAndWidens(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		requested proto.TextRange
		wantText  string
		wantRange proto.TextRange
	}{
		{
			name: "negative start keeps intersecting text", text: strings.Repeat("x", 20),
			requested: proto.TextRange{Location: -300, Length: 310},
			wantText:  strings.Repeat("x", 10), wantRange: proto.TextRange{Location: 0, Length: 10},
		},
		{
			name: "past end is empty", text: strings.Repeat("x", 20),
			requested: proto.TextRange{Location: 25, Length: 5},
			wantRange: proto.TextRange{Location: 20, Length: 0},
		},
		{
			name: "negative length is empty", text: strings.Repeat("x", 20),
			requested: proto.TextRange{Location: 5, Length: -3},
			wantRange: proto.TextRange{Location: 5, Length: 0},
		},
		{
			name: "start inside surrogate pair", text: "a👍b",
			requested: proto.TextRange{Location: 2, Length: 1},
			wantText:  "👍", wantRange: proto.TextRange{Location: 1, Length: 2},
		},
		{
			name: "end inside surrogate pair", text: "a👍b",
			requested: proto.TextRange{Location: 0, Length: 2},
			wantText:  "a👍", wantRange: proto.TextRange{Location: 0, Length: 3},
		},
		{
			name: "empty inside surrogate pair moves start", text: "a👍b",
			requested: proto.TextRange{Location: 2},
			wantRange: proto.TextRange{Location: 1, Length: 0},
		},
		{
			name: "family stays whole", text: "x👨‍👩‍👧‍👦y",
			requested: proto.TextRange{Location: 5, Length: 1},
			wantText:  "👨‍👩‍👧‍👦", wantRange: proto.TextRange{Location: 1, Length: 11},
		},
		{
			name: "flag stays whole", text: "🇨🇳!",
			requested: proto.TextRange{Location: 2, Length: 1},
			wantText:  "🇨🇳", wantRange: proto.TextRange{Location: 0, Length: 4},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := cutReadRange(test.text, test.requested)
			wantLength := len(utf16.Encode([]rune(test.text)))
			if got.Text != test.wantText || got.Range != test.wantRange || got.Length != wantLength {
				t.Fatalf("cutReadRange() = %+v, want text %q range %+v length %d", got, test.wantText, test.wantRange, wantLength)
			}
		})
	}
}

func TestCutReadRangeCapCanWiden(t *testing.T) {
	text := strings.Repeat("z", 3999) + "👍" + strings.Repeat("x", 6000)
	got := cutReadRange(text, proto.TextRange{Location: 0, Length: 5000})
	if got.Range != (proto.TextRange{Location: 0, Length: 4001}) || !strings.HasSuffix(got.Text, "👍") {
		t.Fatalf("capped read = range %+v suffix %q", got.Range, got.Text[len(got.Text)-4:])
	}
}

func TestReadCeilingCountsUTF16Units(t *testing.T) {
	if !withinReadCeiling(strings.Repeat("x", maxFieldTextUnits)) {
		t.Fatal("exactly 200000 UTF-16 units must remain readable")
	}
	if withinReadCeiling(strings.Repeat("x", maxFieldTextUnits-1) + "👍") {
		t.Fatal("a read past 200000 UTF-16 units must be refused")
	}
}
