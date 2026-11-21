package main

import (
	"testing"
	"time"
)

func TestMascotOutfitOn(t *testing.T) {
	for _, tc := range []struct {
		day, want string
	}{
		{"2026-11-21", ""},
		{"2026-12-14", ""},
		{"2026-12-15", outfitChristmas},
		{"2026-12-25", outfitChristmas},
		{"2026-12-26", outfitChristmas},
		{"2026-12-27", ""},
		{"2026-12-31", outfitNewYear},
		{"2027-01-02", outfitNewYear},
		{"2027-01-03", ""},
		{"2027-02-04", ""},
		{"2027-02-05", outfitSpring}, // the eve
		{"2027-02-12", outfitSpring},
		{"2027-02-13", ""},
		{"2028-01-25", outfitSpring},
	} {
		day, err := time.ParseInLocation("2006-01-02 15:04", tc.day+" 23:30", time.Local)
		if err != nil {
			t.Fatal(err)
		}
		if got := mascotOutfitOn(day); got != tc.want {
			t.Errorf("%s: %q, want %q", tc.day, got, tc.want)
		}
		if got := mascotPieces(mascotOutfitOn(day)); (len(got) > 0) != (tc.want != "") {
			t.Errorf("%s: %d pieces for %q", tc.day, len(got), tc.want)
		}
	}
}

func TestMascotMovesRestAtTheirEnds(t *testing.T) {
	rest := mascotPose{orange: 1, peach: 1}
	for move := range mascotMoves {
		if got := mascotPoseAt(move, 0); got != rest {
			t.Errorf("move %d starts at %+v", move, got)
		}
		end := mascotPoseAt(move, 1)
		if end.tilt > 1e-4 || end.tilt < -1e-4 || end.blink || end.orange != 1 || end.peach != 1 {
			t.Errorf("move %d ends at %+v", move, end)
		}
		blinked := false
		for step := range 100 {
			blinked = blinked || mascotPoseAt(move, float32(step)/100).blink
		}
		if !blinked {
			t.Errorf("move %d never blinks", move)
		}
	}
}
