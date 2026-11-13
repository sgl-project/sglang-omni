//go:build windows

package main

import (
	"unicode/utf16"

	"github.com/go-text/typesetting/segmenter"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

type graphemeBoundary struct {
	runeOffset int
	unitOffset int
}

func withinReadCeiling(text string) bool {
	return len(utf16.Encode([]rune(text))) <= maxFieldTextUnits
}

func cutReadRange(text string, requested proto.TextRange) proto.ReadResult {
	runes := []rune(text)
	totalUnits := len(utf16.Encode(runes))
	start := min(max(0, requested.Location), totalUnits)
	requestedEnd := requested.Location
	if requested.Length > 0 {
		if requested.Location > int(^uint(0)>>1)-requested.Length {
			requestedEnd = int(^uint(0) >> 1)
		} else {
			requestedEnd = requested.Location + requested.Length
		}
	}
	end := min(max(start, requestedEnd), totalUnits, start+proto.MaxRead)
	boundaries := textGraphemeBoundaries(runes)
	widenedStart := start
	widenedEnd := end
	startRune := len(runes)
	endRune := len(runes)
	for _, boundary := range boundaries {
		if boundary.unitOffset <= start {
			widenedStart = boundary.unitOffset
			startRune = boundary.runeOffset
		}
		if boundary.unitOffset >= end {
			widenedEnd = boundary.unitOffset
			endRune = boundary.runeOffset
			break
		}
	}
	if end == start {
		widenedEnd = widenedStart
		endRune = startRune
	}
	return proto.ReadResult{
		Text:   string(runes[startRune:endRune]),
		Range:  proto.TextRange{Location: widenedStart, Length: widenedEnd - widenedStart},
		Length: totalUnits,
	}
}

func textGraphemeBoundaries(runes []rune) []graphemeBoundary {
	boundaries := []graphemeBoundary{{}}
	var segmentation segmenter.Segmenter
	segmentation.Init(runes)
	iterator := segmentation.GraphemeIterator()
	runeOffset := 0
	unitOffset := 0
	for iterator.Next() {
		grapheme := iterator.Grapheme()
		runeOffset += len(grapheme.Text)
		unitOffset += len(utf16.Encode(grapheme.Text))
		boundaries = append(boundaries, graphemeBoundary{runeOffset: runeOffset, unitOffset: unitOffset})
	}
	return boundaries
}
