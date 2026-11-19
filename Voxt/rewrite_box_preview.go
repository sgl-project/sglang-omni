package main

import (
	"strings"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

const boxTextCollapseLines = 3

// boxTextSegment is one typed or pasted part of the box's input, in request order.
type boxTextSegment struct {
	Text   string
	Source string
	Pasted bool
	Chars  int
	Lines  int
}

func newBoxTextSegment(text, source string, pasted bool) boxTextSegment {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := 0
	if text != "" {
		lines = strings.Count(text, "\n") + 1
	}
	return boxTextSegment{Text: text, Source: source, Pasted: pasted, Chars: utf8.RuneCountInString(text), Lines: lines}
}

func (a *app) boxPasteText(text string) bool {
	segment := newBoxTextSegment(text, i18n.T("Pasted text"), true)
	if !a.box.open || segment.Lines <= boxTextCollapseLines {
		return false
	}
	if a.box.text != "" {
		a.box.textSegments = append(a.box.textSegments, newBoxTextSegment(a.box.text, "", false))
	}
	a.box.textSegments = append(a.box.textSegments, segment)
	a.box.text = ""
	a.boxSync()
	return true
}

func (a *app) boxTextContent() string {
	var content strings.Builder
	for _, segment := range a.box.textSegments {
		content.WriteString(segment.Text)
	}
	content.WriteString(a.box.text)
	return content.String()
}

func (a *app) boxRemoveText(index int) {
	if index < 0 || index >= len(a.box.textSegments) || !a.box.textSegments[index].Pasted {
		return
	}
	a.box.textSegments = append(a.box.textSegments[:index], a.box.textSegments[index+1:]...)
	a.boxSync()
}

// boxImageInfo is the source and decoded pixel size shown beside one image.
type boxImageInfo struct {
	Source string
	Width  int
	Height int
}

func boxImageCaption(bitmap *boxBitmap, source string) boxImageInfo {
	info := boxImageInfo{Source: source}
	if bitmap != nil {
		info.Width, info.Height = bitmap.Size()
	}
	return info
}
