package main

import (
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

const (
	boxTextCollapseLines = 3
	boxTextCollapseChars = 300
	boxTextShownChars    = 10000
	boxMaxTextBytes      = 32 << 20
)

// boxTextSegment is one editable typed or removable pasted part, in request order.
type boxTextSegment struct {
	ID     int
	Text   string
	Source string
	Pasted bool
	File   bool
	Chars  int
	Lines  int
}

func newBoxTextSegment(text, source string, pasted bool) boxTextSegment {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := 0
	if text != "" {
		lines = strings.Count(strings.TrimSuffix(text, "\n"), "\n") + 1
	}
	return boxTextSegment{Text: text, Source: source, Pasted: pasted, Chars: utf8.RuneCountInString(text), Lines: lines}
}

func (a *app) newBoxTextSegment(text, source string, pasted bool) boxTextSegment {
	segment := newBoxTextSegment(text, source, pasted)
	a.box.nextTextID++
	segment.ID = a.box.nextTextID
	return segment
}

func (a *app) boxPasteText(text string) bool {
	return a.boxPasteTextAt(-1, text, len([]rune(a.box.text)), len([]rune(a.box.text)))
}

// boxPasteTextAt replaces the selected runes in one editor with a collapsed
// pasted block. Index -1 is the trailing editor; other indexes are typed segments.
func (a *app) boxPasteTextAt(index int, text string, start, end int) bool {
	if !a.box.open {
		return false
	}
	if len(text) > boxMaxTextBytes {
		a.box.err = i18n.T("This text is too large.")
		a.boxSync()
		return false
	}
	segment := newBoxTextSegment(text, i18n.T("Pasted text"), true)
	if segment.Lines <= boxTextCollapseLines || index < -1 || index >= len(a.box.textSegments) {
		return false
	}
	editor := a.box.text
	if index >= 0 {
		if a.box.textSegments[index].Pasted {
			return false
		}
		editor = a.box.textSegments[index].Text
	}
	runes := []rune(editor)
	start, end = max(0, min(start, len(runes))), max(0, min(end, len(runes)))
	if start > end {
		start, end = end, start
	}
	prefix, suffix := string(runes[:start]), string(runes[end:])
	pasted := a.newBoxTextSegment(segment.Text, segment.Source, true)
	if index == -1 {
		if prefix != "" {
			a.box.textSegments = append(a.box.textSegments, a.newBoxTextSegment(prefix, "", false))
		}
		a.box.textSegments = append(a.box.textSegments, pasted)
		a.box.text = suffix
	} else {
		typed := a.newBoxTextSegment(prefix, "", false)
		typed.ID = a.box.textSegments[index].ID
		a.box.textSegments[index] = typed
		insert := []boxTextSegment{pasted, a.newBoxTextSegment(suffix, "", false)}
		a.box.textSegments = slices.Insert(a.box.textSegments, index+1, insert...)
	}
	a.box.textSegmentsDirty = true
	a.boxSync()
	return true
}

func (a *app) boxTextContent() string {
	if a.box.textCacheValid && !a.box.textSegmentsDirty && a.box.cachedEditor == a.box.text {
		return a.box.textCache
	}
	var content strings.Builder
	afterFile := false
	add := func(text string, file bool) {
		if text == "" {
			return
		}
		// An imported file starts and ends on its own line, or two files glue together.
		if (file || afterFile) && content.Len() > 0 && !strings.HasSuffix(content.String(), "\n") {
			content.WriteByte('\n')
		}
		content.WriteString(text)
		afterFile = file
	}
	for _, segment := range a.box.textSegments {
		add(segment.Text, segment.File)
	}
	add(a.box.text, false)
	a.box.textCache = content.String()
	a.box.cachedEditor = a.box.text
	a.box.textCacheValid = true
	a.box.textSegmentsDirty = false
	return a.box.textCache
}

func (a *app) boxSegmentChanged(index int, text string) {
	if index < 0 || index >= len(a.box.textSegments) || a.box.textSegments[index].Pasted {
		return
	}
	segment := a.newBoxTextSegment(text, "", false)
	segment.ID = a.box.textSegments[index].ID
	a.box.textSegments[index] = segment
	a.box.textSegmentsDirty = true
	a.boxShare.store(a.boxTextContent(), a.box.images)
}

func (a *app) boxRemoveText(index int) {
	if index < 0 || index >= len(a.box.textSegments) || !a.box.textSegments[index].Pasted {
		return
	}
	a.box.textSegments = append(a.box.textSegments[:index], a.box.textSegments[index+1:]...)
	a.box.textSegmentsDirty = true
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
