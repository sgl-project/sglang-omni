package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// answerInputPreview keeps one original that went to the language model.
type answerInputPreview struct {
	Label  string
	Text   string
	Image  *boxBitmap
	Width  int
	Height int
	Chars  int
	Lines  int
}

func answerTextPreview(text, label string) answerInputPreview {
	segment := newBoxTextSegment(text, label, true)
	return answerInputPreview{Label: label, Text: segment.Text, Chars: segment.Chars, Lines: segment.Lines}
}

func answerInputPreviews(snapshot session.Snapshot, box *rewriteBox) []answerInputPreview {
	boxOwned := snapshot.ImageHint == "box" && box != nil && (snapshot.ID == box.session || snapshot.ID == box.sending)
	automatic := snapshot.ImageHint != "" && snapshot.ImageHint == snapshot.ImagePreviewKind && len(snapshot.ImagePreview) > 0
	if !boxOwned && !automatic {
		return nil
	}
	var previews []answerInputPreview
	if snapshot.Selection != "" {
		label := i18n.T("Selected text")
		if snapshot.SelectionVia == "copy" {
			label = i18n.T("Copied text")
		}
		previews = append(previews, answerTextPreview(snapshot.Selection, label))
	}
	if snapshot.Typed != "" {
		previews = append(previews, answerTextPreview(snapshot.Typed, i18n.T("Text")))
	}
	if boxOwned {
		for index, bitmap := range box.thumbs {
			info := boxImageInfo{Source: i18n.T("Pasted image")}
			if index < len(box.imageInfo) {
				info = box.imageInfo[index]
			}
			previews = append(previews, answerInputPreview{Label: info.Source, Image: bitmap, Width: info.Width, Height: info.Height})
		}
	}
	if automatic {
		bitmap, _ := decodeBoxBitmap(snapshot.ImagePreview)
		label := i18n.T("Copied image")
		if snapshot.ImageHint == "window" {
			label = i18n.T("Screenshot")
		}
		info := boxImageCaption(bitmap, label)
		previews = append(previews, answerInputPreview{Label: label, Image: bitmap, Width: info.Width, Height: info.Height})
	}
	return previews
}
