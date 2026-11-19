package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

var boxDialogOpen = mygo.Dialog.Open
var boxReadPDF = macx.ReadPDF

const boxMaxInputTextBytes int64 = 1 << 20
const boxMaxInputPDFFileBytes int64 = 32 << 20

// boxLanding is one converted input waiting for its turn in boxHeld.
type boxLanding struct {
	gen        int
	jpeg       []byte
	jpegs      [][]byte
	text       string
	totalPages int
	errText    string
	source     string
}

func (a *app) boxChooseImages() {
	a.boxChoose(true)
}

func (a *app) boxChooseFile() {
	a.boxChoose(false)
}

// boxChoose opens the system picker off the UI thread and returns to the
// same box generation before changing its content.
func (a *app) boxChoose(images bool) {
	if !a.box.open {
		return
	}
	gen := a.boxGen
	title := i18n.T("Choose file…")
	filter := mygo.FileFilter{Name: i18n.T("Text or PDF"), Extensions: []string{"txt", "md", "pdf"}}
	if images {
		title = i18n.T("Choose images…")
		filter = mygo.FileFilter{Name: i18n.T("Images"), Extensions: []string{"png", "jpg", "jpeg", "gif"}}
	}
	var parent *mygo.Window
	if native, ok := a.box.win.(*mygo.Window); ok {
		parent = native
	} else {
		parent = a.win
	}
	go func() {
		files, err := boxDialogOpen(mygo.OpenDialogOptions{Parent: parent, Title: title, Multiple: images, Filters: []mygo.FileFilter{filter}})
		a.post(func() {
			if gen != a.boxGen || !a.box.open {
				return
			}
			if err != nil {
				a.box.err = i18n.T("Could not choose files: %v", err)
				a.boxSync()
				return
			}
			if images {
				a.boxDrop(files)
				return
			}
			if len(files) > 0 {
				a.boxReadInputFile(files[0])
			}
		})
	}()
}

// boxPasteClipboard completes the text half of Paste clipboard after
// boxPaste has declined image and Finder clipboard formats.
func (a *app) boxPasteClipboard(services uiServices, index, start, end int) {
	a.boxPasteClipboardTextAt(services.ReadClipboard(), index, start, end)
}

func (a *app) boxPasteClipboardTextAt(text string, index, start, end int) {
	if !a.box.open || text == "" || index < -1 || index >= len(a.box.textSegments) {
		return
	}
	if index >= 0 && a.box.textSegments[index].Pasted {
		return
	}
	editor := a.box.text
	focusID := -1
	if index >= 0 {
		editor = a.box.textSegments[index].Text
		focusID = a.box.textSegments[index].ID
	}
	runes := []rune(editor)
	start, end = max(0, min(start, len(runes))), max(0, min(end, len(runes)))
	if start > end {
		start, end = end, start
	}
	collapsed := a.boxPasteTextAt(index, text, start, end)
	if collapsed && index >= 0 {
		focusID = a.box.textSegments[index+2].ID
	}
	if !collapsed {
		inserted := string(runes[:start]) + text + string(runes[end:])
		if index == -1 {
			a.box.text = inserted
		} else {
			a.boxSegmentChanged(index, inserted)
		}
		a.boxSync()
	}
	caret := start + utf8.RuneCountInString(text)
	if collapsed {
		caret = 0
	}
	a.box.ui.pasteFocusID = focusID
	a.box.ui.pasteCaret = caret
}

func (a *app) boxReadInputFile(path string) {
	gen := a.boxGen
	seq := a.boxReserve()
	source := filepath.Base(path)
	a.async(func() {
		landing := boxLanding{source: source}
		info, err := os.Stat(path)
		extension := strings.ToLower(filepath.Ext(path))
		maxBytes := boxMaxInputTextBytes
		if extension == ".pdf" {
			maxBytes = boxMaxInputPDFFileBytes
		}
		switch {
		case err != nil:
			landing.errText = i18n.T("This file could not be read: %s", source)
		case info.Size() > maxBytes:
			landing.errText = i18n.T("This file is too large: %s", source)
		default:
			switch extension {
			case ".txt", ".md":
				data, err := os.ReadFile(path)
				if err != nil || !utf8.Valid(data) {
					landing.errText = i18n.T("This file could not be read: %s", source)
				} else {
					landing.text = string(data)
				}
			case ".pdf":
				text, pages, totalPages, err := boxReadPDF(path, boxMaxImages)
				if err != nil {
					landing.errText = i18n.T("This file could not be read: %s", source)
					break
				}
				landing.text = strings.TrimSpace(strings.ReplaceAll(text, "\f", "\n"))
				if len(pages) > 0 {
					landing.totalPages = totalPages
					for _, page := range pages {
						jpeg, err := imgconv.JPEG(page)
						if err != nil {
							landing.errText = i18n.T("This file could not be read: %s", source)
							landing.jpegs = nil
							break
						}
						landing.jpegs = append(landing.jpegs, jpeg)
					}
				}
			default:
				landing.errText = i18n.T("This file type is not supported: %s", source)
			}
		}
		if landing.errText == "" && len(landing.text) > int(boxMaxInputTextBytes) {
			landing.errText = i18n.T("This file is too large: %s", source)
			landing.text = ""
			landing.totalPages = 0
			landing.jpegs = nil
		}
		if landing.errText == "" && strings.TrimSpace(landing.text) == "" && landing.totalPages == 0 {
			landing.errText = i18n.T("This file has no text: %s", source)
		}
		a.post(func() { a.boxConvertedLanding(gen, seq, landing) })
	})
}

func (a *app) boxAddFileText(text, source string) {
	if a.box.text != "" {
		a.box.textSegments = append(a.box.textSegments, a.newBoxTextSegment(a.box.text, "", false))
		a.box.text = ""
	}
	segment := a.newBoxTextSegment(text, source, true)
	segment.File = true
	a.box.textSegments = append(a.box.textSegments, segment)
	a.box.textSegmentsDirty = true
	a.boxSync()
}

// boxConverted takes one converted picture on the main thread (errText
// when it would not read, "" on success or a silent skip).
func (a *app) boxConverted(gen, seq int, jpeg []byte, errText string, source ...string) {
	name := ""
	if len(source) > 0 {
		name = source[0]
	}
	a.boxConvertedLanding(gen, seq, boxLanding{jpeg: jpeg, errText: errText, source: name})
}

// boxConvertedLanding applies a converted input only in its own box life
// and reservation order. A send queued behind it runs after it lands.
func (a *app) boxConvertedLanding(gen, seq int, landing boxLanding) {
	var conversionErr error
	landing.gen = gen
	if a.boxHeld == nil {
		a.boxHeld = map[int]boxLanding{}
	}
	a.boxHeld[seq] = landing
	for {
		land, ok := a.boxHeld[a.boxNext]
		if !ok {
			break
		}
		delete(a.boxHeld, a.boxNext)
		a.boxPending--
		a.boxNext++
		if land.gen != a.boxGen {
			continue
		}
		if land.errText != "" {
			conversionErr = errors.New(land.errText)
			a.box.err = land.errText
			a.boxSendQueued = false
		} else {
			if land.text != "" || len(land.jpeg) > 0 || len(land.jpegs) > 0 {
				conversionErr = nil
				a.box.err = ""
				a.boxShare.clearError()
			}
			if land.text != "" {
				a.boxAddFileText(land.text, land.source)
			}
			if land.totalPages > 0 {
				take := min(len(land.jpegs), max(0, boxMaxImages-len(a.box.images)))
				for _, jpeg := range land.jpegs[:take] {
					a.boxAttach(jpeg, land.source)
				}
				if skipped := land.totalPages - take; skipped > 0 {
					a.box.err = i18n.N(skipped, "Skipped %d PDF page beyond the 4-image limit", "Skipped %d PDF pages beyond the 4-image limit", skipped)
					a.box.limit = true
				}
			} else if land.text == "" {
				a.boxAttach(land.jpeg, land.source)
			}
		}
		a.boxSync()
	}
	// Wake the session only after boxSync has stored every landed input.
	a.boxShare.done(conversionErr)
	if a.boxPending == 0 && a.boxSendQueued {
		a.boxSendQueued = false
		a.boxSend()
	}
}
