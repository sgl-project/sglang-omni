package main

// The input box's view (rewrite_box.go): the Rewrite card's dark face
// with the selected text, an editable text box, the pictures'
// thumbnails (each with a remove button), the voice line for the spoken
// instruction, and a line for an error.

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type boxBitmap = ui.Bitmap

// rewriteBoxLocalUI owns widget state that is not part of the box's content.
type rewriteBoxLocalUI struct {
	text          string
	segmentText   map[int]*string
	paste         *boxPasteUI
	pasteFocusID  int
	pasteCaret    int
	expandedText  map[string]bool
	previewImage  int
	addOpen       bool
	editorID      int
	editorStart   int
	editorEnd     int
	editorActive  bool
	inputScroll   ui.ScrollState
	turnScroll    ui.ScrollState
	turnsKey      string
	textScroll    ui.ScrollState
	previewScroll ui.ScrollState
}

type boxPasteUI struct {
	index    int
	original string
	prefix   string
	suffix   string
	suffixID int
	start    int
	end      int
}

var decodeBoxBitmap = ui.DecodeBitmap

// boxThumb is a thumbnail's size in the box.
const boxThumb = 72

type boxDropWindow interface {
	boxWindow
	OnFileDrop(func(*mygo.FileDropEvent)) func()
	OnMove(func()) func()
	SetAlwaysOnTop(bool)
}

var newBoxNativeWindow = func(options mygo.WindowOptions) boxDropWindow { return mygo.NewWindow(options) }

// newBoxWindow creates the input box's native panel.
func newBoxWindow(state func() rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI, options mygo.WindowOptions) boxWindow {
	options.Content = hudContent(func(context *ui.Context) { rewriteBoxView(context, state(), actions, local) })
	options.Vibrancy = mygo.VibrancyHUD
	options.Panel = true
	options.NonActivating = false
	options.DisableClose = true
	window := newBoxNativeWindow(options)
	// A panel floats over every app's windows; the card stays at the
	// windows' level, so the user can put another window over it.
	window.SetAlwaysOnTop(false)
	window.OnFileDrop(func(event *mygo.FileDropEvent) { actions.drop(event.Paths) })
	window.OnMove(actions.moved)
	return window
}

// rewriteBoxView draws the input box from its state, actions and local text.
func rewriteBoxView(context *ui.Context, state rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI) {
	context.SetTheme(overlayTheme(state.Palette))
	defer context.Root().Background(ui.Transparent)
	hadPaste := local.paste != nil
	applyBoxPaste(local, state)
	if !hadPaste && local.text != state.Text {
		local.text = state.Text
	}
	root := ui.Column(context).Fill().Padding(layout.Pad)
	if context.Shortcut(0, ui.KeyEscape) {
		if local.addOpen {
			local.addOpen = false
		} else {
			actions.escape()
		}
	}
	root.Children(func() {
		faceStyle(context, ui.Column(context.Key("box-face")).Fill().Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border).Gap(8), state.Radius, state.Alpha).DragWindow().Children(func() {
			// The header: what it is, Send while it records or a resend
			// waits, Close.
			ui.Row(context).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
				ui.Icon(context, icons.VoxtAnswer.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Shrink(0).Label(i18n.T("Rewrite"))
				ui.Text(context, i18n.T("Rewrite")).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine()

				if state.Recording {
					cardWave(context, state.Levels[:], state.Recording, state.Heard, state.Motion, state.Phase)
				}
				ui.Box(context).Grow(1).MinWidth(0)
				if state.CanSend {
					if cardCapsule(context.Key("box-send"), i18n.T("Send"), i18n.T("Send (Enter)")) {
						actions.textChanged(local.text)
						actions.send()
					}
				}
				if card := state.Card; card != nil && !card.Failed && card.CanCopy {
					cardInsertCopy(context, card, actions.card)
				}
				if cardIconButton(context.Key("box-close"), i18n.T("Close"), "", icons.Close, true) {
					actions.close()
				}
			})
			ui.Box(context).Height(1).Shrink(0).Margin(layout.CardRuleTop, 0, 0, 0).Background(ovCardRule)

			if state.Subtitle != "" {
				ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
					if state.SubtitleError {
						ui.Icon(context, icons.DangerCircle.Linear).Size(12, 12).TextColor(tok(context.Theme()).Danger).Shrink(0)
					}
					ui.Text(context, state.Subtitle).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…").Tooltip(state.Subtitle)
				})
			}
			if state.ShowTip && len(state.Thumbnails) == 0 { // a picture needs the room
				ui.Row(context.Key("box-tip")).Shrink(0).Padding(6, 8, 6, 8).Gap(8).Radius(6).Background(ui.RGBA(255, 255, 255, 0.06)).AlignItems(ui.Center).Children(func() {
					ui.Text(context, i18n.T("Speak or type what to do. Paste or drop images to go with it.")).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0)
					if cardCapsule(context.Key("box-got-it"), i18n.T("Got it"), "") {
						actions.dismissTip()
					}
				})
			}

			if boxFullImagePreview(context, state, local) {
				return
			}

			// The answer and the turns before it, as the answer card draws
			// them: the card is the same one from the shortcut on (wall 122).
			if state.Card != nil {
				boxConversation(context, state, actions, local)
			}

			// The text box stays at the card's foot: the list above it takes
			// the room left, empty before the first answer.
			inputs := ui.Scroll(context.Key("box-input-scroll")).MinHeight(0).MinWidth(0).TrackScroll(&local.inputScroll)
			if !state.Conversation {
				inputs.Grow(1)
			}
			inputs.Children(func() {
				ui.Column(context).MinWidth(0).Gap(8).Children(func() {
					// The text selected when the shortcut fired: the input, as the
					// model gets it first.
					if state.Selection != "" {
						boxSelectionBlockView(context, state.SelectionBlock, local, actions.removeSelection)
					}
					for index, segment := range state.TextSegments {
						if segment.Pasted {
							boxTextBlockView(context, "segment-"+strconv.Itoa(segment.ID), segment, local, func() { actions.removeText(index) })
							continue
						}
						if local.segmentText == nil {
							local.segmentText = map[int]*string{}
						}
						text := local.segmentText[segment.ID]
						if text == nil {
							text = new(string)
							local.segmentText[segment.ID] = text
							*text = segment.Text
						}
						if !hadPaste && *text != segment.Text {
							*text = segment.Text
						}
						boxTextEditor(context, index, text, state, actions, local)
					}
				})
			})

			// The pictures stay reachable while earlier text inputs scroll.
			if len(state.Thumbnails) > 0 {
				ui.Row(context).Shrink(0).Gap(8).AlignItems(ui.Start).Children(func() {
					for index, bitmap := range state.Thumbnails {
						info := boxImageInfo{Source: i18n.T("Image")}
						if index < len(state.ImageInfo) {
							info = state.ImageInfo[index]
						}
						var remove func()
						remove = actions.removeAutoImage
						if index < state.ManualImages {
							remove = func() { actions.remove(index) }
						}
						boxThumbView(context.Key(index), bitmap, info, remove, func() { local.previewImage = index + 1 })
					}
				})
			}

			if state.Limit {
				ui.Text(context, i18n.T("At most %d images", boxMaxImages)).FontSize(12).TextColor(tok(context.Theme()).Danger).Shrink(0)
			}

			// The trailing text box stays visible while earlier inputs scroll.
			boxTextEditor(context, -1, &local.text, state, actions, local)

			// The voice line carries actual speech, not another hint.
			if state.VoiceLine != "" {
				ui.Row(context).Shrink(0).Gap(6).AlignItems(ui.Center).Children(func() {
					ui.Icon(context, icons.VoxtTranscribe.Bold).Size(14, 14).TextColor(ovText2).Shrink(0).Label(i18n.T("The spoken instruction"))
					ui.Text(context, state.VoiceLine).FontSize(12).TextColor(ovText2).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
				})
			}

			if sizeStepButton(context.Key("box-size"), state.SizeStep) {
				actions.cycleSize()
			}
			ui.Row(context.Key("box-footer")).Shrink(0).Gap(5).AlignItems(ui.Center).Children(func() {
				add := dropButton(context.Key("box-add"), icons.Add, i18n.T("Add"))
				if add.Clicked() {
					local.addOpen = !local.addOpen
				}
				ui.Popover(context, add, &local.addOpen, func() {
					ui.Column(context).MinWidth(180).Children(func() {
						if menuRow(context.Key("box-choose-images"), i18n.T("Choose images…"), false) {
							local.addOpen = false
							actions.chooseImages()
						}
						if menuRow(context.Key("box-choose-file"), i18n.T("Choose file…"), false) {
							local.addOpen = false
							actions.chooseFile()
						}
						if menuRow(context.Key("box-paste-clipboard"), i18n.T("Paste clipboard"), false) {
							local.addOpen = false
							services := context.Services()
							if !actions.paste(services) {
								index, start, end := boxClipboardEditor(actions.viewState(), local)
								actions.pasteClipboard(services, index, start, end)
								local.text = actions.viewState().Text
							}
						}
					})
				})
				keycap(context, "Enter")
				ui.Text(context, i18n.T("send")).FontSize(11).TextColor(ovText2).SingleLine()
				ui.Text(context, "·").FontSize(11).TextColor(ovText2).SingleLine()
				keycap(context, "Esc")
				ui.Text(context, i18n.T("close")).FontSize(11).TextColor(ovText2).SingleLine()
			})
		})
	})
	actions.textChanged(local.text)
}

func boxClipboardEditor(state rewriteBoxViewState, local *rewriteBoxLocalUI) (index, start, end int) {
	index, start, end = -1, len([]rune(local.text)), len([]rune(local.text))
	if !local.editorActive {
		return index, start, end
	}
	start, end = local.editorStart, local.editorEnd
	for segmentIndex, segment := range state.TextSegments {
		if segment.ID == local.editorID {
			return segmentIndex, start, end
		}
	}
	return index, start, end
}

func applyBoxPaste(local *rewriteBoxLocalUI, state rewriteBoxViewState) {
	paste := local.paste
	if paste == nil {
		return
	}
	local.paste = nil
	before, after := paste.prefix, paste.suffix
	current := local.text
	if paste.index >= 0 {
		current = *local.segmentText[state.TextSegments[paste.index].ID]
	}
	inserted := ""
	if current != paste.original && strings.HasPrefix(current, before) && strings.HasSuffix(current, after) {
		inserted = current[len(before) : len(current)-len(after)]
	}
	local.pasteCaret = utf8.RuneCountInString(inserted)
	if paste.index == -1 {
		local.text = inserted + paste.suffix
		local.pasteFocusID = -1
		return
	}
	originalID := state.TextSegments[paste.index].ID
	*local.segmentText[originalID] = paste.prefix
	local.pasteFocusID = originalID
	if paste.suffixID == -2 {
		local.pasteFocusID = state.TextSegments[paste.index+2].ID
		local.segmentText[local.pasteFocusID] = new(string)
		*local.segmentText[local.pasteFocusID] = inserted + paste.suffix
	} else {
		*local.segmentText[originalID] = paste.prefix + inserted
	}
}

func boxTextEditor(context *ui.Context, index int, text *string, state rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI) {
	key := "box-text-" + strconv.Itoa(index)
	textArea := ui.TextAreaBase(context.Key(key), text).
		Label(i18n.T("More text for the rewrite")).
		Placeholder(state.EditorPlaceholder).
		Lines(1, 4).FontSize(layout.CardTextSize).TextColor(ovCardText)
	if index == -1 {
		// The same panel as the selected text above it: a field to type
		// in, not a loose line of grey text.
		textArea.Lines(2, 6).Padding(6, 8, 6, 8).Radius(6).Background(ui.RGBA(255, 255, 255, 0.06))
	}
	changed := func() {
		if index == -1 {
			actions.textChanged(*text)
			return
		}
		actions.segmentChanged(index, *text)
	}
	textArea.OnTextKey(func(key ui.TextKey) bool {
		switch {
		case key.Key == ui.KeyEnter && key.Mods == 0:
			changed()
			actions.send()
			return true
		case key.Key == ui.KeyV && key.Mods == ui.Cmd:
			return actions.paste(context.Services())
		}
		return false
	})
	editorID := index
	if index >= 0 {
		editorID = state.TextSegments[index].ID
	}
	start, end := textArea.TextSelection()
	if textArea.Focused() {
		local.editorID, local.editorStart, local.editorEnd = editorID, start, end
		local.editorActive = true
	}
	if local.pasteFocusID == editorID {
		start, end = local.pasteCaret, local.pasteCaret
	}
	textArea.OnPaste(func(paste string) bool {
		original := *text
		if local.paste != nil {
			applyBoxPaste(local, actions.viewState())
			state = actions.viewState()
			focusID := local.pasteFocusID
			if focusID == -1 {
				index, original = -1, local.text
			} else {
				for segmentIndex, segment := range state.TextSegments {
					if segment.ID == focusID {
						index, original = segmentIndex, *local.segmentText[focusID]
						break
					}
				}
			}
			start, end = local.pasteCaret, local.pasteCaret
		}
		if len(paste) > boxMaxTextBytes {
			actions.pasteText(index, paste, start, end)
			return true
		}
		if !actions.pasteText(index, paste, start, end) {
			return actions.paste(context.Services())
		}
		suffixID := -1
		if index >= 0 {
			suffixID = -2 // every earlier-segment paste creates a typed suffix
		}
		local.paste = &boxPasteUI{index: index, original: original, prefix: string([]rune(original)[:start]), suffix: string([]rune(original)[end:]), suffixID: suffixID, start: start, end: end}
		return true
	})
	if local.pasteFocusID == editorID {
		textArea.Focus()
		textArea.SetTextSelection(local.pasteCaret, local.pasteCaret)
		local.pasteFocusID, local.pasteCaret = 0, 0
	} else if index == -1 && state.FocusText {
		textArea.Focus()
		actions.focused()
	}
	changed()
}

// boxThumbView is one picture's thumbnail, with a full-size preview
// (which names its source and pixel size) on click and a remove
// button over the corner.
func boxThumbView(context *ui.Context, bitmap *ui.Bitmap, info boxImageInfo, remove func(), preview func()) {
	label := i18n.T("Remove this image")
	ui.Column(context).Width(boxThumb + 12).Shrink(0).Gap(3).Children(func() {
		ui.Box(context).Width(boxThumb).Height(boxThumb).Shrink(0).Children(func() {
			previewLabel := i18n.T("Preview %s", info.Source)
			if ui.ButtonBase(context.Key("preview")).Width(boxThumb).Height(boxThumb).Radius(6).Label(previewLabel).Tooltip(previewLabel).Children(func() {
				if bitmap != nil {
					ui.Image(context, bitmap).Width(boxThumb).Height(boxThumb).Fit(ui.Cover).Radius(6)
				} else {
					ui.Box(context).Width(boxThumb).Height(boxThumb).Radius(6).Background(ui.RGBA(255, 255, 255, 0.1))
				}
			}).Clicked() {
				preview()
			}
			if remove != nil {
				if ui.ButtonBase(context.Key("rm")).Absolute().Top(-4).Right(-4).Size(16, 16).Radius(8).Center().
					Background(ui.RGBA(0, 0, 0, 0.85)).Border(1, ui.RGBA(255, 255, 255, 0.24)).Label(label).Tooltip(label).
					Children(func() {
						ui.Icon(context, icons.Close.Linear).Size(9, 9).TextColor(ui.RGBA(255, 255, 255, 0.94))
					}).Clicked() {
					remove()
				}
			}
		})
	})
}

// boxConversation draws the answer card's blocks (the conversation, the
// answer being written) in the Rewrite card, scrolling under its fixed
// height.
func boxConversation(context *ui.Context, state rewriteBoxViewState, actions rewriteBoxActions, local *rewriteBoxLocalUI) {
	card, count := state.Card, sessionCardBlockCount(state.Card)
	if key := card.ID + "/" + strconv.Itoa(count); key != local.turnsKey || card.Writing && local.turnScroll.Y >= local.turnScroll.MaxY-2 {
		local.turnsKey, local.turnScroll.Y = key, math.MaxFloat32 // the newest turn is in view
	}
	overlayLocal := actions.cardLocal()
	ui.Scroll(context.Key("box-turns")).Grow(1).MinHeight(0).MinWidth(0).TrackScroll(&local.turnScroll).Children(func() {
		ui.Column(context).Padding(0, layout.CardInset, 0, 0).Children(func() {
			for index := range count {
				cardBlockView(context.Key(index), state.CardSessionID, card, sessionCardBlockAt(card, index), index, state.CardWidth, actions.card, overlayLocal, actions.cardMeasured)
			}
		})
	})
}
