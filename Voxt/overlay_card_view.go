package main

// The answer card's widget tree, drawn from its cached DTO and UI-local state.

import (
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func overlayCard(context *ui.Context, state *sessionOverlayViewState, layoutState sessionOverlayLayout, actions sessionOverlayActions, local sessionOverlayLocal, animation sessionOverlayAnimation, onMeasured func([]sessionOverlayMeasurement)) {
	card := state.Card
	if card == nil {
		return
	}
	messageHeight := 0
	if card.Message != "" {
		messageHeight = layout.CardMsgGap + layout.CardMsgH
	}
	bodyHeight := float32(max(layoutState.Height-layout.CardChrome-sessionCardAttachmentHeight(card)-messageHeight, 0))
	ui.Column(context).Fill().Padding(layout.CardPadTop-layout.Border, layout.CardPadX-layout.Border, layout.CardPadY-layout.Border).DragWindow().Children(func() {
		ui.Row(context).Height(layout.CardHeadH).Shrink(0).Gap(layout.CardHeadGap).AlignItems(ui.Center).Children(func() {
			cardHeader(context, card, layoutState.Top, actions, local, animation)
		})
		ui.Box(context).Height(1).Shrink(0).Margin(layout.CardRuleTop, 0, 0, 0).Background(ovCardRule)
		cardAttachView(context, card, local)
		cardBody(context, state.SessionID, card, bodyHeight, layoutState.Width, layoutState.CardScrolls, layoutState.Alpha, actions, local, onMeasured)
		cardSizeButton(context, card, actions)
		if card.Message != "" {
			ui.Row(context).Height(layout.CardMsgH).Shrink(0).Margin(layout.CardMsgGap, 0, 0, 0).Padding(0, layout.CardSizeBtn+layout.CardSizeBtnInset-layout.CardPadX+layout.CardHeadGap, 0, 0).Gap(6).AlignItems(ui.Center).Children(func() {
				icon, color := icons.InfoCircle, ovText2
				if card.Failed {
					icon, color = icons.DangerCircle, tok(context.Theme()).Danger
				}
				ui.Icon(context, icon.Linear).Size(12, 12).TextColor(color).Shrink(0)
				ui.Text(context, card.Message).FontSize(12).TextColor(ovText2).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1).Tooltip(card.Message)
			})
		}
	})
}

func cardSizeButton(context *ui.Context, card *sessionCardViewState, actions sessionOverlayActions) {
	if sizeStepButton(context.Key("ov-card-size"), card.SizeStep) {
		actions.cycleOverlayCardSize()
	}
}

// cardAttachView says which picture went to the language model.
func cardAttachView(context *ui.Context, card *sessionCardViewState, local sessionOverlayLocal) {
	if card.InputCount > 0 && len(local.Overlay.cardPreviews) > 0 {
		cardInputChips(context, local)
		return
	}
	thumbnails := local.Overlay.cardThumbs
	if card.Attachment == "box" && len(thumbnails) > 0 {
		ui.Row(context).Height(layout.CardAttachThumb).Shrink(0).Margin(layout.CardAttachH-layout.CardAttachThumb, 0, 0, 0).Gap(6).AlignItems(ui.Center).Label(i18n.T("Images sent with the instruction")).Children(func() {
			for index, bitmap := range thumbnails {
				if bitmap != nil {
					ui.Image(context.Key(index), bitmap).Size(layout.CardAttachThumb, layout.CardAttachThumb).Fit(ui.Cover).Radius(4).Shrink(0)
				}
			}
		})
		return
	}
	var label string
	var icon icons.Icon
	switch card.Attachment {
	case "clipboard":
		label, icon = i18n.T("Copied image"), icons.ClipboardList
	case "window":
		label, icon = i18n.T("Screenshot"), icons.WindowFrame
	}
	if label == "" {
		return
	}
	ui.Row(context).Height(layout.CardAttachThumb).Shrink(0).Margin(layout.CardAttachH-layout.CardAttachThumb, 0, 0, 0).Gap(6).AlignItems(ui.Center).Label(label).Children(func() {
		ui.Icon(context, icon.Linear).Size(14, 14).TextColor(ovText2).Shrink(0)
		ui.Text(context, label).FontSize(12).TextColor(ovText2).SingleLine().Shrink(0)
	})
}

func sizeStepButton(context *ui.Context, step int) bool {
	label, icon := i18n.T("Make the card larger"), icons.MaximizeSquareMinimalistic
	if step+1 == len(layout.CardSizes) {
		label, icon = i18n.T("Make the card smaller"), icons.MinimizeSquareMinimalistic
	}
	button := ui.ButtonBase(context).Absolute().Right(layout.CardSizeBtnInset).Bottom(layout.CardSizeBtnInset).
		Size(layout.CardSizeBtn, layout.CardSizeBtn).Center().Label(label).Tooltip(label)
	alpha := float32(0.6)
	if button.Hovered() {
		alpha = 0.95
	}
	return button.Children(func() {
		ui.Icon(context, icon.Linear).Size(layout.CardSizeBtnIcon, layout.CardSizeBtnIcon).TextColor(ui.RGBA(255, 255, 255, alpha))
	}).Clicked()
}

func cardHeader(context *ui.Context, card *sessionCardViewState, top bool, actions sessionOverlayActions, local sessionOverlayLocal, animation sessionOverlayAnimation) {
	icon := icons.VoxtAnswer
	if card.Conversation {
		icon = icons.VoxtTranscribe
	}
	ui.Icon(context, icon.Bold).Size(layout.CardIcon, layout.CardIcon).TextColor(ui.RGB(255, 255, 255)).Opacity(ovCardIconA(card)).Shrink(0).Label(sessionCardKindLabel(card.Kind))
	if card.Conversation {
		cardWave(context, card.Levels[:], card.Recording, card.Heard, animation.Motion, animation.Phase)
		ui.Spacer(context)
	} else {
		if !sessionCardLanguageCapsule(card) {
			ui.Text(context, card.Title).FontSize(layout.CardTitleSize).FontWeight(600).TextColor(ui.RGB(255, 255, 255)).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
		}
		if card.Busy {
			busy := i18n.T("Translating…")
			switch {
			case card.Writing && card.Kind == "transcribe":
				busy = i18n.T("Cleaning up…")
			case card.Writing && card.Kind == "rewrite":
				busy = i18n.T("Writing…")
			}
			ui.Icon(context, icons.VoxtSpinner.Linear).Size(12, 12).TextColor(ui.RGBA(255, 255, 255, 0.88)).Shrink(0).
				Rotate(float32(math.Mod(animation.Phase*400, 360))).Label(busy)
		}
		ui.Box(context).Grow(1).MinWidth(0)
	}
	if sessionCardLanguageCapsule(card) {
		cardLanguage(context, card, actions, local, top)
	}
	switch {
	case card.CanStopAnswer:
		if cardCapsule(context.Key("ov-card-stop-answer"), i18n.T("Stop"), i18n.T("Stop writing and keep the text so far")) {
			actions.stopLLM()
		}
	case card.CanUseTranscript:
		if cardCapsule(context.Key("ov-card-use-transcript"), i18n.T("Use transcript"), "") {
			actions.skipLLM()
		}
	case card.CanStop:
		if cardCapsule(context.Key("ov-card-stop"), i18n.T("Done"), i18n.T("Stop recording the follow-up")) {
			actions.stopSession()
		}
	case card.CanFollowUp:
		if cardCapsule(context.Key("ov-card-continue"), i18n.T("Continue"), card.FollowTooltip) {
			actions.followUpAnswer("")
		}
	case card.CanRetry:
		if cardCapsule(context.Key("ov-card-retry"), i18n.T("Try again"), "") {
			actions.retryAnswer("")
		}
	}
	if card.ShowThinking && cardIconButton(context.Key("ov-card-think"), i18n.T("Thinking level"), card.ThinkingLabel, icons.Tuning2, card.CanThink) {
		actions.toggleThinkMenu()
	}
	if !card.Failed {
		cardInsertCopy(context, card, actions)
	}
	if cardIconButton(context.Key("ov-card-close"), i18n.T("Close"), "", icons.Close, true) {
		actions.closeAnswer()
	}
}

// cardInsertCopy are the answer's Insert and Copy buttons (the answer
// card's header, and the Rewrite card's).
func cardInsertCopy(context *ui.Context, card *sessionCardViewState, actions sessionOverlayActions) {
	{
		if cardIconButton(context.Key("ov-card-insert"), i18n.T("Insert"), card.InsertTooltip, icons.Pen, card.CanInsert) {
			actions.insertAnswer()
		}
		copyIcon, copyLabel, copyTip := icons.Copy, i18n.T("Copy"), card.CopyTooltip
		if card.Copied {
			copyIcon, copyLabel, copyTip = icons.CopyCheck, i18n.T("Copied"), i18n.T("Copied")
		}
		if copyTip == "" {
			copyTip = copyLabel
		}
		if cardIconButton(context.Key("ov-card-copy"), copyLabel, copyTip, copyIcon, card.CanCopy) {
			actions.copyAnswer()
		}
	}
}

func sessionCardKindLabel(kind string) string {
	switch kind {
	case "translate":
		return i18n.T("Translate")
	case "rewrite":
		return i18n.T("Rewrite")
	}
	return i18n.T("Transcribe")
}

func ovCardIconA(card *sessionCardViewState) float32 {
	if card.Conversation {
		return 0.92
	}
	return 1
}

func sessionCardLanguageCapsule(card *sessionCardViewState) bool {
	return card.Kind == "translate" && !card.Failed
}

func cardIconButton(context *ui.Context, label, tip string, icon icons.Icon, enabled bool) bool {
	button := ui.ButtonBase(context).Size(layout.CardBtn, layout.CardBtn).Radius(layout.CardBtn / 2).Center().Shrink(0).Label(label).Disabled(!enabled)
	if tip != "" {
		button.Tooltip(tip)
	}
	hover := enabled && button.Hovered()
	background, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0)
	switch {
	case !enabled:
		background = ui.RGBA(255, 255, 255, 0.04)
	case hover:
		background, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	button.Background(background).Border(1, line)
	alpha := float32(0.92)
	if !enabled {
		alpha = 0.42 * 0.92
	}
	return button.Children(func() {
		ui.Icon(context, icon.Linear).Size(layout.CardBtnIcon, layout.CardBtnIcon).TextColor(ui.RGBA(255, 255, 255, alpha))
	}).Clicked()
}

func cardCapsule(context *ui.Context, label, tip string) bool {
	button := ui.ButtonBase(context).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Center().Shrink(0).Label(label)
	if tip != "" {
		button.Tooltip(tip)
	}
	background, line := ui.RGBA(255, 255, 255, 0.08), ui.RGBA(255, 255, 255, 0.1)
	if button.Hovered() {
		background, line = ui.RGBA(255, 255, 255, 0.16), ui.RGBA(255, 255, 255, 0.18)
	}
	button.Background(background).Border(1, line)
	return button.Children(func() {
		ui.Text(context, label).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.94)).SingleLine()
	}).Clicked()
}

func cardLanguage(context *ui.Context, card *sessionCardViewState, actions sessionOverlayActions, local sessionOverlayLocal, top bool) {
	open := local.Card.langOpen && !local.Card.thinkMenu && card.CanLanguage
	button := ui.ButtonBase(context.Key("ov-card-lang")).Height(layout.CardBtn).Padding(0, 10-layout.Border).Radius(layout.CardBtn / 2).Shrink(0).
		Label(i18n.T("Translate into")).Disabled(!card.CanLanguage).Checked(open)
	line, background := ui.RGBA(255, 255, 255, 0.12), ui.RGBA(255, 255, 255, 0.08)
	switch {
	case open:
		line, background = tok(context.Theme()).Accent.Alpha(0.5), ui.RGBA(255, 255, 255, 0.14)
	case card.CanLanguage && button.Hovered():
		line = tok(context.Theme()).Accent.Alpha(0.28)
	}
	button.Background(background).Border(1, line)
	alpha := float32(1)
	if !card.CanLanguage && !card.Busy {
		alpha = 0.6
	}
	chevron := icons.AltArrowDown
	if open && !top {
		chevron = icons.AltArrowUp
	}
	defer func() {
		if button.Clicked() {
			actions.toggleLangMenu()
		}
	}()
	button.Children(func() {
		ui.Row(context).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(context, card.LanguageLabel).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.92*alpha)).
				SingleLine().Ellipsis("…").MaxWidth(layout.CardLangMaxW).MinWidth(0)
			ui.Icon(context, chevron.Bold).Size(10, 10).TextColor(ui.RGBA(255, 255, 255, 0.72*alpha)).Shrink(0)
		})
	})
}

func cardWave(context *ui.Context, levels []float64, recording, heard, motion bool, phase float64) {
	const scaleX, scaleY = 0.84, 0.82
	bars := layout.WaveBars(levels)
	quiet := [...]float32{3.2, 3.9, 4.6, 5.1, 4.2, 3.5, 4.4, 4.9}
	label := i18n.T("Input level")
	ui.Row(context.Key("ov-card-wave")).Width(layout.CardWaveW).Height(layout.CardIcon).Shrink(0).Gap(layout.WaveBarGap * scaleX).AlignItems(ui.Center).Label(label).Children(func() {
		for index, level := range bars {
			height := quiet[index%len(quiet)] * scaleY
			if recording && heard {
				height = (layout.WaveBarMinH + (layout.WaveBarMaxH-layout.WaveBarMinH)*float32(level)) * scaleY
				if motion {
					height += float32(layout.WaveIdle(index, level, phase)) * scaleY
				}
			}
			ui.Box(context).Width(layout.WaveBarW*scaleX).Height(height).Radius(layout.WaveBarR).Gradient(ui.RGBA(255, 255, 255, 0.98), ui.RGBA(255, 255, 255, 0.8), 180).Shrink(0)
		}
	})
}

func cardBody(context *ui.Context, sessionID string, card *sessionCardViewState, bodyHeight float32, width int, scrolls bool, faceAlpha float32, actions sessionOverlayActions, local sessionOverlayLocal, onMeasured func([]sessionOverlayMeasurement)) {
	if cardInputOriginal(context, card, local, bodyHeight) {
		return
	}
	if !card.Conversation && !card.Failed && bodyHeight > 0 && sessionCardBlockCount(card) > 0 {
		block := sessionCardBlockAt(card, 0)
		request := newSessionOverlayPreviewRequest(context, sessionID, card, block, -1, ovCardText, bodyHeight, width)
		// The page is as translucent as the card's face, over the same
		// material: an opaque page drew a darker box (#3836 feedback).
		request.background = tok(context.Theme()).OverlayFace.Alpha(faceAlpha)
		if actions.overlayTranscriptResult(context.Key("ov-transcript"), request) {
			return
		}
	}
	scroll := &local.Overlay.cardScroll
	key := card.ID + "/" + sessionID
	if card.Writing && card.ID == "" {
		key = "/" + sessionID
	}
	if card.Conversation {
		key = card.ID + "/conversation"
	}
	follow := card.Conversation || card.Writing
	if key != local.Overlay.cardScrollKey {
		local.Overlay.cardScrollKey = key
		scroll.X, scroll.Y = 0, 0
		if follow {
			scroll.Y = math.MaxFloat32
		}
	} else if follow && scroll.Y >= scroll.MaxY-2 {
		scroll.Y = math.MaxFloat32
	}
	blocks := func() {
		// The scroller's room is in the card's padding, right of the text.
		ui.Column(context).Padding(0, layout.CardInset, 0, 0).Children(func() {
			for index := 0; index < sessionCardBlockCount(card); index++ {
				cardBlockView(context.Key(index), sessionID, card, sessionCardBlockAt(card, index), index, width, actions, local, onMeasured)
			}
		})
	}
	if !scrolls {
		// The card grows to its text until the screen's room runs out
		// (overlay_size.go): text past a body with room is a frame the
		// card has not grown to yet. The body clips it there instead of
		// scrolling, which flashed the scroll bar under the pointer and,
		// following the end, moved the text a line up and back each time
		// the card grew (#3836 feedback).
		scroll.X, scroll.Y, scroll.MaxX, scroll.MaxY = 0, 0, 0, 0
		ui.Column(context.Key("ov-card-clip")).Height(bodyHeight).Shrink(0).Margin(layout.CardRuleGap, -layout.CardInset, 0, 0).ClipY().Children(blocks)
		return
	}
	ui.Scroll(context.Key("ov-card-body")).Height(bodyHeight).Shrink(0).Margin(layout.CardRuleGap, -layout.CardInset, 0, 0).TrackScroll(scroll).Children(blocks)
}

func cardBlockView(context *ui.Context, sessionID string, card *sessionCardViewState, block sessionCardBlock, index, width int, actions sessionOverlayActions, local sessionOverlayLocal, onMeasured func([]sessionOverlayMeasurement)) {
	text, _, _ := sessionCardBlockText(block, width)
	gap := sessionCardBlockGap(block, index)
	switch {
	case block.Kind == sessionCardBubble:
		cardBubbleView(context, sessionID, card, block, index, gap, width, actions, local, onMeasured)
	case block.Markdown:
		cardMarkdown(context, local.Overlay, text, ovCardText, block.Live, onMeasured).Margin(gap, 0, 0, 0)
	case block.Kind == sessionCardProblem:
		ui.Row(context).Gap(layout.ProblemGap).AlignItems(ui.Start).Margin(gap, 0, 0, 0).Children(func() {
			ui.Icon(context, icons.DangerCircle.Bold).Size(layout.ProblemIcon, layout.ProblemIcon).TextColor(tok(context.Theme()).Danger).Shrink(0)
			ui.Text(context, text.text).FontSize(text.size).FontWeight(text.weight).TextColor(ovCardText).Grow(1).MinWidth(0)
		})
	default:
		color := ovCardText
		if block.Kind == sessionCardProblemMore || block.Kind == sessionCardSource {
			color = ovText2
		}
		textView := dictionaryText(context, text.text, block.Highlights).FontSize(text.size).TextColor(color).Margin(gap, 0, 0, 0)
		if text.weight != 0 {
			textView.FontWeight(text.weight)
		}
		if block.Selectable {
			textView.Selectable()
		}
	}
}

// cardBubbleView is a conversation's turn (Voxt's
// RewriteConversationBubble): the user's input fits its text on the right,
// up to 340 wide in the small card; an answer takes that width and, under
// the pointer, shows a Copy icon right of it, in the room the bubble
// leaves, so the icon never covers the text.
func cardBubbleView(context *ui.Context, sessionID string, card *sessionCardViewState, block sessionCardBlock, index int, gap float32, width int, actions sessionOverlayActions, local sessionOverlayLocal, onMeasured func([]sessionOverlayMeasurement)) {
	text, title, quote := sessionCardBlockText(block, width)
	justify := ui.Start
	background, line := ui.RGBA(255, 255, 255, 0.06), ui.RGBA(255, 255, 255, 0.08)
	titleAlpha, textAlpha := float32(0.64), float32(0.92)
	if block.User {
		justify, background, line, titleAlpha = ui.End, ui.RGBA(255, 255, 255, 0.1), ui.RGBA(255, 255, 255, 0.12), 0.7
	}
	if block.Live {
		// An answer being written keeps its text's full strength (#4105).
		if block.User {
			textAlpha = 0.84
		} else {
			titleAlpha = 0.54
		}
	}
	if block.User {
		textWidth, _ := context.MeasureText(0, text.span())
		titleWidth, _ := context.MeasureText(0, title.span())
		quoteWidth, _ := context.MeasureText(0, quote.span())
		text.wrap = layout.BubbleTextWidth(textWidth, titleWidth, quoteWidth, text.wrap)
	}
	row := ui.Row(context).Justify(justify).AlignItems(ui.Start).Gap(bubbleCopyGap).Margin(gap, 0, 0, 0)
	// The whole row, so the pointer can move from the answer to its Copy.
	hover := !block.User && !block.Live && row.Hovered()
	row.Children(func() {
		column := ui.Column(context.Key("bubble")).Width(text.wrap+2*layout.BubblePadX).Shrink(0).Padding(layout.BubblePadY-layout.Border, layout.BubblePadX-layout.Border).Gap(layout.BubbleTitleGap).
			Radius(layout.BubbleRadius).Background(background).Border(1, line)
		if block.User {
			column.AlignItems(ui.End)
		}
		column.Children(func() {
			if block.Title != "" {
				ui.Text(context, block.Title).FontSize(layout.BubbleTitleSize).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, titleAlpha)).SingleLine().Ellipsis("…").MaxWidth(text.wrap)
			}
			textColor := ui.RGBA(255, 255, 255, textAlpha)
			request := newSessionOverlayPreviewRequest(context, sessionID, card, block, index, textColor, 0, width)
			switch {
			case !block.User && actions.overlayTranscriptResult(context.Key("preview"), request):
			case block.Markdown:
				cardMarkdown(context, local.Overlay, text, textColor, block.Live, onMeasured)
			default:
				textView := ui.Text(context, block.Text).FontSize(layout.BubbleTextSize).FontWeight(500).TextColor(textColor)
				if block.User {
					textView.TextAlign(ui.End)
				}
				if !block.Live {
					textView.Selectable()
				}
			}
			if block.Quote != "" {
				cardQuoteView(context, block, quote.size, text.wrap, actions)
			}
		})
		if hover {
			bubbleCopyButton(context, block, local, actions)
		}
	})
}

// bubbleCopyGap is the room between an answer bubble and its Copy icon.
const bubbleCopyGap = 6

// bubbleCopyButton is an answer bubble's Copy: an icon only, its name in
// the tooltip; a check while the answer is the one just copied.
func bubbleCopyButton(context *ui.Context, block sessionCardBlock, local sessionOverlayLocal, actions sessionOverlayActions) {
	label, icon := i18n.T("Copy"), icons.Copy
	if local.Overlay.bubbleCopied == block.Text {
		label, icon = i18n.T("Copied"), icons.CopyCheck
	}
	button := ui.ButtonBase(context.Key("copy")).Size(24, 24).Shrink(0).Radius(12).
		Background(ui.RGBA(0, 0, 0, 0.74)).Border(1, ui.RGBA(255, 255, 255, 0.12)).Label(label).Tooltip(label)
	if button.Children(func() {
		ui.Row(context).Grow(1).Justify(ui.Center).AlignItems(ui.Center).Children(func() {
			ui.Icon(context, icon.Linear).Size(13, 13).TextColor(ui.RGBA(255, 255, 255, 0.94))
		})
	}).Clicked() {
		actions.copyBubble(block)
	}
}

// cardQuoteView is a follow-up's selection under its instruction, as wide
// as the bubble's text.
func cardQuoteView(context *ui.Context, block sessionCardBlock, size, wrap float32, actions sessionOverlayActions) {
	ui.Text(context, block.Quote).FontSize(size).TextColor(ui.RGBA(255, 255, 255, 0.64)).Width(wrap).Selectable()
	if !block.Cut {
		return
	}
	label := i18n.T("Show all")
	if block.Open {
		label = i18n.T("Show less")
	}
	if ui.ButtonBase(context.Key("quote")).Height(quoteToggleH).Label(label).Children(func() {
		ui.Text(context, label).FontSize(11).FontWeight(600).TextColor(ui.RGBA(255, 255, 255, 0.78)).SingleLine()
	}).Clicked() {
		actions.toggleQuote(block.Turn)
	}
}
