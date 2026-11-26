package main

import (
	"cmp"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// A conversation's turns as the card draws them: each instruction, with
// what it sent, above its answer.

func sessionCardConversationBlocks(card answerCard, chat bool) []sessionCardBlock {
	blocks := make([]sessionCardBlock, 0, 2*len(card.turns)+1)
	for turn, answer := range card.turns {
		if user, ok := sessionCardUserBlock(card.ui.quotesOpen, turn, answer.Instruction, answer.Selection, false, chat); ok {
			if chat {
				user.Images = sessionCardTurnImages(card.previews, turn)
			}
			blocks = append(blocks, user)
		}
		blocks = append(blocks, sessionCardAnswerBlock(sessionCardTurnKey(turn, "answer"), answer.Answer.Title, answer.Answer.Content, false))
	}
	if card.problem != nil {
		if user, ok := sessionCardUserBlock(card.ui.quotesOpen, len(card.turns), card.input.Raw, card.input.Selection, false, chat); ok {
			blocks = append(blocks, user)
		}
	}
	return blocks
}

func buildSessionCardLiveTail(input sessionCardProjectionInput, follow bool) ([2]sessionCardBlock, int) {
	var live [2]sessionCardBlock
	if !input.Answer.open && !input.Chat {
		if input.Writing {
			live[0] = sessionCardWritingBlock(input.ShownDraft, input.Snapshot.Kind)
			return live, 1
		}
		return live, 0
	}
	if sessionCardRedoWriting(input) {
		live[0] = sessionCardWritingBlock(input.ShownDraft, input.Answer.kind)
		return live, 1
	}
	if !follow && input.Answer.open { // the first turn is the Rewrite card's own
		return live, 0
	}
	snapshot := input.Snapshot
	shownDraft := input.ShownDraft
	if input.HideLiveText {
		snapshot.Partial, shownDraft = "", ""
	}
	count := 0
	said := cmpOr(strings.TrimSpace(snapshot.Raw), strings.TrimSpace(snapshot.Partial))
	if input.Chat && !input.Answer.open {
		said = cmp.Or(strings.TrimSpace(snapshot.Raw), strings.TrimSpace(snapshot.Spoken), said) // what the model was asked
	}
	selection := cmpOr(snapshot.Selection, snapshot.Typed) // the first turn's typed text is its source
	turn := len(input.Answer.turns)
	if !input.Answer.open {
		turn = 0
	}
	if user, ok := sessionCardUserBlock(input.Answer.ui.quotesOpen, turn, said, selection, true, input.Chat); ok {
		user.Preview = strings.TrimSpace(snapshot.Raw) == "" && strings.TrimSpace(snapshot.Partial) != ""
		if input.Chat {
			user.Images = sessionCardTurnImages(input.previews(), turn)
		}
		live[0], count = user, 1
	}
	if snapshot.State != session.Recording {
		live[count] = sessionCardAnswerBlock(sessionCardLiveAnswerKey, snapshot.Title, shownDraft, true)
		count++
	}
	return live, count
}

// sessionCardUserBlock is a turn's user side; it has no heading, its side
// and fill say whose it is. A chat quotes the first turn's selection too;
// the answer card names it in its input row instead.
func sessionCardUserBlock(quotesOpen map[int]bool, turn int, said, selection string, live, chat bool) (sessionCardBlock, bool) {
	block := sessionCardBlock{Key: sessionCardTurnKey(turn, "user"), Kind: sessionCardBubble, Turn: turn,
		Text: strings.TrimSpace(said), Live: live, User: true, Selectable: !live}
	if turn > 0 || chat {
		block.FullQuote = strings.TrimSpace(selection)
		block.Open = quotesOpen[turn]
		block.Quote, block.Cut = quoteText(selection, block.Open)
	}
	if block.Text == "" {
		block.Text = "…"
	}
	return block, strings.TrimSpace(said) != "" || block.Quote != ""
}

// previews are the card's inputs and then the live follow-up's: the
// view's list, which blocks' Images index.
func (input sessionCardProjectionInput) previews() []answerInputPreview {
	if len(input.LivePreviews) == 0 {
		return input.Answer.previews
	}
	return append(slices.Clone(input.Answer.previews), input.LivePreviews...)
}

// sessionCardTurnImages are the pictures turn sent, as indices into previews.
func sessionCardTurnImages(previews []answerInputPreview, turn int) []int {
	var images []int
	for index, preview := range previews {
		if preview.Image != nil && preview.Turn == turn {
			images = append(images, index)
		}
	}
	return images
}

// boxWriting says whether the Rewrite card's first instruction is sent and
// its answer on the way: the card shows the turn at once, the answer
// under it. A window picture still waiting for review keeps the composer.
func (a *app) boxWriting() bool {
	s := a.cardSnapshot() // a Transcribe sends the streaming answer to the background
	return a.box.open && a.box.shown && !a.answer.open && s.ID != "" && s.ID == a.boxSessionID() && s.Kind == store.KindRewrite &&
		s.State == session.Composing && s.Card && !s.Try && !(s.ImagePreviewKind == "window" && len(s.ImagePreview) > 0 && strings.TrimSpace(s.Draft) == "")
}

// boxLivePreviews are the pictures the box's live session sends, as the
// turn they go with.
func (a *app) boxLivePreviews(input sessionCardProjectionInput) []answerInputPreview {
	s := input.Snapshot
	if !input.Chat || s.ID == "" || s.ID != a.boxSessionID() || !s.State.Live() || s.ID == input.Answer.id {
		return nil
	}
	turn := 0
	if input.Answer.open {
		turn = len(input.Answer.turns)
	}
	var images []answerInputPreview
	for _, preview := range answerInputPreviews(s, &a.box) {
		if preview.Image != nil {
			preview.Turn = turn
			images = append(images, preview)
		}
	}
	return images
}
