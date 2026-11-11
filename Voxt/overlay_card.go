package main

// The answer card, redone after Voxt's WaveformAnswerCard
// (10-15, docs/behavior.md §5.24): where a Translate or Rewrite result
// that was not inserted waits, where an answer streams while the model
// writes it, and where a Rewrite answer takes follow-ups.
//
//	header   Voxt's answer icon (20), the title (15, semibold) and Voxt's
//	         arc spinner while it works; then its actions, 24 high:
//	         Translate's language capsule (it opens our own menu,
//	         overlay_lang.go), Continue (a Rewrite follow-up), Insert, Copy,
//	         Close (24 circles); a conversation shows Voxt's microphone and a
//	         small waveform in the title's place. 12 over it, 12 under it to
//	         the rule (Voxt's 16 and 8 looked off, 10-15)
//	rule     1 DIP, white 8 %, at 48 (Voxt's), 10 over the text
//	text     14 medium, white 90 %, selectable, as high as it is up to
//	         Voxt's 220, then it scrolls; a conversation's turns as
//	         bubbles instead
//	line     a small one under it, when there is something to say (why
//	         Insert did not work, another language that did not come)
//
// 440 wide (Voxt's card is 560, measured on the Mac: it fills its 560
// window; the user found it too wide, then 480 too, 10-15), 18 of padding on the sides and 16 under the text: as high as its text,
// so a short answer leaves no empty space (Voxt's card is always 220 of
// text high, 295 in all). It grows and shrinks about the overlay's anchor,
// as the pill does (overlay.go). It does not move, and nothing in it takes
// the keyboard (a non-activating window): the cancel shortcut (the helper's event tap)
// closes it, the Rewrite shortcut follows up.

import (
	"math"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// Colours of the card: Voxt's.
var (
	ovCardText = ui.RGBA(255, 255, 255, 0.9)
	ovCardRule = ui.RGBA(255, 255, 255, 0.08)
)

// cardView is what the answer card shows: an ended session's card, or
// the answer the model is writing into it.
type cardView struct {
	kind  store.Kind
	title string
	text  string
	// busy: another language, or a follow-up, is on its way (the
	// spinner, or the conversation's waveform). A first answer is written
	// in the pill: the card comes with it.
	busy bool
	// recording: a follow-up records (the conversation's waveform moves).
	recording bool
	levels    []float64
	heard     bool
	failed    *session.Problem
	source    []string // a failure: what it worked on
	msg       string   // the small line under the text
	why       string   // why it is in the card (Insert's tip)
	lang      string   // Translate's target
	target    string   // the app it came from
	// conversation: the turns of a Rewrite conversation, as bubbles.
	conversation bool
	bubbles      []cardBubble
	// What the header offers.
	canInsert, canCopy, canFollowUp, canStop, canRetry, canLang bool
	copied                                                      bool
	follow                                                      string // the Rewrite shortcut's glyphs
	// size is the card's, its feature's (Features.CardSizes).
	size layout.CardSize
	// writing: Rewrite's first answer as the model writes it
	// (writingAnswer), the card at its full height from the first words:
	// it does not grow line by line (10-15).
	writing bool
}

// cardBubble is one turn's side in a conversation.
type cardBubble struct {
	user        bool
	title, text string
	live        bool // being said, or written
	// quote: a follow-up's text selected since, under its instruction
	// (#3816), shortened unless open; turn is its index in the turns.
	quote     string
	cut, open bool
	turn      int
}

// quoteMax is how much of a follow-up's selection its bubble shows
// before Show all: runes, lines.
const quoteMax, quoteLines = 160, 3

// quoteText is sel as a bubble shows it: whole when open, else cut to
// quoteMax runes or quoteLines lines, and whether it was cut.
func quoteText(sel string, open bool) (string, bool) {
	sel = strings.TrimSpace(sel)
	short := sel
	if lines := strings.SplitN(short, "\n", quoteLines+1); len(lines) > quoteLines {
		short = strings.Join(lines[:quoteLines], "\n")
	}
	if r := []rune(short); len(r) > quoteMax {
		short = string(r[:quoteMax])
	}
	if short == sel {
		return sel, false
	}
	if open {
		return sel, true
	}
	return strings.TrimSpace(short) + "…", true
}

// userBubble is a turn's user side: what was said, and the text selected
// since for a follow-up (the first turn's selection is the card's text).
func (card answerCard) userBubble(turn int, said, sel string, live bool) (cardBubble, bool) {
	b := cardBubble{user: true, title: i18n.T("You"), text: strings.TrimSpace(said), live: live, turn: turn}
	if turn > 0 {
		b.open = card.quotesOpen[turn]
		b.quote, b.cut = quoteText(sel, b.open)
	}
	return b, b.text != "" || b.quote != ""
}

// cardNow is the card as it shows now.
func (a *app) cardNow() cardView {
	s := a.sess
	if a.prefs.Overlay.HideLiveText {
		s.Partial, s.Draft = "", "" // "Show live text" off: no words in a follow-up's bubbles either
	}
	if !a.answer.open {
		if a.writingAnswer() {
			s := a.pillSession()
			return cardView{kind: s.Kind, title: uitext.CardTitle(s.Kind, s.Title), text: a.shownDraft(s), busy: true, writing: true,
				target: s.Target.Name, size: layout.CardSizes[a.prefs.Features.CardSize(string(s.Kind))]}
		}
		return cardView{size: layout.CardSizes[0]} // no card: the pill shows the session
	}
	card := a.answer
	own := a.cardSessionLive()
	redo := own && s.ID == card.redo
	follow := own && !redo
	v := cardView{kind: card.kind, text: card.text, failed: card.problem, msg: card.msg, why: card.why, lang: card.lang,
		target: card.target.Name, copied: card.copied, busy: own, conversation: card.conversation || follow,
		size: layout.CardSizes[a.prefs.Features.CardSize(string(card.kind))]}
	v.title = uitext.CardTitle(card.kind, card.title)
	if redo {
		v.lang = card.redoLang
	}
	v.canCopy = card.text != ""
	v.canInsert = card.problem == nil && card.text != "" && !own
	v.canLang = card.kind == store.KindTranslate && card.problem == nil && !s.State.Live()
	v.canFollowUp = a.canFollowUp()
	v.canStop = follow && s.State == session.Recording
	v.canRetry = card.problem != nil && !s.State.Live()
	v.follow = a.featureKeys(hotkey.Rewrite)
	if v.failed != nil && !v.conversation {
		// What it worked on: the instruction, then the selection.
		in := card.input
		if card.kind == store.KindRewrite && in.Raw != "" && in.Selection != "" {
			v.source = append(v.source, in.Raw)
		}
		if src := cmpOr(in.Selection, in.Raw); src != "" {
			v.source = append(v.source, src)
		}
	}
	if !v.conversation {
		return v
	}
	v.recording = follow && s.State == session.Recording
	v.levels, v.heard = s.Levels, !s.HeardAt.IsZero()
	for i, t := range card.turns {
		if b, ok := card.userBubble(i, t.Instruction, t.Selection, false); ok {
			v.bubbles = append(v.bubbles, b)
		}
		v.bubbles = append(v.bubbles, cardBubble{title: t.Answer.Title, text: t.Answer.Content})
	}
	switch {
	case follow:
		if b, ok := card.userBubble(len(card.turns), cmpOr(strings.TrimSpace(s.Raw), strings.TrimSpace(s.Partial)), s.Selection, true); ok {
			v.bubbles = append(v.bubbles, b)
		}
		if s.State != session.Recording {
			v.bubbles = append(v.bubbles, cardBubble{title: s.Title, text: a.shownDraft(s), live: true})
		}
	case card.problem != nil:
		// The follow-up that failed: what was said; why, under the card.
		if b, ok := card.userBubble(len(card.turns), card.input.Raw, card.input.Selection, false); ok {
			v.bubbles = append(v.bubbles, b)
		}
		if v.msg == "" {
			v.msg, _, _ = sessionProblemText(card.problem)
		}
	}
	return v
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// writingAnswer says whether Rewrite's first answer is being written, its
// first words in, and will end in the card: the pill hands over to the
// answer card, which the answer streams into, instead of a loader until
// the whole answer is in (#3644). Not with Show live text off, nor for a
// follow-up (its card is open already), a try (the page shows it) or an
// answer that goes in at once (the card switch off: the pill shows it).
func (a *app) writingAnswer() bool {
	s := a.pillSession()
	return !a.answer.open && s.Kind == store.KindRewrite && s.State == session.Composing && s.Card && !s.Try && strings.TrimSpace(s.Draft) != ""
}

// cardRecording says whether the card shows a follow-up recording (its
// waveform moves).
func (a *app) cardRecording() bool {
	return a.answer.open && a.cardSessionLive() && a.sess.ID != a.answer.redo && a.sess.State == session.Recording
}

// cardBlock is one piece of the card's text, as it lays out: a text
// (wrapped where it draws), a failure's line, or a conversation's bubble.
type cardBlock struct {
	text  ovSized
	color ui.Color
	gap   float32 // before it
	// selectable: the answer (its text can be selected and copied).
	selectable bool
	// problem: a failure's line, its icon before it.
	problem bool
	// bubble: a conversation's turn, title its heading (text "" for
	// none), user whose side, live being said or written.
	bubble     bool
	title      ovSized
	user, live bool
	// quote: a follow-up's selection under the text; cut, open: Show all
	// or Show less under it, for the turn's.
	quote     ovSized
	cut, open bool
	turn      int
}

// quoteGap is the space above a bubble's selection and its Show all
// (the bubble column's gap), quoteToggleH that line's height.
const quoteGap, quoteToggleH, quoteSize = layout.BubbleTitleGap, 16, 12

// cardBlocks are the pieces of v's text, in order.
func cardBlocks(v cardView) []cardBlock {
	bodyW := float32(v.size.BodyW())
	switch {
	case v.conversation:
		var out []cardBlock
		inner := float32(v.size.BubbleW() - 2*layout.BubblePadX)
		for i, b := range v.bubbles {
			text := strings.TrimSpace(b.text)
			if text == "" {
				text = "…"
			}
			blk := cardBlock{bubble: true, user: b.user, live: b.live, text: ovSized{text, layout.BubbleTextSize, 500, inner},
				cut: b.cut, open: b.open, turn: b.turn}
			if b.quote != "" {
				blk.quote = ovSized{b.quote, quoteSize, 0, inner}
			}
			// An answer without a title has no heading (Voxt's).
			if title := strings.TrimSpace(b.title); title != "" {
				blk.title = ovSized{title, layout.BubbleTitleSize, 600, 0}
			}
			if i > 0 {
				blk.gap = layout.BubbleGap
			}
			out = append(out, blk)
		}
		return out
	case v.failed != nil:
		text, more, _ := sessionProblemText(v.failed)
		out := []cardBlock{{problem: true, color: ovCardText,
			text: ovSized{text, layout.CardTextSize, layout.CardTextWeight, bodyW - layout.ProblemIcon - layout.ProblemGap}}}
		if more != "" {
			out = append(out, cardBlock{text: ovSized{more, 12, 0, bodyW}, color: ovText2, gap: 4, selectable: true})
		}
		for _, src := range v.source {
			out = append(out, cardBlock{text: ovSized{strings.TrimSpace(src), 13, 0, bodyW}, color: ovText2, gap: 10, selectable: true})
		}
		return out
	}
	text := strings.TrimSpace(v.text)
	return []cardBlock{{text: ovSized{text, layout.CardTextSize, layout.CardTextWeight, bodyW}, color: ovCardText, selectable: text != ""}}
}

// cardTexts are the texts the card's height depends on.
func cardTexts(v cardView) []ovSized {
	var out []ovSized
	for _, b := range cardBlocks(v) {
		out = append(out, b.text)
		if b.title.text != "" {
			out = append(out, b.title)
		}
		if b.quote.text != "" {
			out = append(out, b.quote)
		}
	}
	return out
}

// blockHeight is how high b lays out, and whether that was measured.
func (a *app) blockHeight(b cardBlock) (float32, bool) {
	h, exact := a.ovTextHeight(b.text)
	if b.problem {
		h = max(h, layout.ProblemIcon)
	}
	if b.bubble {
		h += 2 * layout.BubblePadY
		if b.title.text != "" {
			th, ok := a.ovTextHeight(b.title)
			h, exact = h+th+layout.BubbleTitleGap, exact && ok
		}
		if b.quote.text != "" {
			qh, ok := a.ovTextHeight(b.quote)
			h, exact = h+quoteGap+qh, exact && ok
			if b.cut {
				h += quoteGap + quoteToggleH
			}
		}
	}
	return h, exact
}

// streamGrowStep is how much the card grows at a time while an answer
// streams into it: four lines of its text.
const streamGrowStep = 4 * 19 // layout.CardTextSize × 1.3, rounded up (guessText)

// cardBodyHeight is the height of v's text in the card: all of it up to
// cardBodyMax (then it scrolls), and whether it was measured.
func (a *app) cardBodyHeight(v cardView) (int, bool) {
	bodyMax := a.cardBodyMax(v)
	var h float32
	exact := true
	for _, b := range cardBlocks(v) {
		bh, ok := a.blockHeight(b)
		h += b.gap + bh
		exact = exact && ok
	}
	if v.writing || v.conversation && v.busy && !v.recording {
		// An answer on its way: the card grows a few lines at a time,
		// not with every line (#3836).
		h = float32(math.Ceil(float64(h/streamGrowStep))) * streamGrowStep
	}
	return min(int(math.Ceil(float64(h))), bodyMax), exact
}

// cardBodyMax is the most text v shows before it scrolls: as much as the
// work area holds beside the overlay's margin, so that the card grows
// with a long answer and stays on the screen (#3644 FB41); never less
// than layout.CardBodyMax.
func (a *app) cardBodyMax(v cardView) int {
	room := a.ov.area.Height - int(math.Round(a.prefs.Overlay.Margin)) - layout.Pad - layout.CardChrome
	if v.msg != "" {
		room -= layout.CardMsgGap + layout.CardMsgH
	}
	return max(room, layout.CardBodyMax)
}

// cardSize is the card's width (its size's) and height, and whether its
// texts were measured.
func (a *app) cardSize() (w, h int, exact bool) {
	v := a.cardNow()
	body, exact := a.cardBodyHeight(v)
	h = layout.CardChrome + body
	if v.msg != "" {
		h += layout.CardMsgGap + layout.CardMsgH
	}
	return v.size.W, h, exact
}

// featureKeys are the glyphs of a feature's first shortcut ("" for none).
func (a *app) featureKeys(feature string) string {
	hs := hotkey.HotkeysOf(a.prefs.Features, feature)
	if len(hs) == 0 {
		return ""
	}
	return uitext.HotkeyText(hs[0].Keys)
}
