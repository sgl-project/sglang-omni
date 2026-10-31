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
// the keyboard (a non-activating window): Escape (the helper's event tap)
// closes it, the Rewrite shortcut follows up.

import (
	"math"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
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
}

// cardBubble is one turn's side in a conversation.
type cardBubble struct {
	user        bool
	title, text string
	live        bool // being said, or written
}

// cardNow is the card as it shows now.
func (a *app) cardNow() cardView {
	s := a.sess
	if !a.answer.open {
		return cardView{} // no card: the pill shows the session
	}
	card := a.answer
	own := a.cardSessionLive()
	redo := own && s.ID == card.redo
	follow := own && !redo
	v := cardView{kind: card.kind, text: card.text, failed: card.problem, msg: card.msg, why: card.why, lang: card.lang,
		target: card.target.Name, copied: card.copied, busy: own, conversation: card.conversation || follow}
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
	for _, t := range card.turns {
		if in := strings.TrimSpace(t.Instruction); in != "" {
			v.bubbles = append(v.bubbles, cardBubble{user: true, title: i18n.T("You"), text: in})
		}
		v.bubbles = append(v.bubbles, cardBubble{title: t.Answer.Title, text: t.Answer.Content})
	}
	switch {
	case follow:
		if said := cmpOr(strings.TrimSpace(s.Raw), strings.TrimSpace(s.Partial)); said != "" {
			v.bubbles = append(v.bubbles, cardBubble{user: true, title: i18n.T("You"), text: said, live: true})
		}
		if s.State != session.Recording {
			v.bubbles = append(v.bubbles, cardBubble{title: s.Title, text: s.Draft, live: true})
		}
	case card.problem != nil:
		// The follow-up that failed: what was said; why, under the card.
		if in := strings.TrimSpace(card.input.Raw); in != "" {
			v.bubbles = append(v.bubbles, cardBubble{user: true, title: i18n.T("You"), text: in})
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
}

// cardBlocks are the pieces of v's text, in order.
func cardBlocks(v cardView) []cardBlock {
	switch {
	case v.conversation:
		var out []cardBlock
		inner := float32(layout.BubbleW - 2*layout.BubblePadX)
		for i, b := range v.bubbles {
			text := strings.TrimSpace(b.text)
			if text == "" {
				text = "…"
			}
			blk := cardBlock{bubble: true, user: b.user, live: b.live, text: ovSized{text, layout.BubbleTextSize, 500, inner}}
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
			text: ovSized{text, layout.CardTextSize, layout.CardTextWeight, layout.CardBodyW - layout.ProblemIcon - layout.ProblemGap}}}
		if more != "" {
			out = append(out, cardBlock{text: ovSized{more, 12, 0, layout.CardBodyW}, color: ovText2, gap: 4, selectable: true})
		}
		for _, src := range v.source {
			out = append(out, cardBlock{text: ovSized{strings.TrimSpace(src), 13, 0, layout.CardBodyW}, color: ovText2, gap: 10, selectable: true})
		}
		return out
	}
	text := strings.TrimSpace(v.text)
	return []cardBlock{{text: ovSized{text, layout.CardTextSize, layout.CardTextWeight, layout.CardBodyW}, color: ovCardText, selectable: text != ""}}
}

// cardTexts are the texts the card's height depends on.
func cardTexts(v cardView) []ovSized {
	var out []ovSized
	for _, b := range cardBlocks(v) {
		out = append(out, b.text)
		if b.title.text != "" {
			out = append(out, b.title)
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
	}
	return h, exact
}

// cardBodyHeight is the height of v's text in the card: all of it up to
// layout.CardBodyMax (then it scrolls), and whether it was measured.
func (a *app) cardBodyHeight(v cardView) (int, bool) {
	var h float32
	exact := true
	for _, b := range cardBlocks(v) {
		bh, ok := a.blockHeight(b)
		h += b.gap + bh
		exact = exact && ok
	}
	return min(int(math.Ceil(float64(h))), layout.CardBodyMax), exact
}

// cardHeight is the card's height (its width is layout.AnswerW), and whether
// its texts were measured.
func (a *app) cardHeight() (int, bool) {
	v := a.cardNow()
	body, exact := a.cardBodyHeight(v)
	h := layout.CardChrome + body
	if v.msg != "" {
		h += layout.CardMsgGap + layout.CardMsgH
	}
	return h, exact
}

// featureKeys are the glyphs of a feature's first shortcut ("" for none).
func (a *app) featureKeys(feature string) string {
	hs := hotkey.HotkeysOf(a.prefs.Features, feature)
	if len(hs) == 0 {
		return ""
	}
	return uitext.HotkeyText(hs[0].Keys)
}
