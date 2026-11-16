package main

import (
	"math"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The meeting window's sizes in DIPs: the face's plus layout.Pad on each
// side for its shadow.
const (
	meetingOvProblemH   = 42 // two lines of the warning and its gap
	meetingOvCollapsedW = 400 + 2*layout.Pad
	meetingOvCollapsedH = layout.PillH + 2*layout.Pad // the dictation pill's (#3986)
	meetingOvMaxFaceH   = 340                         // then the transcript scrolls
	meetingOvWhoH       = 15                          // a speaker's name, 11 DIPs, and its gap
	meetingOvRowGap     = 8
	meetingOvMenuRowH   = 40 // a menu row: its name and a hint under it
)

// meetingOvWidths are the expanded face's widths by size step
// (Features.CardSize("meeting")), stepped from the corner as the answer
// card's are; the smallest holds the header.
var meetingOvWidths = [settings.CardSizeSteps]int{520, 600, 760}

// meetingOvChrome is the expanded face's height beside its transcript:
// the card's padding, header and rule.
const meetingOvChrome = layout.CardPadTop + layout.CardHeadH + layout.CardRuleTop + 1 + layout.CardRuleGap + layout.CardPadY

// meetingOverlayLayout is the meeting floating window's size.
type meetingOverlayLayout struct {
	Width, Height int
	Collapsed     bool
}

// meetingOverlayModel is the pure view model for the meeting overlay.
type meetingOverlayModel struct {
	Layout        meetingOverlayLayout
	ProblemKey    string
	ShowBar       bool
	ShowExpanded  bool
	ShowAsking    bool
	ShowFinishing bool
}

// overlayModel builds the overlay model from the meeting snapshot: size
// is the expanded face's step (meetingOvWidths), menu whether a menu
// shows in place of the transcript. Asking and finishing always show the
// capsule. The expanded face is as high as its lines, up to
// meetingOvMaxFaceH (the answer card's way).
func overlayModel(s meeting.Snapshot, collapsed bool, size int, menu meetingMenu) meetingOverlayModel {
	asking, finishing := s.State == meeting.StateAsking, s.State == meeting.StateFinishing
	pill := collapsed || asking || finishing
	m := meetingOverlayModel{
		Layout: meetingOverlayLayout{Width: meetingOvCollapsedW, Height: meetingOvCollapsedH, Collapsed: collapsed},
		ShowBar: s.State == meeting.StateRecording || s.State == meeting.StatePaused ||
			s.State == meeting.StateStarting || finishing,
		ShowExpanded:  !pill && s.State.Active(),
		ShowAsking:    asking,
		ShowFinishing: finishing,
	}
	if s.Problem != nil {
		m.ProblemKey = s.Problem.Key
	}
	problem := m.ProblemKey != "" && meetingOverlayProblemText(m.ProblemKey) != "" && !asking && !finishing
	if !pill {
		w := meetingOvWidths[min(max(size, 0), len(meetingOvWidths)-1)]
		body := meetingOverlayBodyHeight(meetingOverlayRows(s), w-2*layout.CardPadX-layout.CardInset)
		if menu != meetingMenuNone {
			body = max(body, menuHeight(menu, s))
		}
		h := meetingOvChrome + body
		if problem {
			h += meetingOvProblemH
		}
		m.Layout.Width, m.Layout.Height = w+2*layout.Pad, min(h, meetingOvMaxFaceH)+2*layout.Pad
	} else if problem {
		m.Layout.Height += meetingOvProblemH
	}
	return m
}

// meetingOverlayRow is one row of the live transcript: lines the same
// speaker said in a row show the name once (first), and a sentence the
// recogniser cut in two is one row.
type meetingOverlayRow struct {
	Key   any // the first segment's ID, or the live line's source
	Who   string
	Text  string
	Me    bool
	Live  bool // the sentence still being heard
	First bool // the first of the speaker's rows: it shows the name
}

// meetingJoinGap is the longest pause inside a sentence the recogniser
// split: a segment that ends without a full stop and the same speaker's
// next one within it are one row.
const meetingJoinGap = 2 * time.Second

// meetingOverlayRows are s's segments and live lines as the overlay shows
// them.
func meetingOverlayRows(s meeting.Snapshot) []meetingOverlayRow {
	var rows []meetingOverlayRow
	var lastEnd time.Duration
	add := func(r meetingOverlayRow, start time.Duration, joinable bool) {
		if n := len(rows); n > 0 {
			prev := &rows[n-1]
			if prev.Who == r.Who {
				if joinable && !prev.Live && !endsSentence(prev.Text) && start-lastEnd <= meetingJoinGap {
					prev.Text = joinMeetingText(prev.Text, r.Text)
					return
				}
				r.First = false
				rows = append(rows, r)
				return
			}
		}
		r.First = true
		rows = append(rows, r)
	}
	for _, seg := range s.Segments {
		if strings.TrimSpace(seg.Text) == "" {
			continue
		}
		me := seg.Speaker == "me" || seg.Speaker == "" && seg.Source == meeting.SourceMic
		add(meetingOverlayRow{Key: seg.ID, Who: meetingSpeaker(seg.Speaker, seg.Source), Text: seg.Text, Me: me}, seg.Start, true)
		lastEnd = seg.End
	}
	for _, src := range []meeting.Source{meeting.SourceMic, meeting.SourceSystem} {
		if t := s.Live[src]; t != "" {
			add(meetingOverlayRow{Key: src, Who: meetingSpeaker("", src), Text: t, Me: src == meeting.SourceMic, Live: true}, 0, false)
		}
	}
	return rows
}

// endsSentence says whether text ends with a sentence's last mark.
func endsSentence(text string) bool {
	t := []rune(strings.TrimRight(text, " \"'”’)）"))
	return len(t) > 0 && strings.ContainsRune(".?!。？！…", t[len(t)-1])
}

// joinMeetingText joins two parts of a sentence: a space between Latin
// words, none between Chinese ones.
func joinMeetingText(a, b string) string {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	ra, rb := []rune(a), []rune(b)
	if len(ra) == 0 || len(rb) == 0 {
		return a + b
	}
	if ra[len(ra)-1] < 0x2E80 && rb[0] < 0x2E80 {
		return a + " " + b
	}
	return a + b
}

// meetingOverlayBodyHeight is how high rows lay out textW wide: at least
// one line (Listening… before any text).
// shortcut: guessed from the font's average widths (guessText), not
// measured as the answer card's are; a little off at the cap, where the
// transcript scrolls anyway.
func meetingOverlayBodyHeight(rows []meetingOverlayRow, textW int) int {
	line := int(math.Ceil(layout.CardTextSize * 1.3))
	if len(rows) == 0 {
		return line
	}
	h := 0
	for i, r := range rows {
		if i > 0 {
			h += meetingOvRowGap
		}
		if r.First {
			h += meetingOvWhoH + 2
		}
		_, th := guessText(ovSized{r.Text, layout.CardTextSize, layout.CardTextWeight, float32(textW)})
		h += int(math.Ceil(float64(th)))
	}
	return h
}

func meetingOverlayProblemText(key string) string {
	if key == "" {
		return ""
	}
	switch key {
	case meeting.ProblemKeySystemSilent:
		return i18n.T("If you see no text when others speak, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording.")
	case meeting.ProblemKeyLiveBacklog:
		return i18n.T("Live captions are falling behind. The final transcript will catch up after you end the meeting.")
	case meeting.ProblemKeyDiskLow:
		return i18n.T("Disk space is running low. End the meeting soon or free space on this Mac.")
	case meeting.ProblemKeyHelperOld:
		return i18n.T("Recording system audio needs a newer Voxt helper. Update Voxt, or record the microphone only.")
	default:
		if strings.HasPrefix(key, "meeting.track.degraded.") {
			return i18n.T("One audio track stopped. Voxt switched to a simpler capture mode.")
		}
		return ""
	}
}

// meetingMenu is the menu the expanded overlay shows over its transcript.
type meetingMenu int

const (
	meetingMenuNone meetingMenu = iota
	meetingMenuMode             // the header's mode capsule: Meeting, Subtitles, Recording
	meetingMenuMore             // "More": translation, copy, cancel
)

// meetingMoreItems are the More menu's items while s records.
type meetingMoreItem int

const (
	moreTranslate meetingMoreItem = iota
	moreCopy
	moreCancel
)

func meetingMoreItems(s meeting.Snapshot) []meetingMoreItem {
	items := []meetingMoreItem{moreTranslate}
	if s.State == meeting.StatePaused {
		items = append(items, moreCopy)
	}
	return append(items, moreCancel)
}

// menuHeight is how high menu lays out.
func menuHeight(menu meetingMenu, s meeting.Snapshot) int {
	switch menu {
	case meetingMenuMode:
		return 3*meetingOvMenuRowH + 2*4
	case meetingMenuMore:
		n := len(meetingMoreItems(s))
		return n*30 + (n-1)*4
	}
	return 0
}
