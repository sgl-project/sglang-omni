package main

// The share card (share_card.go): Voxt in a hat beside a sentence said to
// it, a picture to copy and send. This file builds what the card shows
// from what share_controller.go keeps: the background, the hat, the
// sentence's phase (empty, listening, working, said) and its footer.

import (
	"fmt"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// shareBackground is one of the card's backgrounds: the three colors its
// gradient runs through, and whether it is dark (the panel and the text
// on it follow).
type shareBackground struct {
	Dark    bool
	A, B, C string
}

// shareBackgrounds are the card's backgrounds: five light, five dark. The
// shuffle picks among all ten, whatever the app's appearance: the picture
// is for someone else's screen.
var shareBackgrounds = [...]shareBackground{
	{false, "#4F6FE8", "#9A7CF2", "#FFAE86"},
	{false, "#21A890", "#4F9BEA", "#A996F7"},
	{false, "#7466EE", "#D877B0", "#FFB389"},
	{false, "#3D86EC", "#6FC6E3", "#FFC090"},
	{false, "#8A66F2", "#5486EC", "#5FCFB4"},
	{true, "#0F4A40", "#14121F", "#4A2A1C"},
	{true, "#1D2C6B", "#12111C", "#4A2340"},
	{true, "#3A1F66", "#14121F", "#5A2A2A"},
	{true, "#0D3A5A", "#10131D", "#2E1F55"},
	{true, "#4A2516", "#16121A", "#1F3A5C"},
}

// The sentence's phases.
const (
	sharePhaseEmpty     = "empty"     // nothing said yet: "想说点什么？", a click records
	sharePhaseListening = "listening" // a try records: the levels and the time
	sharePhaseWorking   = "working"   // the try is recognised or rewritten
	sharePhaseSaid      = "said"      // the sentence, its length and its day
)

type shareCardViewState struct {
	Background int  `json:"background"`
	Dark       bool `json:"dark"`
	// Outfit is a built-in hat (mascotPieces); Hat a drawn one, by its
	// name in the wardrobe. At most one is set.
	Outfit string `json:"outfit,omitempty"`
	Hat    string `json:"hat,omitempty"`
	// Dressed counts the hats put on, so the card's Voxt reacts to each.
	Dressed uint64    `json:"dressed"`
	Phase   string    `json:"phase"`
	Levels  []float64 `json:"levels,omitempty"`
	Elapsed string    `json:"elapsed,omitempty"`
	Text    string    `json:"text,omitempty"`
	// Meta is the said sentence's footer: its length and its day.
	Meta   string `json:"meta,omitempty"`
	Copied bool   `json:"copied,omitempty"`
	// Notice says why a try did not start (a model missing, say).
	Notice string                `json:"notice,omitempty"`
	Studio *shareStudioViewState `json:"studio,omitempty"`
}

type shareCardInput struct {
	Background  int
	Outfit, Hat string
	Dressed     uint64
	Text        string        // the kept sentence, "" before one
	Audio       time.Duration // how long it was said
	Said        time.Time     // when
	Mine        bool          // the session is the card's try
	Snapshot    session.Snapshot
	Now         time.Time
	Copied      bool
	Notice      string
	Studio      *shareStudioViewState
}

func buildShareCardViewState(translator *i18n.Translator, input shareCardInput) shareCardViewState {
	index := input.Background % len(shareBackgrounds)
	if index < 0 {
		index += len(shareBackgrounds)
	}
	state := shareCardViewState{
		Background: index, Dark: shareBackgrounds[index].Dark, Outfit: input.Outfit, Hat: input.Hat,
		Dressed: input.Dressed, Phase: sharePhaseEmpty, Copied: input.Copied, Notice: input.Notice, Studio: input.Studio,
	}
	if input.Text != "" {
		state.Phase = sharePhaseSaid
		state.Text = input.Text
		state.Meta = shareMeta(translator, input.Audio, input.Said)
	}
	if !input.Mine {
		return state
	}
	switch snapshot := input.Snapshot; {
	case snapshot.State == session.Recording:
		state.Phase = sharePhaseListening
		state.Levels = snapshot.Levels
		state.Elapsed = tryElapsedText(snapshot.Elapsed(input.Now))
	case snapshot.State.Live():
		state.Phase = sharePhaseWorking
	}
	return state
}

// shareMeta is a said sentence's footer: "0:11 · 2027年2月6日".
func shareMeta(translator *i18n.Translator, audio time.Duration, said time.Time) string {
	seconds := int(audio.Round(time.Second) / time.Second)
	return fmt.Sprintf("%d:%02d", seconds/60, seconds%60) + " · " + homeActivityDate(translator, said)
}

// shareSentence is what a finished try leaves on the card: its result,
// how long it was said and when; ok false when it left nothing (failed,
// cancelled, nothing heard).
func shareSentence(snapshot session.Snapshot) (text string, audio time.Duration, said time.Time, ok bool) {
	if snapshot.State != session.Done || snapshot.Result == "" {
		return "", 0, time.Time{}, false
	}
	start := snapshot.HeardAt
	if start.IsZero() {
		start = snapshot.StartedAt
	}
	if !snapshot.StoppedAt.IsZero() && snapshot.StoppedAt.After(start) {
		audio = snapshot.StoppedAt.Sub(start)
	}
	said = snapshot.StoppedAt
	if said.IsZero() {
		said = time.Now()
	}
	return snapshot.Result, audio, said, true
}
