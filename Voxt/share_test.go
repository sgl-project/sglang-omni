package main

import (
	"bytes"
	"image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/activity"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestShareCardPhases(t *testing.T) {
	tr := i18n.CurrentTranslator()
	now := time.Date(2027, 2, 6, 10, 0, 11, 0, time.Local)
	empty := buildShareCardViewState(tr, shareCardInput{Background: 12})
	if empty.Phase != sharePhaseEmpty || empty.Background != 2 || empty.Dark {
		t.Errorf("empty card: %+v", empty)
	}
	rec := session.Snapshot{ID: "s", State: session.Recording, StartedAt: now.Add(-3 * time.Second), Levels: []float64{.2, .5}}
	listening := buildShareCardViewState(tr, shareCardInput{Background: 6, Mine: true, Snapshot: rec, Now: now})
	if listening.Phase != sharePhaseListening || !listening.Dark || len(listening.Levels) != 2 || listening.Elapsed == "" {
		t.Errorf("listening card: %+v", listening)
	}
	// Another session's recording leaves the card as it is.
	if other := buildShareCardViewState(tr, shareCardInput{Text: "hi", Snapshot: rec, Now: now}); other.Phase != sharePhaseSaid {
		t.Errorf("another session's recording took the card: %+v", other)
	}
	said := buildShareCardViewState(tr, shareCardInput{Text: "hi", Audio: 11 * time.Second, Said: now})
	if said.Phase != sharePhaseSaid || said.Meta != "0:11 · Feb 6, 2027" {
		t.Errorf("said card: %+v", said)
	}
}

func TestShareSentenceFromTry(t *testing.T) {
	start := time.Date(2027, 2, 6, 10, 0, 0, 0, time.Local)
	done := session.Snapshot{ID: "s", State: session.Done, Result: "hello", StartedAt: start, HeardAt: start.Add(time.Second), StoppedAt: start.Add(8 * time.Second)}
	text, audio, said, ok := shareSentence(done)
	if !ok || text != "hello" || audio != 7*time.Second || !said.Equal(done.StoppedAt) {
		t.Errorf("sentence: %q %v %v %v", text, audio, said, ok)
	}
	for _, s := range []session.State{session.Failed, session.Cancelled} {
		if _, _, _, ok := shareSentence(session.Snapshot{State: s, Result: "x"}); ok {
			t.Errorf("a %s try left a sentence", s)
		}
	}
}

func TestShareCardTakesItsTry(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.openShare()
	if !a.share.open || (a.share.outfit == "" && a.share.hat == "") {
		t.Fatalf("opened without a hat: %+v", a.share)
	}
	if got := a.tryPlace(store.KindTranslate, proto.App{PID: os.Getpid()}); got != placeShare {
		t.Errorf("a shortcut with the card open runs at %v, want the card", got)
	}
	a.openStudio()
	if got := a.tryPlace(store.KindTranscribe, proto.App{PID: os.Getpid()}); got == placeShare {
		t.Error("a shortcut while drawing a hat fills the card")
	}
	a.studioCancel()
	now := time.Now()
	a.shareTryStarted("try-1", nil)
	a.sess = session.Snapshot{ID: "try-1", State: session.Done, Result: "said to the card", StartedAt: now.Add(-5 * time.Second), StoppedAt: now}
	if m := a.shareModel(now); m.Phase != sharePhaseSaid || m.Text != "said to the card" {
		t.Errorf("the card's try left %+v", m)
	}
	// A later session of someone else's leaves the sentence.
	a.sess = session.Snapshot{ID: "other", State: session.Done, Result: "typed elsewhere"}
	if m := a.shareModel(now); m.Text != "said to the card" {
		t.Errorf("another session replaced the sentence: %q", m.Text)
	}
	a.closeShare()
	if a.share.open || a.tryPlace(store.KindTranscribe, proto.App{PID: os.Getpid()}) == placeShare {
		t.Error("the card still takes shortcuts once closed")
	}
}

func TestShareShuffleAndCopy(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.openShare()
	bg, outfit, dressed := a.share.background, a.share.outfit, a.share.dressed
	a.shuffleShare()
	if a.share.background == bg || (a.share.outfit == outfit && a.share.hat == "") || a.share.dressed == dressed {
		t.Errorf("shuffle kept the background or hat: %d→%d, %q→%q/%q", bg, a.share.background, outfit, a.share.outfit, a.share.hat)
	}
	var copied []byte
	defer func(old func([]byte) error) { shareClipboard = old }(shareClipboard)
	shareClipboard = func(b []byte) error { copied = b; return nil }
	a.share.text, a.share.audio, a.share.said = "hello", 3*time.Second, time.Now()
	a.copyShare()
	img, err := png.Decode(bytes.NewReader(copied))
	if err != nil {
		t.Fatalf("the copied picture: %v", err)
	}
	if b := img.Bounds(); b.Dx() != 2*shareCardW || b.Dy() != 2*shareCardH {
		t.Errorf("copied %v, want the card at 2×", b)
	}
	if !a.shareModel(a.now()).Copied {
		t.Error("Copy shows no tick")
	}
}

func TestShareStudioDrawsUndoesAndWears(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.openShare()
	a.openStudio()
	a.studioPoint(100, -50, true)
	a.studioPoint(200, -150, false)
	a.studioPoint(300, -50, false)
	a.studioEnd()
	st := a.share.studio
	if len(st.strokes) != 1 || len(st.strokes[0].Points) != 4 {
		t.Fatalf("a stroke of three points: %+v", st.strokes)
	}
	a.studioEraser()
	a.studioPoint(200, -150, true)
	a.studioEnd()
	if len(st.strokes) != 2 {
		t.Errorf("erasing the middle split the stroke into %d", len(st.strokes))
	}
	a.studioPoint(360, 50, true) // nothing there: no undo step
	a.studioEnd()
	a.studioUndo()
	if len(st.strokes) != 1 {
		t.Errorf("undo left %d strokes, want the one before erasing", len(st.strokes))
	}
	a.studioLoad("outfit:" + outfitChristmas)
	a.studioWear()
	if a.share.outfit != outfitChristmas || a.share.hat != "" || a.share.studio != nil {
		t.Errorf("a wardrobe outfit put on as itself: %q %q", a.share.outfit, a.share.hat)
	}
	a.openStudio()
	a.studioPoint(150, -100, true)
	a.studioPoint(250, -100, false)
	a.studioWear()
	if a.share.hat == "" || a.share.outfit != "" {
		t.Fatalf("a drawn hat was not put on: %+v", a.share)
	}
	if _, err := os.Stat(filepath.Join(a.hatsDir(), a.share.hat)); err != nil {
		t.Errorf("the drawn hat was not saved: %v", err)
	}
	if got := readWardrobe(a.hatsDir()); len(got) != 1 || got[0] != a.share.hat {
		t.Errorf("wardrobe %v", got)
	}
	if a.shareBitmap("hat:"+a.share.hat) == nil {
		t.Error("the saved hat does not decode")
	}
}

func TestInsightsNumbers(t *testing.T) {
	now := time.Date(2027, 2, 6, 18, 0, 0, 0, time.Local)
	var log activity.Log
	for _, back := range []int{0, 1, 2, 5, 6, 7, 8} { // runs of 3 and 4
		log.Add(now.AddDate(0, 0, -back), activity.Result{Chars: 400, Spoken: 400, Audio: 2 * time.Minute})
	}
	ins := buildInsightsViewState(i18n.CurrentTranslator(), log, now)
	parts := func(s insightsStatViewState) string {
		out := ""
		for _, p := range s.Parts {
			out += p.Value + " " + p.Unit + " "
		}
		return out
	}
	// 2,800 characters typed at 40 a minute is 70 minutes; 14 were spoken.
	for i, want := range []string{"2,800 characters ", "56 minutes ", "200 characters a minute ", "14 minutes "} {
		if got := parts(ins.Stats[i]); got != want {
			t.Errorf("stat %d (%s) = %q, want %q", i, ins.Stats[i].Label, got, want)
		}
	}
	for i, want := range []string{"7 days ", "3 days ", "4 days "} {
		if got := parts(ins.Days[i]); got != want {
			t.Errorf("day stat %d (%s) = %q, want %q", i, ins.Days[i].Label, got, want)
		}
	}
	if len(ins.Weeks) != insightsWeeks || len(ins.Months) != insightsWeeks {
		t.Errorf("%d weeks, %d month labels", len(ins.Weeks), len(ins.Months))
	}
	if got := insightsDuration(i18n.CurrentTranslator(), 154*time.Minute); len(got) != 2 || got[0].Value != "2" || got[1].Value != "34" {
		t.Errorf("2 h 34 min shows as %+v", got)
	}
}

// TestShareClickThrough clicks the cards' controls as a user does: Home's
// Share, the corner buttons, the prompt, the studio's board and tools,
// the wardrobe, Put on, then the insights card's Copy and a click
// outside each card.
func TestShareClickThrough(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(1100, 720)
	a.navigate("/home")
	frames(tt)
	var copied [][]byte
	defer func(old func([]byte) error) { shareClipboard = old }(shareClipboard)
	shareClipboard = func(b []byte) error { copied = append(copied, b); return nil }
	control := func(label string) {
		t.Helper()
		frames(tt)
		if err := tt.ClickControl(label); err != nil {
			t.Fatalf("click %q: %v", label, err)
		}
		frames(tt)
	}
	control("Share a card of Voxt and you")
	if !a.share.open {
		t.Fatal("Share did not open the card")
	}
	bg := a.share.background
	control("Another background and hat")
	if a.share.background == bg {
		t.Error("the shuffle kept the background")
	}
	control("Copy picture")
	if len(copied) != 1 {
		t.Fatalf("Copy copied %d pictures", len(copied))
	}
	control("What would you like to say?")
	if a.share.tryID == "" && a.share.block == nil {
		t.Error("the prompt neither started a try nor said why not")
	}
	control("Draw a hat")
	if a.share.studio == nil {
		t.Fatal("Draw a hat did not open the studio")
	}
	r, ok := tt.Find("My Voxt!")
	if !ok {
		t.Fatal("no board")
	}
	x, y := r.X+r.W/2, r.Y+r.H+40
	tt.Press(x, y)
	tt.Move(x+40, y+60)
	tt.Move(x+80, y+20)
	tt.Release(x+80, y+20)
	frames(tt)
	if st := a.share.studio; len(st.strokes) != 1 || len(st.strokes[0].Points) < 3 {
		t.Fatalf("dragging on the board drew %+v", a.share.studio.strokes)
	}
	control("See it on Voxt")
	control("Back to drawing")
	if a.share.studio.peek {
		t.Error("the preview did not go back to drawing")
	}
	control("Wardrobe")
	control("Wear this hat")
	if st := a.share.studio; st.closet || st.base != "outfit:"+shareOutfits[0] || len(st.strokes) != 0 {
		t.Errorf("the first wardrobe hat is not on the board: %+v", st)
	}
	control("Put on")
	if a.share.studio != nil || a.share.outfit != shareOutfits[0] {
		t.Errorf("Put on left the studio %v, outfit %q", a.share.studio != nil, a.share.outfit)
	}
	tt.ClickAt(4, 4) // outside the card
	frames(tt)
	if a.share.open {
		t.Error("a click outside left the card open")
	}
	a.openInsightsShare()
	control("Copy picture")
	if len(copied) != 2 {
		t.Errorf("the insights card's Copy copied %d pictures in all", len(copied))
	}
	tt.ClickAt(4, 4)
	frames(tt)
	if a.share.insights.open {
		t.Error("a click outside left the insights card open")
	}
}
