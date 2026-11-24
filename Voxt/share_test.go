package main

import (
	"bytes"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
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

// TestShareCardShortcutsOnHome: a shortcut pressed with the card open on
// Home records into the card, and closing the card cancels that try. The
// card shows only on Home, so a card left open takes no shortcut on
// another page.
func TestShareCardShortcutsOnHome(t *testing.T) {
	a, tt := pageTester(t, "/home")
	readyModels(a)
	drain := queuePosts(a)
	a.openShare()
	frames(tt)
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: selfFront()})
	drain()
	frames(tt)
	if a.sess.ID == "" || !a.sess.Try || a.share.tryID != a.sess.ID {
		t.Fatalf("the shortcut did not start the card's try: %+v, the card's try %q", a.sess, a.share.tryID)
	}
	a.closeShare()
	drain()
	frames(tt)
	if a.sess.State.Live() {
		t.Errorf("closing the card left its try %s", a.sess.State)
	}
	a.openShare()
	a.navigate("/history")
	if got := a.tryPlace(store.KindRewrite, *selfFront()); got == placeShare {
		t.Error("the card, not shown off Home, takes the Rewrite shortcut")
	}
	a.navigate("/rewrite")
	if got := a.tryPlace(store.KindRewrite, *selfFront()); got != placeFeature {
		t.Errorf("Rewrite's page with the card left open: %v, want its own try", got)
	}
	a.navigate("/home")
	if got := a.tryPlace(store.KindTranscribe, *selfFront()); got != placeShare {
		t.Errorf("back on Home: %v, want the card", got)
	}
}

// TestShareWardrobeKeepsSixty: putting on a drawn hat past
// shareWardrobeMax drops the oldest, on disk and in the wardrobe.
func TestShareWardrobeKeepsSixty(t *testing.T) {
	a, _, _ := newTestApp(t)
	dir := a.hatsDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := range shareWardrobeMax {
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%019d.png", i+1)), []byte("old"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	a.openShare()
	a.openStudio()
	a.studioPoint(150, -100, true)
	a.studioWear()
	got := readWardrobe(dir)
	if len(got) != shareWardrobeMax || got[0] != a.share.hat || slices.Contains(got, fmt.Sprintf("%019d.png", 1)) {
		t.Errorf("wardrobe on disk after a 61st hat: %d hats, first %q", len(got), got[0])
	}
	if !slices.Equal(a.share.wardrobe, got) {
		t.Errorf("the wardrobe shown differs from the one on disk: %d vs %d hats", len(a.share.wardrobe), len(got))
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
	if b := img.Bounds(); b.Dx() != 3*shareCardW || b.Dy() != 3*shareCardH {
		t.Errorf("copied %v, want the card at 3×", b)
	}
	// Square and opaque to its corners: chat apps lay transparent corners
	// on white.
	if r, g, b, alpha := img.At(0, 0).RGBA(); alpha>>8 != 255 || (r>>8 > 250 && g>>8 > 250 && b>>8 > 250) {
		t.Errorf("the picture's corner is %d,%d,%d alpha %d, want the card's background", r>>8, g>>8, b>>8, alpha>>8)
	}
	if _, _, _, alpha := img.At(3*shareCardW/2, 3*shareCardH/2).RGBA(); alpha>>8 != 255 {
		t.Errorf("the card's middle has alpha %d, want opaque", alpha>>8)
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
	// The hat is a transparent picture: nothing around the stroke, the
	// stroke's color where it was drawn.
	data, err := os.ReadFile(filepath.Join(a.hatsDir(), a.share.hat))
	if err != nil {
		t.Fatal(err)
	}
	hat, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, alpha := hat.At(4, 4).RGBA(); alpha != 0 {
		t.Errorf("the hat's corner is not transparent (alpha %d)", alpha>>8)
	}
	// The stroke ran from (150, -100) to (250, -100): its middle, at 2×.
	r, g, b, alpha := hat.At(int((200-shareHatBox[0])*2), int((-100-shareHatBox[1])*2)).RGBA()
	if alpha>>8 < 250 || r>>8 < 180 || g>>8 > 90 || b>>8 > 100 {
		t.Errorf("the stroke's middle is %d,%d,%d alpha %d, want the opaque red pen", r>>8, g>>8, b>>8, alpha>>8)
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
	insights := &a.share.insights
	control("Next card")
	control("Card 4 of 6")
	control("Previous card")
	if insights.card != 2 || insights.slide != -1 {
		t.Errorf("Next, the fourth dot and Previous show card %d from side %d, want the third from the left", insights.card, insights.slide)
	}
	tt.Key(0, ui.KeyRight)
	frames(tt)
	tt.Key(0, ui.KeyLeft)
	frames(tt)
	tt.Key(0, ui.KeyLeft)
	frames(tt)
	if insights.card != 1 {
		t.Errorf("→ ← ← from the third card show card %d, want the second", insights.card)
	}
	control("Next card")
	theme := insights.themes[2]
	control("Another color")
	if insights.themes[2] == theme || insights.themes[3] != 3 {
		t.Errorf("Another color left the themes %v", insights.themes)
	}
	control("Copy picture")
	if len(copied) != 2 {
		t.Errorf("the insights card's Copy copied %d pictures in all", len(copied))
	}
	saved := filepath.Join(t.TempDir(), "streak.png")
	defer func(old func(*app, string, func(string))) { chooseInsightsFile = old }(chooseInsightsFile)
	chooseInsightsFile = func(_ *app, name string, then func(string)) {
		if want := "Voxt-streak-" + activity.DayKey(a.now()) + ".png"; name != want {
			t.Errorf("Save offers the name %q, want %q", name, want)
		}
		then(saved)
	}
	control("Save picture")
	waitFor(t, func() { frames(tt) }, func() bool { _, err := os.Stat(saved); return err == nil })
	if f, err := os.Open(saved); err != nil {
		t.Errorf("Save wrote no picture: %v", err)
	} else {
		config, err := png.DecodeConfig(f)
		_ = f.Close() // read only
		if err != nil || config.Width != 3*insightsCardW || config.Height != 3*insightsCardH {
			t.Errorf("the saved picture is %d×%d (%v), want the card at 3×", config.Width, config.Height, err)
		}
	}
	var opened []string
	defer func(old func(string)) { openInsightsLink = old }(openInsightsLink)
	openInsightsLink = func(address string) { opened = append(opened, address) }
	control("Share to LinkedIn")
	if len(copied) != 3 || !slices.Equal(opened, []string{insightsLinkedInPost}) {
		t.Errorf("Share to LinkedIn copied %d pictures in all and opened %q", len(copied), opened)
	}
	tt.ClickAt(4, 4)
	frames(tt)
	if insights.open {
		t.Error("a click outside left the insights cards open")
	}
}

// TestInsightsFaces builds each card from a log: its number, what it
// says beside it, and what its entrance grows to.
func TestInsightsFaces(t *testing.T) {
	now := time.Date(2027, 2, 6, 18, 0, 0, 0, time.Local) // a Saturday
	var log activity.Log
	for _, back := range []int{0, 1, 2, 5, 6, 7, 8} { // runs of 3 and 4
		log.Add(now.AddDate(0, 0, -back), activity.Result{Chars: 400, Spoken: 400, Audio: 2 * time.Minute})
	}
	face := func(kind string) insightsFaceViewState {
		return buildInsightsFace(i18n.CurrentTranslator(), log, now, kind, 0)
	}
	if f := face(insightsCardHighlights); len(f.Stats) != 4 || len(f.Days) != 3 || len(f.Weeks) != insightsWeeks || f.Date != "Feb 6, 2027" {
		t.Errorf("the overview has %d totals, %d day counts and %d weeks, dated %q", len(f.Stats), len(f.Days), len(f.Weeks), f.Date)
	}
	// 2,800 characters are five pages of 500.
	if f := face(insightsCardCharacters); f.Number != 2800 || f.Line != "About 5 pages of a book" ||
		!slices.Equal(f.Notes, []string{"7 dictations", "7 active days"}) {
		t.Errorf("the characters card says %d, %q, %q", f.Number, f.Line, f.Notes)
	}
	week := ""
	streak := face(insightsCardStreak)
	for _, day := range streak.Week {
		mark := "-"
		if day.Active {
			mark = "+"
		}
		week += day.Label + mark
	}
	if streak.Number != 3 || week != "S+M+T-W-T+F+S+" || !streak.Week[6].Today || streak.Move != 2 {
		t.Errorf("the streak card says %d days over %q, its mascot's move %d", streak.Number, week, streak.Move)
	}
	// 200 characters a minute is five typing speeds, of the dial's eight.
	if f := face(insightsCardSpeed); f.Number != 200 || f.Times != "5×" || f.Dial != 5.0/insightsDialTimes {
		t.Errorf("the speed card says %d, %q, the dial at %v", f.Number, f.Times, f.Dial)
	}
	// Typing 2,800 characters takes 70 minutes; saying them took 14.
	saved := face(insightsCardSaved)
	if len(saved.Bars) != 2 || saved.Bars[0].Value != "1 hour 10 minutes" || saved.Bars[1].Value != "14 minutes" || saved.Bars[1].Fraction != .2 {
		t.Errorf("the saved card's bars are %+v", saved.Bars)
	}
	if f := face(insightsCardDays); len(f.Weeks) != insightsWeeks || len(f.Stats) != 2 || f.Stats[1].Parts[0].Value != "4" {
		t.Errorf("the days card has %d weeks and %+v", len(f.Weeks), f.Stats)
	}
	// Nothing yet: a dash for the speed, an empty dial, no bar.
	if f := buildInsightsFace(i18n.CurrentTranslator(), activity.Log{}, now, insightsCardSpeed, 0); f.Known || f.Times != "—" || f.Dial != 0 {
		t.Errorf("the speed card of an empty log says %+v", f)
	}
	if f := buildInsightsFace(i18n.CurrentTranslator(), activity.Log{}, now, insightsCardCharacters, 0); f.Line != "The first page is under way" {
		t.Errorf("the characters card of an empty log says %q", f.Line)
	}
	if f := buildInsightsFace(i18n.CurrentTranslator(), activity.Log{}, now, insightsCardSaved, 0); f.Bars[0].Fraction != 0 || f.Bars[1].Fraction != 0 {
		t.Errorf("the saved card of an empty log has bars %+v", f.Bars)
	}
	for at, want := range map[time.Duration]string{0: "0", 650 * time.Millisecond: "2,450", insightsSettled: "2,800"} {
		if got := insightsCount(2800, insightsProgress(at, 150*time.Millisecond, time.Second)); got != want {
			t.Errorf("the count toward 2,800 shows %q after %v, want %q", got, at, want)
		}
	}
}

// TestInsightsEntrance plays a card's entrance: the speed's dial is
// empty when the card comes up and filled once it has settled, a turn of
// the cards starts the entrance over, and Reduce motion shows the card
// settled at once.
func TestInsightsEntrance(t *testing.T) {
	now := time.Date(2027, 2, 6, 18, 0, 0, 0, time.Local)
	var log activity.Log
	log.Add(now, activity.Result{Chars: 400, Spoken: 400, Audio: 2 * time.Minute})
	face := buildInsightsFace(i18n.CurrentTranslator(), log, now, insightsCardSpeed, 3)
	// The dial's part of the card: only its fill has a color there, on
	// the white tile under the grey track and the dark text.
	filled := func(elapsed time.Duration) int {
		img := ui.Render(func(c *ui.Context) { insightsCardFace(c, &face, elapsed, false) }, insightsCardW, insightsCardH, 1)
		n := 0
		for y := 100; y < 270; y++ {
			for x := 330; x < 550; x++ {
				c := img.RGBAAt(x, y)
				if int(max(c.R, c.G, c.B))-int(min(c.R, c.G, c.B)) > 60 {
					n++
				}
			}
		}
		return n
	}
	if n := filled(0); n != 0 {
		t.Errorf("the dial has %d filled pixels as the card comes up", n)
	}
	if n := filled(insightsSettled); n < 500 {
		t.Errorf("the dial has %d filled pixels once the card settled", n)
	}

	a, _, _ := newTestApp(t)
	a.now = func() time.Time { return now }
	a.openInsightsShare()
	now = now.Add(time.Minute)
	a.showInsightsShare(3) // the speed
	if state := a.insightsModel(now, false); state.Elapsed != 0 || state.Face.Kind != insightsCardSpeed || state.Slide != 1 {
		t.Errorf("the speed card comes up as %+v", state)
	}
	if state := a.insightsModel(now, true); state.Elapsed != insightsSettled {
		t.Errorf("Reduce motion shows the card %v into its entrance", state.Elapsed)
	}
}
