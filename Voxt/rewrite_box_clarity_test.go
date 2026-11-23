package main

import (
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const rewriteBoxTipText = "Speak or type what to do. Paste or drop images to go with it."

func TestRewriteBoxSubtitle(t *testing.T) {
	for _, test := range []struct {
		name      string
		selection string
		want      string
	}{
		{name: "selected text", selection: "source text", want: "Listening… say how to change the selected text"},
		{name: "no selection", want: "Listening… say what to write"},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.box = rewriteBox{open: true, shown: true, session: "v1"}
			snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: test.selection})
			tt := hudTester(a.boxView, boxWinW, boxWinH)
			frames(tt)
			if !hasExact(tt, test.want) {
				t.Fatalf("subtitle is missing from %q", firstTexts(tt))
			}
			if !has(tt, "Input level") {
				t.Fatalf("the recording level is missing from %q", firstTexts(tt))
			}
		})
	}

	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, err: "send failed"}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if !hasExact(tt, "send failed") {
		t.Fatalf("the send error is not the subtitle: %q", firstTexts(tt))
	}
}

func TestRewriteBoxFooterAndPlaceholder(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1"}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording})
	if got := a.boxViewState().EditorPlaceholder; got != "Or type here" {
		t.Errorf("placeholder %q", got)
	}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	for _, want := range []string{"Enter", "send", "Esc", "close"} {
		if !hasExact(tt, want) {
			t.Errorf("%q is missing from %q", want, firstTexts(tt))
		}
	}
	for _, unwanted := range []string{"Type or paste more text, or drop images", "Say the instruction, then Enter sends", "paste or drop up to 4 images"} {
		if has(tt, unwanted) {
			t.Errorf("old hint %q remains in %q", unwanted, firstTexts(tt))
		}
	}
}

func TestRewriteBoxSurfaceMatchesAnswer(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Overlay.Radius = 23
	a.prefs.Overlay.Opacity = 0.71
	a.prefs.Appearance.Palette = "#b25280"
	a.ov.mode, a.ov.h = ovAnswer, boxWinH
	answer := a.currentSessionOverlayLayout()
	box := a.boxViewState()
	if box.Radius != answer.Radius || box.Alpha != answer.Alpha || box.Palette != answer.Palette {
		t.Fatalf("box surface (%v, %v, %q), answer surface (%v, %v, %q)", box.Radius, box.Alpha, box.Palette, answer.Radius, answer.Alpha, answer.Palette)
	}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	tt.SetDark(false)
	frames(tt)
	got := tt.Image().RGBAAt(boxWinW/2, layout.Pad+2)
	want := tok(overlayTheme(answer.Palette)).OverlayFace.Alpha(answer.Alpha)
	want.R = uint8(float32(want.R)*float32(want.A)/255 + 0.5)
	want.G = uint8(float32(want.G)*float32(want.A)/255 + 0.5)
	want.B = uint8(float32(want.B)*float32(want.A)/255 + 0.5)
	if got.R != want.R || got.G != want.G || got.B != want.B || got.A != want.A {
		t.Fatalf("box face is %+v, answer face is %+v", got, want)
	}

	native := &fakeBoxDropWindow{}
	var options mygo.WindowOptions
	previous := newBoxNativeWindow
	newBoxNativeWindow = func(got mygo.WindowOptions) boxDropWindow {
		options = got
		return native
	}
	defer func() { newBoxNativeWindow = previous }()
	newBoxWindow(a.boxViewState, a.boxActions(), &rewriteBoxLocalUI{}, floatingWindowOptions("Rewrite", boxWinW, boxWinH, nil))
	shared := floatingWindowOptions("Rewrite", boxWinW, boxWinH, nil)
	if options.Frameless != shared.Frameless || options.Transparent != shared.Transparent || options.DisableShadow != shared.DisableShadow || options.AlwaysOnTop != shared.AlwaysOnTop || options.SkipTaskbar != shared.SkipTaskbar {
		t.Fatalf("box options differ from the answer window: %+v", options)
	}
	// An ordinary window, as any app's: a click on it brings VoxtNew in
	// front, so a window manager such as Rectangle acts on it.
	if options.Vibrancy != mygo.VibrancyHUD || options.Panel || options.NonActivating || !options.DisableClose {
		t.Fatalf("box vibrancy or window options are wrong: %+v", options)
	}
}

func TestRewriteBoxSyncsAnswerVibrancy(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.ov.native = true
	a.prefs.Overlay.Radius = 23
	native := &fakeBoxDropWindow{}
	previous := newBoxNativeWindow
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	defer func() { newBoxNativeWindow = previous }()
	a.box = rewriteBox{open: true, shown: true, winShown: true} // already placed: no screen to ask in a test
	a.boxSync()
	want := mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(faceRadius(a.prefs.Overlay, ovAnswer, boxWinH)), Dark: true}
	if native.style != want {
		t.Fatalf("box vibrancy %+v, answer vibrancy %+v", native.style, want)
	}
}

func openRewriteBoxForTipTest(t *testing.T, dir string) *sessionApp {
	t.Helper()
	chat := &fakeChat{text: "unused"}
	set, models, _ := fileSetup(t, "make it shorter", chat)
	a := newSessionAppIn(t, dir, "", set, models)
	target := proto.App{PID: 42}
	a.postQueue(func() {
		a.openBox(target, session.StartRequest{Kind: store.KindRewrite, Target: target})
	})
	a.pump("the Rewrite box", func() bool { return a.box.open && a.sess.State == session.Recording })
	return a
}

func TestRewriteBoxTipFirstThreeOpensPersist(t *testing.T) {
	dir := t.TempDir()
	for _, test := range []struct {
		name string
		want bool
	}{
		{name: "first open", want: true},
		{name: "second open", want: true},
		{name: "third open", want: true},
		{name: "fourth open", want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			a := openRewriteBoxForTipTest(t, dir)
			tt := hudTester(a.boxView, boxWinW, boxWinH)
			if got := has(tt, rewriteBoxTipText); got != test.want {
				t.Fatalf("tip visible %v, want %v (texts %q)", got, test.want, firstTexts(tt))
			}
		})
	}
}

func TestRewriteBoxTipGotItPersists(t *testing.T) {
	dir := t.TempDir()
	t.Run("dismiss", func(t *testing.T) {
		a := openRewriteBoxForTipTest(t, dir)
		tt := hudTester(a.boxView, boxWinW, boxWinH)
		click(t, tt, "Got it")
		if has(tt, rewriteBoxTipText) {
			t.Fatal("Got it did not dismiss the tip")
		}
	})
	t.Run("after restart", func(t *testing.T) {
		a := openRewriteBoxForTipTest(t, dir)
		tt := hudTester(a.boxView, boxWinW, boxWinH)
		if has(tt, rewriteBoxTipText) {
			t.Fatal("the dismissed tip returned after a restart")
		}
	})
}

func TestAnswerCardDropsUnsentBoxImages(t *testing.T) {
	newBoxApp := func(t *testing.T) *app {
		a, _, _ := newTestApp(t)
		jpeg := boxJPEGOf(t, 20, 10)
		bitmap, _ := ui.DecodeBitmap(jpeg)
		a.box = rewriteBox{open: true, session: "r1", images: [][]byte{jpeg}, thumbs: []*boxBitmap{bitmap}}
		return a
	}
	assertNoBoxImage := func(t *testing.T, a *app) {
		t.Helper()
		if len(a.answer.images) != 0 || len(a.answer.ui.thumbs) != 0 {
			t.Fatalf("unsent images remained in the answer state: %d, %d", len(a.answer.images), len(a.answer.ui.thumbs))
		}
		for _, preview := range a.answer.previews {
			if preview.Image != nil {
				t.Fatalf("an unsent image remained in the previews: %+v", preview)
			}
		}
		if got := sessionCardAttachmentHeight(a.cachedCardRef()); got != 0 {
			t.Fatalf("the answer reserved %d pixels for an unsent image", got)
		}
		tt := cardTester(t, a)
		frames(tt)
		if has(tt, "Images sent with the instruction") {
			t.Fatal("the answer claims the model received an image it dropped")
		}
	}

	t.Run("answer", func(t *testing.T) {
		a := newBoxApp(t)
		snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
			Raw: "make it formal", Selection: "source text", Result: "Formal answer."})
		assertNoBoxImage(t, a)
	})

	t.Run("redo", func(t *testing.T) {
		a := newBoxApp(t)
		snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
			Raw: "make it formal", Selection: "source text", Result: "First answer.", ImageHint: "box"})
		a.endRedo(session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done,
			Raw: "make it formal", Selection: "source text", Result: "Second answer."})
		assertNoBoxImage(t, a)
	})

	t.Run("redo streaming", func(t *testing.T) {
		a := newBoxApp(t)
		snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
			Raw: "make it formal", Selection: "source text", Result: "First answer.", ImageHint: "box"})
		a.answer.redo = "r2"
		a.sess = session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Composing, Card: true, Draft: "Second answer.", Selection: "source text"}
		a.refreshSessionOverlay(time.Now())
		if card := a.cachedCard(); card.Attachment != "" || card.ThumbnailCount != 0 {
			t.Fatalf("the streaming redo kept %q and %d unsent images", card.Attachment, card.ThumbnailCount)
		}
		if len(a.ov.ui.cardThumbs) != 0 {
			t.Fatal("the streaming redo kept the box thumbnails")
		}
		for _, preview := range a.ov.ui.cardPreviews {
			if preview.Image != nil {
				t.Fatalf("the streaming redo kept an unsent image: %+v", preview)
			}
		}
	})

	t.Run("streaming", func(t *testing.T) {
		attachment, count := sessionCardAttachment(sessionCardProjectionInput{
			Snapshot: session.Snapshot{ID: "r1"}, Writing: true, BoxSessionID: "r1", BoxThumbnailCount: 1,
		})
		if attachment != "" || count != 0 {
			t.Fatalf("the streaming card kept %q and %d unsent images", attachment, count)
		}
	})
}
