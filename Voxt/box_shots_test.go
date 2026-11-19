package main

// The input box's snapshots (snapshot_test.go, TestOverlayShots), and
// the small bitmap the picture rows' shots use.

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// shotBitmap is a small bitmap for the card's and the box's picture
// rows.
func shotBitmap(t *testing.T) *ui.Bitmap {
	t.Helper()
	var out bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := range 16 {
		for x := range 16 {
			img.Set(x, y, color.RGBA{200, 40 + uint8(x*8), 40 + uint8(y*8), 255})
		}
	}
	if err := jpeg.Encode(&out, img, nil); err != nil {
		t.Fatal(err)
	}
	bmp, err := ui.DecodeBitmap(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return bmp
}

// boxShots renders the input box (rewrite_box_view.go) in its states:
// recording with the selected text in the card, typed text and two
// pictures with the instruction being heard, and at the pictures'
// limit with an error kept for the resend.
func boxShots(t *testing.T, a *app, out, suffix string, dark bool) {
	t.Helper()
	desk := ui.RGB(246, 246, 248)
	if dark {
		desk = ui.RGB(30, 31, 36)
	}
	states := []struct {
		name  string
		setup func()
	}{
		{"box-recording-selection", func() {
			a.box.session = "shot-box"
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording,
				Selection: "hey, the release slips to thursday, sorry!"})
		}},
		{"box-typed-images", func() {
			a.box.session = "shot-box"
			a.box.text = "and the tickets too"
			a.box.thumbs = []*ui.Bitmap{shotBitmap(t), shotBitmap(t)}
			a.box.images = [][]byte{{1}, {2}}
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording,
				Selection: "hey, the release slips to thursday, sorry!", Partial: "make it more formal"})
		}},
		{"box-pasted-collapsed", func() {
			a.box.session = "shot-box"
			a.boxPasteText("first line\nsecond line\nthird line\nfourth line")
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording, Partial: "make it more formal"})
		}},
		{"box-pasted-expanded", func() {
			a.box.session = "shot-box"
			a.boxPasteText("first line\nsecond line\nthird line\nfourth line")
			a.box.ui.expandedText = map[string]bool{"segment-0": true}
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording, Partial: "make it more formal"})
		}},
		{"box-image-caption", func() {
			a.box.session = "shot-box"
			a.boxAttach(boxJPEGOf(t, 32, 20), "clipboard")
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording, Partial: "make it more formal"})
		}},
		{"box-image-preview", func() {
			a.box.session = "shot-box"
			a.boxAttach(boxJPEGOf(t, 32, 20), "clipboard")
			a.box.ui.previewImage = 1
			snap(a, session.Snapshot{ID: "shot-box", Kind: store.KindRewrite, State: session.Recording, Partial: "make it more formal"})
		}},
		{"box-limit-error", func() {
			a.box.thumbs = []*ui.Bitmap{shotBitmap(t), shotBitmap(t), shotBitmap(t), shotBitmap(t)}
			a.box.images = [][]byte{{1}, {2}, {3}, {4}}
			a.box.instruction = "make it more formal"
			a.box.selection = "hey, the release slips to thursday, sorry!"
			a.box.limit = true
			a.box.err = "The language model failed: bad request (400)"
		}},
	}
	for _, st := range states {
		a.box = rewriteBox{open: true, shown: true}
		if st.setup != nil {
			st.setup()
		}
		tt := hudTester(func(c *ui.Context) {
			ui.Box(c).Absolute().Fill().Background(desk)
			ui.Box(c).Absolute().Fill().Children(func() { a.boxView(c) })
		}, boxWinW, boxWinH)
		tt.SetDark(true)
		tt.SetScale(2)
		frames(tt)
		f, err := os.Create(filepath.Join(out, "overlay-"+st.name+"-"+suffix+".png"))
		if err != nil {
			t.Fatal(err)
		}
		png.Encode(f, tt.Image())
		f.Close()
	}
	a.box = rewriteBox{}
}
