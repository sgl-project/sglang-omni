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
// empty with the voice hint, with pictures and the instruction ready
// (Send), and at the pictures' limit with an error kept.
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
		{"box-open", nil},
		{"box-images-instruction", func() {
			a.box.text = "the release slips to thursday, sorry!"
			a.box.thumbs = []*ui.Bitmap{shotBitmap(t), shotBitmap(t)}
			a.box.images = [][]byte{{1}, {2}}
			a.box.instruction = "make it more formal"
		}},
		{"box-limit-error", func() {
			a.box.thumbs = []*ui.Bitmap{shotBitmap(t), shotBitmap(t), shotBitmap(t), shotBitmap(t)}
			a.box.images = [][]byte{{1}, {2}, {3}, {4}}
			a.box.instruction = "make it more formal"
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
