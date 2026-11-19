//go:build darwin

package macx

import (
	"bytes"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func TestReadPDF(t *testing.T) {
	text, pages, pageCount, err := ReadPDF("../../testdata/rewrite-box/text.pdf", 4)
	if err != nil {
		t.Fatal(err)
	}
	if pageCount != 2 || len(pages) != 0 || !strings.Contains(text, "First line") || !strings.Contains(text, "Sixth line") {
		t.Fatalf("text PDF: pageCount %d, %d rendered pages, text %q", pageCount, len(pages), text)
	}

	text, pages, pageCount, err = ReadPDF("../../testdata/rewrite-box/scanned.pdf", 4)
	if err != nil {
		t.Fatal(err)
	}
	if text != "" || pageCount != 6 || len(pages) != 4 {
		t.Fatalf("scanned PDF: text %q, pageCount %d, %d rendered pages", text, pageCount, len(pages))
	}
	var previous color.RGBA
	for index, page := range pages {
		image, err := png.Decode(bytes.NewReader(page))
		if err != nil {
			t.Fatalf("page %d is not a PNG: %v", index+1, err)
		}
		bounds := image.Bounds()
		if bounds.Dx() == 0 || bounds.Dy() == 0 || bounds.Dx() > 1600 || bounds.Dy() > 1600 {
			t.Fatalf("page %d dimensions %dx%d", index+1, bounds.Dx(), bounds.Dy())
		}
		center := color.RGBAModel.Convert(image.At(bounds.Dx()/2, bounds.Dy()/2)).(color.RGBA)
		if center.A != 255 {
			t.Fatalf("page %d is blank: %+v", index+1, center)
		}
		if index > 0 && (center.R <= previous.R || center.G <= previous.G || center.B <= previous.B) {
			t.Fatalf("pages are out of order: page %d is %+v after %+v", index+1, center, previous)
		}
		previous = center
	}
}
