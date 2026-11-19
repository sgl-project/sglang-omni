package main

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
)

type fakeBoxDropWindow struct {
	fakeBoxWindow
	onDrop func(*mygo.FileDropEvent)
}

func (f *fakeBoxDropWindow) OnFileDrop(fn func(*mygo.FileDropEvent)) func() {
	f.onDrop = fn
	return func() { f.onDrop = nil }
}

func TestRewriteBoxWindowAcceptsFileDrops(t *testing.T) {
	native := &fakeBoxDropWindow{}
	previous := newBoxNativeWindow
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	defer func() { newBoxNativeWindow = previous }()

	var dropped []string
	newBoxWindow(
		func() rewriteBoxViewState { return rewriteBoxViewState{} },
		rewriteBoxActions{drop: func(paths []string) { dropped = paths }},
		&rewriteBoxLocalUI{},
	)
	if native.onDrop == nil {
		t.Fatal("the panel does not accept file drops")
	}
	want := []string{"/tmp/picture.png"}
	native.onDrop(&mygo.FileDropEvent{Paths: want})
	if !slices.Equal(dropped, want) {
		t.Fatalf("dropped %v, want %v", dropped, want)
	}
}

func TestRewriteBoxHeldImageKeepsItsCapacity(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	gen := a.boxGen
	first := a.boxReserve()
	second := a.boxReserve()
	a.boxConverted(gen, second, boxJPEG(t), "")

	dir := t.TempDir()
	var files []string
	for _, name := range []string{"first.png", "second.png"} {
		name = filepath.Join(dir, name)
		if err := os.WriteFile(name, boxPNG(t), 0o600); err != nil {
			t.Fatal(err)
		}
		files = append(files, name)
	}
	files = append(files, filepath.Join(dir, "unreadable-excess.png"))
	reserved := a.boxSeq
	a.boxDrop(files)
	if got := a.boxSeq - reserved; got != 2 {
		t.Fatalf("reserved %d more images while one was held, want 2", got)
	}
	a.boxConverted(gen, first, boxJPEG(t), "")
	if len(a.box.images) != boxMaxImages || a.box.err != "" || !a.box.limit {
		t.Fatalf("images %d, error %q, limit %v", len(a.box.images), a.box.err, a.box.limit)
	}
}

func TestRewriteBoxDropRejectsExcessBeforeConversion(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, images: [][]byte{boxJPEG(t), boxJPEG(t), boxJPEG(t)}}
	dir := t.TempDir()
	valid := filepath.Join(dir, "valid.png")
	if err := os.WriteFile(valid, boxPNG(t), 0o600); err != nil {
		t.Fatal(err)
	}
	a.boxDrop([]string{valid, filepath.Join(dir, "missing.png")})
	if len(a.box.images) != boxMaxImages {
		t.Fatalf("%d images after the drop", len(a.box.images))
	}
	if !a.box.limit || a.box.err != "" {
		t.Fatalf("the excess file was converted: limit %v, error %q", a.box.limit, a.box.err)
	}
}

func TestRewriteBoxPasteRejectsExcessBeforeConversion(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, images: [][]byte{boxJPEG(t), boxJPEG(t), boxJPEG(t), boxJPEG(t)}}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.SetClipboardData(transfer.New(transfer.NewItem(transfer.Lazy(transfer.PNG, func() ([]byte, error) {
		return nil, errors.New("the excess image was read")
	}))))
	tt.Command("paste")
	frames(tt)
	if !a.box.limit || a.box.err != "" || a.boxPending != 0 {
		t.Fatalf("the excess image was converted: limit %v, error %q, pending %d", a.box.limit, a.box.err, a.boxPending)
	}
}

func TestRewriteBoxHTMLAndPNGClipboardPastesImage(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.SetClipboardData(transfer.New(transfer.NewItem(
		transfer.Bytes(transfer.HTML, []byte("<img src=\"picture.png\">")),
		transfer.Bytes(transfer.PNG, boxPNG(t)),
	)))
	tt.Command("paste")
	frames(tt)
	if len(a.box.images) != 1 {
		t.Fatalf("%d images after an HTML+PNG paste, want 1", len(a.box.images))
	}
}

// TestRewriteBoxOversizedClipboardImageIsRefused: a pasted picture past
// the converter's limit is refused when read, before any conversion.
func TestRewriteBoxOversizedClipboardImageIsRefused(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.SetClipboardData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, make([]byte, imgconv.MaxFileBytes+1)))))
	tt.Command("paste")
	frames(tt)
	if len(a.box.images) != 0 || a.boxSeq != 0 || !has(tt, "This image is too large.") {
		t.Fatalf("oversized paste: %d images, %d conversions, texts %q", len(a.box.images), a.boxSeq, firstTexts(tt))
	}
}

func TestRewriteBoxClipboardImageFailureShowsError(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.SetClipboardData(transfer.New(transfer.NewItem(transfer.Lazy(transfer.PNG, func() ([]byte, error) {
		return nil, errors.New("clipboard read failed")
	}))))
	tt.Command("paste")
	frames(tt)
	if a.box.err == "" || !has(tt, "This image could not be read.") {
		t.Fatalf("clipboard failure is silent: box error %q, texts %q", a.box.err, firstTexts(tt))
	}
}
