package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
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
		floatingWindowOptions("Rewrite", boxWinW, boxWinH, nil),
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
	a, fs, _ := newTestApp(t)
	fs.handlers[proto.MethodClipboardImage] = func(json.RawMessage) (any, error) {
		return proto.ClipboardImageResult{JPEG: boxJPEG(t)}, nil
	}
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

func TestRewriteBoxOversizeClipboardDoesNotReadPNG(t *testing.T) {
	a, fs, _ := newTestApp(t)
	fs.handlers[proto.MethodClipboardImage] = func(json.RawMessage) (any, error) {
		return proto.ClipboardImageResult{JPEG: make([]byte, imgconv.MaxFileBytes+1)}, nil
	}
	a.box = rewriteBox{open: true, shown: true, focusText: true, images: [][]byte{boxJPEG(t)}}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	read := false
	tt.SetClipboardData(transfer.New(transfer.NewItem(transfer.Lazy(transfer.PNG, func() ([]byte, error) {
		read = true
		return nil, errors.New("the raw PNG was read")
	}))))
	tt.Command("paste")
	frames(tt)
	if read {
		t.Fatal("the oversized clipboard PNG was read before rejection")
	}
	if a.box.err != "This image is too large." || a.boxPending != 0 || len(a.box.images) != 1 {
		t.Fatalf("oversize clipboard image was not rejected or changed the existing input: error %q, pending %d, images %d", a.box.err, a.boxPending, len(a.box.images))
	}
}

func TestBoxTextTooLarge(t *testing.T) {
	if boxTextTooLarge("small") {
		t.Fatal("ordinary text was rejected")
	}
	if !boxTextTooLarge(strings.Repeat("x", (32<<20)+1)) {
		t.Fatal("oversize text was accepted")
	}
}

func TestRewriteBoxOversizeTextKeepsExistingInput(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, text: "keep me", images: [][]byte{boxJPEG(t)}}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.SetClipboard(strings.Repeat("x", (32<<20)+1))
	tt.Command("paste")
	frames(tt)
	if a.box.text != "keep me" || len(a.box.images) != 1 {
		t.Fatalf("oversize text replaced the existing input: text %q, images %d", a.box.text, len(a.box.images))
	}
	if a.box.err != "This text is too large." {
		t.Fatalf("oversize text error %q", a.box.err)
	}
}

func TestRewriteBoxClipboardImageFailureShowsError(t *testing.T) {
	a, fs, _ := newTestApp(t)
	fs.handlers[proto.MethodClipboardImage] = func(json.RawMessage) (any, error) {
		return nil, errors.New("clipboard read failed")
	}
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

// TestRewriteBoxPastesCopiedPictures: ⌘V puts a picture in the card from
// every clipboard the helper reads, not only a PNG. An image file copied
// in Finder is the picture although Finder puts its name there as text
// (it pasted the name), and a clipboard holding only a JPEG is a picture
// too (it pasted nothing). Another copied file pastes its name, and a
// spreadsheet range's text wins over the picture of itself it brings.
func TestRewriteBoxPastesCopiedPictures(t *testing.T) {
	finder := func(name string) transfer.Data {
		return transfer.New(transfer.NewItem(
			transfer.Bytes(transfer.FileList, []byte("file:///Users/me/Desktop/"+name)),
			transfer.Bytes(transfer.Text, []byte(name)),
			transfer.Bytes(transfer.PNG, boxPNG(t)), // the file's icon
		))
	}
	for _, c := range []struct {
		name      string
		clipboard transfer.Data
		image     bool
		text      string
	}{
		{"an image file copied in Finder", finder("photo.png"), true, ""},
		{"a HEIC file copied in Finder", finder("IMG_0042.HEIC"), true, ""},
		{"a JPEG alone", transfer.New(transfer.NewItem(transfer.Bytes("image/jpeg", boxJPEG(t)))), true, ""},
		{"another file copied in Finder", finder("notes.pdf"), false, "notes.pdf"},
		{"a spreadsheet range", transfer.New(transfer.NewItem(
			transfer.Bytes(transfer.Text, []byte("Q3 revenue")),
			transfer.Bytes(transfer.PNG, boxPNG(t)),
		)), false, "Q3 revenue"},
	} {
		t.Run(c.name, func(t *testing.T) {
			a, fs, _ := newTestApp(t)
			asked := 0
			fs.handlers[proto.MethodClipboardImage] = func(json.RawMessage) (any, error) {
				asked++
				return proto.ClipboardImageResult{JPEG: boxJPEG(t)}, nil
			}
			a.box = rewriteBox{open: true, shown: true, focusText: true}
			tt := hudTester(a.boxView, boxWinW, boxWinH)
			frames(tt)
			tt.SetClipboardData(c.clipboard)
			tt.Command("paste")
			frames(tt)
			if got := len(a.box.images) == 1 && asked == 1; got != c.image {
				t.Errorf("%d images, the helper asked %d times; want a picture %v", len(a.box.images), asked, c.image)
			}
			if got := a.boxTextContent(); got != c.text {
				t.Errorf("the card's text %q, want %q", got, c.text)
			}
		})
	}
}
