package main

import (
	"errors"
	"slices"
	"testing"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
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
