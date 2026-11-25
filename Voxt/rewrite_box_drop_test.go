package main

import (
	"bytes"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestRewriteBoxDropOrderAcrossBatches: a big picture of a first batch
// converts after a second batch's small one (each batch converts on
// its own goroutine). The box and the request must keep the drop
// order: the first batch's picture stays first.
func TestRewriteBoxDropOrderAcrossBatches(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.speak()
	slow, fast := boxJPEGOf(t, 8, 8), boxJPEGOf(t, 16, 16)
	a.postQueue(func() {
		gen := a.boxGen
		first := a.boxReserve()  // batch 1's big picture, still converting
		second := a.boxReserve() // batch 2's, dropped meanwhile
		a.boxConverted(gen, second, fast, "")
		if len(a.box.images) != 0 {
			t.Errorf("%d images before the first batch's picture landed", len(a.box.images))
		}
		a.boxConverted(gen, first, slow, "")
		if len(a.box.images) != 2 || !bytes.Equal(a.box.images[0], slow) || !bytes.Equal(a.box.images[1], fast) {
			t.Errorf("the box lost the drop order: %q", a.box.images)
		}
	})
	a.postQueue(a.boxSend) // Enter: finish the recording and send
	a.pump("the answer", func() bool { return a.answer.open })
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	imgs, _ := reqs[0]["images"].([][]byte)
	if len(imgs) != 2 || !bytes.Equal(imgs[0], slow) || !bytes.Equal(imgs[1], fast) {
		t.Errorf("the request lost the drop order: %q", imgs)
	}
}

// TestRewriteBoxDropKeepsBatchOrder: files dropped in two batches land
// in the box in the drop order, and the pending count returns to zero.
func TestRewriteBoxDropKeepsBatchOrder(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	dir := t.TempDir()
	var files []string
	var want [][]byte
	for i, side := range []int{8, 16, 24} {
		var raw bytes.Buffer
		if err := png.Encode(&raw, image.NewRGBA(image.Rect(0, 0, side, side))); err != nil {
			t.Fatal(err)
		}
		f := filepath.Join(dir, strings.Repeat("f", i+1)+".png")
		if err := os.WriteFile(f, raw.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		enc, err := imgconv.JPEG(raw.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		files, want = append(files, f), append(want, enc)
	}
	a.postQueue(func() { a.boxDrop(files[:1]) })
	a.postQueue(func() { a.boxDrop(files[1:]) })
	a.pump("the pictures", func() bool { return a.boxPending == 0 && len(a.box.images) == 3 })
	for i, enc := range want {
		if !bytes.Equal(a.box.images[i], enc) {
			t.Errorf("image %d is not the %dth dropped file", i, i+1)
		}
	}
}
