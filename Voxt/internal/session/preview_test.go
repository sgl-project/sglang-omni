package session

import (
	"bytes"
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestBoxAutomaticInputsCanBeSuppressed(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday."}
	var clipReads, shots int
	h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendClipboardImage, s.Privacy.SendWindow = true, true
		c.ClipboardImage = func(context.Context) ([]byte, error) { clipReads++; return []byte{0xff, 0xd8, 'c', 0xff, 0xd9}, nil }
		c.Window = func(context.Context, int) ([]byte, error) { shots++; return []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, nil }
		c.BoxAutoSuppressed = func() (bool, bool) { return true, true }
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?"}}); err != nil {
		t.Fatal(err)
	}
	end := h.rec.wait(t)
	if end.State != Done || end.Selection != "" || end.ImageHint != "" {
		t.Fatalf("suppressed input survived: end %s, selection %q, hint %q", end.State, end.Selection, end.ImageHint)
	}
	if clipReads != 0 || shots != 0 {
		t.Fatalf("automatic captures still ran: clipboard %d, window %d", clipReads, shots)
	}
	reqs := model.requests()
	if len(reqs) != 1 || len(reqs[0].Images) != 0 {
		t.Fatalf("images sent after suppression: %+v", reqs)
	}
}

func TestBoxWindowPreviewCanBeRemovedBeforeSending(t *testing.T) {
	shot := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	model := &fakeLLM{text: "We ship on Friday."}
	var removed atomic.Bool
	var shots int
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		c.Select = &fakeSelector{text: "ship friday?"}
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { shots++; return shot, nil }
		c.BoxAutoSuppressed = func() (bool, bool) { return false, removed.Load() }
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.rec.waitFor(t, "the window preview while recording", func(s Snapshot) bool {
		return s.State == Recording && s.ImagePreviewKind == "window" && bytes.Equal(s.ImagePreview, shot)
	})
	if got := len(model.requests()); got != 0 {
		t.Fatalf("%d requests before recording stopped", got)
	}
	removed.Store(true)
	h.speak(id, 3)
	end := h.rec.wait(t)
	if end.State != Done || end.ImageHint != "" || end.ImagePreviewKind != "" || len(end.ImagePreview) != 0 {
		t.Fatalf("removed preview survived: end %s, hint %q, preview %q %d bytes", end.State, end.ImageHint, end.ImagePreviewKind, len(end.ImagePreview))
	}
	reqs := model.requests()
	if len(reqs) != 1 || len(reqs[0].Images) != 0 {
		t.Fatalf("images sent after removal: %+v", reqs)
	}
	if shots != 1 {
		t.Fatalf("%d window captures, want one", shots)
	}
}

func TestLateWindowPreviewDoesNotReplaceBoxImages(t *testing.T) {
	shot, own := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, []byte{0xff, 0xd8, 'o', 0xff, 0xd9}
	model := &fakeLLM{text: "We ship on Friday.", pause: 200 * time.Millisecond}
	releaseCapture := make(chan struct{})
	var shots atomic.Int64
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		c.Select = &fakeSelector{text: "ship friday?"}
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) {
			shots.Add(1)
			<-releaseCapture
			return shot, nil
		}
		c.BoxInput = func(context.Context) (string, [][]byte, error) { return "", [][]byte{own}, nil }
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 3)
	h.rec.waitFor(t, "the box image request", func(s Snapshot) bool { return s.ImageHint == "box" })
	close(releaseCapture)
	end := h.rec.wait(t)
	if end.State != Done || end.ImageHint != "box" || end.ImagePreviewKind != "" || len(end.ImagePreview) != 0 {
		t.Fatalf("late capture replaced the box input: end %s, hint %q, preview %q %d bytes", end.State, end.ImageHint, end.ImagePreviewKind, len(end.ImagePreview))
	}
	if shots.Load() != 1 {
		t.Fatalf("%d window captures, want one", shots.Load())
	}
}

func TestAutomaticImagePreviewFollowsTheImageSent(t *testing.T) {
	clip, shot, own := []byte{0xff, 0xd8, 'c', 0xff, 0xd9}, []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, []byte{0xff, 0xd8, 'o', 0xff, 0xd9}
	for _, tc := range []struct {
		name, wantKind string
		images         [][]byte
		want           []byte
	}{
		{name: "clipboard", wantKind: "clipboard", want: clip},
		{name: "box image wins", images: [][]byte{own}, wantKind: "", want: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "We ship on Friday."}
			h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: "ship friday?"}
				s.Privacy.SendClipboardImage = true
				c.ClipboardImage = func(context.Context) ([]byte, error) { return clip, nil }
			})
			id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack, Images: tc.images})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			end := h.rec.wait(t)
			if end.ImagePreviewKind != tc.wantKind || !bytes.Equal(end.ImagePreview, tc.want) {
				t.Fatalf("preview %q %q, want %q %q", end.ImagePreviewKind, end.ImagePreview, tc.wantKind, tc.want)
			}
		})
	}

	model := &fakeLLM{text: "We ship on Friday."}
	h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return shot, nil }
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?"}}); err != nil {
		t.Fatal(err)
	}
	end := h.rec.wait(t)
	if end.ImagePreviewKind != "window" || !bytes.Equal(end.ImagePreview, shot) {
		t.Fatalf("window preview %q %q", end.ImagePreviewKind, end.ImagePreview)
	}
}
