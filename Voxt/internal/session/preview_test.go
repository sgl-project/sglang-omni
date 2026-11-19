package session

import (
	"bytes"
	"context"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

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
