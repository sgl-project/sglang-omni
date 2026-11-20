package session

import (
	"bytes"
	"context"
	"strings"
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

func TestBoxWindowPreviewCanBeRemovedBeforeRequest(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday."}
	shot := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	reviewStarted, approve := make(chan struct{}), make(chan struct{})
	var suppressed atomic.Bool
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return shot, nil }
		c.BoxInput = func(context.Context) (string, [][]byte, error) { return "", nil, nil }
		c.BoxAutoSuppressed = func() (bool, bool) { return false, suppressed.Load() }
		c.BoxReview = func(ctx context.Context) error {
			close(reviewStarted)
			select {
			case <-approve:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 3)
	select {
	case <-reviewStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("the request did not wait for the window preview's review")
	}
	preview := h.rec.waitFor(t, "the removable window preview", func(s Snapshot) bool {
		return s.State == Recognizing && s.ImagePreviewKind == "window" && bytes.Equal(s.ImagePreview, shot)
	})
	if len(model.requests()) != 0 || preview.ImageHint != "" {
		t.Fatalf("window image was sent before its review: requests %d, hint %q", len(model.requests()), preview.ImageHint)
	}
	suppressed.Store(true)
	close(approve)
	end := h.rec.wait(t)
	if end.State != Done || end.ImageHint != "" {
		t.Fatalf("suppressed window image survived: end %s, hint %q", end.State, end.ImageHint)
	}
	reqs := model.requests()
	if len(reqs) != 1 || len(reqs[0].Images) != 0 {
		t.Fatalf("images sent after removing the preview: %+v", reqs)
	}
}

// TestBoxInputTypedDuringWindowReviewIsSent: text typed and a text
// block pasted into the box while the window preview waits for approval,
// and a picture pasted there, go with the request; the box's picture
// wins over the window's.
func TestBoxInputTypedDuringWindowReviewIsSent(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday."}
	shot, pasted := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, []byte{0xff, 0xd8, 'p', 0xff, 0xd9}
	typed := "and cc the team\n\nRelease notes:\n- faster start\n- fewer crashes"
	approve := make(chan struct{})
	var approved atomic.Bool
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return shot, nil }
		c.BoxInput = func(context.Context) (string, [][]byte, error) {
			if approved.Load() {
				return typed, [][]byte{pasted}, nil
			}
			return "", nil, nil
		}
		c.BoxReview = func(ctx context.Context) error {
			select {
			case <-approve:
				approved.Store(true)
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 3)
	h.rec.waitFor(t, "the window preview", func(s Snapshot) bool { return s.ImagePreviewKind == "window" })
	close(approve)
	end := h.rec.wait(t)
	if end.State != Done || end.Typed != typed || end.ImagePreviewKind != "" {
		t.Fatalf("end %s, typed %q, preview %q", end.State, end.Typed, end.ImagePreviewKind)
	}
	reqs := model.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; !strings.Contains(user, typed) {
		t.Errorf("typed text missing from the user message:\n%s", user)
	}
	if !equalBytes(reqs[0].Images, [][]byte{pasted}) {
		t.Errorf("images %q, want the box's picture only", reqs[0].Images)
	}
}

func TestBoxInputClearedDuringWindowReviewIsNotSent(t *testing.T) {
	model := &fakeLLM{text: `{"title":"Friday","content":"We ship on Friday."}`}
	approve := make(chan struct{})
	var approved atomic.Bool
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, nil }
		c.BoxInput = func(context.Context) (string, [][]byte, error) {
			if approved.Load() {
				return "", nil, nil
			}
			return "delete me", nil, nil
		}
		c.BoxReview = func(ctx context.Context) error {
			select {
			case <-approve:
				approved.Store(true)
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 3)
	h.rec.waitFor(t, "the window preview", func(s Snapshot) bool { return s.ImagePreviewKind == "window" })
	close(approve)
	end := h.rec.wait(t)
	if end.State != Done || end.Typed != "" {
		t.Fatalf("cleared text survived: end %s, typed %q", end.State, end.Typed)
	}
	reqs := model.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; strings.Contains(user, "delete me") {
		t.Errorf("cleared text in the user message:\n%s", user)
	}
}

// TestBoxClosedRetryKeepsTypedText: a retry from the answer card starts
// after its input box has closed, so an empty box read must not erase the
// typed text carried by the retry.
func TestBoxClosedRetryKeepsTypedText(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday."}
	h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
		c.BoxInput = func(context.Context) (string, [][]byte, error) { return "", nil, nil }
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?", Typed: "and cc the team"}}); err != nil {
		t.Fatal(err)
	}
	end := h.rec.wait(t)
	if end.State != Done || end.Typed != "and cc the team" {
		t.Fatalf("closed retry lost its typed text: end %s, typed %q", end.State, end.Typed)
	}
	reqs := model.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; !strings.Contains(user, "and cc the team") {
		t.Errorf("typed text missing from the user message:\n%s", user)
	}
}

// TestBoxResendClearedDuringWindowReviewIsNotSent: a send again carries
// the text typed before; the user clears the box while its window
// screenshot waits for approval, and the old text stays out.
func TestBoxResendClearedDuringWindowReviewIsNotSent(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday."}
	approve := make(chan struct{})
	var approved atomic.Bool
	h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, nil }
		c.BoxInput = func(context.Context) (string, [][]byte, error) {
			if approved.Load() {
				return "", nil, nil
			}
			return "delete me", nil, nil
		}
		c.BoxReview = func(ctx context.Context) error {
			select {
			case <-approve:
				approved.Store(true)
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?", Typed: "delete me"}}); err != nil {
		t.Fatal(err)
	}
	h.rec.waitFor(t, "the window preview", func(s Snapshot) bool { return s.ImagePreviewKind == "window" })
	close(approve)
	end := h.rec.wait(t)
	if end.State != Done || end.Typed != "" {
		t.Fatalf("cleared text survived the resend: end %s, typed %q", end.State, end.Typed)
	}
	reqs := model.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; strings.Contains(user, "delete me") {
		t.Errorf("cleared text in the user message:\n%s", user)
	}
}

func TestBoxWindowReviewDoesNotBlockMissingLanguageModel(t *testing.T) {
	review := make(chan struct{}, 1)
	h := newHarness(t, &fakeFile{text: "make it polite"}, nil, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, nil }
		c.BoxReview = func(context.Context) error {
			review <- struct{}{}
			return nil
		}
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?"}}); err != nil {
		t.Fatal(err)
	}
	end := h.rec.wait(t)
	if end.State != Failed || end.Problem == nil || end.Problem.Code != ProblemNoLLM {
		t.Fatalf("end %s, problem %+v", end.State, end.Problem)
	}
	select {
	case <-review:
		t.Fatal("the missing model error waited for an image review")
	default:
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
