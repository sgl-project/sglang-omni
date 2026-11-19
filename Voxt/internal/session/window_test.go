package session

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestWindowPicture: with Privacy.SendWindow on, the language model gets
// a picture of the target app's window (taken once, for that app), with
// Transcribe's clean-up and Translate's answer; off, in a try, or when
// the picture fails, it gets the text alone and the session goes on;
// Transcribe without its clean-up calls no model and takes no picture.
func TestWindowPicture(t *testing.T) {
	jpeg := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	for _, tc := range []struct {
		name    string
		on      bool
		try     bool
		err     error
		kind    store.Kind
		want    [][]byte
		noClean bool
	}{
		{"transcribe", true, false, nil, store.KindTranscribe, [][]byte{jpeg}, false},
		{"transcribe without clean-up", true, false, nil, store.KindTranscribe, nil, true},
		{"translate", true, false, nil, store.KindTranslate, [][]byte{jpeg}, false},
		{"off", false, false, nil, store.KindTranscribe, nil, false},
		{"a try", true, true, nil, store.KindTranscribe, nil, false},
		{"no permission", true, false, errors.New("permission_denied"), store.KindTranscribe, nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "Hello."}
			var pids []int
			h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = !tc.noClean
				s.Privacy.SendWindow = tc.on
				c.Window = func(_ context.Context, pid int) ([]byte, error) {
					pids = append(pids, pid)
					if tc.err != nil {
						return nil, tc.err
					}
					return jpeg, nil
				}
			})
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack, Try: tc.try})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			if end := h.rec.wait(t); end.State != Done {
				t.Fatalf("end %s %+v", end.State, end.Problem)
			}
			reqs := model.requests()
			if len(reqs) == 0 && !tc.noClean {
				t.Fatal("no request to the language model")
			}
			for _, r := range reqs {
				if !equalBytes(r.Images, tc.want) {
					t.Errorf("images %q, want %q", r.Images, tc.want)
				}
			}
			if wantShots := tc.on && !tc.try && !tc.noClean; (len(pids) == 1 && pids[0] == slack.PID) != wantShots || len(pids) > 1 {
				t.Errorf("pictures taken of %v", pids)
			}
		})
	}
}

// TestClipboardImage: with Privacy.SendClipboardImage on, Rewrite reads
// the clipboard's picture before the selection (whose ⌘C would replace
// it) and sends it with the text, instead of the window's; off, with no
// picture or when reading fails, the model gets the text (or the window)
// and the rewrite lands. Translate and a try never read the clipboard.
func TestClipboardImage(t *testing.T) {
	clip, shot := []byte{0xff, 0xd8, 'c', 0xff, 0xd9}, []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	for _, tc := range []struct {
		name         string
		kind         store.Kind
		on, window   bool
		try          bool
		pic          []byte
		err          error
		want         [][]byte
		wantReadClip bool
	}{
		{name: "rewrite", kind: store.KindRewrite, on: true, pic: clip, want: [][]byte{clip}, wantReadClip: true},
		{name: "off", kind: store.KindRewrite, pic: clip},
		{name: "wins over the window", kind: store.KindRewrite, on: true, window: true, pic: clip, want: [][]byte{clip}, wantReadClip: true},
		{name: "no picture: the window", kind: store.KindRewrite, on: true, window: true, want: [][]byte{shot}, wantReadClip: true},
		{name: "reading fails", kind: store.KindRewrite, on: true, err: errors.New("helper gone"), wantReadClip: true},
		{name: "a try", kind: store.KindRewrite, on: true, try: true, pic: clip},
		{name: "translate", kind: store.KindTranslate, on: true, pic: clip},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "We ship on Friday."}
			sel := &fakeSelector{text: "ship friday?"}
			var reads, selBefore int
			h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
				c.Select = sel
				s.Privacy.SendClipboardImage, s.Privacy.SendWindow = tc.on, tc.window
				c.Window = func(context.Context, int) ([]byte, error) { return shot, nil }
				c.ClipboardImage = func(context.Context) ([]byte, error) {
					sel.mu.Lock()
					reads, selBefore = reads+1, len(sel.calls)
					sel.mu.Unlock()
					return tc.pic, tc.err
				}
			})
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack, Try: tc.try})
			if err != nil {
				t.Fatal(err)
			}
			if tc.kind == store.KindRewrite {
				h.speak(id, 3)
			}
			if end := h.rec.wait(t); end.State != Done || end.Result != "We ship on Friday." {
				t.Fatalf("end %s %q %+v", end.State, end.Result, end.Problem)
			}
			reqs := model.requests()
			if len(reqs) == 0 {
				t.Fatal("no request to the language model")
			}
			for _, r := range reqs {
				if !equalBytes(r.Images, tc.want) {
					t.Errorf("images %q, want %q", r.Images, tc.want)
				}
			}
			if (reads == 1) != tc.wantReadClip || reads > 1 || selBefore != 0 {
				t.Errorf("clipboard read %d times, after %d selection reads", reads, selBefore)
			}
		})
	}
}

// equalBytes reports whether two lists of pictures hold the same bytes.
func equalBytes(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

// TestBoxImages: a Rewrite sent from the input box attaches the box's
// own pictures, required, and neither reads the clipboard nor takes the
// window's picture; an empty box keeps today's rule (the clipboard's
// picture, else the window's), and the snapshot says which picture went
// (Snapshot.ImageHint).
func TestBoxImages(t *testing.T) {
	box1, box2 := []byte{0xff, 0xd8, '1', 0xff, 0xd9}, []byte{0xff, 0xd8, '2', 0xff, 0xd9}
	clip, shot := []byte{0xff, 0xd8, 'c', 0xff, 0xd9}, []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	for _, tc := range []struct {
		name                 string
		images               [][]byte
		clipOn, windowOn     bool
		clip                 []byte
		want                 [][]byte
		wantHint             string
		wantReadClip, wantSh bool
	}{
		{name: "box pictures win", images: [][]byte{box1, box2}, clipOn: true, windowOn: true, clip: clip,
			want: [][]byte{box1, box2}, wantHint: "box"},
		{name: "empty box: the clipboard", clipOn: true, windowOn: true, clip: clip,
			want: [][]byte{clip}, wantHint: "clipboard", wantReadClip: true},
		{name: "empty box: the window", windowOn: true, want: [][]byte{shot}, wantHint: "window", wantSh: true},
		{name: "empty box, nothing on", wantHint: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "We ship on Friday."}
			var reads, shots int
			h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
				s.Privacy.SendClipboardImage, s.Privacy.SendWindow = tc.clipOn, tc.windowOn
				c.Window = func(context.Context, int) ([]byte, error) { shots++; return shot, nil }
				c.ClipboardImage = func(context.Context) ([]byte, error) { reads++; return tc.clip, nil }
			})
			if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
				Text: &TextInput{Raw: "make it polite", Selection: "ship friday?"}, Images: tc.images}); err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "We ship on Friday." {
				t.Fatalf("end %s %q %+v", end.State, end.Result, end.Problem)
			}
			if end.ImageHint != tc.wantHint {
				t.Errorf("hint %q, want %q", end.ImageHint, tc.wantHint)
			}
			reqs := model.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d requests", len(reqs))
			}
			if !equalBytes(reqs[0].Images, tc.want) {
				t.Errorf("images %q, want %q", reqs[0].Images, tc.want)
			}
			if wantRequired := len(tc.images) > 0; reqs[0].ImagesRequired != wantRequired {
				t.Errorf("required %v, want %v", reqs[0].ImagesRequired, wantRequired)
			}
			if reads != b2i(tc.wantReadClip) || shots != b2i(tc.wantSh) {
				t.Errorf("clipboard reads %d, window pictures %d", reads, shots)
			}
		})
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// TestImageHintClearedWhenDropped: the model refused the picture and
// answered from the text alone (llm.Response.ImagesDropped); the card
// must not claim a picture went with the request.
func TestImageHintClearedWhenDropped(t *testing.T) {
	model := &fakeLLM{text: "We ship on Friday.", drop: true}
	h := newHarness(t, &fakeFile{text: "unused"}, model, func(c *Config, s *settings.Settings) {
		s.Privacy.SendWindow = true
		c.Window = func(context.Context, int) ([]byte, error) { return []byte{0xff, 0xd8, 'w', 0xff, 0xd9}, nil }
	})
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?"}}); err != nil {
		t.Fatal(err)
	}
	end := h.rec.wait(t)
	if end.State != Done || end.ImageHint != "" {
		t.Fatalf("end %s hint %q", end.State, end.ImageHint)
	}
}

// TestBoxInput: the input box's session records and reads the selection
// as ever; when the answer is asked for, the box's typed text and
// pictures (Config.BoxInput) go with it, the pictures required, in one
// fixed order: the selection, the typed text, the spoken instruction. The
// early window preview is not sent once the box has pictures. A send again
// (Text) carries the same bytes in the same position, so the server's
// prefix cache reuses them.
func TestBoxInput(t *testing.T) {
	box1, box2 := []byte{0xff, 0xd8, '1', 0xff, 0xd9}, []byte{0xff, 0xd8, '2', 0xff, 0xd9}
	shot := []byte{0xff, 0xd8, 'w', 0xff, 0xd9}
	model := &fakeLLM{text: "We ship on Friday."}
	sel := &fakeSelector{text: "ship friday?"}
	var shots int
	h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
		c.Select = sel
		s.Privacy.SendWindow, s.Privacy.SendClipboardImage = true, true
		c.Window = func(context.Context, int) ([]byte, error) { shots++; return shot, nil }
		c.ClipboardImage = func(context.Context) ([]byte, error) { return nil, nil }
		c.BoxInput = func(context.Context) (string, [][]byte, error) {
			return "and cc the team", [][]byte{box1, box2}, nil
		}
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 3)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "We ship on Friday." {
		t.Fatalf("end %s %q %+v", end.State, end.Result, end.Problem)
	}
	if end.ImageHint != "box" || end.Typed != "and cc the team" || end.Selection != "ship friday?" {
		t.Errorf("hint %q, typed %q, selection %q", end.ImageHint, end.Typed, end.Selection)
	}
	want := "<selected_text>\nship friday?\n</selected_text>\n\n<typed_text>\nand cc the team\n</typed_text>\n\n<instruction>\nmake it polite\n</instruction>"
	reqs := model.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; user != want {
		t.Errorf("user message:\n%s", user)
	}
	if wantChars := len("ship friday?" + "and cc the team" + "make it polite"); reqs[0].InputChars != wantChars {
		t.Errorf("input chars %d, want %d", reqs[0].InputChars, wantChars)
	}
	if !equalBytes(reqs[0].Images, [][]byte{box1, box2}) || !reqs[0].ImagesRequired {
		t.Errorf("images %q, required %v", reqs[0].Images, reqs[0].ImagesRequired)
	}
	if shots != 1 || end.ImagePreviewKind != "" || len(end.ImagePreview) != 0 {
		t.Errorf("%d window pictures taken, preview %q %d bytes; want one capture and no preview sent", shots, end.ImagePreviewKind, len(end.ImagePreview))
	}
	// A send again (Text): the same bytes in the same position.
	if _, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack,
		Text: &TextInput{Raw: "make it polite", Selection: "ship friday?", Typed: "and cc the team"}}); err != nil {
		t.Fatal(err)
	}
	if end := h.rec.wait(t); end.State != Done {
		t.Fatalf("resend end %s %+v", end.State, end.Problem)
	}
	reqs = model.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests after the resend", len(reqs))
	}
	if !equalBytes(reqs[1].Images, reqs[0].Images) || !reqs[1].ImagesRequired {
		t.Errorf("resend images %q, required %v", reqs[1].Images, reqs[1].ImagesRequired)
	}
	if user := reqs[1].Messages[len(reqs[1].Messages)-1].Content; user != want {
		t.Errorf("resend's user message:\n%s", user)
	}
	if shots != 1 {
		t.Errorf("%d window pictures taken, want only the early preview", shots)
	}
}

// TestBoxInputTypedOnly: with nothing selected, the box's typed text is
// the source (the selection's place); an empty box is the speech alone,
// as ever.
func TestBoxInputTypedOnly(t *testing.T) {
	for _, tc := range []struct {
		name, typed string
		wantUser    string
		wantFormat  llm.Format
	}{
		{"typed text is the source", "a rough draft", "<selected_text>\na rough draft\n</selected_text>\n\n<instruction>\nmake it polite\n</instruction>", llm.FormatText},
		{"an empty box is the speech", "", "<instruction>\nmake it polite\n</instruction>", llm.FormatJSON},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "We ship on Friday."}
			if tc.wantFormat == llm.FormatJSON {
				model.text = `{"title":"Friday","content":"We ship on Friday."}`
			}
			h := newHarness(t, &fakeFile{text: "make it polite"}, model, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{}
				c.BoxInput = func(context.Context) (string, [][]byte, error) { return tc.typed, nil, nil }
			})
			id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Box: true, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			if end := h.rec.wait(t); end.State != Done {
				t.Fatalf("end %s %+v", end.State, end.Problem)
			}
			reqs := model.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d requests", len(reqs))
			}
			if user := reqs[0].Messages[len(reqs[0].Messages)-1].Content; user != tc.wantUser {
				t.Errorf("user message:\n%s", user)
			}
			if reqs[0].Format != tc.wantFormat {
				t.Errorf("format %v, want %v", reqs[0].Format, tc.wantFormat)
			}
		})
	}
}
