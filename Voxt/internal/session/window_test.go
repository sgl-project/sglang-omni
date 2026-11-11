package session

import (
	"bytes"
	"context"
	"errors"
	"testing"

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
		want    []byte
		noClean bool
	}{
		{"transcribe", true, false, nil, store.KindTranscribe, jpeg, false},
		{"transcribe without clean-up", true, false, nil, store.KindTranscribe, nil, true},
		{"translate", true, false, nil, store.KindTranslate, jpeg, false},
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
				if !bytes.Equal(r.Image, tc.want) {
					t.Errorf("image %q, want %q", r.Image, tc.want)
				}
			}
			if wantShots := tc.on && !tc.try && !tc.noClean; (len(pids) == 1 && pids[0] == slack.PID) != wantShots || len(pids) > 1 {
				t.Errorf("pictures taken of %v", pids)
			}
		})
	}
}
