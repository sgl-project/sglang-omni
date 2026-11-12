package main

import (
	"bytes"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestWindowScreenshotEndToEnd: with "Send the window to the language
// model" on, the helper's screenshot of the app in front (Slack, pid 812)
// goes with the text to the language model; without Screen Recording the
// model gets the text alone and the translation still lands.
func TestWindowScreenshotEndToEnd(t *testing.T) {
	for _, tc := range []struct {
		name, opts string
		image      bool
	}{
		{"allowed", "", true},
		{"no Screen Recording", ",no-screen", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{text: "早上好"}
			set, keys, _ := fileSetup(t, "unused", chat)
			ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
			a := newSessionApp(t, helpertest.Opt("selection", "Good morning")+","+ev+tc.opts, func(s *settings.Settings) {
				set(s)
				s.Features.Translate.Target = "zh-Hans"
				s.Privacy.SendWindow = true
			}, keys)
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != session.Done || a.sess.Result != "早上好" {
				t.Fatalf("end %+v", a.sess)
			}
			reqs := chat.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d chats", len(reqs))
			}
			img, _ := reqs[0]["image"].([]byte)
			if want := helpertest.Shot(812); tc.image != bytes.Equal(img, want) || !tc.image && img != nil {
				t.Errorf("image %q, want it %v\n%s", img, tc.image, a.logText())
			}
		})
	}
}
