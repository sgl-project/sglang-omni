package main

import (
	"bytes"
	"encoding/json"
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

// TestClipboardImageEndToEnd: with "Send the copied image with Rewrite"
// on and nothing selected, the clipboard picture goes to the model. Text
// selections stay text-only unless the user deliberately enables text and
// image together. A copied placeholder returns the newly selected image,
// never the earlier clipboard image; literal placeholder text sends none.
func TestClipboardImageEndToEnd(t *testing.T) {
	for _, tc := range []struct {
		name, opts, sel   string
		viaCopy, on, both bool
		legacy            bool
		want              []byte
	}{
		{name: "a copied image, no selection", opts: ",clipboard-image", on: true, want: helpertest.ClipImage},
		{name: "existing settings send text and copied image", opts: ",clipboard-image", sel: "this", on: true, legacy: true, want: helpertest.ClipImage},
		{name: "a fresh selection stays out", opts: ",clipboard-image", sel: "this", on: true},
		{name: "a copied image placeholder returns the new image", opts: ",clipboard-image,copy-selection-image", sel: "[Image]", viaCopy: true, on: true, want: helpertest.Shot(812)},
		{name: "literal placeholder text through copy stays text", opts: ",clipboard-image", sel: "[Image]", viaCopy: true, on: true},
		{name: "deliberate text and image sends both", opts: ",clipboard-image", sel: "this", on: true, both: true, want: helpertest.ClipImage},
		{name: "privacy off drops an image placeholder", opts: ",clipboard-image,copy-selection-image", sel: "[Image]", viaCopy: true},
		{name: "privacy off drops mixed content even with text and image on", opts: ",clipboard-image,copy-selection-image", sel: "caption [Image]", viaCopy: true, both: true},
		{name: "nothing copied", sel: "this", on: true},
		{name: "off", opts: ",clipboard-image"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{text: "A red square."}
			set, keys, _ := fileSetup(t, "describe the picture", chat)
			ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
			opts := ev + tc.opts
			if tc.viaCopy {
				opts = helpertest.Opt("copy-selection", tc.sel) + "," + opts
			} else if tc.sel != "" {
				opts = helpertest.Opt("selection", tc.sel) + "," + opts
			}
			a := newSessionApp(t, opts, func(s *settings.Settings) {
				set(s)
				s.Privacy.SendClipboardImage = tc.on
				if tc.legacy {
					s.Version = 1
					var document map[string]any
					raw, _ := json.Marshal(s)
					if err := json.Unmarshal(raw, &document); err != nil {
						t.Fatal(err)
					}
					delete(document["privacy"].(map[string]any), "sendClipboardImageWithSelection")
					raw, _ = json.Marshal(document)
					migrated, err := settings.Migrate(raw)
					if err != nil {
						t.Fatal(err)
					}
					*s = migrated
				} else {
					s.Privacy.SendClipboardImageWithSelection = tc.both
				}
			}, keys)
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != session.Done || a.sess.Result != "A red square." {
				t.Fatalf("end %+v", a.sess)
			}
			reqs := chat.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d chats", len(reqs))
			}
			img, _ := reqs[0]["image"].([]byte)
			if !bytes.Equal(img, tc.want) {
				t.Errorf("image %q, want %q\n%s", img, tc.want, a.logText())
			}
		})
	}
}
