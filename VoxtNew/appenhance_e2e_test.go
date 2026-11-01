package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// hotkeyIn is hotkeyLine with another app in front.
func hotkeyIn(afterMS int, bundleID string) string {
	front := helpertest.Apps[bundleID]
	b, _ := json.Marshal(helpertest.Script{AfterMS: afterMS, Method: proto.MethodHotkeyEvent,
		Params: mustJSON(proto.HotkeyEventParams{Binding: "transcribe/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, T: 1, Front: &front})})
	return string(b)
}

// e2eRules: Gmail by its address (⌘↩ after inserting), Slack by app
// (↩), and Chrome itself.
func e2eRules() appctx.Rules {
	return appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{
		{ID: "g-mail", Name: "Mail", Enabled: true, Prompt: "An email: complete sentences.", URLs: []string{"mail.google.com/mail/*"},
			AutoKey: "Command+Return", Features: appctx.DefaultFeatures},
		{ID: "g-chat", Name: "Chat", Enabled: true, Prompt: "A chat message: short.", Apps: []appctx.App{{BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}},
			AutoKey: "Return", Features: appctx.DefaultFeatures},
		{ID: "g-chrome", Name: "Chrome", Enabled: true, Prompt: "Browsing.", Apps: []appctx.App{{BundleID: "com.google.Chrome", Name: "Chrome"}},
			Features: appctx.DefaultFeatures},
		{ID: "g-linear", Name: "Issues", Enabled: true, Prompt: "An issue tracker.", URLs: []string{"linear.app"},
			Features: appctx.DefaultFeatures},
	}}
}

// TestSessionAppEnhancement runs sessions over the fake helper process:
// the address of Chrome's tab is read through browser.url and matches the
// Mail group, whose scene goes into the prompt and whose ⌘↩ follows the
// paste; the history keeps the rule; Chrome without an address falls back
// to its own group; Slack matches by app.
func TestSessionAppEnhancement(t *testing.T) {
	for _, tc := range []struct {
		name, front, opts string
		group             string
		source            appctx.Source
		host              string
		scene             string
		press             []proto.KeyPress
	}{
		{"url", "com.google.Chrome", "", "g-mail", appctx.SourceURL, "mail.google.com", "An email: complete sentences.",
			[]proto.KeyPress{{Key: 36, Modifiers: []string{proto.ModCommand}}}},
		{"browser app", "com.google.Chrome", "no-url", "g-chrome", appctx.SourceBrowserApp, "", "Browsing.", nil},
		{"app", "com.tinyspeck.slackmacgap", "", "g-chat", appctx.SourceApp, "", "A chat message: short.", []proto.KeyPress{{Key: 36}}},
		// Arc builds its pages' tree only once app.configure listed it.
		{"arc tree", "company.thebrowser.Browser", "no-tree=company.thebrowser.Browser", "g-linear", appctx.SourceURL, "linear.app", "An issue tracker.", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{text: "Hello."}
			set, keys, _ := fileSetup(t, "hello", chat)
			opts := script(t, hotkeyIn(300, tc.front), hotkeyIn(800, tc.front)) + ",front=" + tc.front
			if tc.opts != "" {
				opts += "," + tc.opts
			}
			a := newSessionApp(t, opts, set, keys)
			a.setRules(e2eRules())
			a.pump("the end", func() bool { return a.sess.ID != "" && a.sess.State.Ended() })
			end := a.sess
			if end.State != session.Done || !end.Inserted || end.Scene == nil {
				t.Fatalf("end %+v", end)
			}
			if end.Scene.GroupID() != tc.group || end.Scene.Source != tc.source || end.Scene.Host != tc.host {
				t.Errorf("scene %+v", end.Scene)
			}
			p := a.pasted()
			if len(p) != 1 || !reflect.DeepEqual(p[0].Press, tc.press) {
				t.Errorf("pasted %+v", p)
			}
			msgs, _ := chat.requests()[0]["messages"].([]any)
			sys, _ := msgs[0].(map[string]any)["content"].(string)
			if !strings.Contains(sys, tc.scene) {
				t.Errorf("no scene in %s", sys)
			}
			es := a.history()
			if len(es) != 1 || es[0].RuleID != tc.group || es[0].RuleSource != string(tc.source) || es[0].URLHost != tc.host {
				t.Errorf("history %+v", es)
			}
			if r, ok := a.urlReads["com.google.Chrome"]; tc.front == "com.google.Chrome" && (!ok || r.ok != (tc.opts == "")) {
				t.Errorf("url reads %+v", a.urlReads)
			}
		})
	}
	// App enhancement off: no scene, no key, no address read.
	chat := &fakeChat{text: "Hello."}
	set, keys, _ := fileSetup(t, "hello", chat)
	a := newSessionApp(t, script(t, hotkeyIn(300, "com.google.Chrome"), hotkeyIn(800, "com.google.Chrome"))+",front=com.google.Chrome", func(s *settings.Settings) {
		set(s)
		s.AppEnhancement.Enabled = false
	}, keys)
	a.setRules(e2eRules())
	a.pump("the end", func() bool { return a.sess.ID != "" && a.sess.State.Ended() })
	if a.sess.Scene == nil || a.sess.Scene.Matched() || len(a.pasted()) != 1 || a.pasted()[0].Press != nil || len(a.urlReads) != 0 {
		t.Errorf("off: %+v %+v", a.sess.Scene, a.pasted())
	}
}
