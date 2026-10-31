package session

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

var (
	chrome = proto.App{PID: 901, BundleID: "com.google.Chrome", Name: "Google Chrome"}
	safari = proto.App{PID: 902, BundleID: "com.apple.Safari", Name: "Safari"}
	notes  = proto.App{PID: 530, BundleID: "com.apple.Notes", Name: "Notes"}
)

// sceneRules: GitHub by address, Chrome as a browser group, Slack as a
// chat group that presses Return, a disabled group for Notes.
func sceneRules() appctx.Rules {
	return appctx.Rules{Version: appctx.Version, Groups: []appctx.Group{
		{ID: "g-code", Name: "Code review", Enabled: true, Prompt: "Use Markdown for code.", URLs: []string{"github.com"}, Features: appctx.DefaultFeatures},
		{ID: "g-browser", Name: "Work browser", Enabled: true, Prompt: "Keep it formal in {{APP_NAME}}.", Apps: []appctx.App{{BundleID: chrome.BundleID, Name: "Chrome"}},
			Features: appctx.DefaultFeatures},
		{ID: "g-chat", Name: "Chat", Enabled: true, Prompt: "Short and casual.", Apps: []appctx.App{{BundleID: slack.BundleID, Name: "Slack"}},
			AutoKey: "Return", Features: appctx.Features{Transcribe: true}},
		{ID: "g-off", Name: "Notes", Enabled: false, Prompt: "unused", Apps: []appctx.App{{BundleID: notes.BundleID}}, Features: appctx.DefaultFeatures},
	}}
}

// fakeURL is browser.url: an address per app, slow when told to.
type fakeURL struct {
	urls  map[string]string
	delay time.Duration
	err   error

	mu    sync.Mutex
	calls []proto.App
}

func (f *fakeURL) read(ctx context.Context, app proto.App) (string, error) {
	f.mu.Lock()
	f.calls = append(f.calls, app)
	f.mu.Unlock()
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	return f.urls[app.BundleID], f.err
}

func (f *fakeURL) n() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

// TestScene runs sessions in each place app enhancement tells apart.
func TestScene(t *testing.T) {
	for _, tc := range []struct {
		name    string
		target  proto.App
		url     *fakeURL
		off     bool // app enhancement off
		group   string
		source  appctx.Source
		host    string
		prompt  string // in the system prompt
		app     string // the scene's app name
		press   []proto.KeyPress
		urlRead bool
	}{
		{name: "url group", target: safari, url: &fakeURL{urls: map[string]string{safari.BundleID: "https://github.com/sgl-project/sglang-omni/VoxtNew"}},
			group: "g-code", source: appctx.SourceURL, host: "github.com", prompt: "Use Markdown for code.", app: "Safari", urlRead: true},
		{name: "browser app group", target: chrome, url: &fakeURL{urls: map[string]string{chrome.BundleID: "https://news.ycombinator.com/"}},
			group: "g-browser", source: appctx.SourceBrowserApp, host: "news.ycombinator.com", prompt: "Keep it formal in Google Chrome.", app: "Google Chrome", urlRead: true},
		{name: "app group with an auto key", target: slack, group: "g-chat", source: appctx.SourceApp, prompt: "Short and casual.", app: "Slack",
			press: []proto.KeyPress{{Key: 36}}},
		{name: "no group", target: safari, url: &fakeURL{urls: map[string]string{safari.BundleID: "https://example.com/"}},
			source: appctx.SourceGlobal, host: "example.com", urlRead: true},
		{name: "disabled group", target: notes, source: appctx.SourceGlobal},
		{name: "url times out", target: chrome, url: &fakeURL{urls: map[string]string{chrome.BundleID: "https://github.com/"}, delay: 5 * time.Second},
			group: "g-browser", source: appctx.SourceBrowserApp, prompt: "Keep it formal in Google Chrome.", app: "Google Chrome", urlRead: true},
		{name: "url fails", target: safari, url: &fakeURL{err: errors.New("no focused window")}, source: appctx.SourceGlobal, urlRead: true},
		{name: "app enhancement off", target: slack, off: true, source: appctx.SourceGlobal},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u := tc.url
			if u == nil {
				u = &fakeURL{}
			}
			h := newHarness(t, &fakeStreamer{final: "hello"}, &fakeLLM{text: "Hello."}, func(c *Config, s *settings.Settings) {
				c.Scene = &SceneConfig{
					Rules:      func() (appctx.Rules, bool) { return sceneRules(), !tc.off },
					IsBrowser:  func(id string) bool { return id == chrome.BundleID || id == safari.BundleID },
					URL:        u.read,
					URLTimeout: 150 * time.Millisecond,
				}
			})
			h.text.focus = proto.FocusResult{PID: tc.target.PID, Role: "AXTextArea", Editable: true}
			id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: tc.target})
			if err != nil {
				t.Fatal(err)
			}
			// The decision shows in the snapshot while it records.
			rec := h.rec.waitFor(t, "a scene", func(s Snapshot) bool { return s.Scene != nil })
			if !tc.off && rec.State != Recording && u.delay == 0 {
				t.Errorf("scene decided in %s", rec.State)
			}
			h.speak(id, 3)
			end := h.rec.wait(t)
			if end.State != Done || end.Scene == nil {
				t.Fatalf("end %+v", end)
			}
			m := *end.Scene
			if m.GroupID() != tc.group || m.Source != tc.source || m.Host != tc.host {
				t.Errorf("scene %q %q %q", m.GroupID(), m.Source, m.Host)
			}
			if (u.n() > 0) != tc.urlRead {
				t.Errorf("browser.url called %d times", u.n())
			}
			sys := h.llmRequest(t)
			if tc.prompt != "" {
				if !strings.Contains(sys, tc.prompt) || !strings.Contains(sys, tc.app) || !strings.Contains(end.Profile, "scene") {
					t.Errorf("no scene %q/%q in the prompt (%s):\n%s", tc.prompt, tc.app, end.Profile, sys)
				}
			} else if strings.Contains(end.Profile, "scene") || strings.Contains(sys, "unused") {
				t.Errorf("a scene without a group: %s", end.Profile)
			}
			h.text.mu.Lock()
			presses := h.text.presses
			h.text.mu.Unlock()
			if len(presses) != 1 || !reflect.DeepEqual(presses[0], tc.press) {
				t.Errorf("presses %+v, want %+v", presses, tc.press)
			}
			es := h.hist.all()
			if len(es) != 1 {
				t.Fatalf("entries %+v", es)
			}
			e := es[0]
			wantName, wantSource := "", ""
			if tc.group != "" {
				wantName, wantSource = sceneRules().Groups[groupIndex(tc.group)].Name, string(tc.source)
			}
			if e.RuleID != tc.group || e.RuleName != wantName || e.RuleSource != wantSource || e.URLHost != tc.host {
				t.Errorf("entry rule %q %q %q host %q", e.RuleID, e.RuleName, e.RuleSource, e.URLHost)
			}
		})
	}
}

func groupIndex(id string) int {
	for i, g := range sceneRules().Groups {
		if g.ID == id {
			return i
		}
	}
	return -1
}

// llmRequest is the one request's system prompt.
func (h *harness) llmRequest(t *testing.T) string {
	t.Helper()
	reqs := h.llm.(*fakeLLM).requests()
	if len(reqs) != 1 {
		t.Fatalf("%d llm requests", len(reqs))
	}
	return reqs[0].Messages[0].Content
}

// TestSceneSkipped: no app enhancement for a try, for VoxtNew itself,
// or for a feature the group does not apply to.
func TestSceneSkipped(t *testing.T) {
	u := &fakeURL{urls: map[string]string{chrome.BundleID: "https://github.com/"}}
	for _, tc := range []struct {
		name string
		req  StartRequest
	}{
		{"try", StartRequest{Kind: store.KindTranscribe, Target: slack, Try: true}},
		{"voxtnew in front", StartRequest{Kind: store.KindTranscribe, Target: proto.App{PID: 4242, BundleID: "dev.hehehai.voxtnew", Name: "VoxtNew"}}},
		{"no target", StartRequest{Kind: store.KindTranscribe}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeStreamer{final: "hello"}, &fakeLLM{text: "Hello."}, func(c *Config, s *settings.Settings) {
				c.Self = 4242
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
					r := sceneRules()
					r.Groups = append(r.Groups, appctx.Group{ID: "all", Name: "All", Enabled: true, Prompt: "Everywhere.",
						Apps: []appctx.App{{BundleID: "dev.hehehai.voxtnew"}}, Features: appctx.DefaultFeatures})
					return r, true
				}, IsBrowser: func(string) bool { return true }, URL: u.read}
			})
			id, err := h.m.Start(tc.req)
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			end := h.rec.wait(t)
			if end.Scene != nil && end.Scene.Matched() || strings.Contains(h.llmRequest(t), "Everywhere.") || strings.Contains(h.llmRequest(t), "Short and casual.") {
				t.Errorf("scene %+v", end.Scene)
			}
			if u.n() != 0 {
				t.Errorf("browser.url called")
			}
		})
	}
}

// TestSceneLLMOff: without a language model the auto key still presses,
// and it waits for the address when it is slow.
func TestSceneLLMOff(t *testing.T) {
	r := sceneRules()
	r.Groups[1].AutoKey = "Command+Return"
	u := &fakeURL{urls: map[string]string{chrome.BundleID: "https://linear.app/"}, delay: 80 * time.Millisecond}
	h := newHarness(t, &fakeFile{text: "hello"}, nil, func(c *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) { return r, true },
			IsBrowser: func(id string) bool { return id == chrome.BundleID }, URL: u.read, URLTimeout: time.Second}
	})
	h.text.focus = proto.FocusResult{PID: chrome.PID, Role: "AXTextArea", Editable: true}
	id, _ := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: chrome})
	h.speak(id, 4)
	end := h.rec.wait(t)
	h.text.mu.Lock()
	presses := h.text.presses
	h.text.mu.Unlock()
	want := []proto.KeyPress{{Key: 36, Modifiers: []string{proto.ModCommand}}}
	if end.Scene == nil || end.Scene.GroupID() != "g-browser" || len(presses) != 1 || !reflect.DeepEqual(presses[0], want) {
		t.Errorf("scene %+v presses %+v", end.Scene, presses)
	}
	if e := h.hist.all(); len(e) != 1 || e[0].URLHost != "linear.app" || e[0].RuleSource != "browserApp" {
		t.Errorf("entry %+v", e)
	}
}

// TestSceneAnswerCard: a result left in the answer card presses nothing.
func TestSceneAnswerCard(t *testing.T) {
	h := newHarness(t, &fakeStreamer{final: "hello"}, &fakeLLM{text: "Hello."}, func(c *Config, s *settings.Settings) {
		c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) { return sceneRules(), true }}
	})
	h.text.focus = proto.FocusResult{PID: slack.PID, Role: "AXButton"}
	id := h.start()
	h.speak(id, 3)
	end := h.rec.wait(t)
	if !end.Card || len(h.text.pastes()) != 0 || end.Scene.GroupID() != "g-chat" {
		t.Errorf("end %+v", end)
	}
}

// TestSceneWrite: Translate and Rewrite take the scene when the group
// applies to them, and press its key only after writing new text, never
// after a translation or a replaced selection.
func TestSceneWrite(t *testing.T) {
	for _, tc := range []struct {
		name     string
		kind     Kind
		sel      string
		features appctx.Features
		answer   string
		scene    bool
		press    bool
	}{
		{"translate, group off for it", store.KindTranslate, "Good morning", appctx.DefaultFeatures, "早上好", false, false},
		{"translate, group on for it", store.KindTranslate, "Good morning", appctx.Features{Translate: true}, "早上好", true, false},
		{"translate speech, group on", store.KindTranslate, "", appctx.Features{Translate: true}, "你好", true, false},
		{"rewrite a selection", store.KindRewrite, "hey team", appctx.DefaultFeatures, "Hello team.", true, false},
		{"rewrite from scratch", store.KindRewrite, "", appctx.DefaultFeatures, `{"title":"Hi","content":"Hello team."}`, true, true},
		{"rewrite, group off for it", store.KindRewrite, "", appctx.Features{Transcribe: true}, `{"title":"Hi","content":"Hello team."}`, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rules := appctx.Rules{Groups: []appctx.Group{{ID: "g", Name: "Chat", Enabled: true, Prompt: "Short and casual.",
				Apps: []appctx.App{{BundleID: slack.BundleID}}, AutoKey: "Return", Features: tc.features}}}
			model := &fakeLLM{text: tc.answer}
			h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: tc.sel}
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) { return rules, true }}
			})
			id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			if tc.sel == "" || tc.kind == store.KindRewrite { // Rewrite always hears its instruction
				h.rec.waitFor(t, "recording", func(s Snapshot) bool { return s.State == Recording })
				h.speak(id, 3)
			}
			end := h.rec.wait(t)
			// Rewrite answers wait in the card (Voxt's); the rest is inserted.
			if end.State != Done || end.Inserted == (tc.kind == store.KindRewrite) || end.Card != (tc.kind == store.KindRewrite) {
				t.Fatalf("end %+v", end)
			}
			sys := h.llmRequest(t)
			if strings.Contains(sys, "Short and casual.") != tc.scene || strings.Contains(end.Profile, "scene") != tc.scene {
				t.Errorf("scene in the prompt: %v (%s)\n%s", !tc.scene, end.Profile, sys)
			}
			h.text.mu.Lock()
			presses := h.text.presses
			h.text.mu.Unlock()
			switch {
			case tc.kind == store.KindRewrite:
				if len(presses) != 0 { // nothing pasted: the card inserts when asked
					t.Errorf("presses %+v", presses)
				}
			case len(presses) != 1 || (presses[0] != nil) != tc.press:
				t.Errorf("presses %+v", presses)
			}
		})
	}
}
