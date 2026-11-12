package session

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestSceneTranscribeCleanup(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		global, want         bool
		mode, browserMode    appctx.Cleanup
		browser, disabled    bool
		featureOff, rulesOff bool
		url                  string
		timeout, note, try   bool
	}{
		{name: "default on", global: true, want: true},
		{name: "default off"},
		{name: "on overrides global off", mode: appctx.CleanupOn, want: true},
		{name: "off overrides global on", global: true, mode: appctx.CleanupOff},
		{name: "invalid inherits", global: true, want: true, mode: "invalid"},
		{name: "disabled group inherits", global: true, want: true, mode: appctx.CleanupOff, disabled: true},
		{name: "feature disabled inherits", global: true, want: true, mode: appctx.CleanupOff, featureOff: true},
		{name: "matching disabled inherits", global: true, want: true, mode: appctx.CleanupOff, rulesOff: true},
		{name: "URL off beats browser on", global: true, mode: appctx.CleanupOff, browserMode: appctx.CleanupOn, browser: true, url: "https://example.com/chat"},
		{name: "URL on beats browser off", mode: appctx.CleanupOn, browserMode: appctx.CleanupOff, browser: true, url: "https://example.com/chat", want: true},
		{name: "URL default beats browser off", global: true, browserMode: appctx.CleanupOff, browser: true, url: "https://example.com/chat", want: true},
		{name: "browser fallback off", global: true, browserMode: appctx.CleanupOff, browser: true, url: "https://other.example/"},
		{name: "URL timeout uses browser off", global: true, mode: appctx.CleanupOn, browserMode: appctx.CleanupOff, browser: true, url: "https://example.com/", timeout: true},
		{name: "note follows global on", global: true, want: true, mode: appctx.CleanupOff, note: true},
		{name: "note follows global off", mode: appctx.CleanupOn, note: true},
		{name: "try follows global on", global: true, want: true, mode: appctx.CleanupOff, try: true},
		{name: "try follows global off", mode: appctx.CleanupOn, try: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := slack
			if tc.browser {
				target = chrome
			}
			g := appctx.Group{ID: "g", Enabled: !tc.disabled, Prompt: "Keep the scene prompt.", TranscribeCleanup: tc.mode,
				Apps: []appctx.App{{BundleID: target.BundleID}}, Features: appctx.Features{Transcribe: !tc.featureOff}}
			rules := appctx.Rules{Groups: []appctx.Group{g}}
			if tc.browser {
				rules.Groups[0].Apps, rules.Groups[0].URLs = nil, []string{"example.com"}
				rules.Groups = append(rules.Groups, appctx.Group{ID: "browser", Enabled: true, Apps: []appctx.App{{BundleID: target.BundleID}}, Features: appctx.DefaultFeatures, TranscribeCleanup: tc.browserMode})
			}
			u := &fakeURL{urls: map[string]string{target.BundleID: tc.url}}
			if tc.timeout {
				u.delay = time.Second
			}
			model := &fakeLLM{text: "Cleaned text."}
			resolved := make(chan bool, 1)
			h := newHarness(t, &fakeFile{text: "hello aura link"}, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = tc.global
				c.Notes = &memNotes{}
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) { return rules, !tc.rulesOff },
					IsBrowser: func(string) bool { return tc.browser }, URL: u.read, URLTimeout: 30 * time.Millisecond}
				base := c.Resolve
				c.Resolve = func(ctx context.Context, k Kind, effective settings.Settings) (Services, error) {
					resolved <- effective.Features.Transcribe.Enhance
					svc, err := base(ctx, k, effective)
					if !effective.Features.Transcribe.Enhance {
						svc.LLM, svc.LLMInfo = nil, Model{}
					}
					return svc, err
				}
			})
			h.text.focus.PID = target.PID
			id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: target, Note: tc.note, Try: tc.try})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			end := h.rec.wait(t)
			if end.State != Done {
				t.Fatalf("end %+v", end)
			}
			if got := <-resolved; got != tc.want {
				t.Errorf("resolver cleanup %v, want %v", got, tc.want)
			}
			wantText := "hello VoxtNew" // dictionary replacement still applies without cleanup
			if tc.want {
				wantText = "Cleaned text."
			}
			if end.Result != wantText || (len(model.requests()) > 0) != tc.want {
				t.Errorf("result %q, chats %d", end.Result, len(model.requests()))
			}
			if h.m.cfg.Settings().Features.Transcribe.Enhance != tc.global {
				t.Error("the session changed the global setting")
			}
			if tc.browser && tc.timeout && (end.Scene == nil || end.Scene.GroupID() != "browser") {
				t.Errorf("timeout scene %+v", end.Scene)
			}
		})
	}
}

func TestSceneCleanupDoesNotDelayRecording(t *testing.T) {
	for _, mode := range []appctx.Cleanup{appctx.CleanupDefault, appctx.CleanupOn} {
		t.Run(string(mode), func(t *testing.T) {
			release, resolved := make(chan struct{}), make(chan struct{}, 1)
			h := newHarness(t, &fakeFile{text: "hello"}, nil, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
					return appctx.Rules{Groups: []appctx.Group{{ID: "url", Enabled: true, URLs: []string{"example.com"}, Features: appctx.DefaultFeatures, TranscribeCleanup: mode}}}, true
				}, IsBrowser: func(string) bool { return true }, URLTimeout: 5 * time.Second,
					URL: func(ctx context.Context, _ proto.App) (string, error) {
						select {
						case <-release:
							return "https://example.com/", nil
						case <-ctx.Done():
							return "", ctx.Err()
						}
					}}
				base := c.Resolve
				c.Resolve = func(ctx context.Context, k Kind, s settings.Settings) (Services, error) {
					resolved <- struct{}{}
					return base(ctx, k, s)
				}
			})
			id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: chrome})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3) // Audio is captured and Stop works while the URL is blocked.
			if mode == appctx.CleanupDefault {
				select {
				case <-resolved:
				case <-time.After(time.Second):
					t.Error("default-only groups delayed model resolution")
				}
			} else {
				select {
				case <-resolved:
					t.Error("model resolved before the effective override was known")
				default:
				}
			}
			close(release)
			if end := h.rec.wait(t); end.State != Done {
				t.Errorf("end %+v", end)
			}
		})
	}
}

func TestSceneCleanupCancelBeforeURL(t *testing.T) {
	var resolves atomic.Int32
	urlEnded := make(chan struct{})
	h := newHarness(t, &fakeFile{text: "hello"}, &fakeLLM{text: "Cleaned."}, func(c *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
			return appctx.Rules{Groups: []appctx.Group{{ID: "url", Enabled: true, URLs: []string{"example.com"}, Features: appctx.DefaultFeatures, TranscribeCleanup: appctx.CleanupOff}}}, true
		}, IsBrowser: func(string) bool { return true }, URLTimeout: 5 * time.Second,
			URL: func(ctx context.Context, _ proto.App) (string, error) {
				<-ctx.Done()
				close(urlEnded)
				return "", ctx.Err()
			}}
		base := c.Resolve
		c.Resolve = func(ctx context.Context, k Kind, s settings.Settings) (Services, error) {
			resolves.Add(1)
			return base(ctx, k, s)
		}
	})
	id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: chrome})
	if err != nil {
		t.Fatal(err)
	}
	h.rec.waitFor(t, "audio capture", func(s Snapshot) bool { return !s.HeardAt.IsZero() })
	h.m.Cancel(id)
	if end := h.rec.wait(t); end.State != Cancelled {
		t.Errorf("end %+v", end)
	}
	select {
	case <-urlEnded:
	case <-time.After(time.Second):
		t.Fatal("URL lookup did not stop on cancellation")
	}
	if resolves.Load() != 0 || len(h.text.pastes()) != 0 {
		t.Errorf("cancelled session resolved %d models or inserted text", resolves.Load())
	}
}

func TestSceneCleanupOffDoesNotWarmOrChat(t *testing.T) {
	model := &prefillLLM{fakeLLM: &fakeLLM{text: "Cleaned."}}
	h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
			r := sceneRules()
			r.Groups[2].TranscribeCleanup = appctx.CleanupOff
			return r, true
		}}
	}) // This fake resolver deliberately returns a model even when cleanup is off.
	id := h.start()
	h.speak(id, 3)
	end := h.rec.wait(t)
	model.mu.Lock()
	warmed := len(model.prefills)
	model.mu.Unlock()
	if end.Result != "hello" || len(model.requests()) != 0 || warmed != 0 {
		t.Errorf("result %q, chats %d, prefills %d", end.Result, len(model.requests()), warmed)
	}
}

func TestSceneCleanupLeavesTranslateAndRewrite(t *testing.T) {
	for _, kind := range []Kind{store.KindTranslate, store.KindRewrite} {
		t.Run(string(kind), func(t *testing.T) {
			answer := `{"title":"Hello","content":"Edited text."}`
			if kind == store.KindTranslate {
				answer = "你好。"
			}
			model := &fakeLLM{text: answer}
			h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = false
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
					return appctx.Rules{Groups: []appctx.Group{{ID: "g", Enabled: true, Apps: []appctx.App{{BundleID: slack.BundleID}}, Prompt: "Keep this scene.",
						Features: appctx.Features{Transcribe: true, Translate: true, Rewrite: true}, TranscribeCleanup: appctx.CleanupOff}}}, true
				}}
			})
			id, err := h.m.Start(StartRequest{Kind: kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			end := h.rec.wait(t)
			if end.State != Done || !strings.Contains(h.llmRequest(t), "Keep this scene.") {
				t.Errorf("end %+v", end)
			}
		})
	}
}

func TestSceneCleanupRetry(t *testing.T) {
	for _, on := range []bool{false, true} {
		t.Run(map[bool]string{false: "off", true: "on"}[on], func(t *testing.T) {
			model := &fakeLLM{text: "Cleaned retry."}
			h, entry := retryHarness(t, &fakeFile{text: "hello"}, model, true)
			s := h.m.cfg.Settings()
			s.Features.Transcribe.Enhance = !on
			h.m.cfg.Settings = func() settings.Settings { return s }
			h.m.cfg.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
				r := sceneRules()
				r.Groups[2].TranscribeCleanup = appctx.CleanupOff
				if on {
					r.Groups[2].TranscribeCleanup = appctx.CleanupOn
				}
				return r, true
			}}
			base := h.m.cfg.Resolve
			h.m.cfg.Resolve = func(ctx context.Context, kind Kind, effective settings.Settings) (Services, error) {
				if effective.Features.Transcribe.Enhance != on {
					t.Error("retry resolver did not receive its app override")
				}
				svc, err := base(ctx, kind, effective)
				if !effective.Features.Transcribe.Enhance {
					svc.LLM = nil
				}
				return svc, err
			}
			out, err := h.m.Retry(context.Background(), RetryRequest{Entry: entry})
			want := "hello"
			if on {
				want = "Cleaned retry."
			}
			if err != nil || out.Result != want || (len(model.requests()) > 0) != on {
				t.Errorf("retry %+v, %v, chats %d", out, err, len(model.requests()))
			}
			if h.m.cfg.Settings().Features.Transcribe.Enhance != !on {
				t.Error("retry changed global cleanup")
			}
		})
	}
}

func TestSceneCleanupRetryKeepsURLRule(t *testing.T) {
	for _, tc := range []struct {
		name, state string
		want        bool
	}{
		{name: "recorded path rule"},
		{name: "current group changed to on", state: "on", want: true},
		{name: "disabled group falls back", state: "disabled", want: true},
		{name: "deleted group falls back", state: "deleted", want: true},
		{name: "inapplicable group falls back", state: "inapplicable", want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "Cleaned retry."}
			h, entry := retryHarness(t, &fakeFile{text: "hello"}, model, true)
			entry.AppBundle, entry.AppName = chrome.BundleID, chrome.Name
			entry.URLHost, entry.RuleID, entry.RuleSource = "example.com", "private", string(appctx.SourceURL)
			private := appctx.Group{ID: "private", Enabled: true, URLs: []string{"example.com/private/*"}, Features: appctx.DefaultFeatures, TranscribeCleanup: appctx.CleanupOff}
			switch tc.state {
			case "on":
				private.TranscribeCleanup = appctx.CleanupOn
			case "disabled":
				private.Enabled = false
			case "inapplicable":
				private.Features.Transcribe = false
			}
			rules := appctx.Rules{Groups: []appctx.Group{{ID: "browser", Enabled: true, Apps: []appctx.App{{BundleID: chrome.BundleID}}, Features: appctx.DefaultFeatures, TranscribeCleanup: appctx.CleanupOn}}}
			if tc.state != "deleted" {
				rules.Groups = append(rules.Groups, private)
			}
			h.m.cfg.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) { return rules, true }}
			base := h.m.cfg.Resolve
			h.m.cfg.Resolve = func(ctx context.Context, kind Kind, effective settings.Settings) (Services, error) {
				if effective.Features.Transcribe.Enhance != tc.want {
					t.Errorf("retry resolver cleanup %v, want %v", effective.Features.Transcribe.Enhance, tc.want)
				}
				svc, err := base(ctx, kind, effective)
				if !effective.Features.Transcribe.Enhance {
					svc.LLM = nil
				}
				return svc, err
			}
			out, err := h.m.Retry(context.Background(), RetryRequest{Entry: entry})
			want := "hello"
			if tc.want {
				want = "Cleaned retry."
			}
			if err != nil || out.Result != want || (len(model.requests()) > 0) != tc.want {
				t.Errorf("retry %+v, %v, chats %d", out, err, len(model.requests()))
			}
		})
	}
}

func TestRetrySceneCleanupScope(t *testing.T) {
	for _, tc := range []struct {
		name           string
		req            StartRequest
		mode, fallback appctx.Cleanup
		want           string
	}{
		{"Transcribe override", StartRequest{Kind: store.KindTranscribe}, appctx.CleanupOff, appctx.CleanupOn, "private"},
		{"Transcribe default over browser off", StartRequest{Kind: store.KindTranscribe}, appctx.CleanupDefault, appctx.CleanupOff, "private"},
		{"default-only groups", StartRequest{Kind: store.KindTranscribe}, appctx.CleanupDefault, appctx.CleanupDefault, "browser"},
		{"same cleanup choice", StartRequest{Kind: store.KindTranscribe}, appctx.CleanupOn, appctx.CleanupOn, "browser"},
		{"Translate unchanged", StartRequest{Kind: store.KindTranslate}, appctx.CleanupOff, appctx.CleanupOn, "browser"},
		{"Rewrite unchanged", StartRequest{Kind: store.KindRewrite}, appctx.CleanupOff, appctx.CleanupOn, "browser"},
		{"Note unchanged", StartRequest{Kind: store.KindTranscribe, Note: true}, appctx.CleanupOff, appctx.CleanupOn, "browser"},
		{"Try unchanged", StartRequest{Kind: store.KindTranscribe, Try: true}, appctx.CleanupOff, appctx.CleanupOn, "browser"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, nil, nil, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = true
				features := appctx.Features{Transcribe: true, Translate: true, Rewrite: true}
				c.Scene = &SceneConfig{Rules: func() (appctx.Rules, bool) {
					return appctx.Rules{Groups: []appctx.Group{
						{ID: "private", Enabled: true, URLs: []string{"example.com/private/*"}, Features: features, TranscribeCleanup: tc.mode},
						{ID: "browser", Enabled: true, Apps: []appctx.App{{BundleID: chrome.BundleID}}, Features: features, TranscribeCleanup: tc.fallback},
					}}, true
				}}
			})
			r := &run{m: h.m, req: tc.req, settings: h.m.cfg.Settings(), ctx: context.Background()}
			r.retryScene(store.Entry{Kind: tc.req.Kind, AppBundle: chrome.BundleID, URLHost: "example.com", RuleID: "private", RuleSource: string(appctx.SourceURL)})
			if match := r.scene(); match.GroupID() != tc.want {
				t.Errorf("retry group %q, want %q", match.GroupID(), tc.want)
			}
		})
	}
}
