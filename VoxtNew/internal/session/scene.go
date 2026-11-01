package session

// App enhancement in a session: which group of internal/appctx the
// session falls in, decided from the app in front when the shortcut was
// pressed (StartRequest.Target) and, for a browser, the address of its
// front tab, read in the background from the start (the helper's
// browser.url, Accessibility only) so that it is there when the
// composition needs it. The matched group gives the composer's Scene
// layer, the key pressed after the insert, and the history's rule; the
// snapshot carries it for the overlay's label.

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// SceneConfig is app enhancement's side of the Manager. Rules and URL are
// required.
type SceneConfig struct {
	// Rules returns the groups and whether app enhancement is on (memory
	// only: it runs as the session starts).
	Rules func() (appctx.Rules, bool)
	// IsBrowser reports whether an app is a browser (nil: none is).
	IsBrowser func(bundleID string) bool
	// URL reads a browser's front tab's address ("" for none).
	URL func(ctx context.Context, app proto.App) (string, error)
	// URLTimeout bounds URL (0: DefaultURLTimeout); without an address in
	// time the browser is matched as an app.
	URLTimeout time.Duration
}

// DefaultURLTimeout is how long a session waits for a browser's address.
const DefaultURLTimeout = time.Second

// sceneState is a run's app enhancement: the match once it is known.
type sceneState struct {
	once     sync.Once
	done     chan struct{}
	mu       sync.Mutex
	match    appctx.Match
	deadline time.Time
	fallback appctx.Match // without the address: the browser as an app
}

// startScene starts deciding the session's group (Manager.Start, before
// the session's goroutine starts).
func (r *run) startScene() {
	sc := &r.sc
	sc.done = make(chan struct{})
	cfg := r.m.cfg.Scene
	target := r.req.Target
	global := appctx.Match{Source: appctx.SourceGlobal, App: target}
	if cfg == nil || cfg.Rules == nil || r.req.Try || target.BundleID == "" || (r.m.cfg.Self != 0 && target.PID == r.m.cfg.Self) {
		r.setScene(global, false)
		return
	}
	rules, on := cfg.Rules()
	if !on {
		r.setScene(global, true)
		return
	}
	feature := string(r.req.Kind)
	in := appctx.Context{App: target}
	if cfg.IsBrowser == nil || !cfg.IsBrowser(target.BundleID) || cfg.URL == nil {
		r.setScene(appctx.Resolve(rules, in, feature), true)
		return
	}
	in.Browser = true
	timeout := cfg.URLTimeout
	if timeout <= 0 {
		timeout = DefaultURLTimeout
	}
	sc.mu.Lock()
	sc.fallback = appctx.Resolve(rules, in, feature)
	sc.deadline = r.m.cfg.Now().Add(timeout)
	sc.mu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(r.ctx, timeout)
		defer cancel()
		got := make(chan string, 1)
		start := time.Now()
		go func() {
			u, err := cfg.URL(ctx, target)
			if err != nil {
				r.logf("info", "no address of %s: %v", target.BundleID, err)
			}
			got <- u
		}()
		select {
		case u := <-got:
			in.URL = u
			m := appctx.Resolve(rules, in, feature)
			r.logf("info", "address of %s read in %v (%s)", target.BundleID, time.Since(start).Round(time.Millisecond), orNone(m.Host))
			r.setScene(m, true)
		case <-ctx.Done():
			r.logf("info", "no address of %s within %v: matched as an app", target.BundleID, timeout)
			r.setScene(sc.fallback, true)
		}
	}()
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

// setScene keeps the first decision and shows it in the snapshot.
func (r *run) setScene(m appctx.Match, publish bool) {
	sc := &r.sc
	sc.once.Do(func() {
		sc.mu.Lock()
		sc.match = m
		sc.mu.Unlock()
		close(sc.done)
		if m.Matched() {
			r.logf("info", "app enhancement: %s by %s", m.GroupName(), m.Source)
		}
		if publish {
			c := cloneMatch(m)
			r.m.update(r, func(s *Snapshot) { s.Scene = &c })
		}
	})
}

// scene is the session's match, waiting for it until the address's
// deadline at most (then the browser counts as an app).
func (r *run) scene() appctx.Match {
	sc := &r.sc
	select {
	case <-sc.done:
	default:
		sc.mu.Lock()
		wait := sc.deadline.Sub(r.m.cfg.Now())
		sc.mu.Unlock()
		select {
		case <-sc.done:
		case <-time.After(max(wait, 0)):
			sc.mu.Lock()
			fb := sc.fallback
			sc.mu.Unlock()
			r.setScene(fb, true)
		case <-r.ctx.Done():
			return appctx.Match{Source: appctx.SourceGlobal, App: r.req.Target}
		}
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.match
}

// knownScene is the match if it is decided, without waiting.
func (r *run) knownScene() (appctx.Match, bool) {
	select {
	case <-r.sc.done:
		r.sc.mu.Lock()
		defer r.sc.mu.Unlock()
		return r.sc.match, true
	default:
		return appctx.Match{}, false
	}
}

// composeScene is the composer's Scene layer: the app and the group's
// prompt, nil when no group matched.
func (r *run) composeScene() *compose.Scene {
	m := r.scene()
	if !m.Matched() {
		return nil
	}
	name := strings.TrimSpace(r.req.Target.Name)
	if name == "" {
		for _, a := range m.Group.Apps {
			if strings.EqualFold(a.BundleID, r.req.Target.BundleID) {
				name = a.Name
			}
		}
	}
	if name == "" && m.Host != "" {
		name = m.Host
	}
	if name == "" {
		name = r.req.Target.BundleID
	}
	return &compose.Scene{AppName: name, Prompt: m.Group.Prompt}
}

// press are the keys the matched group presses after an insert: after a
// transcription, and after a rewrite that wrote new text where the cursor
// is; never after a translation or a rewrite that replaced a selection
// (replace), where a Return would act on text the user did not write.
func (r *run) press(replace bool) []proto.KeyPress {
	switch {
	case r.req.Kind == store.KindTranscribe:
	case r.req.Kind == store.KindRewrite && !replace:
	default:
		return nil
	}
	m := r.scene()
	if !m.Matched() {
		return nil
	}
	keys, err := appctx.ParseAutoKey(m.Group.AutoKey)
	if err != nil {
		r.logf("warning", "auto key %q of %s: %v", m.Group.AutoKey, m.GroupName(), err)
		return nil
	}
	return keys
}

func cloneMatch(m appctx.Match) appctx.Match {
	if m.Group != nil {
		g := *m.Group
		m.Group = &g
	}
	return m
}
