package main

// App enhancement as the app holds it: the groups
// (internal/appctx), kept in the store's document appctx:rules and in a
// rule book sessions read from their goroutines; the reads of browser
// addresses (the helper's browser.url, Accessibility only), what they
// told about each browser; and the live preview of the App enhancement
// page (the app in front, its address, the group it falls in).

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// selfBundleID is VoxtNew's bundle id (mygo.json).
const selfBundleID = "dev.hehehai.voxtnew"

// urlTimeout is how long a session waits for a browser's address.
const urlTimeout = 800 * time.Millisecond

// ruleBook is the groups as sessions read them: a copy, swapped whole.
// Goroutine-safe.
type ruleBook struct {
	mu    sync.Mutex
	rules appctx.Rules
	// saveMu orders the writes to the store: each writes the newest rules.
	saveMu sync.Mutex
}

func (b *ruleBook) get() appctx.Rules {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rules
}

func (b *ruleBook) set(r appctx.Rules) {
	b.mu.Lock()
	b.rules = cloneRules(r)
	b.mu.Unlock()
}

// save writes the newest rules to docs.
func (b *ruleBook) save(ctx context.Context, docs store.Documents) error {
	b.saveMu.Lock()
	defer b.saveMu.Unlock()
	data, err := b.get().Marshal()
	if err != nil {
		return err
	}
	return docs.Put(ctx, appctx.Key, data)
}

// cloneRules copies the groups and their lists.
func cloneRules(r appctx.Rules) appctx.Rules {
	out := appctx.Rules{Version: r.Version}
	for _, g := range r.Groups {
		g.Apps = append([]appctx.App(nil), g.Apps...)
		g.URLs = append([]string(nil), g.URLs...)
		out.Groups = append(out.Groups, g)
	}
	return out
}

// urlRead is what the last read of a browser's address at a shortcut
// told: whether it gave one, and when.
type urlRead struct {
	ok  bool
	err string
	at  time.Time
}

// aePreview is the App enhancement page's preview: an address read of
// the app in front (or the last app before VoxtNew).
type aePreview struct {
	app     proto.App
	reading bool
	read    bool // a read finished for app
	url     string
	err     string
	seq     int
}

// appEnhanceState is App enhancement's state.
type appEnhanceState struct {
	aeBook  *ruleBook
	aeRules appctx.Rules // the page's copy (main thread)
	aeErr   string       // the document could not be read
	// lastOther is the last app in front that is not VoxtNew: the
	// preview's app while VoxtNew's window is in front.
	lastOther proto.App
	// urlReads are the last address read of each browser at a shortcut.
	urlReads  map[string]urlRead
	aePrev    aePreview
	aeSaveSeq int
	aePage    aePageState // appenhance_page.go
	// treesSent is the app.configure last sent (its bundle ids), so the
	// same list is not sent twice; "" after a helper (re)start.
	treesSent string
}

// openAppEnhancement reads the groups (once, while the app starts).
func (a *app) openAppEnhancement() {
	a.aeBook = &ruleBook{}
	a.urlReads = map[string]urlRead{}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	data, ok, err := a.lib.store.Documents().Get(ctx, appctx.Key)
	r := appctx.Rules{Version: appctx.Version}
	if err == nil && ok {
		r, err = appctx.Parse(data)
	}
	if err != nil {
		log.Printf("app enhancement: %v", err)
		a.aeErr = err.Error()
	}
	a.aeRules = r
	a.aeBook.set(r)
}

// setRules changes the groups: the page's copy and the sessions' at once,
// the store after a moment (typing in a field saves once it pauses).
func (a *app) setRules(r appctx.Rules) {
	a.aeRules = r.Normalized()
	a.aeRules.Version = appctx.Version
	a.aeBook.set(a.aeRules)
	a.aeSaveSeq++
	seq := a.aeSaveSeq
	save := func() {
		book, docs := a.aeBook, a.lib.store.Documents()
		a.async(func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := book.save(ctx, docs); err != nil {
				a.post(func() { a.toast(errText(err)) })
			}
		})
	}
	if a.win == nil {
		save()
		return
	}
	time.AfterFunc(400*time.Millisecond, func() {
		a.post(func() {
			if a.aeSaveSeq == seq {
				save()
			}
		})
	})
}

// sceneConfig is app enhancement for the session manager: the rule book,
// the settings' switch and browsers, and browser.url through the helper.
func (a *app) sceneConfig() *session.SceneConfig {
	book, st, sup := a.aeBook, a.settings, a.host.sup
	return &session.SceneConfig{
		Rules: func() (appctx.Rules, bool) { return book.get(), st.Get().AppEnhancement.Enabled },
		IsBrowser: func(id string) bool {
			return appctx.IsBrowser(id, st.Get().AppEnhancement.Browsers)
		},
		URL: func(ctx context.Context, app proto.App) (string, error) {
			res, err := readURL(ctx, sup, app)
			read := urlRead{ok: err == nil && res.URL != "", err: res.Error, at: time.Now()}
			if err != nil {
				read.err = err.Error()
			}
			id := app.BundleID
			a.post(func() { a.urlReads[id] = read })
			if err == nil && res.URL == "" && res.Error != "" {
				err = errors.New(res.Error)
			}
			return res.URL, err
		},
		URLTimeout: urlTimeout,
	}
}

// readURL is browser.url for app.
func readURL(ctx context.Context, sup interface {
	Call(context.Context, string, string, any, func(proto.Message)) (json.RawMessage, error)
}, app proto.App) (proto.BrowserURLResult, error) {
	var out proto.BrowserURLResult
	raw, err := sup.Call(ctx, "", proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: app.BundleID, PID: app.PID}, nil)
	if err == nil {
		err = json.Unmarshal(raw, &out)
	}
	return out, err
}

// syncTrees tells the helper which browsers need Chromium's accessibility
// tree turned on (app.configure): the Chromium ones while app
// enhancement is on, so browser.url finds the tab's web area; none while
// it is off (a tree costs the browser memory and time).
func (a *app) syncTrees() {
	if !a.helperReady() {
		a.treesSent = ""
		return
	}
	var ids []string
	if a.prefs.AppEnhancement.Enabled {
		ids = appctx.TreeOnDemand(a.prefs.AppEnhancement.Browsers)
	}
	key := "[" + strings.Join(ids, ",") + "]"
	if key == a.treesSent {
		return
	}
	a.treesSent = key
	helperCall(a, proto.MethodAppConfigure, proto.AppConfigureParams{ManualAccessibility: ids}, 5*time.Second,
		func(_ proto.Empty, err error) {
			if err != nil {
				a.treesSent = ""
				a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityWarning, Source: helper.LogSource, Message: "app.configure: " + err.Error()})
			}
		})
}

// isSelf reports whether app is VoxtNew.
func isSelf(app proto.App) bool {
	return app.PID != 0 && app.PID == os.Getpid() || strings.EqualFold(app.BundleID, selfBundleID)
}

// frontChanged follows app.activated (helperlink.go): the last app that
// is not VoxtNew, and the preview while the page shows.
func (a *app) frontChanged(front proto.App) {
	if isSelf(front) || front.PID == 0 {
		return
	}
	a.lastOther = front
	if a.currentPage() == "app-enhancement" {
		a.refreshPreview(false)
	}
}

// previewApp is the app the preview shows: the app in front, or the last
// one before VoxtNew.
func (a *app) previewApp() proto.App {
	if a.front.PID != 0 && !isSelf(a.front) {
		return a.front
	}
	return a.lastOther
}

// refreshPreview reads the preview app's address (a browser's), unless
// it was read already (force: again).
func (a *app) refreshPreview(force bool) {
	app := a.previewApp()
	p := &a.aePrev
	if !force && p.app == app && (p.read || p.reading) {
		return
	}
	p.seq++
	*p = aePreview{app: app, seq: p.seq}
	if app.BundleID == "" || !appctx.IsBrowser(app.BundleID, a.prefs.AppEnhancement.Browsers) || !a.helperReady() {
		p.read = true
		return
	}
	p.reading = true
	seq := p.seq
	helperCall(a, proto.MethodBrowserURL, proto.BrowserURLParams{BundleID: app.BundleID, PID: app.PID}, 2*time.Second,
		func(res proto.BrowserURLResult, err error) {
			if a.aePrev.seq != seq {
				return
			}
			a.aePrev.reading, a.aePrev.read = false, true
			a.aePrev.url, a.aePrev.err = res.URL, res.Error
			if err != nil {
				a.aePrev.err = err.Error()
			}
		})
}

// previewMatch is the group the preview app falls in, for Transcribe.
func (a *app) previewMatch() appctx.Match {
	app := a.previewApp()
	in := appctx.Context{App: app, Browser: appctx.IsBrowser(app.BundleID, a.prefs.AppEnhancement.Browsers)}
	if in.Browser && a.aePrev.app == app {
		in.URL = a.aePrev.url
	}
	return appctx.Resolve(a.aeRules, in, appctx.Transcribe)
}

// newGroupID makes a group's id.
func newGroupID() string { return "g-" + strings.ToLower(core.NewID(time.Now())) }
