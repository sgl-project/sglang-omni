package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// ghServer is GitHub's releases API: list (JSON) with etag, or status
// instead when it is not 200, or no answer for hang. A request with the
// etag in If-None-Match gets 304. It keeps every request.
type ghServer struct {
	mu       sync.Mutex
	list     string
	etag     string
	status   int
	header   http.Header
	hang     bool
	requests []*http.Request
}

func (s *ghServer) start(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.requests = append(s.requests, r)
		list, etag, status, header, hang := s.list, s.etag, s.status, s.header, s.hang
		s.mu.Unlock()
		for k, v := range header {
			w.Header()[k] = v
		}
		switch {
		case hang:
			<-r.Context().Done()
		case status != 0:
			w.WriteHeader(status)
		case etag != "" && r.Header.Get("If-None-Match") == etag:
			w.WriteHeader(http.StatusNotModified)
		default:
			if etag != "" {
				w.Header().Set("ETag", etag)
			}
			fmt.Fprint(w, list)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/repos/sgl-project/sglang-omni/releases?per_page=30"
}

func (s *ghServer) set(f func(s *ghServer)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f(s)
}

func (s *ghServer) last() *http.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests[len(s.requests)-1]
}

// ghList is a releases answer: tag, then "draft", "pre" or "nodmg" flags.
func ghList(rels ...[]string) string {
	var out []map[string]any
	for _, r := range rels {
		tag := r[0]
		v := strings.TrimPrefix(tag, "voxt-v")
		m := map[string]any{"tag_name": tag, "draft": slices.Contains(r, "draft"), "prerelease": slices.Contains(r, "pre"),
			"body": "  ## What's new in " + v + "\n\n- Faster.  \n", "html_url": "https://github.com/sgl-project/sglang-omni/releases/tag/" + tag,
			"assets": []map[string]string{{"name": "Voxt-" + v + "-macOS.zip", "browser_download_url": "https://dl/zip-" + v}}}
		if !slices.Contains(r, "nodmg") {
			m["assets"] = append(m["assets"].([]map[string]string), map[string]string{"name": "Voxt-" + v + "-arm64.dmg", "browser_download_url": "https://dl/dmg-" + v})
		}
		out = append(out, m)
	}
	b, _ := json.Marshal(out)
	return string(b)
}

// The releases of today (2.0.3 the newest) and one later.
var (
	todayReleases = ghList([]string{"voxt-v2.0.3"}, []string{"voxt-v2.0.1"}, []string{"v0.2.0"}, []string{"voxt-v1.16.3"})
	newerReleases = ghList([]string{"voxt-v2.0.10-beta.1", "pre"}, []string{"voxt-v3.0.0", "draft"}, []string{"voxt-v2.0.4"},
		[]string{"voxt-v2.0.3"}, []string{"v9.9.9"}, []string{"voxt-v2.0.10-beta.2", "pre", "nodmg"})
)

// TestParseVersion: versions compare as numbers, a beta before its
// release; other forms are not Voxt's versions.
func TestParseVersion(t *testing.T) {
	order := []string{"1.16.3", "2.0.3", "2.0.4-beta.1", "2.0.4-beta.2", "2.0.4-beta.10", "2.0.4", "2.0.9", "2.0.10", "2.1.0", "10.0.0"}
	for i := 1; i < len(order); i++ {
		a, _ := parseVersion(order[i-1])
		b, ok := parseVersion(order[i])
		if !ok || slices.Compare(a[:], b[:]) >= 0 {
			t.Errorf("%s < %s: ok %v, %v %v", order[i-1], order[i], ok, a, b)
		}
	}
	for _, s := range []string{"", "2.0", "v2.0.3", "2.0.3-dev", "2.0.3-rc.1", "2.0.3 "} {
		if _, ok := parseVersion(s); ok {
			t.Errorf("%q parsed", s)
		}
	}
	if !checksReleases(false, "2.0.3") || !checksReleases(false, "2.0.4-beta.1") || checksReleases(true, "2.0.3") || checksReleases(false, "0.0.0-dev") {
		t.Error("checksReleases")
	}
}

// TestReleaseCheck: the check against GitHub's API on loopback: the
// newest Voxt release, betas only when asked, drafts and other tags
// skipped, the disk image else the page; the ETag (304), the rate limit,
// a timeout and an answer that does not decode. The request carries the
// User-Agent and the ETag, nothing else.
func TestReleaseCheck(t *testing.T) {
	srv := &ghServer{list: todayReleases, etag: `"e1"`}
	url := srv.start(t)
	store := settings.NewMemory()
	r := newReleaseUpdates(store, url, "2.0.3", nil)
	client := &http.Client{}
	check := func() (release, bool, error) { return r.check(context.Background(), client) }

	rel, newer, err := check()
	if err != nil || newer || rel.Version != "2.0.3" {
		t.Fatalf("up to date: %+v %v %v", rel, newer, err)
	}
	if last := store.Get().Updates.LastCheck; time.Since(last) > time.Minute {
		t.Errorf("last check %v", last)
	}
	req := srv.last()
	if req.Header.Get("User-Agent") != "Voxt/2.0.3" || req.Header.Get("If-None-Match") != "" || req.URL.RawQuery != "per_page=30" {
		t.Errorf("first request %v %q", req.Header, req.URL.RawQuery)
	}
	for k := range req.Header {
		if k != "User-Agent" && k != "Accept-Encoding" {
			t.Errorf("sends %s", k)
		}
	}
	// Unchanged: 304, the list kept.
	if rel, newer, err = check(); err != nil || newer || rel.Version != "2.0.3" || srv.last().Header.Get("If-None-Match") != `"e1"` {
		t.Errorf("304: %+v %v %v %v", rel, newer, err, srv.last().Header)
	}

	srv.set(func(s *ghServer) { s.list, s.etag = newerReleases, `"e2"` })
	rel, newer, err = check()
	if err != nil || !newer || rel.Version != "2.0.4" || rel.Download != "https://dl/dmg-2.0.4" || rel.Notes != "## What's new in 2.0.4\n\n- Faster." {
		t.Errorf("newer: %+v %v %v", rel, newer, err)
	}
	r.SetPrereleases(true)
	if rel, newer, err = check(); err != nil || !newer || rel.Version != "2.0.10-beta.2" ||
		rel.Download != "https://github.com/sgl-project/sglang-omni/releases/tag/voxt-v2.0.10-beta.2" {
		t.Errorf("beta, no disk image: %+v %v %v", rel, newer, err)
	}
	r.SetPrereleases(false)

	srv.set(func(s *ghServer) {
		s.status, s.header = http.StatusForbidden, http.Header{"X-Ratelimit-Remaining": {"0"}}
	})
	if _, _, err = check(); err != errRateLimited || !strings.Contains(releaseErrText(err), "Try again in an hour") {
		t.Errorf("rate limit: %v", err)
	}
	srv.set(func(s *ghServer) { s.header = nil })
	if _, _, err = check(); err == nil || err == errRateLimited || !strings.Contains(err.Error(), "403") {
		t.Errorf("403: %v", err)
	}
	srv.set(func(s *ghServer) { s.status, s.list, s.etag = 0, `[{"tag_name":`, "" })
	if _, _, err = check(); err == nil || !strings.Contains(releaseErrText(err), "could not read") {
		t.Errorf("malformed: %v", err)
	}
	srv.set(func(s *ghServer) { s.hang = true })
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, _, err = r.check(ctx, client); err == nil || !strings.Contains(releaseErrText(err), "did not answer in time") {
		t.Errorf("timeout: %v", err)
	}
	srv.set(func(s *ghServer) { s.hang, s.list = false, `[]` })
	if rel, newer, err = check(); err != nil || newer || rel.Version != "" {
		t.Errorf("no release: %+v %v %v", rel, newer, err)
	}
}

// newReleaseApp is a test app of version 2.0.3 without an update feed,
// over srv.
func newReleaseApp(t *testing.T, srv *ghServer) (*app, *ui.Tester) {
	t.Helper()
	a, _, tt := newTestApp(t)
	a.upd = newReleaseUpdates(a.settings, srv.start(t), "2.0.3", a.checkReleasesNow)
	a.connectUpdates()
	return a, tt
}

// opened records the addresses opened in the browser.
func opened(t *testing.T) *[]string {
	var got []string
	old := openExternal
	openExternal = func(u string) { got = append(got, u) }
	t.Cleanup(func() { openExternal = old })
	return &got
}

// TestReleaseDialog: Check for Updates… without an update feed opens the
// dialog in the app: checking, then the new version's notes with
// Download (its disk image) and Later, up to date, or the failure with
// Try again. Escape and Cancel close it.
func TestReleaseDialog(t *testing.T) {
	srv := &ghServer{list: newerReleases}
	a, tt := newReleaseApp(t, srv)
	got := opened(t)
	a.navigate("/settings/about")

	// Checking: the answer waits.
	q := queuePosts(a)
	click(t, tt, "Check for Updates…")
	if !has(tt, "Checking for updates…") || !has(tt, "Cancel") {
		t.Fatalf("checking: %q", firstTexts(tt))
	}
	click(t, tt, "Cancel")
	if a.relOpen || has(tt, "Checking for updates…") {
		t.Error("Cancel did not close it")
	}
	q()
	a.postQueue = nil
	frames(tt)
	if a.relOpen {
		t.Error("the late answer opened it")
	}

	a.checkForUpdates() // the menus' and ⌘K's
	if !has(tt, "Voxt 2.0.4 is available") || !has(tt, "You have Voxt 2.0.3.") || !has(tt, "Faster.") || !hasPrefix(tt, "Download opens its disk image.") {
		t.Fatalf("available: %q", firstTexts(tt))
	}
	click(t, tt, "Download")
	if a.relOpen || !slices.Equal(*got, []string{"https://dl/dmg-2.0.4"}) {
		t.Errorf("Download: open %v, opened %v", a.relOpen, *got)
	}
	a.checkForUpdates()
	click(t, tt, "Later")
	if a.relOpen {
		t.Error("Later did not close it")
	}
	if slices.ContainsFunc(a.notices, func(n core.Notification) bool { return n.ID == problemRelease }) {
		t.Error("a check the user asked for made a card")
	}

	srv.set(func(s *ghServer) { s.list = todayReleases })
	a.checkForUpdates()
	if !has(tt, "You're up to date") || !has(tt, "Voxt 2.0.3 is the newest version.") {
		t.Fatalf("up to date: %q", firstTexts(tt))
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.relOpen {
		t.Error("Escape did not close it")
	}

	srv.set(func(s *ghServer) { s.status = http.StatusInternalServerError })
	a.checkForUpdates()
	if !has(tt, "Couldn't check for updates") || !hasPrefix(tt, "Voxt could not read the list of releases on GitHub") {
		t.Fatalf("failed: %q", firstTexts(tt))
	}
	srv.set(func(s *ghServer) { s.status = 0 })
	click(t, tt, "Try again")
	if !has(tt, "You're up to date") {
		t.Errorf("try again: %q", firstTexts(tt))
	}
	click(t, tt, "Done")
	if a.relOpen {
		t.Error("Done did not close it")
	}
}

// TestReleaseAutoCheck: the background check, a day after the last, makes
// one card and no window; Download opens the disk image; the card
// dismissed stays away for that version, after a restart too, and a newer
// version gets one. Up to date withdraws it, a failure says nothing, and
// it waits a day between checks and does nothing when turned off.
func TestReleaseAutoCheck(t *testing.T) {
	srv := &ghServer{list: newerReleases}
	dir := t.TempDir()
	a, _, tt := newTestAppIn(t, dir)
	a.upd = newReleaseUpdates(a.settings, srv.start(t), "2.0.3", a.checkReleasesNow)
	a.connectUpdates()
	got := opened(t)
	cards := func(a *app) []core.Notification {
		return slices.DeleteFunc(slices.Clone(a.notices), func(n core.Notification) bool { return n.ID != problemRelease })
	}

	a.autoCheckReleases()
	a.autoCheckReleases() // a day has not passed: no second request
	if c := cards(a); len(c) != 1 || c[0].Title != "Voxt 2.0.4 is available" || a.relOpen || len(srv.requests) != 1 {
		t.Fatalf("cards %+v, dialog %v, requests %d", c, a.relOpen, len(srv.requests))
	}
	a.navigate("/home")
	if !has(tt, "Voxt 2.0.4 is available") || !has(tt, "Download") {
		t.Fatalf("card: %q", firstTexts(tt))
	}
	click(t, tt, "Download")
	if !slices.Equal(*got, []string{"https://dl/dmg-2.0.4"}) || len(cards(a)) != 0 {
		t.Errorf("Download: opened %v, cards %+v", *got, cards(a))
	}
	if d := a.settings.Get().Updates.Dismissed; d != "2.0.4" {
		t.Errorf("dismissed %q", d)
	}

	// The same version a day later, after a restart: no card.
	a.quit()
	a.lib.close()
	a, _, tt = newTestAppIn(t, dir)
	a.upd = newReleaseUpdates(a.settings, srv.url(), "2.0.3", a.checkReleasesNow)
	a.connectUpdates()
	a.settings.Update(context.Background(), func(s *settings.Settings) { s.Updates.LastCheck = time.Now().Add(-25 * time.Hour) })
	a.autoCheckReleases()
	if len(cards(a)) != 0 || len(srv.requests) != 2 {
		t.Fatalf("dismissed version: cards %+v, requests %d", cards(a), len(srv.requests))
	}

	// A newer one: a card, which × dismisses.
	srv.set(func(s *ghServer) { s.list = ghList([]string{"voxt-v2.0.5"}) })
	a.settings.Update(context.Background(), func(s *settings.Settings) { s.Updates.LastCheck = time.Time{} })
	a.autoCheckReleases()
	if c := cards(a); len(c) != 1 || c[0].Version != "2.0.5" {
		t.Fatalf("newer: %+v", c)
	}

	// After the restart into 2.0.5, the kept card goes.
	a.quit()
	a.lib.close()
	a, _, _ = newTestAppIn(t, dir)
	a.keepReleaseCard("2.0.4")
	if len(cards(a)) != 1 {
		t.Fatalf("the card was not kept: %+v", a.notices)
	}
	a.keepReleaseCard("2.0.5")
	if len(cards(a)) != 0 {
		t.Errorf("the installed version's card stayed: %+v", cards(a))
	}

	// Back on 2.0.3: failures say nothing; off, nothing is asked.
	a.upd = newReleaseUpdates(a.settings, srv.url(), "2.0.3", a.checkReleasesNow)
	srv.set(func(s *ghServer) { s.status = http.StatusBadGateway })
	a.settings.Update(context.Background(), func(s *settings.Settings) { s.Updates.LastCheck = time.Time{} })
	n := len(a.notices)
	a.autoCheckReleases()
	if len(a.notices) != n || a.relOpen || len(a.toasts) != 0 {
		t.Errorf("a failure showed: %+v %v %q", a.notices, a.relOpen, a.toasts)
	}
	a.upd.SetAutomaticChecks(false)
	before := len(srv.requests)
	a.autoCheckReleases()
	if len(srv.requests) != before {
		t.Error("checked while off")
	}
}

func (s *ghServer) url() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return "http://" + s.requests[0].Host + s.requests[0].URL.RequestURI()
}

// TestReleaseAbout: Settings → About without an update feed: the last
// check, Check for Updates…, the automatic check and Beta updates, no
// Install updates automatically, and how to install a new version; the
// menus, the menu bar icon and ⌘K have Check for Updates….
func TestReleaseAbout(t *testing.T) {
	srv := &ghServer{list: todayReleases}
	a, tt := newReleaseApp(t, srv)
	a.navigate("/settings/about")
	for _, s := range []string{"Check for updates", "Not checked yet.", "Check automatically", "Once a day. A card says when a new version is out.", "Beta updates"} {
		if !has(tt, s) {
			t.Errorf("about lacks %q: %q", s, firstTexts(tt))
		}
	}
	if !hasPrefix(tt, "Voxt does not install updates itself") || has(tt, "Install updates automatically") || has(tt, "This version does not update automatically.") {
		t.Errorf("about: %q", firstTexts(tt))
	}
	click(t, tt, "Check for Updates…")
	click(t, tt, "Done")
	if !hasPrefix(tt, "Last checked") || has(tt, "Not checked yet.") {
		t.Errorf("last check: %q", firstTexts(tt))
	}
	click(t, tt, "Check for updates automatically")
	if !a.settings.Get().Updates.NoAutoCheck {
		t.Error("the switch did not turn the check off")
	}
	see(t, tt, "Get beta versions")
	click(t, tt, "Get beta versions")
	if r := a.releases(); !r.pre {
		t.Error("beta updates not applied")
	}

	if !slices.ContainsFunc(a.trayItems(), func(it *mygo.MenuItem) bool { return it.Label == "Check for Updates…" }) {
		t.Error("no tray item")
	}
	if !slices.ContainsFunc(a.commands(), func(c command) bool { return c.id == "check-updates" }) {
		t.Error("no palette command")
	}
}

// TestReleaseLive asks the real GitHub API, read-only, as 2.0.3 would:
// VOXT_LIVE_RELEASES=1 go test -run TestReleaseLive -v .
func TestReleaseLive(t *testing.T) {
	if os.Getenv("VOXT_LIVE_RELEASES") == "" {
		t.Skip("set VOXT_LIVE_RELEASES=1 to ask api.github.com")
	}
	r := newReleaseUpdates(settings.NewMemory(), releasesURL, "2.0.3", nil)
	rel, newer, err := r.check(context.Background(), &http.Client{Timeout: releaseTimeout})
	t.Logf("newest %s, newer than 2.0.3: %v, download %s, err %v", rel.Version, newer, rel.Download, err)
	if err != nil || rel.Version == "" {
		t.Fatal("no release")
	}
}
