package main

// Update checks without an update feed (2.0.x, docs/release.md): a
// release build whose mygo.json has no feed cannot update itself, but it
// can tell the user that a newer version is out. releaseUpdates is the
// updateService of such a build (main.go): it reads GitHub's releases of
// the repository and keeps the newest voxt-vX.Y.Z (and voxt-vX.Y.Z-beta.N
// while Beta updates is on) that is not a draft. The request carries
// User-Agent Voxt/<version> and the ETag of the last answer, nothing
// else, through the proxy the downloads use.
//
// Check for Updates… (the menus, ⌘K, Settings → About) opens a dialog
// that checks, then shows the new version's notes with Download (its
// disk image, else its release page), says the app is up to date, or why
// the check failed. Once a day, and a minute after launch when the last
// check is older than that, a check runs in the background: a newer
// version becomes a card in the sidebar (Download, ×), never a window;
// the card dismissed stays away for that version. A failed background
// check shows nothing: the log says it. The preferences (automatic
// checks, the last check, the version dismissed) are in settings.Updates.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// releasesURL lists the repository's releases, newest first; tests use
// their server's.
const releasesURL = "https://api.github.com/repos/sgl-project/sglang-omni/releases?per_page=30"

// releaseTimeout bounds a check; tests shorten it.
var releaseTimeout = 10 * time.Second

// releaseEvery is the least time between two background checks, which are
// looked for every releaseTick, the first releaseFirstWait after launch.
const (
	releaseEvery     = 24 * time.Hour
	releaseTick      = time.Hour
	releaseFirstWait = time.Minute
)

// maxReleases bounds GitHub's answer; maxNotes the notes the dialog shows.
const (
	maxReleases = 8 << 20
	maxNotes    = 8000
)

// The card of a newer version: its ID, and the prefix of its action's
// link, followed by the address to download from.
const (
	problemRelease = "update/available"
	linkDownload   = "action:download:"
)

// openExternal opens an address in the browser; tests record it.
var openExternal = func(u string) { go mygo.Shell.OpenExternal(u) }

// errRateLimited is GitHub's limit on requests without an account (60
// an hour for an address) reached.
var errRateLimited = errors.New("GitHub's rate limit reached")

var (
	releaseTag     = regexp.MustCompile(`^voxt-v(\d+\.\d+\.\d+(?:-beta\.\d+)?)$`)
	releaseVersion = regexp.MustCompile(`^(\d+)\.(\d+)\.(\d+)(?:-beta\.(\d+))?$`)
)

// version is X.Y.Z or X.Y.Z-beta.N as numbers: major, minor, patch, 1 for
// a release (0 a beta, which comes before its release), the beta's N.
type version [5]int

// parseVersion reads X.Y.Z or X.Y.Z-beta.N.
func parseVersion(s string) (version, bool) {
	m := releaseVersion.FindStringSubmatch(s)
	if m == nil {
		return version{}, false
	}
	var v version
	for i, p := range m[1:4] {
		v[i], _ = strconv.Atoi(p)
	}
	if m[4] == "" {
		v[3] = 1
	} else {
		v[4], _ = strconv.Atoi(m[4])
	}
	return v, true
}

// checksReleases reports whether a build of version v without an update
// feed checks GitHub's releases: a release or a beta, not a development
// build.
func checksReleases(dev bool, v string) bool { return !dev && releaseVersion.MatchString(v) }

// release is a version found: its notes (Markdown) and where to get it.
type release struct {
	Version  string
	Notes    string
	Download string // its disk image, else its release page
}

// ghRelease is a release as GitHub's API lists it.
type ghRelease struct {
	Tag        string `json:"tag_name"`
	Draft      bool   `json:"draft"`
	Prerelease bool   `json:"prerelease"`
	Body       string `json:"body"`
	Page       string `json:"html_url"`
	Assets     []struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
	} `json:"assets"`
}

// fetchReleases reads the list at address; etag is the last answer's (""
// none). An unchanged list (304) returns nil and etag.
func fetchReleases(ctx context.Context, client *http.Client, address, agent, etag string) ([]ghRelease, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return nil, etag, err
	}
	req.Header.Set("User-Agent", agent)
	if etag != "" {
		req.Header.Set("If-None-Match", etag)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, etag, err
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusNotModified:
		return nil, etag, nil
	case resp.StatusCode == http.StatusTooManyRequests,
		resp.StatusCode == http.StatusForbidden && resp.Header.Get("X-RateLimit-Remaining") == "0":
		return nil, etag, errRateLimited
	case resp.StatusCode/100 != 2:
		return nil, etag, fmt.Errorf("GitHub answered HTTP %d", resp.StatusCode)
	}
	var list []ghRelease
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxReleases)).Decode(&list); err != nil {
		return nil, etag, fmt.Errorf("GitHub's answer: %w", err)
	}
	if list == nil {
		list = []ghRelease{} // null: no release, still an answer
	}
	return list, resp.Header.Get("ETag"), nil
}

// newestRelease is the newest Voxt release of list that is not a draft,
// betas included when beta is.
func newestRelease(list []ghRelease, beta bool) (release, version, bool) {
	var best release
	var bestV version
	found := false
	for _, r := range list {
		m := releaseTag.FindStringSubmatch(r.Tag)
		if r.Draft || m == nil {
			continue
		}
		v, _ := parseVersion(m[1])
		if (r.Prerelease || v[3] == 0) && !beta {
			continue
		}
		if found && slices.Compare(v[:], bestV[:]) <= 0 {
			continue
		}
		best, bestV, found = release{Version: m[1], Notes: trimNotes(r.Body), Download: r.Page}, v, true
		for _, as := range r.Assets {
			if strings.HasSuffix(as.Name, ".dmg") && strings.Contains(as.Name, m[1]) {
				best.Download = as.URL
				break
			}
		}
	}
	return best, bestV, found
}

// trimNotes is a release's notes for the dialog: no surrounding space, at
// most maxNotes characters.
func trimNotes(s string) string {
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > maxNotes {
		s = strings.TrimSpace(string(r[:maxNotes])) + "…"
	}
	return s
}

// releaseUpdates is the updateService of a release build without an
// update feed.
type releaseUpdates struct {
	store   settings.Service
	url     string
	version string // the running one
	open    func() // CheckNow: the app's dialog (main thread)

	mu      sync.Mutex
	pre     bool        // SetPrereleases
	etag    string      // of list
	list    []ghRelease // the last answer; nil: none yet
	changes []*func()
}

func newReleaseUpdates(store settings.Service, url, version string, open func()) *releaseUpdates {
	return &releaseUpdates{store: store, url: url, version: version, open: open}
}

func (r *releaseUpdates) Prefs() updatePrefs {
	u := r.store.Get().Updates
	return updatePrefs{Enabled: true, CheckOnly: true, AutoCheck: !u.NoAutoCheck, LastCheck: u.LastCheck}
}

func (r *releaseUpdates) SetAutomaticChecks(on bool) {
	r.save(func(u *settings.Updates) { u.NoAutoCheck = !on })
}

// SetAutomaticDownloads: nothing installs (About has no such switch).
func (r *releaseUpdates) SetAutomaticDownloads(bool) {}

func (r *releaseUpdates) SetPrereleases(on bool) {
	r.mu.Lock()
	r.pre = on
	r.mu.Unlock()
}

func (r *releaseUpdates) CheckNow() { r.open() }

func (r *releaseUpdates) OnChange(fn func()) func() {
	p := &fn
	r.mu.Lock()
	r.changes = append(r.changes, p)
	r.mu.Unlock()
	return func() {
		r.mu.Lock()
		r.changes = slices.DeleteFunc(r.changes, func(q *func()) bool { return q == p })
		r.mu.Unlock()
	}
}

// save changes the preferences, then tells OnChange's functions.
func (r *releaseUpdates) save(fn func(*settings.Updates)) {
	if err := r.store.Update(context.Background(), func(s *settings.Settings) { fn(&s.Updates) }); err != nil {
		log.Printf("updates: %v", err)
	}
	r.mu.Lock()
	changes := slices.Clone(r.changes)
	r.mu.Unlock()
	for _, c := range changes {
		(*c)()
	}
}

// check reads the releases (any goroutine) and returns the newest one and
// whether it is newer than the running version; a success is the last
// check.
func (r *releaseUpdates) check(ctx context.Context, client *http.Client) (release, bool, error) {
	r.mu.Lock()
	etag, list, beta := r.etag, r.list, r.pre
	r.mu.Unlock()
	if list == nil {
		etag = ""
	}
	got, etag, err := fetchReleases(ctx, client, r.url, "Voxt/"+r.version, etag)
	if err != nil {
		return release{}, false, err
	}
	if got != nil {
		list = got
		r.mu.Lock()
		r.etag, r.list = etag, got
		r.mu.Unlock()
	}
	rel, v, ok := newestRelease(list, beta)
	r.save(func(u *settings.Updates) { u.LastCheck = time.Now() })
	return rel, ok && r.newer(v), nil
}

// newer reports whether v is newer than the running version.
func (r *releaseUpdates) newer(v version) bool {
	running, _ := parseVersion(r.version)
	return slices.Compare(v[:], running[:]) > 0
}

// releaseState is the dialog of Check for Updates… (main thread).
type releaseState struct {
	relOpen  bool
	relDone  bool // the check finished: relErr or relFound
	relSeq   int  // the dialog's check: an older one's answer is late
	relErr   error
	relFound release
	relNewer bool
	relMD    markdown.Options
}

// releases is the check of GitHub's releases, nil in builds with another
// updater or none.
func (a *app) releases() *releaseUpdates {
	r, _ := a.upd.(*releaseUpdates)
	return r
}

// runReleaseCheck checks in the background, then calls done on the main
// thread.
func (a *app) runReleaseCheck(done func(release, bool, error)) {
	r := a.releases()
	if r == nil {
		return
	}
	client := a.networkFor(a.prefs).Client
	a.async(func() {
		defer client.CloseIdleConnections()
		ctx, cancel := context.WithTimeout(context.Background(), releaseTimeout)
		defer cancel()
		rel, newer, err := r.check(ctx, client)
		a.post(func() { done(rel, newer, err) })
	})
}

// checkReleasesNow is Check for Updates…: the dialog, checking.
func (a *app) checkReleasesNow() {
	a.relSeq++
	seq := a.relSeq
	a.relOpen, a.relDone = true, false
	a.runReleaseCheck(func(rel release, newer bool, err error) {
		if seq != a.relSeq {
			return
		}
		a.relDone, a.relFound, a.relNewer, a.relErr = true, rel, newer, err
		if err == nil && !newer {
			a.setProblem(problemRelease, "", nil)
		}
	})
}

// autoCheckReleases is the background check, when it is on and the last
// one is a day old: a newer version becomes a card.
func (a *app) autoCheckReleases() {
	r := a.releases()
	if r == nil {
		return
	}
	if u := r.store.Get().Updates; u.NoAutoCheck || time.Since(u.LastCheck) < releaseEvery {
		return
	}
	a.runReleaseCheck(func(rel release, newer bool, err error) {
		if err != nil {
			log.Printf("updates: %v", err)
			return
		}
		if !newer {
			a.setProblem(problemRelease, "", nil)
			return
		}
		if r.store.Get().Updates.Dismissed == rel.Version {
			return
		}
		a.setProblem(problemRelease, rel.Version, &core.Notification{Severity: core.SeverityInfo,
			Title: i18n.T("Voxt %s is available", rel.Version), Body: i18n.T("Install it from its disk image, or with brew upgrade --cask voxt."),
			Link: linkDownload + rel.Download, Version: rel.Version})
	})
}

// watchReleases runs the background check; main starts it once the
// window is open.
func (a *app) watchReleases() {
	if a.releases() == nil {
		return
	}
	go func() {
		time.Sleep(releaseFirstWait)
		for {
			a.post(a.autoCheckReleases)
			time.Sleep(releaseTick)
		}
	}()
}

// keepReleaseCard drops a card kept from before a restart whose version
// is not newer than running (it was installed since); main calls it.
func (a *app) keepReleaseCard(running string) {
	i := slices.IndexFunc(a.notices, func(n core.Notification) bool { return n.ID == problemRelease })
	if i < 0 {
		return
	}
	v, ok := parseVersion(a.notices[i].Version)
	now, _ := parseVersion(running)
	if !ok || slices.Compare(v[:], now[:]) <= 0 {
		a.setProblem(problemRelease, "", nil)
	}
}

// releaseCardGone remembers the version of a dismissed card.
func (a *app) releaseCardGone(n core.Notification) {
	if n.ID == problemRelease && n.Version != "" {
		v := n.Version
		a.updateSettings(func(s *settings.Settings) { s.Updates.Dismissed = v })
	}
}

// releaseErrText says why a check failed.
func releaseErrText(err error) string {
	var ne net.Error
	switch {
	case errors.Is(err, errRateLimited):
		return i18n.T("GitHub limits how often Voxt can check. Try again in an hour.")
	case errors.As(err, &ne) && ne.Timeout(), errors.Is(err, context.DeadlineExceeded):
		return i18n.T("GitHub did not answer in time. Check your connection, or the proxy in Speech models → Network.")
	}
	return i18n.T("Voxt could not read the list of releases on GitHub: %v", err)
}

// releaseDialog is Check for Updates…'s dialog: checking, then the new
// version, up to date, or the failure. Escape, × and Cancel close it.
func (a *app) releaseDialog(c *ui.Context) {
	r := a.releases()
	if !a.relOpen || r == nil {
		return
	}
	t := c.Theme()
	title, sub := i18n.T("Checking for updates…"), ""
	switch {
	case !a.relDone:
	case a.relErr != nil:
		title = i18n.T("Couldn't check for updates")
	case a.relNewer:
		title, sub = i18n.T("Voxt %s is available", a.relFound.Version), i18n.T("You have Voxt %s.", r.version)
	default:
		title, sub = i18n.T("You're up to date"), i18n.T("Voxt %s is the newest version.", r.version)
	}
	dialogFrame(c, "release", &a.relOpen, title, sub, 520, func() {
		switch {
		case !a.relDone:
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Spinner(c).Size(px(t, 16), px(t, 16))
				caption(c, i18n.T("Looking for the newest release on GitHub."))
			})
		case a.relErr != nil:
			ui.Text(c, releaseErrText(a.relErr)).Selectable()
		case a.relNewer:
			if a.relFound.Notes != "" {
				markdown.View(c, a.relFound.Notes, &a.relMD)
			}
			caption(c, i18n.T("Download opens its disk image. With Homebrew, run brew upgrade --cask voxt instead."))
		}
	}, func() {
		ui.Spacer(c)
		switch {
		case !a.relDone:
			if ui.Button(c.Key("release-cancel"), i18n.T("Cancel")).Clicked() {
				a.relOpen = false
			}
		case a.relErr != nil:
			if ui.Button(c.Key("release-cancel"), i18n.T("Cancel")).Clicked() {
				a.relOpen = false
			}
			if primaryButton(c.Key("release-retry"), i18n.T("Try again")).Clicked() {
				a.checkReleasesNow()
			}
		case a.relNewer:
			if ui.Button(c.Key("release-later"), i18n.T("Later")).Clicked() {
				a.relOpen = false
			}
			if primaryButton(c.Key("release-download"), i18n.T("Download")).Clicked() {
				a.relOpen = false
				openExternal(a.relFound.Download)
			}
		default:
			if primaryButton(c.Key("release-ok"), i18n.T("Done")).Clicked() {
				a.relOpen = false
			}
		}
	})
}
