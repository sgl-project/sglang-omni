// Package meetdetect decides which of the apps recording from the
// microphone is a meeting (Settings → Meetings → Detect meetings): the
// helper's mic.watch says which processes record, this package picks the
// meeting apps among them and remembers which ones the user was already
// asked about. A browser is a meeting only on a meeting's page, which the
// app reads (browser.url) and gives to Web.
//
// Pure logic: no helper, no clock of its own.
package meetdetect

import (
	"cmp"
	"maps"
	"net/url"
	"slices"
	"strings"
	"time"
)

// Meeting apps, as App and Web name them; the app has their names.
const (
	Zoom     = "zoom"
	Teams    = "teams"
	Tencent  = "tencent"
	Feishu   = "feishu"
	DingTalk = "dingtalk"
	WeCom    = "wecom"
	Slack    = "slack"
	Discord  = "discord"
	Webex    = "webex"
	Meet     = "meet"
)

// native are the meeting apps' bundle ids. A chat app with voice messages
// only (WeChat) is not one: each message would ask.
var native = []struct{ bundle, app string }{
	{"us.zoom.xos", Zoom},
	{"com.microsoft.teams2", Teams},
	{"com.microsoft.teams", Teams},
	{"com.tencent.meeting", Tencent},
	{"com.electron.lark", Feishu},
	{"com.bytedance.macos.feishu", Feishu},
	{"com.larksuite.larkApp", Feishu},
	{"com.alibaba.DingTalkMac", DingTalk},
	{"com.tencent.WeWorkMac", WeCom},
	{"com.tinyspeck.slackmacgap", Slack},
	{"com.hnc.Discord", Discord},
	{"Cisco-Systems.Spark", Webex},
	{"com.webex.meetingmanager", Webex},
}

// web are the hosts of meetings held in a browser (the host itself or a
// subdomain of it).
var web = []struct{ host, app string }{
	{"meet.google.com", Meet},
	{"teams.microsoft.com", Teams},
	{"teams.live.com", Teams},
	{"teams.cloud.microsoft", Teams},
	{"zoom.us", Zoom},
	{"meeting.tencent.com", Tencent},
	{"voovmeeting.com", Tencent},
}

// webKit records for Safari's pages (and other WebKit views): its bundle
// id names no browser, so Safari's front tab is read.
const (
	webKit = "com.apple.WebKit.GPU"
	safari = "com.apple.Safari"
)

// Quiet is how long an app must stay off the microphone before its next
// use asks again: an app that lets go of the microphone while muted asks
// once a meeting, not at every unmute.
const Quiet = 2 * time.Minute

// Hold is how long an app must record before it is a candidate: a
// microphone test or a voice clip of a moment is no meeting, and with
// Start recording automatically it would start one.
const Hold = 3 * time.Second

// User is a process recording from an input (the helper's mic.watch).
type User struct {
	PID      int
	BundleID string
}

// Candidate is an app recording now that the user was not asked about:
// a meeting app (App), or a browser whose front tab says whether it is a
// meeting (Browser: App is "" until Web names it).
type Candidate struct {
	Key     string // the app's bundle id: what Settle takes
	App     string
	Browser string // the browser's bundle id, for browser.url
}

// owns reports whether a process with bundle id belongs to the app with
// bundle id app: the app itself or a helper of it ("<app>.helper").
func owns(app, id string) bool {
	if len(id) < len(app) || !strings.EqualFold(id[:len(app)], app) {
		return false
	}
	return len(id) == len(app) || id[len(app)] == '.'
}

// classify names the candidate a recording process stands for; browsers
// are the bundle ids of the browsers the app knows.
func classify(id string, browsers []string) (Candidate, bool) {
	if id == "" {
		return Candidate{}, false
	}
	for _, n := range native {
		if owns(n.bundle, id) {
			return Candidate{Key: n.bundle, App: n.app}, true
		}
	}
	if strings.EqualFold(id, webKit) {
		return Candidate{Key: safari, Browser: safari}, true
	}
	best := ""
	for _, b := range browsers {
		// The longest id wins: Chrome Beta's helper is not Chrome's.
		if owns(b, id) && len(b) > len(best) {
			best = b
		}
	}
	if best == "" {
		return Candidate{}, false
	}
	return Candidate{Key: best, Browser: best}, true
}

// Web names the meeting a browser's tab at address holds.
func Web(address string) (app string, ok bool) {
	address = strings.TrimSpace(address)
	if !strings.Contains(address, "://") {
		address = "https://" + address
	}
	u, err := url.Parse(address)
	if err != nil {
		return "", false
	}
	host := strings.ToLower(strings.TrimSuffix(u.Hostname(), "."))
	for _, w := range web {
		if host == w.host || strings.HasSuffix(host, "."+w.host) {
			return w.app, true
		}
	}
	return "", false
}

// Detector remembers which apps recording now are settled: the user was
// asked, or there is nothing to ask. The zero value is ready. Not safe
// for concurrent use (the app's main thread).
type Detector struct {
	// Hold replaces the package's Hold when it is not zero (tests).
	Hold time.Duration
	// settled has an app's key while it records (the zero time), and for
	// Quiet after it stopped (when it stopped).
	settled map[string]time.Time
	// since is when each app that records now started.
	since   map[string]time.Time
	present []string
}

// Update takes the processes recording at now and returns the candidates
// to decide about, each app once, by key, and how long until an app that
// just started has recorded for Hold (0: none waits; call Update again
// then). An app that stopped recording more than Quiet ago is forgotten:
// its next use is a candidate again.
func (d *Detector) Update(users []User, browsers []string, now time.Time) (found []Candidate, wait time.Duration) {
	d.present = d.present[:0]
	for _, u := range users {
		c, ok := classify(u.BundleID, browsers)
		if !ok || slices.Contains(d.present, c.Key) {
			continue
		}
		d.present = append(d.present, c.Key)
		found = append(found, c)
	}
	if d.since == nil {
		d.since = map[string]time.Time{}
	}
	maps.DeleteFunc(d.since, func(key string, _ time.Time) bool { return !slices.Contains(d.present, key) })
	for _, key := range d.present {
		if _, recording := d.since[key]; !recording {
			d.since[key] = now
		}
	}
	for key, left := range d.settled {
		switch {
		case !left.IsZero() && now.Sub(left) > Quiet:
			// Long gone: forgotten, also when it records again just now
			// (Update runs only when something changes).
			delete(d.settled, key)
		case slices.Contains(d.present, key):
			d.settled[key] = time.Time{}
		case left.IsZero():
			d.settled[key] = now
		}
	}
	hold := cmp.Or(d.Hold, Hold)
	found = slices.DeleteFunc(found, func(c Candidate) bool {
		if _, done := d.settled[c.Key]; done {
			return true
		}
		if left := hold - now.Sub(d.since[c.Key]); left > 0 {
			if wait == 0 || left < wait {
				wait = left
			}
			return true
		}
		return false
	})
	slices.SortFunc(found, func(a, b Candidate) int { return strings.Compare(a.Key, b.Key) })
	return found, wait
}

// Settle marks key as decided for this use: Update returns it again only
// after the app stayed off the microphone for Quiet.
func (d *Detector) Settle(key string) {
	if d.settled == nil {
		d.settled = map[string]time.Time{}
	}
	d.settled[key] = time.Time{}
}

// Forget undoes Settle: an app that still records is a candidate again at
// the next Update (the user asked to be asked later).
func (d *Detector) Forget(key string) { delete(d.settled, key) }

// Recording reports whether the app with key records now (as of the last
// Update).
func (d *Detector) Recording(key string) bool { return slices.Contains(d.present, key) }
