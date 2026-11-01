package appctx

import (
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// Source says how a group matched.
type Source string

const (
	// SourceURL: one of its URL rules matched the browser tab's address.
	SourceURL Source = "url"
	// SourceBrowserApp: the browser in front is one of its apps (no URL
	// rule matched, or the address could not be read).
	SourceBrowserApp Source = "browserApp"
	// SourceApp: the app in front (not a browser) is one of its apps.
	SourceApp Source = "app"
	// SourceGlobal: no group; the feature's own rules only.
	SourceGlobal Source = "global"
)

// Context is where a session started: the app in front when its shortcut
// was pressed and, for a browser, its front tab's URL.
type Context struct {
	App proto.App
	// Browser: App is a browser (IsBrowser); URL is its front tab's
	// address, "" when it could not be read in time.
	Browser bool
	URL     string
}

// Match is the group a session falls in.
type Match struct {
	Source Source
	// Group is the matched group (a copy); nil for SourceGlobal.
	Group *Group
	// Pattern is the URL rule that matched (SourceURL).
	Pattern string
	// Host is the tab's host when an address was read, whatever matched.
	Host string
	// App is the app in front.
	App proto.App
}

// Matched reports whether a group matched.
func (m Match) Matched() bool { return m.Group != nil }

// GroupID and GroupName are the matched group's, "" for none.
func (m Match) GroupID() string {
	if m.Group == nil {
		return ""
	}
	return m.Group.ID
}

func (m Match) GroupName() string {
	if m.Group == nil {
		return ""
	}
	return m.Group.Name
}

// Resolve decides the group of a session of feature in ctx (see the
// package comment). Disabled groups and groups that do not apply to
// feature are skipped as if they were not there.
func Resolve(r Rules, ctx Context, feature string) Match {
	m := Match{Source: SourceGlobal, App: ctx.App}
	addr, hasAddr := Address{}, false
	if ctx.Browser {
		addr, hasAddr = ParseAddress(ctx.URL)
		if hasAddr {
			m.Host = addr.Host
		}
	}
	var live []Group
	for _, g := range r.Groups {
		if g.Enabled && g.Features.Has(feature) {
			live = append(live, g)
		}
	}
	// 1. The most specific URL rule, then the earlier group.
	if hasAddr {
		best, bestRank := -1, -1
		pattern := ""
		for i, g := range live {
			for _, p := range g.URLs {
				if !MatchPattern(p, addr) {
					continue
				}
				if s := specificity(p); s > bestRank {
					best, bestRank, pattern = i, s, p
				}
			}
		}
		if best >= 0 {
			g := live[best]
			m.Source, m.Group, m.Pattern = SourceURL, &g, pattern
			return m
		}
	}
	// 2 and 3. The group the app is a member of.
	for _, g := range live {
		if g.HasApp(ctx.App.BundleID) {
			g := g
			m.Group = &g
			m.Source = SourceApp
			if ctx.Browser {
				m.Source = SourceBrowserApp
			}
			return m
		}
	}
	return m
}

// Browser is a browser whose front tab VoxtNew reads.
type Browser struct {
	BundleID string
	Name     string
	// Engine is "webkit", "chromium" or "gecko": how its window shows the
	// address to Accessibility (the helper reads them alike).
	Engine string
}

// Engines.
const (
	WebKit   = "webkit"
	Chromium = "chromium"
	Gecko    = "gecko"
)

// Builtin are the browsers VoxtNew knows; the user adds others by bundle
// id (settings.AppEnhancement.Browsers).
var Builtin = []Browser{
	{"com.apple.Safari", "Safari", WebKit},
	{"com.apple.SafariTechnologyPreview", "Safari Technology Preview", WebKit},
	{"com.google.Chrome", "Google Chrome", Chromium},
	{"com.google.Chrome.beta", "Google Chrome Beta", Chromium},
	{"com.google.Chrome.dev", "Google Chrome Dev", Chromium},
	{"com.google.Chrome.canary", "Google Chrome Canary", Chromium},
	{"org.chromium.Chromium", "Chromium", Chromium},
	{"com.microsoft.edgemac", "Microsoft Edge", Chromium},
	{"com.microsoft.edgemac.Beta", "Microsoft Edge Beta", Chromium},
	{"com.microsoft.edgemac.Dev", "Microsoft Edge Dev", Chromium},
	{"com.microsoft.edgemac.Canary", "Microsoft Edge Canary", Chromium},
	{"com.brave.Browser", "Brave Browser", Chromium},
	{"com.brave.Browser.beta", "Brave Browser Beta", Chromium},
	{"com.brave.Browser.nightly", "Brave Browser Nightly", Chromium},
	{"company.thebrowser.Browser", "Arc", Chromium},
	{"company.thebrowser.dia", "Dia", Chromium},
	{"com.vivaldi.Vivaldi", "Vivaldi", Chromium},
	{"com.operasoftware.Opera", "Opera", Chromium},
	{"org.mozilla.firefox", "Firefox", Gecko},
	{"app.zen-browser.zen", "Zen", Gecko},
}

// Browsers are the built-in browsers, then the user's (those not built in).
func Browsers(custom []settings.Browser) []Browser {
	out := append([]Browser(nil), Builtin...)
	for _, b := range custom {
		id := strings.TrimSpace(b.BundleID)
		if id == "" || builtin(id) || hasBrowser(out, id) {
			continue
		}
		name := strings.TrimSpace(b.Name)
		if name == "" {
			name = id
		}
		out = append(out, Browser{BundleID: id, Name: name})
	}
	return out
}

// TreeOnDemand are the browsers whose pages the helper must ask for an
// accessibility tree (app.configure's manualAccessibility): the built-in
// Chromium ones and the user's (their engine is unknown; most are
// Chromium, and the attribute is harmless elsewhere). Chromium builds a
// page's tree only when a client asks: without it, Arc shows browser.url
// no web area.
func TreeOnDemand(custom []settings.Browser) []string {
	var out []string
	for _, b := range Browsers(custom) {
		if b.Engine == Chromium || b.Engine == "" {
			out = append(out, b.BundleID)
		}
	}
	return out
}

// IsBrowser reports whether bundleID is a built-in browser or one of the
// user's.
func IsBrowser(bundleID string, custom []settings.Browser) bool {
	if bundleID == "" {
		return false
	}
	return hasBrowser(Browsers(custom), bundleID)
}

// IsBuiltin reports whether bundleID is a built-in browser.
func IsBuiltin(bundleID string) bool { return builtin(bundleID) }

func builtin(id string) bool { return hasBrowser(Builtin, id) }

func hasBrowser(bs []Browser, id string) bool {
	for _, b := range bs {
		if strings.EqualFold(b.BundleID, id) {
			return true
		}
	}
	return false
}
