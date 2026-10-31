package appctx

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func TestParseAddress(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string // "" = not an address
	}{
		{"https://github.com/hehehai/voxtnew?tab=readme#top", "github.com/hehehai/voxtnew"},
		{"http://WWW.Example.COM", "example.com/"},
		{"https://www.example.com:8443/a/b/", "example.com/a/b/"},
		{"github.com/x", "github.com/x"},
		{"  https://mail.google.com/mail/u/0/#inbox  ", "mail.google.com/mail/u/0/"},
		{"chrome://settings/privacy", "settings/privacy"},
		{"https://docs.google.com/document/d/1AbC/edit", "docs.google.com/document/d/1abc/edit"},
		{"https://zh.wikipedia.org/wiki/%E8%AF%AD%E9%9F%B3", "zh.wikipedia.org/wiki/语音"},
		{"https://example.com./", "example.com/"},
		{"about:blank", ""},
		{"mailto:me@example.com", ""},
		{"file:///Users/me/a.html", ""},
		{"", ""},
		{"not a url", ""},
	} {
		a, ok := ParseAddress(tc.in)
		got := ""
		if ok {
			got = a.String()
		}
		if got != tc.want {
			t.Errorf("ParseAddress(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestCanonicalPattern(t *testing.T) {
	for in, want := range map[string]string{
		"github.com":                  "github.com/*",
		"GitHub.com/":                 "github.com/*",
		"https://www.github.com/x/":   "github.com/x/*",
		"github.com/x":                "github.com/x",
		"github.com/x/*":              "github.com/x/*",
		"*.notion.so":                 "*.notion.so/*",
		"localhost:3000/app":          "localhost/app",
		"example.com/search?q=x":      "example.com/search*",
		"  mail.google.com/mail/*  ":  "mail.google.com/mail/*",
		"http://example.com:80/a#b":   "example.com/a*",
		"*://example.com/*":           "example.com/*",
		"https://example.com./path/":  "example.com/path/*",
		"https://example.com/路径/":     "example.com/路径/*",
		"https://WWW.EXAMPLE.COM/Abc": "example.com/abc",
	} {
		if got := CanonicalPattern(in); got != want {
			t.Errorf("CanonicalPattern(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCheckPattern(t *testing.T) {
	for in, want := range map[string]error{
		"github.com":           nil,
		"*.notion.so/*":        nil,
		"localhost:5173":       nil,
		"https://a.b/c?d":      nil,
		"例子.中国/路径":             nil,
		"":                     ErrPatternEmpty,
		"   ":                  ErrPatternEmpty,
		"git hub.com":          ErrPatternSpace,
		"ftp://example.com":    ErrPatternScheme,
		"*":                    ErrPatternHost,
		"*/foo":                ErrPatternHost,
		"github":               ErrPatternHost,
		"/path/only":           ErrPatternHost,
		"exa$mple.com":         ErrPatternHost,
		"example..com":         ErrPatternHost,
		".example.com":         ErrPatternHost,
		"https:///nohost/path": ErrPatternHost,
	} {
		if got := CheckPattern(in); !errors.Is(got, want) && got != want {
			t.Errorf("CheckPattern(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestMatchPattern(t *testing.T) {
	for _, tc := range []struct {
		pattern, url string
		want         bool
	}{
		{"github.com", "https://github.com/hehehai/voxtnew", true},
		{"github.com", "https://github.com", true},
		{"github.com", "https://gist.github.com/x", false},
		{"github.com", "https://notgithub.com/", false},
		{"github.com/hehehai/*", "https://github.com/hehehai/voxtnew", true},
		{"github.com/hehehai/*", "https://github.com/other/x", false},
		{"github.com/hehehai", "https://github.com/hehehai", true},
		{"github.com/hehehai", "https://github.com/hehehai/voxtnew", false},
		{"*.notion.so", "https://www.notion.so/page", true},
		{"*.notion.so", "https://team.notion.so/page", true},
		{"*.notion.so", "https://notion.so/", true},
		{"*.notion.so", "https://notion.solutions/", false},
		{"mail.google.com/mail/*", "https://mail.google.com/mail/u/0/#inbox", true},
		{"mail.google.com/mail/*", "https://mail.google.com/chat/", false},
		{"*.google.com/*/edit", "https://docs.google.com/document/d/1/edit", true},
		{"docs.google.com/*/edit", "https://docs.google.com/document/d/1/edit?usp=sharing", true},
		{"localhost:3000", "http://localhost:5173/app", true}, // ports are not matched
		{"WWW.Linear.APP/team/*", "https://linear.app/team/issue/AUR-12", true},
		{"linear.app", "about:blank", false},
		{"bad rule", "https://bad/", false},
		{"https://example.com/a*b", "https://example.com/a/x/b", true},
		{"https://example.com/a*b", "https://example.com/a/x/c", false},
	} {
		a, _ := ParseAddress(tc.url)
		if got := MatchPattern(tc.pattern, a); got != tc.want {
			t.Errorf("MatchPattern(%q, %q) = %v, want %v", tc.pattern, tc.url, got, tc.want)
		}
	}
}

func TestGlob(t *testing.T) {
	for _, tc := range []struct {
		p, s string
		want bool
	}{
		{"", "", true}, {"*", "", true}, {"*", "abc", true}, {"a*", "a", true}, {"a*c", "abbbc", true},
		{"a*c", "abbb", false}, {"*a*a*", "banana", true}, {"**", "x", true}, {"a", "b", false}, {"abc", "ab", false},
		{"a*b*c", "aXbYbZc", true}, {"a*b*c", "aXcYb", false},
	} {
		if got := glob(tc.p, tc.s); got != tc.want {
			t.Errorf("glob(%q, %q) = %v", tc.p, tc.s, got)
		}
	}
}

// fixture groups: a URL group for GitHub, a chat group (Slack, Teams),
// a browser group (Chrome) with its own URL rule, a disabled group.
func fixture() Rules {
	return Rules{Version: Version, Groups: []Group{
		{ID: "g-github", Name: "Code review", Enabled: true, URLs: []string{"github.com"}, Features: DefaultFeatures},
		{ID: "g-chat", Name: "Chat", Enabled: true, Apps: []App{{"com.tinyspeck.slackmacgap", "Slack"}, {"com.microsoft.teams2", "Teams"}},
			AutoKey: "Return", Features: Features{Transcribe: true}},
		{ID: "g-work", Name: "Work browser", Enabled: true, Apps: []App{{"com.google.Chrome", "Google Chrome"}},
			URLs: []string{"github.com/hehehai/*", "linear.app"}, Features: Features{Transcribe: true, Translate: true, Rewrite: true}},
		{ID: "g-off", Name: "Off", Enabled: false, Apps: []App{{"com.apple.mail", "Mail"}}, URLs: []string{"mail.google.com"}, Features: DefaultFeatures},
		{ID: "g-translate", Name: "Docs (translate)", Enabled: true, URLs: []string{"docs.google.com"}, Apps: []App{{"com.apple.Notes", "Notes"}},
			Features: Features{Translate: true}},
	}}
}

func TestResolve(t *testing.T) {
	chrome := proto.App{PID: 1, BundleID: "com.google.Chrome", Name: "Google Chrome"}
	safari := proto.App{PID: 2, BundleID: "com.apple.Safari", Name: "Safari"}
	slack := proto.App{PID: 3, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}
	mail := proto.App{PID: 4, BundleID: "com.apple.mail", Name: "Mail"}
	notes := proto.App{PID: 5, BundleID: "com.apple.Notes", Name: "Notes"}
	for _, tc := range []struct {
		name    string
		ctx     Context
		feature string
		group   string
		source  Source
		pattern string
		host    string
	}{
		{"url rule in a browser", Context{App: safari, Browser: true, URL: "https://github.com/golang/go"}, Transcribe, "g-github", SourceURL, "github.com", "github.com"},
		{"more specific rule wins over an earlier group", Context{App: safari, Browser: true, URL: "https://github.com/hehehai/voxtnew"}, Transcribe, "g-work", SourceURL, "github.com/hehehai/*", "github.com"},
		{"url rule of the browser's own group", Context{App: chrome, Browser: true, URL: "https://linear.app/team"}, Transcribe, "g-work", SourceURL, "linear.app", "linear.app"},
		{"browser app group when no rule matches", Context{App: chrome, Browser: true, URL: "https://news.ycombinator.com/"}, Transcribe, "g-work", SourceBrowserApp, "", "news.ycombinator.com"},
		{"browser app group without an address", Context{App: chrome, Browser: true}, Transcribe, "g-work", SourceBrowserApp, "", ""},
		{"browser app group on an unreadable address", Context{App: chrome, Browser: true, URL: "about:blank"}, Transcribe, "g-work", SourceBrowserApp, "", ""},
		{"another browser, no group", Context{App: safari, Browser: true, URL: "https://news.ycombinator.com/"}, Transcribe, "", SourceGlobal, "", "news.ycombinator.com"},
		{"app group", Context{App: slack}, Transcribe, "g-chat", SourceApp, "", ""},
		{"a URL is ignored for an app that is not a browser", Context{App: slack, URL: "https://github.com/"}, Transcribe, "g-chat", SourceApp, "", ""},
		{"disabled group by app", Context{App: mail}, Transcribe, "", SourceGlobal, "", ""},
		{"disabled group by url", Context{App: safari, Browser: true, URL: "https://mail.google.com/"}, Transcribe, "", SourceGlobal, "", "mail.google.com"},
		{"feature off: chat does not translate", Context{App: slack}, Translate, "", SourceGlobal, "", ""},
		{"feature off: rewrite of chat", Context{App: slack}, Rewrite, "", SourceGlobal, "", ""},
		{"feature on: rewrite in GitHub", Context{App: safari, Browser: true, URL: "https://github.com/x"}, Rewrite, "g-github", SourceURL, "github.com", "github.com"},
		{"feature off skips to the next match", Context{App: chrome, Browser: true, URL: "https://docs.google.com/d"}, Transcribe, "g-work", SourceBrowserApp, "", "docs.google.com"},
		{"translate-only group by url", Context{App: chrome, Browser: true, URL: "https://docs.google.com/d"}, Translate, "g-translate", SourceURL, "docs.google.com", "docs.google.com"},
		{"translate-only group by app", Context{App: notes}, Translate, "g-translate", SourceApp, "", ""},
		{"no app", Context{}, Transcribe, "", SourceGlobal, "", ""},
		{"bundle ids ignore case", Context{App: proto.App{BundleID: "COM.TINYSPECK.SLACKMACGAP"}}, Transcribe, "g-chat", SourceApp, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := Resolve(fixture(), tc.ctx, tc.feature)
			if m.GroupID() != tc.group || m.Source != tc.source || m.Pattern != tc.pattern || m.Host != tc.host {
				t.Errorf("got group %q source %q pattern %q host %q", m.GroupID(), m.Source, m.Pattern, m.Host)
			}
			if m.Matched() != (tc.group != "") {
				t.Errorf("Matched = %v", m.Matched())
			}
			if m.App != tc.ctx.App {
				t.Errorf("App = %+v", m.App)
			}
		})
	}
}

func TestResolveFirstGroupOnATie(t *testing.T) {
	r := Rules{Groups: []Group{
		{ID: "a", Enabled: true, URLs: []string{"example.com/*"}, Apps: []App{{BundleID: "x.app"}}, Features: DefaultFeatures},
		{ID: "b", Enabled: true, URLs: []string{"example.com"}, Apps: []App{{BundleID: "x.app"}}, Features: DefaultFeatures},
	}}
	if m := Resolve(r, Context{App: proto.App{BundleID: "b.browser"}, Browser: true, URL: "https://example.com/x"}, Transcribe); m.GroupID() != "a" {
		t.Errorf("url tie: %q", m.GroupID())
	}
	if m := Resolve(r, Context{App: proto.App{BundleID: "x.app"}}, Transcribe); m.GroupID() != "a" {
		t.Errorf("app tie: %q", m.GroupID())
	}
	// The match is a copy.
	m := Resolve(r, Context{App: proto.App{BundleID: "x.app"}}, Transcribe)
	m.Group.Name = "changed"
	if r.Groups[0].Name != "" {
		t.Error("Resolve returned the rules' own group")
	}
}

func TestDocument(t *testing.T) {
	r, err := Parse(nil)
	if err != nil || len(r.Groups) != 0 || r.Version != Version {
		t.Fatalf("empty: %+v %v", r, err)
	}
	if _, err := Parse([]byte("{")); err == nil {
		t.Error("bad JSON parsed")
	}
	in := Rules{Groups: []Group{
		{ID: "a", Name: "  Chat ", Enabled: true, Apps: []App{{" com.x ", "X"}, {"COM.X", "dup"}, {"", "blank"}},
			URLs: []string{" github.com ", "https://github.com/", "", "linear.app"}, AutoKey: " Return ", Features: DefaultFeatures},
		{ID: "", Name: "no id"},
		{ID: "a", Name: "dup id"},
	}}
	data, err := in.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	out, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	want := Rules{Version: Version, Groups: []Group{{ID: "a", Name: "Chat", Enabled: true, Apps: []App{{"com.x", "X"}},
		URLs: []string{"github.com", "linear.app"}, AutoKey: "Return", Features: DefaultFeatures}}}
	if !reflect.DeepEqual(out, want) {
		t.Errorf("round trip:\n got %+v\nwant %+v", out, want)
	}
	empty, _ := Rules{}.Marshal()
	if string(empty) != `{"version":1,"groups":[]}` {
		t.Errorf("empty document %s", empty)
	}
	if g := NewGroup("n", "New"); !g.Enabled || g.Features != DefaultFeatures || g.Features.Translate {
		t.Errorf("NewGroup %+v", g)
	}
	if _, ok := out.Group("a"); !ok {
		t.Error("Group(a)")
	}
	if _, ok := out.Group("zz"); ok {
		t.Error("Group(zz)")
	}
}

func TestAutoKey(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want []proto.KeyPress
		err  error
	}{
		{"", nil, nil},
		{"Return", []proto.KeyPress{{Key: 36}}, nil},
		{"command+return", []proto.KeyPress{{Key: 36, Modifiers: []string{proto.ModCommand}}}, nil},
		{"Shift+Control+Return", []proto.KeyPress{{Key: 36, Modifiers: []string{proto.ModControl, proto.ModShift}}}, nil},
		{"RightCommand+Return", []proto.KeyPress{{Key: 36, Modifiers: []string{proto.ModCommand}}}, nil},
		{"Tab", []proto.KeyPress{{Key: 48}}, nil},
		{"Command", nil, ErrAutoKey},
		{"Fn+Return", nil, ErrAutoKey},
		{"Return+Tab", nil, hotkey.ErrTwoKeys},
		{"Bogus", nil, hotkey.ErrUnknown},
	} {
		got, err := ParseAutoKey(tc.in)
		if !errors.Is(err, tc.err) || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ParseAutoKey(%q) = %+v, %v", tc.in, got, err)
		}
	}
	for in, want := range map[string]string{"command+return": "Command+Return", "RightCommand+Return": "Command+Return", "x+y+": "x+y+"} {
		if got := CanonicalAutoKey(in); got != want {
			t.Errorf("CanonicalAutoKey(%q) = %q", in, got)
		}
	}
}

func TestBrowsers(t *testing.T) {
	custom := []settings.Browser{{BundleID: "org.example.browser", Name: "Example"}, {BundleID: "com.google.chrome"}, {BundleID: " "}, {BundleID: "net.x.y"}}
	if !IsBrowser("com.brave.Browser", nil) || !IsBrowser("com.microsoft.edgemac", nil) || !IsBrowser("company.thebrowser.Browser", nil) {
		t.Error("Chromium family")
	}
	if IsBrowser("org.example.browser", nil) || !IsBrowser("org.example.browser", custom) {
		t.Error("custom browser")
	}
	if IsBrowser("", custom) || IsBrowser("com.tinyspeck.slackmacgap", custom) {
		t.Error("not a browser")
	}
	bs := Browsers(custom)
	if len(bs) != len(Builtin)+2 || bs[len(bs)-1].Name != "net.x.y" || bs[len(bs)-2].Name != "Example" {
		t.Errorf("Browsers: %+v", bs[len(Builtin):])
	}
	if !IsBuiltin("com.apple.Safari") || IsBuiltin("org.example.browser") {
		t.Error("IsBuiltin")
	}
}

func TestTreeOnDemand(t *testing.T) {
	ids := TreeOnDemand([]settings.Browser{{BundleID: "org.example.browser"}, {BundleID: "com.google.chrome"}})
	for _, want := range []string{"com.google.Chrome", "com.google.Chrome.canary", "org.chromium.Chromium", "com.microsoft.edgemac",
		"com.brave.Browser", "company.thebrowser.Browser", "company.thebrowser.dia", "com.vivaldi.Vivaldi", "com.operasoftware.Opera",
		"org.example.browser"} {
		if !slices.Contains(ids, want) {
			t.Errorf("TreeOnDemand lacks %s: %v", want, ids)
		}
	}
	for _, not := range []string{"com.apple.Safari", "org.mozilla.firefox", "app.zen-browser.zen", "com.google.chrome"} {
		if slices.Contains(ids, not) {
			t.Errorf("TreeOnDemand has %s", not)
		}
	}
}
