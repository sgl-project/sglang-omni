// Package appctx is app enhancement: groups of apps and website
// addresses, each with a scene prompt the composer layers on top of the
// feature's own rules, an optional key to press after inserting, and the
// features it applies to; and the decision of which group a session falls
// in, from the app that was in front when its shortcut was pressed and,
// for a browser, the address of its front tab.
//
// Everything here is pure: the address comes from the helper's
// browser.url (Accessibility only) through the caller, the groups from the
// store's document Key. Order of matching (Resolve):
//
//  1. a group whose URL rule matches the tab's address (the most specific
//     rule wins, then the earlier group);
//  2. the group the browser app itself is a member of;
//  3. the group the app in front is a member of (any app);
//  4. none: the feature's global rules only.
//
// Imports: helper/proto, hotkey.
package appctx

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
)

// Key is the store document holding the groups.
const Key = "appctx:rules"

// Version is the version of the document Rules writes.
const Version = 1

// Features a group can apply to: store.Kind's values.
const (
	Transcribe = "transcribe"
	Translate  = "translate"
	Rewrite    = "rewrite"
)

// Rules is the document: the groups, in the user's order.
type Rules struct {
	Version int     `json:"version"`
	Groups  []Group `json:"groups"`
}

// Group is a scene: apps and website addresses that share a prompt.
type Group struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Enabled bool   `json:"enabled"`
	// Prompt is the scene's rules, layered on the feature's own (the
	// composer's Scene layer); it may use the composer's variables.
	Prompt string `json:"prompt,omitempty"`
	// Apps are members by bundle id; URLs the address rules ("github.com",
	// "*.notion.so/*", "mail.google.com/mail/*").
	Apps []App    `json:"apps,omitempty"`
	URLs []string `json:"urls,omitempty"`
	// AutoKey is pressed after the result is inserted ("Return",
	// "Command+Return", internal/hotkey's grammar); "" for none.
	AutoKey string `json:"autoKey,omitempty"`
	// Features are the features it applies to.
	Features Features `json:"features"`
}

// App is a member app: its bundle id, and its name as it was added.
type App struct {
	BundleID string `json:"bundleId"`
	Name     string `json:"name,omitempty"`
}

// Features says which features a group applies to.
type Features struct {
	Transcribe bool `json:"transcribe"`
	Translate  bool `json:"translate"`
	Rewrite    bool `json:"rewrite"`
}

// Has reports whether feature is on ("" counts as Transcribe).
func (f Features) Has(feature string) bool {
	switch feature {
	case Translate:
		return f.Translate
	case Rewrite:
		return f.Rewrite
	}
	return f.Transcribe
}

// DefaultFeatures: Transcribe and Rewrite on, Translate off (a scene's
// tone rarely belongs in a translation).
var DefaultFeatures = Features{Transcribe: true, Rewrite: true}

// NewGroup is a new, enabled group named name with the default features.
func NewGroup(id, name string) Group {
	return Group{ID: id, Name: name, Enabled: true, Features: DefaultFeatures}
}

// Parse reads the document; empty data is no groups. Groups without an
// id are dropped; apps and URL rules are trimmed and deduplicated.
func Parse(data []byte) (Rules, error) {
	r := Rules{Version: Version}
	if len(strings.TrimSpace(string(data))) == 0 {
		return r, nil
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return Rules{Version: Version}, fmt.Errorf("appctx: %w", err)
	}
	r.Version = Version
	return r.Normalized(), nil
}

// Marshal writes the document.
func (r Rules) Marshal() ([]byte, error) {
	r.Version = Version
	r = r.Normalized()
	if r.Groups == nil {
		r.Groups = []Group{}
	}
	return json.Marshal(r)
}

// Normalized is r with groups without an id dropped, names trimmed, and
// each group's apps and rules trimmed and without duplicates.
func (r Rules) Normalized() Rules {
	out := Rules{Version: r.Version}
	seen := map[string]bool{}
	for _, g := range r.Groups {
		if g.ID == "" || seen[g.ID] {
			continue
		}
		seen[g.ID] = true
		out.Groups = append(out.Groups, g.Normalized())
	}
	return out
}

// Normalized is g with its fields trimmed, its apps and URL rules without
// blanks and duplicates (the first one kept).
func (g Group) Normalized() Group {
	g.Name = strings.TrimSpace(g.Name)
	g.AutoKey = strings.TrimSpace(g.AutoKey)
	var apps []App
	for _, a := range g.Apps {
		a.BundleID = strings.TrimSpace(a.BundleID)
		a.Name = strings.TrimSpace(a.Name)
		if a.BundleID == "" || slices.ContainsFunc(apps, func(b App) bool { return strings.EqualFold(b.BundleID, a.BundleID) }) {
			continue
		}
		apps = append(apps, a)
	}
	g.Apps = apps
	var urls []string
	for _, u := range g.URLs {
		u = strings.TrimSpace(u)
		if u == "" || slices.ContainsFunc(urls, func(v string) bool { return CanonicalPattern(v) == CanonicalPattern(u) }) {
			continue
		}
		urls = append(urls, u)
	}
	g.URLs = urls
	return g
}

// Group returns the group with id.
func (r Rules) Group(id string) (Group, bool) {
	for _, g := range r.Groups {
		if g.ID == id {
			return g, true
		}
	}
	return Group{}, false
}

// HasApp reports whether bundleID is a member.
func (g Group) HasApp(bundleID string) bool {
	if bundleID == "" {
		return false
	}
	for _, a := range g.Apps {
		if strings.EqualFold(a.BundleID, bundleID) {
			return true
		}
	}
	return false
}

// ErrAutoKey: an auto key that is not one key with modifiers.
var ErrAutoKey = errors.New("appctx: the auto key must be one key, with modifiers or not")

// ParseAutoKey reads an auto key ("Return", "Command+Return",
// "Shift+Return"); "" is none (nil). fn and sides are not posted.
func ParseAutoKey(keys string) ([]proto.KeyPress, error) {
	keys = strings.TrimSpace(keys)
	if keys == "" {
		return nil, nil
	}
	c, err := hotkey.Parse(keys)
	if err != nil {
		return nil, err
	}
	if c.ModifierOnly() {
		return nil, ErrAutoKey
	}
	if _, fn := c.Mods[proto.ModFn]; fn {
		return nil, ErrAutoKey
	}
	k := proto.KeyPress{Key: c.Code}
	for _, f := range []string{proto.ModControl, proto.ModOption, proto.ModShift, proto.ModCommand} {
		if _, ok := c.Mods[f]; ok {
			k.Modifiers = append(k.Modifiers, f)
		}
	}
	return []proto.KeyPress{k}, nil
}

// CanonicalAutoKey is keys in canonical spelling ("command+return" is
// "Command+Return"), or keys as written when it does not parse.
func CanonicalAutoKey(keys string) string {
	keys = strings.TrimSpace(keys)
	c, err := hotkey.Parse(keys)
	if err != nil {
		return keys
	}
	// Sides are not posted: write the families only.
	for f := range c.Mods {
		c.Mods[f] = proto.SideAny
	}
	return c.String()
}
