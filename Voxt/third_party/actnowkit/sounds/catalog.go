// Package sounds is the family's cue sounds: a catalogue of sounds an app
// ships (catalog.json and the audio files, read from an fs.FS the app
// passes in: actnowkit carries no audio), the choice of a sound for each of
// the app's events (a sound of the catalogue, a macOS system sound, or
// none), filtering by group, tag and text for a picker, and a player with
// low latency for quick repeats (AVAudioPlayer on macOS through purego; a
// silent output elsewhere).
//
//	cat, err := sounds.Load(appResources, "sounds")     // sounds/catalog.json
//	player := sounds.NewPlayer(cat, nil)                 // nil: the platform's output
//	ev := sounds.Event{ID: "record.start", Default: "tally", Fallback: "Pop"}
//	player.Preload(prefs.Choice(ev))
//	player.PlayEvent(ev, prefs)                          // the user's choice, at their volume
//	player.Preview("chime", prefs.Volume)                // the picker's ▶, cutting off the last preview
//
// The family's catalogue lives in a private repository (the sounds are
// licensed for use inside apps, not for redistribution on their own); apps
// copy it into their bundles. A build without it (a source build of an open
// source app) passes a nil catalogue: every event then plays its Fallback
// system sound.
package sounds

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strings"
	"sync"
	"time"
)

// Schema is the schema of catalog.json.
const Schema = "actnow.sounds.catalog/1"

// CatalogFile is the catalogue's file name in its directory.
const CatalogFile = "catalog.json"

// Text is a name in the interface languages.
type Text struct {
	En string `json:"en"`
	Zh string `json:"zh,omitempty"`
}

// In is the text in lang (a BCP-47 tag: "zh", "zh-Hans", "zh-CN" … pick
// Chinese, anything else English); English when there is no translation.
func (t Text) In(lang string) string {
	if t.Zh != "" && (lang == "zh" || strings.HasPrefix(strings.ToLower(lang), "zh-") || strings.HasPrefix(strings.ToLower(lang), "zh_")) {
		return t.Zh
	}
	return t.En
}

// Group is a group of sounds (a picker's section).
type Group struct {
	ID   string `json:"id"`
	Name Text   `json:"name"`
}

// Tag is a label sounds carry, for filtering ("soft", "bright", "short" …).
type Tag struct {
	ID   string `json:"id"`
	Name Text   `json:"name"`
	// Featured tags are the picker's filter chips; the others still match
	// a text search.
	Featured bool `json:"featured,omitempty"`
}

// Use is a suggested use of sounds ("record.start", "success", "poke" …):
// an app's events name the uses they suit, and the picker offers those
// sounds first.
type Use struct {
	ID   string `json:"id"`
	Name Text   `json:"name"`
}

// Source says where a sound comes from.
type Source struct {
	Name     string `json:"name"`               // "React Bits Sound Lab"
	ID       string `json:"id,omitempty"`       // the source's id of the sound
	Title    string `json:"title,omitempty"`    // the source's name of the sound
	Exported string `json:"exported,omitempty"` // date of the export, 2006-01-02
	SHA256   string `json:"sha256,omitempty"`   // of the exported file, before processing
}

// Sound is one sound of the catalogue.
type Sound struct {
	ID    string   `json:"id"`
	File  string   `json:"file"` // relative to the catalogue's directory
	Name  Text     `json:"name"`
	Group string   `json:"group"`
	Tags  []string `json:"tags,omitempty"`
	// Duration is how long the file plays, in seconds.
	Duration float64 `json:"duration"`
	// Audible is how long it is heard (to its last sample above −60 dBFS),
	// in seconds; 0 = Duration. An app that waits for a cue to end (a
	// start-of-recording cue before muting the output) waits this long.
	Audible float64  `json:"audible,omitempty"`
	Desc    Text     `json:"desc,omitzero"`
	Uses    []string `json:"uses,omitempty"`
	// Pair is the sound that goes with this one (a start cue's stop cue,
	// a switch's on and off).
	Pair string `json:"pair,omitempty"`
	// Replaces lists macOS system sounds ("Pop", "Tink") this sound is the
	// closest to: a setting that named one migrates to this sound.
	Replaces []string `json:"replaces,omitempty"`
	SHA256   string   `json:"sha256,omitempty"` // of File
	Source   *Source  `json:"source,omitempty"`
}

// Length is the sound's audible length (Audible, or Duration).
func (s Sound) Length() time.Duration {
	d := s.Audible
	if d <= 0 {
		d = s.Duration
	}
	return time.Duration(d * float64(time.Second))
}

// HasTag reports whether the sound carries tag.
func (s Sound) HasTag(tag string) bool { return slices.Contains(s.Tags, tag) }

// Catalog is a catalogue of sounds and where to read their files.
type Catalog struct {
	Schema  string  `json:"schema"`
	Version string  `json:"version"`
	Groups  []Group `json:"groups"`
	Tags    []Tag   `json:"tags,omitempty"`
	Uses    []Use   `json:"uses,omitempty"`
	Sounds  []Sound `json:"sounds"`
	// Notice is the attribution the app shows (About, acknowledgements).
	Notice string `json:"notice,omitempty"`

	fsys fs.FS
	dir  string
	byID map[string]int

	mu   sync.Mutex
	data map[string][]byte // files read and checked
}

// Errors.
var (
	ErrNoCatalog = errors.New("sounds: no catalogue")
	ErrUnknown   = errors.New("sounds: unknown sound")
)

// Load reads dir/catalog.json from fsys (dir "" or "." for the root) and
// checks it: the schema, unique ids, known groups, files inside dir.
// Missing files are found when a sound is read (Data).
func Load(fsys fs.FS, dir string) (*Catalog, error) {
	if fsys == nil {
		return nil, ErrNoCatalog
	}
	dir = path.Clean(strings.TrimPrefix(dir, "/"))
	b, err := fs.ReadFile(fsys, path.Join(dir, CatalogFile))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("%w: %v", ErrNoCatalog, err)
		}
		return nil, err
	}
	return Parse(b, fsys, dir)
}

// Parse reads a catalogue from b; its files are read from fsys under dir.
func Parse(b []byte, fsys fs.FS, dir string) (*Catalog, error) {
	c := &Catalog{}
	if err := json.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("sounds: catalog.json: %w", err)
	}
	if c.Schema != Schema {
		return nil, fmt.Errorf("sounds: catalog.json: schema %q, want %q", c.Schema, Schema)
	}
	c.fsys, c.dir = fsys, path.Clean(dir)
	groups := map[string]bool{}
	for _, g := range c.Groups {
		if g.ID == "" || groups[g.ID] {
			return nil, fmt.Errorf("sounds: catalog.json: group %q empty or repeated", g.ID)
		}
		groups[g.ID] = true
	}
	c.byID = map[string]int{}
	for i, s := range c.Sounds {
		switch {
		case !validID(s.ID):
			return nil, fmt.Errorf("sounds: catalog.json: bad sound id %q", s.ID)
		case IsSystem(s.ID) || s.ID == None:
			return nil, fmt.Errorf("sounds: catalog.json: sound id %q is reserved", s.ID)
		case c.byID[s.ID] != 0:
			return nil, fmt.Errorf("sounds: catalog.json: sound %q repeated", s.ID)
		case !groups[s.Group]:
			return nil, fmt.Errorf("sounds: catalog.json: sound %q: unknown group %q", s.ID, s.Group)
		case s.File == "" || path.IsAbs(s.File) || strings.HasPrefix(path.Clean(s.File), ".."):
			return nil, fmt.Errorf("sounds: catalog.json: sound %q: bad file %q", s.ID, s.File)
		}
		c.byID[s.ID] = i + 1
	}
	return c, nil
}

func validID(id string) bool {
	if id == "" || len(id) > 64 {
		return false
	}
	for _, r := range id {
		if !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.') {
			return false
		}
	}
	return true
}

// Sound returns the sound id. A nil catalogue has none.
func (c *Catalog) Sound(id string) (Sound, bool) {
	if c == nil {
		return Sound{}, false
	}
	if i := c.byID[id]; i > 0 {
		return c.Sounds[i-1], true
	}
	return Sound{}, false
}

// Group returns the group id.
func (c *Catalog) Group(id string) (Group, bool) {
	if c == nil {
		return Group{}, false
	}
	i := slices.IndexFunc(c.Groups, func(g Group) bool { return g.ID == id })
	if i < 0 {
		return Group{}, false
	}
	return c.Groups[i], true
}

// GroupName is the group's name in lang (the id for a group the catalogue
// does not describe).
func (c *Catalog) GroupName(id, lang string) string {
	if g, ok := c.Group(id); ok {
		return g.Name.In(lang)
	}
	return id
}

// TagName is the tag's name in lang (the id itself for a tag the
// catalogue does not describe).
func (c *Catalog) TagName(id, lang string) string {
	if c != nil {
		if i := slices.IndexFunc(c.Tags, func(t Tag) bool { return t.ID == id }); i >= 0 {
			return c.Tags[i].Name.In(lang)
		}
	}
	return id
}

// UseName is the use's name in lang (the id for one not described).
func (c *Catalog) UseName(id, lang string) string {
	if c != nil {
		if i := slices.IndexFunc(c.Uses, func(u Use) bool { return u.ID == id }); i >= 0 {
			return c.Uses[i].Name.In(lang)
		}
	}
	return id
}

// ForSystem is the sound that replaces the macOS system sound name
// (Sound.Replaces), for migrating settings; "" when none does.
func (c *Catalog) ForSystem(name string) string {
	if c == nil {
		return ""
	}
	for _, s := range c.Sounds {
		if slices.ContainsFunc(s.Replaces, func(r string) bool { return strings.EqualFold(r, name) }) {
			return s.ID
		}
	}
	return ""
}

// Data reads the sound's file (once; later calls share the bytes, which
// callers must not change) and checks its SHA-256 when the catalogue
// gives one.
func (c *Catalog) Data(id string) ([]byte, error) {
	s, ok := c.Sound(id)
	if !ok {
		return nil, fmt.Errorf("%w %q", ErrUnknown, id)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if b, ok := c.data[id]; ok {
		return b, nil
	}
	b, err := fs.ReadFile(c.fsys, path.Join(c.dir, s.File))
	if err != nil {
		return nil, fmt.Errorf("sounds: %s: %w", id, err)
	}
	if s.SHA256 != "" {
		sum := sha256.Sum256(b)
		if hex.EncodeToString(sum[:]) != strings.ToLower(s.SHA256) {
			return nil, fmt.Errorf("sounds: %s: %s does not match its SHA-256", id, s.File)
		}
	}
	if c.data == nil {
		c.data = map[string][]byte{}
	}
	c.data[id] = b
	return b, nil
}

// Check reads every sound's file and returns the problems (missing files,
// SHA-256 mismatches): for an app's tests of its bundled copy.
func (c *Catalog) Check() []error {
	if c == nil {
		return []error{ErrNoCatalog}
	}
	var errs []error
	for _, s := range c.Sounds {
		if _, err := c.Data(s.ID); err != nil {
			errs = append(errs, err)
		}
	}
	return errs
}

// Subset is a catalogue of the sounds keep reports true for, with the
// groups that still have sounds (an app that ships some of the family's
// sounds builds its catalog.json this way: WriteJSON).
func (c *Catalog) Subset(keep func(Sound) bool) *Catalog {
	out := &Catalog{Schema: c.Schema, Version: c.Version, Tags: c.Tags, Uses: c.Uses, Notice: c.Notice, fsys: c.fsys, dir: c.dir, byID: map[string]int{}}
	used := map[string]bool{}
	for _, s := range c.Sounds {
		if keep(s) {
			out.Sounds = append(out.Sounds, s)
			out.byID[s.ID] = len(out.Sounds)
			used[s.Group] = true
		}
	}
	for _, g := range c.Groups {
		if used[g.ID] {
			out.Groups = append(out.Groups, g)
		}
	}
	return out
}

// JSON is the catalogue as catalog.json.
func (c *Catalog) JSON() ([]byte, error) {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
