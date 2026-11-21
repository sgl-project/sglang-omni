package sounds

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

// A choice is what a setting holds for an event:
//
//	""             the event's default
//	"none"         no sound
//	"system:Ping"  a macOS system sound by name (NSSound's names)
//	"chime"        a sound of the catalogue
const (
	None         = "none"
	SystemPrefix = "system:"
)

// System is the choice of the macOS system sound name.
func System(name string) string { return SystemPrefix + name }

// IsSystem reports whether choice names a system sound.
func IsSystem(choice string) bool { return strings.HasPrefix(choice, SystemPrefix) }

// SystemName is the system sound a choice names ("" for other choices).
func SystemName(choice string) string {
	if IsSystem(choice) {
		return strings.TrimPrefix(choice, SystemPrefix)
	}
	return ""
}

// Event is something an app plays a sound for ("record.start", "pick",
// "poke"). Apps declare their events; the user's choices are Prefs.
type Event struct {
	ID   string
	Name Text
	// Default is the choice of a fresh install: a sound of the catalogue
	// (or None for an event that is off until the user picks a sound).
	Default string
	// Fallback is the system sound played for a catalogue sound that
	// this build does not have (no catalogue: a source build without the
	// family's sounds): "Pop". "" = silence.
	Fallback string
	// Uses are the catalogue's uses that suit the event: the picker
	// suggests their sounds first.
	Uses []string
}

// Prefs are an app's sound settings: on or off, volume, and a choice per
// event (only those that differ from the event's default are kept). Apps
// keep it in their own settings file; it is not shared with the family.
type Prefs struct {
	On     bool              `json:"on"`
	Volume int               `json:"volume"` // percent, 0–100
	Events map[string]string `json:"events,omitempty"`
}

// DefaultVolume is a fresh install's volume: cues should not compete with
// the user's own audio.
const DefaultVolume = 70

// VolumeSteps are the volumes a picker offers (its volume button's list,
// loudest first in the pop-up).
var VolumeSteps = []int{0, 25, 50, 75, 100}

// NearestVolumeStep is the step of VolumeSteps closest to v (the one a
// picker shows as chosen for a volume set before the steps).
func NearestVolumeStep(v int) int {
	best := VolumeSteps[0]
	for _, s := range VolumeSteps {
		if abs(s-v) < abs(best-v) {
			best = s
		}
	}
	return best
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// VolumeLevel is how a volume is drawn: 0 muted, 1 low (up to 50 %), 2
// high.
func VolumeLevel(v int) int {
	switch {
	case v <= 0:
		return 0
	case v <= 50:
		return 1
	}
	return 2
}

// DefaultPrefs are a fresh install's prefs: on, at DefaultVolume, every
// event on its default.
func DefaultPrefs() Prefs { return Prefs{On: true, Volume: DefaultVolume} }

// Choice is the choice for ev: the user's, or ev.Default.
func (p Prefs) Choice(ev Event) string {
	if c, ok := p.Events[ev.ID]; ok && c != "" {
		return c
	}
	return ev.Default
}

// Choose sets the choice for ev; choosing its default removes the entry.
func (p *Prefs) Choose(ev Event, choice string) {
	if choice == "" || choice == ev.Default {
		delete(p.Events, ev.ID)
		if len(p.Events) == 0 {
			p.Events = nil
		}
		return
	}
	if p.Events == nil {
		p.Events = map[string]string{}
	} else {
		p.Events = maps.Clone(p.Events)
	}
	p.Events[ev.ID] = choice
}

// Normalize clamps the volume and drops choices that are malformed.
func (p *Prefs) Normalize() {
	p.Volume = min(100, max(0, p.Volume))
	for id, c := range p.Events {
		if c == "" || len(c) > 80 || strings.ContainsAny(c, "/\\") || (!IsSystem(c) && c != None && !validID(c)) || c == SystemPrefix {
			delete(p.Events, id)
		}
	}
}

// Kind is what a resolved choice plays.
type Kind int

const (
	KindNone    Kind = iota // nothing
	KindCatalog             // a sound of the catalogue
	KindSystem              // a macOS system sound
)

// Target is what an event plays: the kind, the sound's id or system name,
// and the volume (0–1).
type Target struct {
	Kind   Kind
	Name   string
	Volume float64
}

// Resolve is what choice plays with cat (nil: no catalogue): a catalogue
// sound it has, a system sound, or fallback (a system sound name) for a
// catalogue sound it lacks.
func Resolve(cat *Catalog, choice, fallback string, volume int) Target {
	v := float64(min(100, max(0, volume))) / 100
	switch {
	case choice == None || choice == "":
		return Target{}
	case IsSystem(choice):
		if n := SystemName(choice); n != "" {
			return Target{KindSystem, n, v}
		}
		return Target{}
	}
	if _, ok := cat.Sound(choice); ok {
		return Target{KindCatalog, choice, v}
	}
	if fallback != "" {
		return Target{KindSystem, fallback, v}
	}
	return Target{}
}

// Target is what ev plays with these prefs (nothing while they are off).
func (p Prefs) Target(cat *Catalog, ev Event) Target {
	if !p.On {
		return Target{}
	}
	return Resolve(cat, p.Choice(ev), ev.Fallback, p.Volume)
}

// MigrateSystem is the choice for a setting that named the macOS system
// sound name before the family's sounds: when the user had left it on
// the app's old default (wasDefault), the event's new default; otherwise
// the system sound itself, kept as the user chose it. An empty name is
// the event's default.
func MigrateSystem(ev Event, name string, wasDefault bool) string {
	if name == "" || wasDefault {
		return ev.Default
	}
	return System(name)
}

// MigrateClosest is the catalogue's closest sound to the system sound name
// (Catalog.ForSystem), or System(name) when none replaces it.
func MigrateClosest(cat *Catalog, name string) string {
	if id := cat.ForSystem(name); id != "" {
		return id
	}
	return System(name)
}

// SystemSoundsDir is the folder of macOS's system sounds.
const SystemSoundsDir = "/System/Library/Sounds"

var soundExts = map[string]bool{".aiff": true, ".aif": true, ".caf": true, ".wav": true, ".m4a": true, ".mp3": true}

// SystemSounds lists the sounds in dir ("" = SystemSoundsDir) by name
// ("Pop" for Pop.aiff), sorted, without repeats; nil when dir cannot be
// read (off macOS).
func SystemSounds(dir string) []string {
	if dir == "" {
		dir = SystemSoundsDir
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, e := range ents {
		name := e.Name()
		ext := filepath.Ext(name)
		if e.IsDir() || strings.HasPrefix(name, ".") || !soundExts[strings.ToLower(ext)] {
			continue
		}
		if base := strings.TrimSuffix(name, ext); base != "" && !seen[base] {
			seen[base] = true
			out = append(out, base)
		}
	}
	sort.Strings(out)
	return out
}

// KnownSystemSounds are the sounds macOS 15–26 ships in
// /System/Library/Sounds.
var KnownSystemSounds = []string{"Basso", "Blow", "Bottle", "Frog", "Funk", "Glass", "Hero", "Morse", "Ping", "Pop", "Purr", "Sosumi", "Submarine", "Tink"}

// systemAudible is how long each system sound is heard (to its last
// sample at or above −60 dBFS), measured on macOS 26.7's files (Auralink's
// table); the files go on in silence for longer. Glass, Funk and Blow
// were not measured.
var systemAudible = map[string]time.Duration{
	"Pop":       215 * time.Millisecond,
	"Tink":      45 * time.Millisecond,
	"Ping":      492 * time.Millisecond,
	"Morse":     180 * time.Millisecond,
	"Submarine": 872 * time.Millisecond,
	"Basso":     169 * time.Millisecond,
	"Bottle":    564 * time.Millisecond,
	"Frog":      331 * time.Millisecond,
	"Hero":      861 * time.Millisecond,
	"Purr":      431 * time.Millisecond,
	"Sosumi":    593 * time.Millisecond,
}

// Length is how long a target is heard: the catalogue sound's audible
// length, the system sound's (0 for one not measured), 0 for none.
func (t Target) Length(cat *Catalog) time.Duration {
	switch t.Kind {
	case KindCatalog:
		if s, ok := cat.Sound(t.Name); ok {
			return s.Length()
		}
	case KindSystem:
		return systemAudible[t.Name]
	}
	return 0
}

// IsKnownSystem reports whether name is one of KnownSystemSounds.
func IsKnownSystem(name string) bool { return slices.Contains(KnownSystemSounds, name) }
