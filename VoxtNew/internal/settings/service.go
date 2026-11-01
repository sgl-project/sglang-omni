package settings

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
)

// Key is the document the settings are stored under (store.Documents).
const Key = "settings:app"

// Documents is where a Store keeps its document; store.Documents
// satisfies it.
type Documents interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Put(ctx context.Context, key string, doc []byte) error
}

// Scales are the UI scales Appearance offers (90, 100, 110 and 125 %).
var Scales = []float64{0.9, 1, 1.1, 1.25}

// Themes are the values of Appearance.Theme.
var Themes = []string{"system", "light", "dark"}

// Sidebar width bounds, DIPs at 100 %.
const (
	MinSidebarWidth = 170
	MaxSidebarWidth = 320
)

// Store implements Service over Documents.
type Store struct {
	docs    Documents
	publish func(Changed)

	update sync.Mutex // serializes Update: read, save, swap
	mu     sync.RWMutex
	cur    Settings
}

var _ Service = (*Store)(nil)

// Open loads the settings document from docs and returns a Store that
// saves there. publish (may be nil) receives Changed after each saved
// update; main wires it to the core.Bus. A document that does not decode
// is replaced by the defaults at the next update: Open returns the Store
// with defaults and the error, for the log.
func Open(ctx context.Context, docs Documents, publish func(Changed)) (*Store, error) {
	s := &Store{docs: docs, publish: publish, cur: Defaults()}
	data, ok, err := docs.Get(ctx, Key)
	if err != nil {
		return s, fmt.Errorf("settings: %w", err)
	}
	if !ok {
		return s, nil
	}
	cur, err := Migrate(data)
	s.cur = cur
	if err != nil {
		return s, fmt.Errorf("settings: %w", err)
	}
	return s, nil
}

// NewMemory returns a Store with the defaults that keeps them in memory,
// for tests.
func NewMemory() *Store {
	s, _ := Open(context.Background(), &memDocs{}, nil)
	return s
}

// Get returns a copy of the current settings, from memory.
func (s *Store) Get() Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return clone(s.cur)
}

// Update applies fn to a copy, normalizes it, saves it, swaps it in and
// publishes Changed. Nothing is saved or published when fn changes
// nothing.
func (s *Store) Update(ctx context.Context, fn func(*Settings)) error {
	s.update.Lock()
	old := s.Get()
	next := clone(old)
	fn(&next)
	next = normalize(next)
	if reflect.DeepEqual(old, next) {
		s.update.Unlock()
		return nil
	}
	data, err := json.MarshalIndent(next, "", "  ")
	if err == nil {
		err = s.docs.Put(ctx, Key, data)
	}
	if err != nil {
		s.update.Unlock()
		return fmt.Errorf("settings: %w", err)
	}
	s.mu.Lock()
	s.cur = next
	s.mu.Unlock()
	s.update.Unlock()
	if s.publish != nil {
		s.publish(Changed{Old: old, New: clone(next)})
	}
	return nil
}

// clone copies s deeply (its slices), so callers never share state.
func clone(s Settings) Settings {
	s.General.OtherLanguages = slices.Clone(s.General.OtherLanguages)
	s.Appearance.ClosedSections = slices.Clone(s.Appearance.ClosedSections)
	s.Audio.Devices = slices.Clone(s.Audio.Devices)
	s.Audio.DeviceNames = maps.Clone(s.Audio.DeviceNames)
	if p := s.Audio.MuteOutput; p != nil {
		on := *p
		s.Audio.MuteOutput = &on
	}
	f := &s.Features
	f.Transcribe.Hotkeys = slices.Clone(f.Transcribe.Hotkeys)
	f.Translate.Hotkeys = slices.Clone(f.Translate.Hotkeys)
	f.Rewrite.Hotkeys = slices.Clone(f.Rewrite.Hotkeys)
	s.AppEnhancement.Browsers = slices.Clone(s.AppEnhancement.Browsers)
	s.Family.Follow = maps.Clone(s.Family.Follow)
	s.Family.Conflicts = slices.Clone(s.Family.Conflicts)
	s.Family.Prompted = slices.Clone(s.Family.Prompted)
	s.Family.KeysSince = maps.Clone(s.Family.KeysSince)
	return s
}

// normalize keeps values in their ranges.
func normalize(s Settings) Settings {
	s.Version = Version
	ap := &s.Appearance
	if !slices.Contains(Themes, ap.Theme) {
		ap.Theme = "system"
	}
	best := 1.0 // snap the scale to the nearest offered one
	for _, v := range Scales {
		if abs(v-ap.Scale) < abs(best-ap.Scale) {
			best = v
		}
	}
	ap.Scale = best
	if ap.SidebarWidth != 0 {
		ap.SidebarWidth = min(max(ap.SidebarWidth, MinSidebarWidth), MaxSidebarWidth)
	}
	ap.ClosedSections = uniq(ap.ClosedSections)
	s.General.MainLanguage = strings.TrimSpace(s.General.MainLanguage)
	s.General.OtherLanguages = uniq(s.General.OtherLanguages)
	s.Audio.Devices = uniq(s.Audio.Devices)
	for id := range s.Audio.DeviceNames { // names of the listed devices only
		if !slices.Contains(s.Audio.Devices, id) {
			delete(s.Audio.DeviceNames, id)
		}
	}
	if len(s.Audio.DeviceNames) == 0 {
		s.Audio.DeviceNames = nil
	}
	o := &s.Overlay
	if o.Position != "top" {
		o.Position = "bottom"
	}
	if o.Opacity == 0 {
		o.Opacity = DefaultOpacity
	}
	o.Opacity = min(max(o.Opacity, 0.5), 1)
	o.Radius = min(max(o.Radius, 0), 40)
	o.Margin = min(max(o.Margin, 0), 400)
	f := &s.Features
	for _, hk := range []*[]Hotkey{&f.Transcribe.Hotkeys, &f.Translate.Hotkeys, &f.Rewrite.Hotkeys} {
		*hk = slices.DeleteFunc(*hk, func(h Hotkey) bool { return strings.TrimSpace(h.Keys) == "" })
		for i := range *hk {
			(*hk)[i] = normalizeHotkey((*hk)[i])
		}
	}
	if f.PasteLast.Keys != "" {
		f.PasteLast = normalizeHotkey(f.PasteLast)
	}
	for _, f := range []*Feature{&f.Transcribe.Feature, &f.Translate.Feature, &f.Rewrite.Feature} {
		f.ASR, f.LLM = strings.TrimSpace(f.ASR), strings.TrimSpace(f.LLM)
	}
	return s
}

func normalizeHotkey(h Hotkey) Hotkey {
	h.Keys = strings.TrimSpace(h.Keys)
	switch h.Trigger {
	case TriggerTap, TriggerDouble, TriggerLong:
	default:
		h.Trigger = TriggerTap
	}
	return h
}

// uniq trims list's strings and drops empty and repeated ones.
func uniq(list []string) []string {
	var out []string
	for _, v := range list {
		if v = strings.TrimSpace(v); v != "" && !slices.Contains(out, v) {
			out = append(out, v)
		}
	}
	return out
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

type memDocs struct {
	mu sync.Mutex
	m  map[string][]byte
}

func (d *memDocs) Get(_ context.Context, key string) ([]byte, bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	v, ok := d.m[key]
	return slices.Clone(v), ok, nil
}

func (d *memDocs) Put(_ context.Context, key string, doc []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.m == nil {
		d.m = map[string][]byte{}
	}
	d.m[key] = slices.Clone(doc)
	return nil
}
