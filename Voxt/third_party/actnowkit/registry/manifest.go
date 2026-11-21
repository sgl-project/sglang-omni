// Package registry is family discovery (spec §6): each app writes its
// manifest apps/<bundle id>.json at every start, and finds the others from
// their manifests, LaunchServices (installed but never started) and
// NSRunningApplication (running).
//
//	m := registry.Manifest{App: "dicta", BundleID: id, Name: "Dicta", Version: v, Path: registry.BundlePath(), …}
//	registry.Write(d, m)
//	found, _ := registry.Discover(d, registry.DefaultSystem)
//	store.Apps = func() map[string]models.AppState { return registry.States(found) }
package registry

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/shortcuts"
)

// Kind is the schema kind of apps/<bundle id>.json.
const Kind = "app"

// Capabilities (spec §6.2); the list may grow.
const (
	CapModelsHF  = "models.hf" // uses the family model library
	CapASRLocal  = "asr.local"
	CapASRRemote = "asr.remote"
	CapTTSLocal  = "tts.local"
	CapLLMLocal  = "llm.local"
	CapLLMRemote = "llm.remote"
	CapDictation = "dictation" // voice input / dictation into other apps
	CapMeeting   = "meeting"
	CapColor     = "color"
	CapAPILocal  = "api.local"
	CapAccount   = "account" // actnow.dev sign-in (spec §12)
	// CapSubtitles: video subtitles (transcribe, translate, style, burn
	// in). Such an app opens <scheme>://import?path=<file> (a video,
	// audio or subtitle file another app hands over).
	CapSubtitles = "subtitles"
	// CapLearning: local language learning from media (a study player
	// over subtitles, word lookup, a vocabulary with spaced review). Such
	// an app opens <scheme>://import?path=<file>[&subtitles=<file>] (a
	// video or audio file another app hands over, with an optional
	// subtitle file to use instead of transcribing it).
	CapLearning = "learning"
	// CapLibrary: a library of media and documents that takes files other
	// apps hand over ("Save to Act"). Such an app opens
	// <scheme>://import?path=<file>[&path=…][&tags=a,b][&note=…][&source=<url>]
	// and <scheme>://new?url=<link> (fills its collect window, no download).
	CapLibrary = "library"
	// CapEmbedLocal: a local multimodal embedding model (search by meaning
	// over text, images, audio and video).
	CapEmbedLocal = "embed.local"
	// CapVLMLocal: a local vision language model (image understanding).
	CapVLMLocal = "vlm.local"
)

// Link keys of Manifest.Links.
const (
	LinkModels    = "models"
	LinkShortcuts = "shortcuts"
	LinkNetwork   = "network"
)

// ModelsInfo is the manifest's "models" object.
type ModelsInfo struct {
	Root  string       `json:"root"` // the $HUB the app uses
	Extra family.Extra `json:"-"`
}

func (m *ModelsInfo) UnmarshalJSON(b []byte) error {
	type plain ModelsInfo
	return family.UnmarshalKnown(b, (*plain)(m), &m.Extra)
}

func (m ModelsInfo) MarshalJSON() ([]byte, error) {
	type plain ModelsInfo
	return family.MarshalKnown(plain(m), m.Extra)
}

// KeychainRef names a Keychain item.
type KeychainRef struct {
	Service string `json:"service"`
	Account string `json:"account"`
}

// ServiceAuth is how a local service authenticates.
type ServiceAuth struct {
	Type     string       `json:"type"` // "bearer", "none"
	Keychain *KeychainRef `json:"keychain,omitempty"`
}

// Service is a local service an app offers the others (spec §11.2).
type Service struct {
	ID        string       `json:"id"`
	Kind      string       `json:"kind"` // "openai-audio", "openai-chat"…
	URL       string       `json:"url"`
	Endpoints []string     `json:"endpoints,omitempty"`
	Auth      *ServiceAuth `json:"auth,omitempty"`
	PortFile  string       `json:"portFile,omitempty"`
	Extra     family.Extra `json:"-"`
}

func (s *Service) UnmarshalJSON(b []byte) error {
	type plain Service
	return family.UnmarshalKnown(b, (*plain)(s), &s.Extra)
}

func (s Service) MarshalJSON() ([]byte, error) {
	type plain Service
	return family.MarshalKnown(plain(s), s.Extra)
}

// Manifest is apps/<bundle id>.json (spec §6.2). Only its app writes it.
type Manifest struct {
	Schema       string               `json:"schema"`
	App          string               `json:"app"`
	BundleID     string               `json:"bundleId"`
	Variant      family.Variant       `json:"variant"`
	Name         string               `json:"name"`
	Version      string               `json:"version"`
	Channel      string               `json:"channel,omitempty"` // "stable" / "beta"
	Path         string               `json:"path"`              // the running .app
	TeamID       string               `json:"teamId,omitempty"`  // "" = ad hoc
	SpecVersion  int                  `json:"specVersion"`
	Updated      time.Time            `json:"updated"`
	PID          int                  `json:"pid,omitempty"` // a hint only (spec §6.3)
	URLScheme    string               `json:"urlScheme,omitempty"`
	Links        map[string]string    `json:"links,omitempty"`
	DataDir      string               `json:"dataDir,omitempty"`
	Capabilities []string             `json:"capabilities"`
	Models       *ModelsInfo          `json:"models,omitempty"`
	Shortcuts    []shortcuts.Shortcut `json:"shortcuts,omitempty"`
	Services     []Service            `json:"services,omitempty"`
	Extra        family.Extra         `json:"-"`
}

func (m *Manifest) UnmarshalJSON(b []byte) error {
	type plain Manifest
	return family.UnmarshalKnown(b, (*plain)(m), &m.Extra)
}

func (m Manifest) MarshalJSON() ([]byte, error) {
	type plain Manifest
	return family.MarshalKnown(plain(m), m.Extra)
}

// Has reports a capability.
func (m Manifest) Has(capability string) bool { return slices.Contains(m.Capabilities, capability) }

// Validate checks the required members (spec §6.2) and the shortcuts'
// keys.
func (m Manifest) Validate() error {
	var missing []string
	for name, v := range map[string]string{"app": m.App, "bundleId": m.BundleID, "variant": string(m.Variant), "name": m.Name, "version": m.Version, "path": m.Path} {
		if v == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		slices.Sort(missing)
		return fmt.Errorf("manifest of %q: missing %s", m.BundleID, strings.Join(missing, ", "))
	}
	if m.Has(CapModelsHF) && (m.Models == nil || m.Models.Root == "") {
		return fmt.Errorf("manifest of %q: models.hf without models.root", m.BundleID)
	}
	for _, s := range m.Shortcuts {
		if _, err := shortcuts.Parse(s.Keys); err != nil {
			return fmt.Errorf("manifest of %q: shortcut %s: %w", m.BundleID, s.ID, err)
		}
	}
	return nil
}

// Write writes the app's manifest (spec §6.2): at every start, and when
// the version, shortcuts, capabilities or services change. Schema,
// specVersion, updated and pid are filled in; variant from the bundle id
// when empty; shortcut keys and triggers are normalised. Members a newer
// version of the app wrote are kept. Posts TopicApps.
func Write(d family.Dir, m Manifest) error {
	if m.Variant == "" {
		if _, v, err := family.ParseBundleID(m.BundleID); err == nil {
			m.Variant = v
		}
	}
	if m.App == "" {
		if a, _, err := family.ParseBundleID(m.BundleID); err == nil {
			m.App = a.ID
		}
	}
	m.Schema = family.SchemaOf(Kind, 1)
	m.SpecVersion = family.SpecVersion
	m.Updated = family.Now()
	if m.PID == 0 {
		m.PID = os.Getpid()
	}
	if m.Capabilities == nil {
		m.Capabilities = []string{}
	}
	for i, s := range m.Shortcuts {
		if k, err := shortcuts.Normalize(s.Keys); err == nil {
			m.Shortcuts[i].Keys = k
		}
		m.Shortcuts[i].Trigger = shortcuts.NormalizeTrigger(s.Trigger)
	}
	if err := m.Validate(); err != nil {
		return err
	}
	if m.Extra == nil {
		var old Manifest
		err := family.Load(d.AppFile(m.BundleID), Kind, 1, &old)
		if errors.Is(err, family.ErrNewerSchema) {
			return err
		}
		if err == nil {
			m.Extra = old.Extra
		}
	}
	if err := family.WriteJSON(d.AppFile(m.BundleID), m); err != nil {
		return err
	}
	_ = d.Post(family.TopicApps)
	return nil
}

// Read reads one manifest.
func Read(d family.Dir, bundleID string) (Manifest, error) {
	var m Manifest
	err := family.Load(d.AppFile(bundleID), Kind, 1, &m)
	if errors.Is(err, family.ErrNewerSchema) {
		err = nil
	}
	return m, err
}

// List reads every manifest. Files that cannot be read (corrupt,
// another kind) are reported in errs and skipped. A manifest whose
// bundleId differs from its file name is skipped too (spec §13: other
// apps' files are untrusted).
func List(d family.Dir) (ms []Manifest, errs []error) {
	entries, err := os.ReadDir(d.AppsDir())
	if err != nil {
		if !os.IsNotExist(err) {
			errs = append(errs, err)
		}
		return nil, errs
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".json") {
			continue
		}
		bid := strings.TrimSuffix(name, ".json")
		m, err := Read(d, bid)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if m.BundleID != bid {
			errs = append(errs, fmt.Errorf("%s: names %q", name, m.BundleID))
			continue
		}
		ms = append(ms, m)
	}
	slices.SortFunc(ms, func(a, b Manifest) int { return strings.Compare(a.BundleID, b.BundleID) })
	return ms, errs
}

// Remove deletes a manifest (cleaning up an uninstalled app, spec §16.1
// scenario 8; models.Store.CleanOrphans does it too).
func Remove(d family.Dir, bundleID string) error {
	err := os.Remove(d.AppFile(bundleID))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = d.Post(family.TopicApps)
	return nil
}
