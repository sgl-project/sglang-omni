// Package settings is the family's shared general settings (spec §9):
// settings.json (interface language, proxy, Hugging Face endpoints, model
// folder), each app's "follow the family" flags, and the first-run
// migration that merges an app's existing values without overriding the
// user's choices.
//
//	fam, _, _ := settings.Read(d)
//	lang := settings.Effective(fam, settings.KeyLanguage, follow[settings.KeyLanguage], own.Language, "")
//	settings.Set(ctx, d, bundleID, settings.KeyProxy, "socks5://127.0.0.1:7897") // when following
package settings

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/family"
)

// Kind is the schema kind of settings.json.
const Kind = "settings"

// Keys of the shared settings (spec §9.1). Phase 2 keys
// ("speech.mainLanguage", "speech.otherLanguages") are kept untouched
// when present.
const (
	KeyLanguage    = "general.language"    // BCP-47; "" = the system's
	KeyProxy       = "network.proxy"       // "" automatic, "direct", or a proxy URL
	KeyHFEndpoints = "network.hfEndpoints" // "auto", "official", "mirror" or URLs
	KeyModelsRoot  = "models.root"         // absolute path; "" = the default rules (spec §7.1)
	// KeyModelsBackup: whether Time Machine backs up the model library
	// ("on"; unset or "off" = excluded, spec §7.12). It is a property of
	// the shared folder, so it has no follow flag: every app applies the
	// family's value (models.ApplyBackup).
	KeyModelsBackup = "models.backup"
)

// Values of models.backup.
const (
	BackupOn  = "on"
	BackupOff = "off"
)

// ModelsBackup reports whether models.backup is on (default off).
func ModelsBackup(s Settings) bool {
	v, _ := s.Get(KeyModelsBackup)
	return v == BackupOn
}

// Keys lists the phase-1 keys.
var Keys = []string{KeyLanguage, KeyProxy, KeyHFEndpoints, KeyModelsRoot}

// Values of network.proxy and network.hfEndpoints.
const (
	ProxyAuto         = ""
	ProxyDirect       = "direct"
	EndpointsAuto     = "auto"
	EndpointsOfficial = "official"
	EndpointsMirror   = "mirror"
	OfficialEndpoint  = "https://huggingface.co"
	MirrorEndpoint    = "https://hf-mirror.com"
)

// General is the "general" object.
type General struct {
	Language *string      `json:"language,omitempty"`
	Extra    family.Extra `json:"-"`
}

func (g *General) UnmarshalJSON(b []byte) error {
	type plain General
	return family.UnmarshalKnown(b, (*plain)(g), &g.Extra)
}

func (g General) MarshalJSON() ([]byte, error) {
	type plain General
	return family.MarshalKnown(plain(g), g.Extra)
}

// Network is the "network" object.
type Network struct {
	Proxy       *string      `json:"proxy,omitempty"`
	HFEndpoints *string      `json:"hfEndpoints,omitempty"`
	Extra       family.Extra `json:"-"`
}

func (n *Network) UnmarshalJSON(b []byte) error {
	type plain Network
	return family.UnmarshalKnown(b, (*plain)(n), &n.Extra)
}

func (n Network) MarshalJSON() ([]byte, error) {
	type plain Network
	return family.MarshalKnown(plain(n), n.Extra)
}

// Models is the "models" object.
type Models struct {
	Root   *string      `json:"root,omitempty"`
	Backup *string      `json:"backup,omitempty"`
	Extra  family.Extra `json:"-"`
}

func (m *Models) UnmarshalJSON(b []byte) error {
	type plain Models
	return family.UnmarshalKnown(b, (*plain)(m), &m.Extra)
}

func (m Models) MarshalJSON() ([]byte, error) {
	type plain Models
	return family.MarshalKnown(plain(m), m.Extra)
}

// Settings is settings.json. A member that is absent is unset (not the
// default): pointers tell the two apart.
type Settings struct {
	Schema    string       `json:"schema"`
	Rev       int          `json:"rev"`
	Updated   time.Time    `json:"updated"`
	UpdatedBy string       `json:"updatedBy"`
	General   *General     `json:"general,omitempty"`
	Network   *Network     `json:"network,omitempty"`
	Models    *Models      `json:"models,omitempty"`
	Extra     family.Extra `json:"-"`
}

func (s *Settings) UnmarshalJSON(b []byte) error {
	type plain Settings
	return family.UnmarshalKnown(b, (*plain)(s), &s.Extra)
}

func (s Settings) MarshalJSON() ([]byte, error) {
	type plain Settings
	return family.MarshalKnown(plain(s), s.Extra)
}

// ErrUnknownKey: not a phase-1 key.
var ErrUnknownKey = errors.New("not a shared setting")

func (s *Settings) field(key string, create bool) (**string, error) {
	switch key {
	case KeyLanguage:
		if s.General == nil {
			if !create {
				return nil, nil
			}
			s.General = &General{}
		}
		return &s.General.Language, nil
	case KeyProxy, KeyHFEndpoints:
		if s.Network == nil {
			if !create {
				return nil, nil
			}
			s.Network = &Network{}
		}
		if key == KeyProxy {
			return &s.Network.Proxy, nil
		}
		return &s.Network.HFEndpoints, nil
	case KeyModelsRoot, KeyModelsBackup:
		if s.Models == nil {
			if !create {
				return nil, nil
			}
			s.Models = &Models{}
		}
		if key == KeyModelsBackup {
			return &s.Models.Backup, nil
		}
		return &s.Models.Root, nil
	}
	return nil, fmt.Errorf("%q: %w", key, ErrUnknownKey)
}

// Get returns a key's value; ok is false when it is unset.
func (s Settings) Get(key string) (value string, ok bool) {
	f, err := s.field(key, false)
	if err != nil || f == nil || *f == nil {
		return "", false
	}
	return **f, true
}

// SetValue sets a key in memory (Set writes the file). The value is
// checked (Validate) and normalised (Normalize).
func (s *Settings) SetValue(key, value string) error {
	value, err := Normalize(key, value)
	if err != nil {
		return err
	}
	f, err := s.field(key, true)
	if err != nil {
		return err
	}
	*f = &value
	return nil
}

// Unset removes a key in memory.
func (s *Settings) Unset(key string) error {
	f, err := s.field(key, false)
	if err != nil {
		return err
	}
	if f != nil {
		*f = nil
	}
	return nil
}

// Normalize checks a value and returns it in the form written to the file:
// trimmed; endpoint URLs without trailing slashes; a cleaned folder path.
func Normalize(key, value string) (string, error) {
	value = strings.TrimSpace(value)
	switch key {
	case KeyLanguage:
		if value != "" && !validLanguage(value) {
			return "", fmt.Errorf("language %q: not a BCP-47 tag", value)
		}
	case KeyProxy:
		if value == ProxyAuto || value == ProxyDirect {
			return value, nil
		}
		u, err := url.Parse(value)
		if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https" && u.Scheme != "socks5" && u.Scheme != "socks5h") {
			return "", fmt.Errorf("proxy %q: want \"\", \"direct\" or an http://, https:// or socks5:// URL", value)
		}
	case KeyHFEndpoints:
		switch value {
		case EndpointsAuto, EndpointsOfficial, EndpointsMirror:
			return value, nil
		case "":
			return EndpointsAuto, nil
		}
		var out []string
		for _, f := range strings.Fields(value) {
			u, err := url.Parse(f)
			if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") {
				return "", fmt.Errorf("endpoint %q: not an http(s) URL", f)
			}
			out = append(out, strings.TrimRight(f, "/"))
		}
		return strings.Join(out, " "), nil
	case KeyModelsRoot:
		if value == "" {
			return "", nil
		}
		if !filepath.IsAbs(value) {
			return "", fmt.Errorf("model folder %q: not an absolute path", value)
		}
		return filepath.Clean(value), nil
	case KeyModelsBackup:
		switch strings.ToLower(value) {
		case "", BackupOff, "false", "no", "0":
			return BackupOff, nil
		case BackupOn, "true", "yes", "1":
			return BackupOn, nil
		}
		return "", fmt.Errorf("models.backup %q: want \"on\" or \"off\"", value)
	default:
		return "", fmt.Errorf("%q: %w", key, ErrUnknownKey)
	}
	return value, nil
}

func validLanguage(tag string) bool {
	parts := strings.Split(tag, "-")
	for i, p := range parts {
		if len(p) == 0 || len(p) > 8 {
			return false
		}
		for _, c := range p {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
				return false
			}
		}
		if i == 0 && (len(p) < 2 || len(p) > 3) {
			return false
		}
	}
	return true
}

// Endpoints turns network.hfEndpoints into the endpoints to race:
// auto → official and mirror; official; mirror; or the listed URLs.
func Endpoints(value string) []string {
	switch strings.TrimSpace(value) {
	case "", EndpointsAuto:
		return []string{OfficialEndpoint, MirrorEndpoint}
	case EndpointsOfficial:
		return []string{OfficialEndpoint}
	case EndpointsMirror:
		return []string{MirrorEndpoint}
	}
	var out []string
	for _, f := range strings.Fields(value) {
		out = append(out, strings.TrimRight(f, "/"))
	}
	return out
}

// Read reads settings.json. readOnly is true when it has a newer major
// schema (show family.TextNewerFormat; do not write). A missing file is
// empty settings; a corrupt one is reported, and treated as empty by
// Update (which quarantines it).
func Read(d family.Dir) (s Settings, readOnly bool, err error) {
	err = family.Load(d.SettingsFile(), Kind, 1, &s)
	switch {
	case family.IsNotExist(err):
		return Settings{}, false, nil
	case errors.Is(err, family.ErrNewerSchema):
		return s, true, nil
	}
	return s, false, err
}

// Update is the read-modify-write of settings.json (spec §9.2): under
// locks/settings.lock, fn changes the settings, rev goes up by one,
// updated/updatedBy are set, the file is replaced atomically and
// TopicSettings posted. fn may return family.ErrNoChange.
func Update(ctx context.Context, d family.Dir, bundleID string, fn func(s *Settings) error) (Settings, error) {
	var s Settings
	changed := false
	err := d.Update(ctx, family.LockSettings, d.SettingsFile(), Kind, 1, &s, func(bool) error {
		if err := fn(&s); err != nil {
			return err
		}
		s.Schema = family.SchemaOf(Kind, 1)
		s.Rev++
		s.Updated, s.UpdatedBy = family.Now(), bundleID
		changed = true
		return nil
	})
	if err == nil && changed {
		_ = d.Post(family.TopicSettings)
	}
	return s, err
}

// Set writes one key (for an app whose flag for it is following).
func Set(ctx context.Context, d family.Dir, bundleID, key, value string) (Settings, error) {
	return Update(ctx, d, bundleID, func(s *Settings) error {
		if cur, ok := s.Get(key); ok {
			if n, err := Normalize(key, value); err == nil && n == cur {
				return family.ErrNoChange
			}
		}
		return s.SetValue(key, value)
	})
}

// Effective is a setting's value for an app (spec §9.3):
// follow ? (family's value if set, else appDefault) : own.
// Environment variables (HF_ENDPOINT) still override the
// result where the app reads them, as before.
func Effective(fam Settings, key string, follow bool, own, appDefault string) string {
	if !follow {
		return own
	}
	if v, ok := fam.Get(key); ok {
		return v
	}
	return appDefault
}

// Follow holds an app's follow flags, kept in the app's own settings
// (e.g. "family.follow.language"). A key that is missing follows: new
// installs follow everything (spec §9.3).
type Follow map[string]bool

// Of reports whether key follows the family.
func (f Follow) Of(key string) bool {
	v, ok := f[key]
	return !ok || v
}

// Local is an app's own value of a shared key, for Migrate.
type Local struct {
	Value string
	// IsDefault: the user never changed it in this app.
	IsDefault bool
	// InUse: the default is in real use and must not silently change —
	// for models.root, the default folder holds this app's models.
	InUse bool
}

// Outcome is what Migrate decided for a key.
type Outcome struct {
	Key    string
	Follow bool // the app's new follow flag
	Wrote  bool // the app's value became the family's
	// Conflict: the family has another value and the app keeps its own;
	// show TextOtherValue once on the setting's row (for models.root,
	// TextMoveModels from the models package).
	Conflict bool
	Family   string // the family's value, when set
}

// Migrate merges an app's values into the family settings at the app's
// first start with the spec (spec §9.4, §15.2 steps 4–5):
//
//   - the family has no value: a value the user changed is written to the
//     family (first come, first served), and the app follows;
//   - the family has the same value: the app follows;
//   - the family has another value: an app on its default (and not InUse)
//     follows — a new install adopts the family's choice; otherwise the
//     app keeps its own (Follow false) and shows the conflict once.
//
// Keys missing from local are left alone. Idempotent.
func Migrate(ctx context.Context, d family.Dir, bundleID string, local map[string]Local) ([]Outcome, error) {
	var out []Outcome
	_, err := Update(ctx, d, bundleID, func(s *Settings) error {
		out = out[:0]
		wrote := false
		for _, key := range Keys {
			l, ok := local[key]
			if !ok {
				continue
			}
			o := Outcome{Key: key}
			own, err := Normalize(key, l.Value)
			if err != nil {
				own = l.Value
			}
			fam, set := s.Get(key)
			o.Family = fam
			switch {
			case !set && l.IsDefault:
				o.Follow = true
			case !set:
				if err := s.SetValue(key, own); err != nil {
					return err
				}
				o.Follow, o.Wrote, o.Family = true, true, own
				wrote = true
			case same(key, fam, own):
				o.Follow = true
			case l.IsDefault && !l.InUse:
				o.Follow = true
			default:
				o.Conflict = true
			}
			out = append(out, o)
		}
		if !wrote {
			return family.ErrNoChange
		}
		return nil
	})
	return out, err
}

func same(key, a, b string) bool {
	if key == KeyModelsRoot {
		return filepath.Clean(a) == filepath.Clean(b) || a == "" && b == ""
	}
	if key == KeyHFEndpoints {
		return strings.Join(Endpoints(a), " ") == strings.Join(Endpoints(b), " ")
	}
	return a == b
}

// Reference texts.
var (
	// TextOtherValue: the one-time prompt on a setting row after a
	// migration conflict (spec §9.4). {setting} is the row's name.
	TextOtherValue = family.Text{
		EN: "Other ActNow apps use a different {setting}.",
		ZH: "其他 ActNow app 使用的{setting}不同。",
	}
	TextUseTheirs = family.Text{EN: "Use theirs", ZH: "改用相同设置"}
	// TextAppliesToAll / TextUseOwnSetting are in family.
)
