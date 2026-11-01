package actnow

// The family's shared settings (spec §9): the interface language, the
// proxy and Hugging Face endpoints of model downloads, the model folder.
// VoxtNew keeps its own values and a follow flag for each (settings.Family);
// while it follows, its value is the family's and a change it makes goes
// to the family's settings.json.

import (
	"context"
	"errors"
	"slices"

	"github.com/hehehai/actnowkit/family"
	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// The shared keys VoxtNew has.
const (
	KeyLanguage    = famsettings.KeyLanguage
	KeyProxy       = famsettings.KeyProxy
	KeyHFEndpoints = famsettings.KeyHFEndpoints
	KeyModelsRoot  = famsettings.KeyModelsRoot
)

// ReadShared reads settings.json again (a notification, VoxtNew in
// front); it reports whether it changed.
func (f *Family) ReadShared() bool {
	s, _, err := famsettings.Read(f.dir)
	if err != nil {
		f.logf("family: settings: %v", err)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	changed := s.Rev != f.shared.Rev || s.Updated != f.shared.Updated
	f.shared = s
	return changed
}

// Shared is settings.json as last read (memory only).
func (f *Family) Shared() famsettings.Settings {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.shared
}

// Own is VoxtNew's own value of a shared key, and its default.
func Own(prefs settings.Settings, key string) (value, def string) {
	switch key {
	case KeyLanguage:
		return prefs.General.Language, ""
	case KeyProxy:
		return prefs.Network.Proxy, famsettings.ProxyAuto
	case KeyHFEndpoints:
		v := prefs.Network.HFEndpoints
		if v == "" {
			v = famsettings.EndpointsAuto
		}
		return v, famsettings.EndpointsAuto
	}
	return "", ""
}

// SetOwn sets VoxtNew's own value of a shared key.
func SetOwn(prefs *settings.Settings, key, value string) {
	switch key {
	case KeyLanguage:
		prefs.General.Language = value
	case KeyProxy:
		prefs.Network.Proxy = value
	case KeyHFEndpoints:
		if value == famsettings.EndpointsAuto {
			value = ""
		}
		prefs.Network.HFEndpoints = value
	}
}

// Effective is the value VoxtNew uses of a shared key (spec §9.3): the
// family's when it follows and the family has one, else its default
// while following, else its own (memory only).
func (f *Family) Effective(prefs settings.Settings, key string) string {
	own, def := Own(prefs, key)
	return famsettings.Effective(f.Shared(), key, prefs.Family.Follows(key), own, def)
}

// Set is the user changing a shared setting in VoxtNew: written to the
// family's file while VoxtNew follows it (nothing to do otherwise: the
// caller keeps VoxtNew's own value).
func (f *Family) Set(ctx context.Context, prefs settings.Settings, key, value string) error {
	if !prefs.Family.Follows(key) || f.readOnly {
		return nil
	}
	if value == "" && key == KeyHFEndpoints {
		value = famsettings.EndpointsAuto
	}
	_, err := famsettings.Set(ctx, f.dir, f.bundleID, key, value)
	if err != nil && !errors.Is(err, family.ErrNoChange) {
		return err
	}
	f.ReadShared()
	return nil
}

// Migrate merges VoxtNew's values into the family's at the first start
// with the protocol (spec §9.4): it sets the follow flags and the keys
// whose one-time notice is due in prefs.Family. Idempotent.
func (f *Family) Migrate(ctx context.Context, prefs *settings.Settings) error {
	if f.readOnly {
		return nil
	}
	local := map[string]famsettings.Local{}
	for _, key := range []string{KeyLanguage, KeyProxy, KeyHFEndpoints} {
		own, def := Own(*prefs, key)
		local[key] = famsettings.Local{Value: own, IsDefault: own == def}
	}
	// VoxtNew has no model folder of its own: it always takes the
	// family's (or the default rules).
	local[KeyModelsRoot] = famsettings.Local{Value: "", IsDefault: true}
	out, err := famsettings.Migrate(ctx, f.dir, f.bundleID, local)
	if err != nil {
		return err
	}
	if prefs.Family.Follow == nil {
		prefs.Family.Follow = map[string]bool{}
	}
	for _, o := range out {
		prefs.Family.Follow[o.Key] = o.Follow
		if o.Conflict && !slices.Contains(prefs.Family.Conflicts, o.Key) {
			prefs.Family.Conflicts = append(prefs.Family.Conflicts, o.Key)
		}
	}
	f.ReadShared()
	return nil
}

// Follow sets whether VoxtNew follows the family for key: off is "Use a
// different setting for this app" (VoxtNew keeps the value it has now as
// its own); on is "Use theirs" (it takes the family's). It also clears
// the key's migration notice.
func (f *Family) Follow(prefs *settings.Settings, key string, on bool) {
	if !on {
		SetOwn(prefs, key, f.Effective(*prefs, key))
	}
	if prefs.Family.Follow == nil {
		prefs.Family.Follow = map[string]bool{}
	}
	prefs.Family.Follow[key] = on
	prefs.Family.Conflicts = slices.DeleteFunc(prefs.Family.Conflicts, func(k string) bool { return k == key })
	if on {
		SetOwn(prefs, key, f.Effective(*prefs, key))
	}
}

// Mirror copies the family's values into VoxtNew's own for the keys it
// follows (when the family has one), so that what reads VoxtNew's
// settings (the interface language) sees them; it reports whether
// anything changed. Memory only.
func (f *Family) Mirror(prefs *settings.Settings) bool {
	shared := f.Shared()
	changed := false
	for _, key := range []string{KeyLanguage, KeyProxy, KeyHFEndpoints} {
		v, ok := shared.Get(key)
		if !ok || !prefs.Family.Follows(key) {
			continue
		}
		if own, _ := Own(*prefs, key); own != v {
			SetOwn(prefs, key, v)
			changed = true
		}
	}
	return changed
}
