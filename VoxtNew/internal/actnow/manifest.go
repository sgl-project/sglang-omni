package actnow

// VoxtNew's manifest (apps/<bundle id>.json, spec §6.2): written at every
// start and when its version, shortcuts, capabilities or model folder
// change; and its shortcuts against the other apps' (spec §10.4).

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// Links are the deep links of VoxtNew's pages other apps open ("Open
// VoxtNew settings" in a shortcut conflict).
var Links = map[string]string{
	registry.LinkShortcuts: "voxtnew://settings/shortcuts",
	registry.LinkModels:    "voxtnew://models",
	registry.LinkNetwork:   "voxtnew://models/network",
}

// Capabilities are VoxtNew's (spec §6.2): the family's model library,
// local speech recognition and language models, dictation into other
// apps.
var Capabilities = []string{registry.CapModelsHF, registry.CapASRLocal, registry.CapLLMLocal, registry.CapDictation}

// Shortcuts are VoxtNew's global shortcuts as the family declares them:
// each feature's bindings (ids "transcribe", "translate", "rewrite", a
// second binding "transcribe.2"…) and "pasteLast", with the time each got
// its keys (prefs.Family.KeysSince). Keys the family's notation cannot
// write are left out.
func Shortcuts(prefs settings.Settings) []shortcuts.Shortcut {
	var out []shortcuts.Shortcut
	add := func(id, title string, h settings.Hotkey, trigger string, exclusive bool) {
		keys, err := shortcuts.Normalize(h.Keys)
		if err != nil || h.Keys == "" {
			return
		}
		s := shortcuts.Shortcut{ID: id, Title: title, Keys: keys, Trigger: shortcuts.NormalizeTrigger(trigger),
			Enabled: true, Mechanism: shortcuts.EventTap, Exclusive: exclusive}
		if since, ok := prefs.Family.KeysSince[id]; ok && since.Keys == keys && !since.Since.IsZero() {
			at := since.Since
			s.EnabledAt = &at
		}
		out = append(out, s)
	}
	f := prefs.Features
	for _, x := range []struct {
		id, title string
		keys      []settings.Hotkey
	}{
		{"transcribe", "Transcribe", f.Transcribe.Hotkeys},
		{"translate", "Translate", f.Translate.Hotkeys},
		{"rewrite", "Rewrite", f.Rewrite.Hotkeys},
	} {
		for i, h := range x.keys {
			id := x.id
			if i > 0 {
				id += "." + strconv.Itoa(i+1)
			}
			trigger := h.Trigger
			if trigger == "" {
				trigger = settings.TriggerTap
			}
			add(id, x.title, h, trigger, false)
		}
	}
	// ⌃⌥V and the like: the helper swallows the keys it pastes on.
	add("pasteLast", "Paste last", f.PasteLast, shortcuts.Press, true)
	return out
}

// StampKeys records in prefs when each declared shortcut got the keys it
// has now (the first time VoxtNew sees them), and forgets those gone; it
// reports whether anything changed. Memory only.
func StampKeys(prefs *settings.Settings, now time.Time) bool {
	seen := map[string]bool{}
	changed := false
	for _, s := range Shortcuts(*prefs) {
		seen[s.ID] = true
		if old, ok := prefs.Family.KeysSince[s.ID]; ok && old.Keys == s.Keys {
			continue
		}
		if prefs.Family.KeysSince == nil {
			prefs.Family.KeysSince = map[string]settings.KeysSince{}
		}
		prefs.Family.KeysSince[s.ID] = settings.KeysSince{Keys: s.Keys, Since: now.UTC().Truncate(time.Second)}
		changed = true
	}
	for id := range prefs.Family.KeysSince {
		if !seen[id] {
			delete(prefs.Family.KeysSince, id)
			changed = true
		}
	}
	return changed
}

// Manifest is VoxtNew's manifest for prefs (memory only).
func (f *Family) Manifest(prefs settings.Settings) registry.Manifest {
	return registry.Manifest{
		App: AppID, BundleID: f.bundleID, Name: "VoxtNew", Version: f.version,
		Path: registry.BundlePath(), TeamID: registry.TeamID(), URLScheme: "voxtnew",
		Links: Links, Capabilities: Capabilities,
		Models:    &registry.ModelsInfo{Root: f.Hub()},
		Shortcuts: Shortcuts(prefs),
	}
}

// WriteManifest writes VoxtNew's manifest when it differs from the one
// last written (at start always). A bundle-less run (no path) writes
// none.
func (f *Family) WriteManifest(m registry.Manifest) error {
	if f.readOnly || m.Path == "" {
		return nil
	}
	key, _ := json.Marshal(m)
	f.mu.Lock()
	same := f.written == string(key)
	f.mu.Unlock()
	if same {
		return nil
	}
	if err := registry.Write(f.dir, m); err != nil {
		return err
	}
	f.mu.Lock()
	f.written = string(key)
	f.mu.Unlock()
	return nil
}

// Conflicts are VoxtNew's shortcuts that collide with an installed
// family app's (spec §10.4), from the last discovery; Prompt says
// VoxtNew is the later of the two, which shows the prompt (memory only).
func (f *Family) Conflicts(prefs settings.Settings) []shortcuts.Conflict {
	return shortcuts.FindConflicts(registry.HolderOf(f.Manifest(prefs)), f.others())
}

// UsedBy lists the installed family apps whose shortcuts collide with
// keys (the recorder's "Also used by …"), from the last discovery.
func (f *Family) UsedBy(keys string) []shortcuts.Hit {
	return shortcuts.UsedBy(keys, f.bundleID, f.others())
}

// others are the shortcut holders of the other installed apps (any
// build of theirs: a developer's Auralink fires on the same keys), not
// VoxtNew's own other builds.
func (f *Family) others() []shortcuts.Holder {
	var apps []registry.Found
	for _, a := range f.Found() {
		if a.App.ID != AppID {
			apps = append(apps, a)
		}
	}
	return registry.Holders(apps)
}
