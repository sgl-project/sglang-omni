package main

// VoxtNew in the ActNow family (internal/actnow, the family spec): the
// family directory opened once the single-instance lock is held, before
// the models (their library is the family's, $HUB); then, off the main
// thread, the first start's migration, the manifest, the discovery of the
// other apps and the notifications it follows. VoxtNew reads the family
// again when it comes to the front, as notifications are only hints.
// Nothing here stops VoxtNew: a failure is logged.
//
// What it gives the pages: a.famApps (the apps found, for "Shared with",
// conflicts), a.famConflicts (VoxtNew's shortcuts that other apps also
// use), and the shared settings VoxtNew follows (the language, the
// downloads' network: family_network.go).

import (
	"context"
	"log"
	"os"
	"reflect"
	"sync/atomic"
	"time"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// familyState is the family as the UI knows it (main thread).
type familyState struct {
	fam *actnow.Family
	// famApps are the family's apps of the last discovery, VoxtNew among
	// them; famConflicts VoxtNew's shortcuts other installed apps use too.
	famApps      []registry.Found
	famConflicts []shortcuts.Conflict
	famStarted   bool
	famReading   atomic.Bool // a re-read on activation runs
	famReadAt    time.Time
}

// familyConfig is how VoxtNew joins the family: tests put the family
// directory and the model library in their data folder.
var familyConfig = func(a *app) actnow.Config {
	return actnow.Config{Version: mygo.App.Version(), Logf: log.Printf}
}

// openFamily opens the family directory and VoxtNew's place in it (a
// little I/O, while the app starts).
func (a *app) openFamily() {
	if a.fam == nil {
		a.fam = actnow.Open(familyConfig(a), a.prefs)
		a.connectFamilyRuntime() // family_runtime.go
	}
}

// startFamily runs the first start's migration (once), writes the
// manifest, discovers the other apps and follows the family's
// notifications; off the main thread.
func (a *app) startFamily() {
	if a.famStarted || a.fam == nil {
		return
	}
	a.famStarted = true
	fam, m, st := a.fam, a.models, a.settings
	prefs := a.prefs
	a.async(func() {
		ctx := context.Background()
		if !fam.ReadOnly() {
			// The library moved (spec v1.0.9): bring the models over from
			// the Hugging Face caches first, once.
			n, bytes, err := m.Import(ctx, fam.ReadBackup())
			if err != nil {
				log.Printf("family: importing models from the Hugging Face cache: %v", err)
			}
			if n > 0 {
				log.Printf("family: brought %d models (%d bytes) from the Hugging Face cache into %s", n, bytes, fam.Hub())
			}
		}
		if !prefs.Family.Migrated && !fam.ReadOnly() {
			if err := m.Migrate(ctx); err != nil { // the models on disk: claims, origin migrated
				log.Printf("family: claims: %v", err)
			}
			if err := fam.Migrate(ctx, &prefs); err != nil { // the shared settings
				log.Printf("family: settings: %v", err)
			}
			prefs.Family.Migrated = true
		}
		actnow.StampKeys(&prefs, time.Now())
		fam.Mirror(&prefs)
		if err := st.Update(ctx, func(s *settings.Settings) {
			s.Family, s.General.Language, s.Network = prefs.Family, prefs.General.Language, prefs.Network
		}); err != nil {
			log.Printf("family: %v", err)
		}
		if err := fam.WriteManifest(fam.Manifest(prefs)); err != nil {
			log.Printf("family: manifest: %v", err)
		}
		fam.ApplyBackup() // Time Machine (spec §7.12)
		found := fam.Discover()
		m.SetNetwork(a.networkFor(prefs))
		m.Refresh() // the names of the apps sharing a model
		a.post(func() { a.familyFound(found) })
		fam.Watch(func(topic string) { a.familyTopic(topic) },
			family.TopicApps, family.TopicSettings, family.TopicModels, family.TopicMemory)
	})
}

// familyTopic follows a notification (on the watcher's goroutine).
func (a *app) familyTopic(topic string) {
	switch topic {
	case family.TopicApps:
		found := a.fam.Discover()
		a.post(func() { a.familyFound(found) })
	case family.TopicSettings:
		if a.fam.ReadShared() {
			a.post(a.familyShared)
		}
		a.fam.ApplyBackup() // models.backup may have changed
	case family.TopicModels:
		a.models.Refresh() // its OnChange redraws
	case family.TopicMemory:
		a.fam.Runtime().OnMemory()
	}
}

// familyActivated reads the family again when VoxtNew comes to the front
// (at most every 5 s): the notifications may have been missed.
func (a *app) familyActivated() {
	if a.fam == nil || !a.famStarted || time.Since(a.famReadAt) < 5*time.Second || !a.famReading.CompareAndSwap(false, true) {
		return
	}
	a.famReadAt = time.Now()
	fam, m := a.fam, a.models
	a.async(func() {
		defer a.famReading.Store(false)
		if fam.ClaimsLost() {
			// The claims went with the family folder (a cleaner): claim
			// the models again (spec §7.6, v1.0.9).
			if err := m.Migrate(context.Background()); err != nil {
				log.Printf("family: claims: %v", err)
			}
		}
		found := fam.Discover()
		shared := fam.ReadShared()
		m.Refresh()
		a.post(func() {
			a.familyFound(found)
			if shared {
				a.familyShared()
			}
		})
	})
}

// familyFound takes a discovery: the apps, the shortcut conflicts.
func (a *app) familyFound(found []registry.Found) {
	a.famApps = found
	a.famConflicts = a.fam.Conflicts(a.prefs)
	a.shortcutPrompt() // settings_shortcuts_family.go: the one-time prompt
}

// familyOthers reports whether the last discovery found a release of
// another ActNow app installed on this Mac (spec §6.3): only then do the
// shared settings speak of the family ("Applies to all ActNow apps"); the
// data still follows the family's files either way.
func (a *app) familyOthers() bool {
	for _, f := range a.famApps {
		if f.Installed && f.Variant == family.Release && f.App.ID != actnow.AppID {
			return true
		}
	}
	return false
}

// familyShared applies the family's settings after settings.json changed:
// those VoxtNew follows become its own (the language), the downloads'
// network is set again.
func (a *app) familyShared() {
	next := a.prefs
	if a.fam.Mirror(&next) {
		a.updateSettings(func(s *settings.Settings) { s.General.Language, s.Network = next.General.Language, next.Network })
	}
	a.models.SetNetwork(a.networkFor(next))
}

// familySettingsChanged follows VoxtNew's own settings: a shared setting
// the user changed goes to the family while VoxtNew follows it; the
// manifest is written again when the shortcuts changed (off the main
// thread).
func (a *app) familySettingsChanged(old, new settings.Settings) {
	if a.fam == nil || !a.famStarted {
		return
	}
	fam := a.fam
	if old.General.Language != new.General.Language || old.Network != new.Network {
		a.async(func() {
			for _, key := range []string{actnow.KeyLanguage, actnow.KeyProxy, actnow.KeyHFEndpoints} {
				was, _ := actnow.Own(old, key)
				now, _ := actnow.Own(new, key)
				if was == now {
					continue
				}
				if err := fam.Set(context.Background(), new, key, now); err != nil {
					log.Printf("family: %s: %v", key, err)
				}
			}
		})
		a.models.SetNetwork(a.networkFor(new))
	}
	if !reflect.DeepEqual(old.Features, new.Features) {
		next := new
		if actnow.StampKeys(&next, time.Now()) {
			a.updateSettings(func(s *settings.Settings) { s.Family.KeysSince = next.Family.KeysSince })
		}
		a.async(func() {
			if err := fam.WriteManifest(fam.Manifest(next)); err != nil {
				log.Printf("family: manifest: %v", err)
			}
		})
		a.famConflicts = fam.Conflicts(next)
		a.conflictsSeen(old) // settings_shortcuts_family.go: the recorder showed them
	}
}

// networkFor is how model downloads reach the Hub under prefs (spec
// §9.1): the family's endpoints and proxy while VoxtNew follows them
// ($HF_ENDPOINT first, as huggingface_hub), the proxy "automatic" through
// the Mac's system proxy (models.SystemProxy).
func (a *app) networkFor(prefs settings.Settings) models.Network {
	var n models.Network
	if os.Getenv("HF_ENDPOINT") == "" {
		n.Endpoints = familyEndpoints(a.fam.Effective(prefs, actnow.KeyHFEndpoints))
	}
	n.Client = models.NewHTTPClient(models.ProxyFor(a.fam.Effective(prefs, actnow.KeyProxy), systemProxy))
	return n
}

// closeFamily removes the runtime file and stops following (quit).
func (a *app) closeFamily() {
	if a.fam != nil {
		a.fam.Close()
	}
}
