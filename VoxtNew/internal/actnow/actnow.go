// Package actnow is VoxtNew's part in the ActNow family of apps (the
// family spec v1.0.3, through actnowkit): the family directory
// (~/Library/Application Support/ActNow), VoxtNew's manifest in it, the
// other family apps it discovers, the shared settings it follows
// (interface language, proxy, Hugging Face endpoints, model folder), its
// shortcuts against theirs, and its runtime file (the models it holds in
// memory, its paste in progress).
//
// The models themselves are internal/models' (in the family's library,
// $HUB); this package gives it the family's Store.
//
// Everything here is goroutine-safe and does I/O on the family's files:
// call it off the main thread, except the getters that say they read
// memory. A failure is logged and never stops VoxtNew: the family is
// optional (spec §15.2).
//
// Imports: settings.
package actnow

import (
	"fmt"
	"log"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// AppID is VoxtNew's id in the family's table.
const AppID = "voxtnew"

// DevBundleID names VoxtNew when it does not run from a bundle (go run,
// tests of the app).
const DevBundleID = "dev.hehehai.voxtnew.dev"

// Config is how VoxtNew joins the family.
type Config struct {
	// BundleID is the running bundle's ("": read from the bundle, else
	// DevBundleID).
	BundleID string
	// Version is VoxtNew's ("0.1.0-beta.1").
	Version string
	// Dir is the family directory (zero: family.Default(), which honours
	// $ACTNOW_FAMILY_DIR).
	Dir family.Dir
	// Hub is the model library ("": the family's models.root, else the
	// environment's rules; tests set one).
	Hub string
	// System finds installed and running apps (nil: registry.DefaultSystem).
	System registry.System
	// Logf logs (nil: log.Printf).
	Logf func(format string, args ...any)
}

// Family is VoxtNew in the family directory.
type Family struct {
	dir      family.Dir
	bundleID string
	version  string
	sys      registry.System
	logf     func(format string, args ...any)
	readOnly bool
	hub      string // Config.Hub
	store    *fammodels.Store
	runtime  *Runtime

	backup atomic.Bool // models.backup, as last read (library.go)

	mu       sync.Mutex
	found    []registry.Found
	shared   famsettings.Settings // settings.json as last read
	written  string               // the manifest last written (its content but the time)
	watchers []*family.Watcher
}

// Open sets up the family directory (creating it and family.json when
// missing, removing temporary files crashes left), resolves the model
// library VoxtNew uses from the family's settings (prefs: VoxtNew's
// follow flags) and drops VoxtNew's claims on models another app deleted
// for all apps. It never fails: a family directory that cannot be set up
// is logged, and VoxtNew goes on with what it can read.
func Open(cfg Config, prefs settings.Settings) *Family {
	f := &Family{dir: cfg.Dir, bundleID: cfg.BundleID, version: cfg.Version, sys: cfg.System, logf: cfg.Logf, hub: cfg.Hub}
	if f.dir.Root == "" {
		f.dir = family.Default()
	}
	if f.sys == nil {
		f.sys = registry.DefaultSystem
	}
	if f.logf == nil {
		f.logf = log.Printf
	}
	if f.bundleID == "" {
		f.bundleID = runningBundleID()
	}
	info, err := f.dir.Ensure(f.bundleID)
	switch {
	case err != nil:
		f.logf("family: %s: %v", f.dir.Root, err)
	case info.ReadOnly():
		f.readOnly = true
		f.logf("family: %s is of spec %d (this VoxtNew knows %d): read only", f.dir.Root, info.SpecVersion, family.SpecVersion)
	default:
		if n := f.dir.CleanAllTemps(); n > 0 {
			f.logf("family: removed %d temporary files", n)
		}
	}
	f.ReadShared()
	f.store = &fammodels.Store{Family: f.dir, Hub: f.hubFor(prefs), BundleID: f.bundleID, AppID: AppID, Apps: f.states}
	if hit, err := f.store.ApplyTombstones(); err != nil {
		f.logf("family: tombstones: %v", err)
	} else if len(hit) > 0 {
		f.logf("family: %d models were deleted for all apps elsewhere", len(hit))
	}
	f.runtime = newRuntime(f)
	return f
}

// runningBundleID is the bundle's identifier, DevBundleID outside one or
// for an identifier outside the family.
func runningBundleID() string {
	if p := registry.BundlePath(); p != "" {
		if id, err := registry.BundleIDAt(p); err == nil {
			if _, _, err := family.ParseBundleID(id); err == nil {
				return id
			}
		}
	}
	return DevBundleID
}

// hubFor is $HUB for VoxtNew's flags (spec §7.1): the family's
// models.root when following it, else the environment's rules (VoxtNew
// has no folder of its own).
func (f *Family) hubFor(prefs settings.Settings) string {
	if f.hub != "" {
		return f.hub
	}
	root, _ := f.Shared().Get(famsettings.KeyModelsRoot)
	hub, _ := fammodels.ResolveHub(prefs.Family.Follows(famsettings.KeyModelsRoot), root, "")
	return hub
}

// Dir is the family directory.
func (f *Family) Dir() family.Dir { return f.dir }

// BundleID is VoxtNew's, as the family knows it.
func (f *Family) BundleID() string { return f.bundleID }

// ReadOnly: a newer ActNow app owns the directory's format; VoxtNew
// writes nothing there (spec §4.1).
func (f *Family) ReadOnly() bool { return f.readOnly }

// Store is VoxtNew's view of the model library (internal/models' Manager
// works through it).
func (f *Family) Store() *fammodels.Store { return f.store }

// Hub is the model library's folder ($HUB). Memory only.
func (f *Family) Hub() string { return f.store.Hub }

// Runtime is VoxtNew's runtime file.
func (f *Family) Runtime() *Runtime { return f.runtime }

// Discover finds the family's apps again (manifests, LaunchServices,
// running processes): at start, when one writes its manifest, and when
// VoxtNew comes to the front (spec §6.3).
func (f *Family) Discover() []registry.Found {
	found, errs := registry.Discover(f.dir, f.sys)
	for _, err := range errs {
		f.logf("family: discover: %v", err)
	}
	f.mu.Lock()
	f.found = found
	f.mu.Unlock()
	return found
}

// Found is the apps the last Discover found, VoxtNew among them (memory
// only).
func (f *Family) Found() []registry.Found {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.found)
}

// states is the Store's view of the apps: the last discovery (an app
// with claims it does not know counts as installed).
func (f *Family) states() map[string]fammodels.AppState {
	return registry.States(f.Found())
}

// AppName is a family app's name for the UI ("AudioSmith", "Dicta
// (dev)"), from its bundle id; the id itself outside the family.
func AppName(bundleID string) string {
	a, v, err := family.ParseBundleID(bundleID)
	if err != nil {
		return bundleID
	}
	if v != family.Release {
		return fmt.Sprintf("%s (%s)", a.Name, v)
	}
	return a.Name
}

// Watch calls fn with each topic posted (family.TopicApps…), on a
// goroutine of its own, until Close. Notifications are hints: VoxtNew
// also reads again when it comes to the front.
func (f *Family) Watch(fn func(topic string), topics ...string) {
	w, err := f.dir.Watch(topics...)
	if err != nil {
		f.logf("family: watch: %v", err)
		return
	}
	f.mu.Lock()
	f.watchers = append(f.watchers, w)
	f.mu.Unlock()
	go func() {
		for t := range w.C {
			fn(t)
		}
	}()
}

// Close stops watching and removes the runtime file (VoxtNew quits).
func (f *Family) Close() {
	f.mu.Lock()
	ws := f.watchers
	f.watchers = nil
	f.mu.Unlock()
	for _, w := range ws {
		w.Close()
	}
	f.runtime.remove()
}
