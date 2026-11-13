package registry

import (
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/shortcuts"
)

// System answers the questions discovery asks macOS. DefaultSystem is the
// real one; tests and the conformance kit supply fakes.
type System interface {
	// AppPaths are where LaunchServices knows apps with this bundle id
	// (NSWorkspace URLsForApplicationsWithBundleIdentifier:).
	AppPaths(bundleID string) []string
	// Running lists the running processes of a bundle id
	// (NSRunningApplication); never decide "running" from a pid alone.
	Running(bundleID string) []Process
	// BundleID reads an app bundle's CFBundleIdentifier.
	BundleID(appPath string) (string, error)
}

// Process is a running app process.
type Process struct {
	PID  int
	Path string
}

// Found is a family app (one bundle id) as discovery sees it.
type Found struct {
	BundleID string
	App      family.App // zero when the bundle id is not in the family table
	Variant  family.Variant
	Name     string
	// Manifest is the app's manifest; nil when it never started with the
	// spec ("installed, not registered").
	Manifest *Manifest
	Path     string // where it is installed ("" when uninstalled)
	// Installed: the manifest's path holds the app, or LaunchServices
	// knows it. A manifest whose app is neither is "uninstalled": its
	// claims become orphans (spec §6.3 step 3, §7.7).
	Installed bool
	Running   bool
	Processes []Process
}

// Discover finds the family apps (spec §6.3): every manifest, plus the
// release bundle ids of the family table that LaunchServices knows
// without a manifest; then which are installed and which run. UIs show
// Release variants only (Visible); dev and test variants still count for
// model claims.
func Discover(d family.Dir, sys System) ([]Found, []error) {
	ms, errs := List(d)
	byID := map[string]*Found{}
	var order []string
	for i := range ms {
		m := &ms[i]
		f := &Found{BundleID: m.BundleID, Manifest: m, Variant: m.Variant, Name: m.Name}
		if a, v, err := family.ParseBundleID(m.BundleID); err == nil {
			f.App = a
			if f.Variant == "" {
				f.Variant = v
			}
		}
		byID[m.BundleID] = f
		order = append(order, m.BundleID)
	}
	for _, a := range family.Apps {
		if _, ok := byID[a.BundleID]; ok {
			continue
		}
		paths := sys.AppPaths(a.BundleID)
		if len(paths) == 0 {
			continue
		}
		byID[a.BundleID] = &Found{BundleID: a.BundleID, App: a, Variant: family.Release, Name: a.Name, Path: paths[0], Installed: true}
		order = append(order, a.BundleID)
	}
	out := make([]Found, 0, len(order))
	for _, id := range order {
		f := byID[id]
		if f.Manifest != nil {
			if p := f.Manifest.Path; p != "" && pathHolds(sys, p, id) {
				f.Path, f.Installed = p, true
			} else if paths := sys.AppPaths(id); len(paths) > 0 {
				f.Path, f.Installed = paths[0], true // moved
			}
		}
		f.Processes = sys.Running(id)
		f.Running = len(f.Processes) > 0
		if f.Running && !f.Installed {
			f.Installed = true
			if f.Path == "" {
				f.Path = f.Processes[0].Path
			}
		}
		if f.Name == "" {
			f.Name = f.App.Name
		}
		out = append(out, *f)
	}
	slices.SortFunc(out, func(a, b Found) int { return strings.Compare(a.BundleID, b.BundleID) })
	return out, errs
}

func pathHolds(sys System, path, bundleID string) bool {
	if st, err := os.Stat(path); err != nil || !st.IsDir() {
		return false
	}
	id, err := sys.BundleID(path)
	return err == nil && strings.EqualFold(id, bundleID)
}

// Visible filters what the UI lists: release variants only (spec §6.3
// step 5).
func Visible(found []Found) []Found {
	var out []Found
	for _, f := range found {
		if f.Variant == family.Release {
			out = append(out, f)
		}
	}
	return out
}

// States turns discovery into models.Store.Apps's map.
func States(found []Found) map[string]models.AppState {
	out := map[string]models.AppState{}
	for _, f := range found {
		st := models.AppState{BundleID: f.BundleID, AppID: f.App.ID, Name: f.Name, Installed: f.Installed, Running: f.Running}
		if f.Manifest != nil && f.Manifest.Models != nil {
			st.ModelsRoot = f.Manifest.Models.Root
		}
		out[f.BundleID] = st
	}
	return out
}

// Holders turns the installed apps' manifests into shortcut holders for
// shortcuts.FindConflicts (installed, running or not: they may start at
// login).
func Holders(found []Found) []shortcuts.Holder {
	var out []shortcuts.Holder
	for _, f := range found {
		if !f.Installed || f.Manifest == nil {
			continue
		}
		out = append(out, shortcuts.Holder{BundleID: f.BundleID, Name: f.Name, Updated: f.Manifest.Updated, Links: f.Manifest.Links, Shortcuts: f.Manifest.Shortcuts})
	}
	return out
}

// HolderOf is a manifest as a shortcut holder (the app's own, before or
// after writing it).
func HolderOf(m Manifest) shortcuts.Holder {
	return shortcuts.Holder{BundleID: m.BundleID, Name: m.Name, Updated: m.Updated, Links: m.Links, Shortcuts: m.Shortcuts}
}

// BundlePath is the .app bundle containing the running executable
// (…/Foo.app/Contents/MacOS/foo → …/Foo.app); "" when not in one.
func BundlePath() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	if r, err := filepath.EvalSymlinks(exe); err == nil {
		exe = r
	}
	return bundleOf(exe)
}

func bundleOf(exe string) string {
	for dir := filepath.Dir(exe); dir != "/" && dir != "." && dir != ""; dir = filepath.Dir(dir) {
		if strings.HasSuffix(dir, ".app") {
			return dir
		}
	}
	return ""
}

// TeamID is the Team ID of the running app's signature ("" when ad hoc
// or not on macOS), for Manifest.TeamID.
func TeamID() string { return teamID() }

// DefaultSystem asks macOS (LaunchServices and NSRunningApplication
// through purego); on other systems it knows no apps, and reads
// Info.plist files itself.
var DefaultSystem System = defaultSystem{}
