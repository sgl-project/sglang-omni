package mygo

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// Set by `mygo build` and `mygo dev` with -ldflags -X: what mygo.config.ts says
// about the app, for platforms whose executables carry no such metadata
// (Linux), and the URL schemes it declares.
var (
	packageName       string
	packageVersion    string
	packageIdentifier string
	packageURLSchemes string // comma separated
	// packageFileExtensions are those of the file associations.
	packageFileExtensions string // comma separated, lower case
)

// bundleInfo is what the backend reads of the app's bundle, which does not
// change while it runs. Public methods such as App.Name ask for it from any
// goroutine.
var bundleInfo = sync.OnceValues(func() (platform.PackageInfo, bool) { return backend().App().Package() })

// packageInfo describes the app as packaged by `mygo build` or `mygo dev`.
func packageInfo() (platform.PackageInfo, bool) {
	if info, ok := bundleInfo(); ok {
		return info, true
	}
	if packageIdentifier != "" {
		return platform.PackageInfo{Name: packageName, Version: packageVersion, Identifier: packageIdentifier}, true
	}
	return platform.PackageInfo{}, false
}

// Deep links: URLs of the schemes the app handles reach OnOpenURL. macOS
// hands them to bundles that declare the scheme (urlSchemes in mygo.config.ts)
// with Apple Events; Windows and Linux start the app with the URL as an
// argument, so the arguments of the launch, and those that a second
// instance forwards (RequestSingleInstanceLock), are searched for them.

// registeredSchemes are the schemes passed to RegisterURLScheme.
var registeredSchemes struct {
	sync.Mutex
	set map[string]bool
}

// isURLScheme reports whether s is a URL scheme: a letter, then letters,
// digits, "+", "-" and ".". (Package regexp would make builds for browsers
// larger.)
func isURLScheme(s string) bool {
	for i, c := range []byte(s) {
		letter := c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
		if !letter && (i == 0 || (c < '0' || c > '9') && c != '+' && c != '-' && c != '.') {
			return false
		}
	}
	return s != ""
}

// handlesScheme reports whether URLs of scheme reach OnOpenURL.
func handlesScheme(scheme string) bool {
	scheme = strings.ToLower(scheme)
	for _, s := range strings.Split(packageURLSchemes, ",") {
		if s != "" && strings.EqualFold(s, scheme) {
			return true
		}
	}
	registeredSchemes.Lock()
	defer registeredSchemes.Unlock()
	return registeredSchemes.set[scheme]
}

// urlArgs returns the command line arguments that are URLs of schemes the
// app handles.
func urlArgs(args []string) []string {
	var urls []string
	for _, a := range args {
		// A single letter is a Windows drive, as in C:\.
		if scheme, _, ok := strings.Cut(a, ":"); ok && len(scheme) > 1 && isURLScheme(scheme) && handlesScheme(scheme) {
			urls = append(urls, a)
		}
	}
	return urls
}

// fileArgs returns the command line arguments that are files the app
// opens, of the extensions of its file associations, as absolute paths;
// relative ones are relative to wd. Linux launchers may pass file URLs.
func fileArgs(args []string, wd string) []string {
	if packageFileExtensions == "" {
		return nil
	}
	exts := strings.Split(packageFileExtensions, ",")
	var files []string
	for _, a := range args {
		p := a
		if u, err := url.Parse(a); err == nil && u.Scheme == "file" {
			p = filepath.FromSlash(u.Path)
			if runtime.GOOS == "windows" {
				p = strings.TrimPrefix(p, `\`) // file:///C:/x
			}
		}
		if !slices.Contains(exts, strings.TrimPrefix(strings.ToLower(filepath.Ext(p)), ".")) {
			continue
		}
		if !filepath.IsAbs(p) {
			p = filepath.Join(wd, p)
		}
		if info, err := os.Stat(p); err == nil && info.Mode().IsRegular() {
			files = append(files, p)
		}
	}
	return files
}

// deliverArgs passes the URLs among args to OnOpenURL and the files to
// OnOpenFile.
func deliverArgs(args []string, wd string) {
	for _, u := range urlArgs(args) {
		fire1(&App.onOpenURL, u)
	}
	for _, f := range fileArgs(args, wd) {
		fire1(&App.onOpenFile, f)
	}
}

// RegisterURLScheme makes the app the handler of the URLs of scheme for the
// current user, e.g. myapp://open?item=1 for "myapp": they reach OnOpenURL,
// including the one the app is started with. Call it before Run, typically
// every time the app starts: registering again is cheap.
//
// On Windows it registers the executable under HKEY_CURRENT_USER, on Linux
// a hidden desktop entry that becomes the default handler (as xdg-mime
// does). On macOS the scheme must be listed in urlSchemes in mygo.config.ts,
// which registers it with the system; this makes the app its default
// handler when others claim it too. Unregister it when the app is
// uninstalled, for example with UnregisterURLScheme.
func (a *Application) RegisterURLScheme(scheme string) error {
	if !isURLScheme(scheme) {
		return fmt.Errorf("mygo: invalid URL scheme %q", scheme)
	}
	scheme = strings.ToLower(scheme)
	registeredSchemes.Lock()
	if registeredSchemes.set == nil {
		registeredSchemes.set = map[string]bool{}
	}
	registeredSchemes.set[scheme] = true
	registeredSchemes.Unlock()
	id, name := appID(), a.Name()
	return onMainValue(func() error { return backend().App().RegisterURLScheme(scheme, id, name) })
}

// UnregisterURLScheme undoes RegisterURLScheme (Linux, Windows); URLs of
// the scheme no longer reach the app unless mygo.config.ts declares it.
func (a *Application) UnregisterURLScheme(scheme string) error {
	if !isURLScheme(scheme) {
		return fmt.Errorf("mygo: invalid URL scheme %q", scheme)
	}
	scheme = strings.ToLower(scheme)
	registeredSchemes.Lock()
	delete(registeredSchemes.set, scheme)
	registeredSchemes.Unlock()
	id, name := appID(), a.Name()
	return onMainValue(func() error { return backend().App().UnregisterURLScheme(scheme, id, name) })
}

// IsURLSchemeRegistered reports whether the system opens URLs of scheme
// with this app.
func (a *Application) IsURLSchemeRegistered(scheme string) bool {
	if !isURLScheme(scheme) {
		return false
	}
	scheme = strings.ToLower(scheme)
	id, name := appID(), a.Name()
	return onMainValue(func() bool { return backend().App().IsURLSchemeRegistered(scheme, id, name) })
}

// launchArgs delivers the URLs and files the app was started with, once it
// is ready, then runs the functions waiting for that (main thread).
func launchArgs() {
	deliverArgs(os.Args[1:], startDir)
	launch.Lock()
	launch.done = true
	fns := launch.fns
	launch.fns = nil
	launch.Unlock()
	for _, fn := range fns {
		fn()
	}
}

var launch struct {
	sync.Mutex
	done bool
	fns  []func()
}

// whenLaunched runs fn on the main thread once the app delivered the
// files and URLs it was started with: right away when it did.
func whenLaunched(fn func()) {
	launch.Lock()
	if !launch.done {
		launch.fns = append(launch.fns, fn)
		launch.Unlock()
		return
	}
	launch.Unlock()
	postMain(fn)
}
