// Package webview2 finds the Microsoft Edge WebView2 Runtime as
// WebView2Loader.dll does. The Windows backend loads the runtime it finds,
// and `mygo doctor` reports it, so the two never disagree; it builds
// everywhere, and finds no installed runtime but on Windows.
package webview2

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Download is where people get the runtime.
const Download = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

// channels are the runtime's EdgeUpdate channels, most stable first, the
// order WebView2Loader.dll tries them in.
var channels = []string{
	"{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", // Stable (Evergreen)
	"{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}", // Beta
	"{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}", // Dev
	"{65C35B14-6C1D-4122-AC46-7148CC9D6497}", // Canary
}

// A runtime's pieces: its browser process at the root of its directory, and
// its client library, which the loader calls, under EBWebView for the
// architecture of the process.
const (
	browserExe = "msedgewebview2.exe"
	clientDir  = "EBWebView"
	clientDLL  = "EmbeddedBrowserWebView.dll"
)

var errNotInstalled = errors.New("the Microsoft Edge WebView2 Runtime is not installed; get it from " + Download)

// Find returns the directory of the runtime a window showing a web page
// gets, and whether it is a fixed version runtime, named by
// WEBVIEW2_BROWSER_EXECUTABLE_FOLDER, rather than an installed one. Without
// a runtime, the error says why.
func Find() (dir string, fixed bool, err error) {
	return find(os.Getenv("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER"), regString)
}

// find is Find with the environment's folder and the registry given: read
// returns a string value of the machine's (HKLM) or the user's (HKCU) key.
func find(fixed string, read func(machine bool, path, name string) string) (string, bool, error) {
	if fixed != "" {
		if ClientDLL(fixed) == "" {
			return "", false, fmt.Errorf("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER=%s holds no WebView2 Runtime: no %s, or no %s", fixed, browserExe, filepath.Join(clientDir, archDir(), clientDLL))
		}
		return fixed, true, nil
	}
	// EdgeUpdate records where it installed a channel's runtime, per machine
	// and per user, and keeps the record after the files are gone.
	stale := ""
	for _, channel := range channels {
		best, bestVersion := "", []int(nil)
		for _, machine := range []bool{true, false} {
			dir := read(machine, `SOFTWARE\Microsoft\EdgeUpdate\ClientState\`+channel, clientDir)
			if dir == "" {
				continue
			}
			if ClientDLL(dir) == "" {
				if stale == "" {
					stale = dir
				}
				continue
			}
			if v := parseVersion(filepath.Base(dir)); best == "" || compareVersions(v, bestVersion) > 0 {
				best, bestVersion = dir, v
			}
		}
		if best != "" {
			return best, false, nil
		}
	}
	if stale != "" {
		return "", false, fmt.Errorf("the WebView2 Runtime registered at %s is missing or incomplete; reinstall it from %s", stale, Download)
	}
	return "", false, errNotInstalled
}

// ClientDLL returns the client library of the runtime in dir, or "" when
// dir holds no runtime. The library alone is none: the Edge browser keeps
// one beside itself, whose environments fail with ERROR_NOT_SUPPORTED, so
// the runtime's browser process must be there too.
func ClientDLL(dir string) string {
	if _, err := os.Stat(filepath.Join(dir, browserExe)); err != nil {
		return ""
	}
	p := filepath.Join(dir, clientDir, archDir(), clientDLL)
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

func archDir() string {
	switch runtime.GOARCH {
	case "arm64":
		return "arm64"
	case "386":
		return "x86"
	}
	return "x64"
}

func parseVersion(s string) []int {
	var v []int
	for _, part := range strings.Split(s, ".") {
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil
		}
		v = append(v, n)
	}
	return v
}

func compareVersions(a, b []int) int {
	for i := 0; i < len(a) || i < len(b); i++ {
		var x, y int
		if i < len(a) {
			x = a[i]
		}
		if i < len(b) {
			y = b[i]
		}
		if x != y {
			if x < y {
				return -1
			}
			return 1
		}
	}
	return 0
}
