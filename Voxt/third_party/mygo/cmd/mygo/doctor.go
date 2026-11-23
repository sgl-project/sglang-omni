package main

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/egoist/mygo/internal/nsis"
	"github.com/egoist/mygo/internal/webview2"
)

func runDoctor(args []string) error {
	flags := newFlags("doctor", "", "Checks the tools and system libraries MyGo needs.")
	if err := flags.Parse(args); err != nil {
		return err
	}
	ok := true
	check := func(name string, good bool, detail string) {
		mark := "\033[32m✓\033[0m"
		if !good {
			mark, ok = "\033[31m✗\033[0m", false
		}
		fmt.Printf("%s %-10s %s\n", mark, name, detail)
	}

	goOut, err := exec.Command("go", "version").Output()
	check("go", err == nil, strings.TrimSpace(string(goOut)))
	if bunOut, err := exec.Command("bun", "--version").Output(); err == nil {
		check("bun", true, strings.TrimSpace(string(bunOut)))
	} else {
		fmt.Println("- bun        not found (optional, used by the frontend template: https://bun.sh)")
	}

	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("sw_vers", "-productVersion").Output()
		v := strings.TrimSpace(string(out))
		major, _ := strconv.Atoi(strings.Split(v, ".")[0])
		check("macOS", err == nil && major >= 12, v+" (WKWebView; macOS 12 or later is required)")
	case "linux":
		out, _ := exec.Command("sh", "-c", "ldconfig -p 2>/dev/null | grep -o 'libwebkit2gtk-4\\.[01]\\.so[.0-9]*' | sort -u").Output()
		libs := strings.TrimSpace(string(out))
		check("webkit", libs != "", or(libs, "WebKitGTK not found: install libwebkit2gtk-4.1-0 (Debian/Ubuntu) or webkit2gtk4.1 (Fedora)"))
	case "windows":
		// The runtime a window showing a web page gets, not the install
		// EdgeUpdate registers, which outlives its files.
		dir, _, err := webview2.Find()
		if err != nil {
			dir = err.Error()
		}
		check("webview2", err == nil, dir)
	default:
		check(runtime.GOOS, false, "no MyGo backend for this platform yet")
	}

	// Tools of optional mygo build features.
	optional := func(name, feature string, found bool, detail string) {
		if found {
			fmt.Printf("\033[32m✓\033[0m %-10s %s\n", name, detail)
		} else {
			fmt.Printf("- %-10s not found (optional: %s)\n", name, feature)
		}
	}
	tool := func(name string) string {
		p, _ := exec.LookPath(name)
		return p
	}
	if compiler := makensis(); compiler != "" || runtime.GOOS != "windows" {
		optional("makensis", "the Windows installer, https://nsis.sourceforge.io", compiler != "", compiler)
	} else {
		fmt.Printf("- %-10s not found (mygo build downloads NSIS %s for the Windows installer)\n", "makensis", nsis.Release.Version)
	}
	if runtime.GOOS == "windows" {
		st := signtool()
		optional("signtool", "signing Windows apps, from the Windows SDK", st != "", st)
	} else {
		ossl := tool("osslsigncode")
		optional("osslsign", "signing Windows apps here (osslsigncode)", ossl != "", ossl)
	}
	gh := tool("gh")
	optional("gh", "mygo build -upload to GitHub releases, https://cli.github.com", gh != "", gh)
	if runtime.GOOS == "darwin" {
		out, _ := exec.Command("security", "find-identity", "-v", "-p", "codesigning").Output()
		ids := strings.Count(string(out), "Developer ID Application")
		optional("codesign", "a Developer ID to sign apps for other Macs", ids > 0, fmt.Sprintf("%d Developer ID identities", ids))
	}

	if !ok {
		return fmt.Errorf("some checks failed")
	}
	return nil
}

func or(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}
