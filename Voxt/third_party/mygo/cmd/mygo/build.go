package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

type buildOptions struct {
	debug        bool
	sign         string // macOS signing identity
	skipDMG      bool
	skipNotarize bool
	pkg          string            // directory of the main package
	work         string            // for generated files
	overlay      map[string]string // go build -overlay entries (the frontend)
}

func runBuild(args []string) error {
	flags := newFlags("build", "[flags] [dir]", `Builds a production app for each platform. It runs buildCommand from
mygo.json, then compiles the app with the frontendDist files embedded, served
at mygo://localhost/. macOS gets a signed .app bundle and a
"<name> <version> <arch>.dmg" disk image, without the arch for
darwin/universal, whose window invites dragging the app to Applications;
other platforms get an executable. Linux also gets a Debian
package, and the app as <name>-<version>-linux-<arch>.tar.gz with
install.sh, which installs it for the user in ~/.local, where it can update
itself. The contents of the
resources directory and the resources listed in mygo.json are copied into
the bundle's Contents/Resources, or next to the executable, and the programs
among them are signed with the app. Directories of resources named after a
platform, such as resources/darwin or resources/linux-amd64, only ship with
that platform's apps; darwin/universal combines darwin-arm64 and
darwin-amd64. MyGo needs no cgo, so any platform can be compiled from any
machine; signing and disk images need macOS. Windows also gets
"<name> Setup <version> <arch>.exe", made with NSIS, which mygo build
downloads on Windows when it is not installed.

Set macos.signingIdentity in mygo.json (or -sign) to a Developer ID to ship
outside the Mac App Store, and macos.notarize to notarize the disk image.

With updates in mygo.json and the key of mygo keygen in
MYGO_UPDATER_PRIVATE_KEY, each platform also gets a signed update archive,
delta updates from the last versions published (updates.deltas), and
update-<platform>.json: publish them where updates point to.

The platform web (or js/wasm) builds an app of native UI for browsers into
<out>/web: index.html, which loads the app, Go's wasm_exec.js and the app's
WebAssembly, named by its content: a static site, to serve from anywhere.`)
	platforms := flags.String("platform", runtime.GOOS+"/"+runtime.GOARCH, "comma separated GOOS/GOARCH targets, e.g. darwin/universal,linux/amd64,windows/amd64, or web for browsers")
	debug := flags.Bool("debug", false, "keep development features such as the web inspector")
	skipBuildCommand := flags.Bool("skip-build-command", false, "do not run buildCommand")
	skipDMG := flags.Bool("skip-dmg", false, "do not create a disk image for macOS")
	skipNotarize := flags.Bool("skip-notarize", false, "do not notarize even when macos.notarize is set")
	sign := flags.String("sign", "", `macOS signing identity (default: macos.signingIdentity, or "-" for ad hoc)`)
	out := flags.String("o", "", "output directory (default: out from mygo.json or dist)")
	upload := flags.Bool("upload", false, "upload the installers and updates to the GitHub release of this version (updates.github), as a draft, or to the bucket of updates.s3")
	if err := flags.Parse(args); err != nil {
		return err
	}
	c, err := loadConfig(dirArg(flags.Args()))
	if err != nil {
		return err
	}
	if *out != "" {
		c.Out = *out
	}
	// The catalogs name the languages the packages declare.
	if _, err := c.languages(); err != nil {
		return err
	}
	if *upload {
		if err := checkUpload(c); err != nil {
			return err
		}
	}
	opts := buildOptions{debug: *debug, sign: c.MacOS.SigningIdentity, skipDMG: *skipDMG, skipNotarize: *skipNotarize}
	if *sign != "" {
		opts.sign = *sign
	}
	if c.MacOS.Notarize != nil && !opts.skipNotarize && opts.sign == "-" {
		return fmt.Errorf("notarization needs a Developer ID signing identity: set macos.signingIdentity or pass -sign")
	}

	con.startBanner("build")
	started := time.Now()
	// The accessors of the catalogs come first, which the app calls, then
	// the TypeScript client: the frontend build type-checks and bundles
	// them.
	if err := generateI18n(c); err != nil {
		return err
	}
	if c.I18n != nil {
		if dirs := compiledDirs(c); dirs != nil {
			checkCatalogImported(c, dirs)
		}
	}
	if err := writeClient(context.Background(), c, false); err != nil {
		return err
	}
	if c.BuildCommand != "" && !*skipBuildCommand {
		t := con.start("Building the frontend")
		if err := t.end(run(t, "web", c.root, c.BuildCommand), "Built the frontend"); err != nil {
			return err
		}
	}
	work, err := os.MkdirTemp("", "mygo-build-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	opts.work = work
	if opts.pkg, err = packageDir(c); err != nil {
		return err
	}
	// Left behind by an interrupted build.
	for _, p := range sysoFiles(opts.pkg) {
		if strings.HasPrefix(filepath.Base(p), "mygo_windows_") {
			os.Remove(p)
		}
	}
	if opts.overlay, err = frontendFiles(c, opts.pkg, work); err != nil {
		return err
	}

	var built []string
	for _, p := range splitList(*platforms) {
		if isWebPlatform(p) {
			paths, err := buildWeb(c, opts)
			built = append(built, paths...)
			if err != nil {
				return err
			}
			continue
		}
		goos, goarch, ok := strings.Cut(p, "/")
		if !ok {
			return fmt.Errorf("invalid platform %q, want GOOS/GOARCH", p)
		}
		paths, err := buildPlatform(c, goos, goarch, opts)
		built = append(built, paths...)
		if err != nil {
			return err
		}
	}
	con.println(buildSummary(c, built, time.Since(started)))
	if *upload {
		return publish(c, built)
	}
	return nil
}

// buildSummary lists what a build made, by directory, with sizes.
func buildSummary(c *Config, artifacts []string, took time.Duration) string {
	var b strings.Builder
	b.WriteString("\n  " + green(con.sym.ok) + " " + bold("Built "+c.Name+" "+c.Version) + dim(" in "+formatDuration(took)) + "\n")
	width := 0
	for _, a := range artifacts {
		width = max(width, utf8.RuneCountInString(filepath.Base(a)))
	}
	dir := ""
	for _, a := range artifacts {
		if d := relPathTo(c.root, filepath.Dir(a)) + "/"; d != dir {
			dir = d
			b.WriteString("\n    " + dim(dir) + "\n")
		}
		name := filepath.Base(a)
		b.WriteString("      " + cyan(name) + strings.Repeat(" ", width-utf8.RuneCountInString(name)) + "  " + dim(fmt.Sprintf("%9s", sizeOf(a))) + "\n")
	}
	return b.String()
}

// buildPlatform builds and packages the app for one platform into
// <out>/<goos>-<goarch> and returns the artifacts. Everything is prepared
// in a staging directory that then replaces the platform directory, so a
// failed build keeps the previous one and nothing stale remains.
func buildPlatform(c *Config, goos, goarch string, opts buildOptions) ([]string, error) {
	out := c.path(c.Out)
	if err := os.MkdirAll(out, 0o755); err != nil {
		return nil, err
	}
	stage, err := os.MkdirTemp(out, ".staging-"+goos+"-"+goarch+"-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(stage)
	if err := os.Chmod(stage, 0o755); err != nil {
		return nil, err
	}
	res, err := c.appResources(goos, goarch)
	if err != nil {
		return nil, err
	}
	if other := c.otherArch(goos, goarch); other != "" {
		warnf("No %s for %s/%s, though there is %s", filepath.Join(resourcesDir, goos+"-"+goarch), goos, goarch, filepath.Join(resourcesDir, other))
	}

	target := goos + "-" + goarch
	if !opts.debug && keepInspector() {
		logf("Keeping the inspector of native UI (MYGO_INSPECTOR=1)")
	}
	ldflags := "-s -w" + packageFlags(c) + updateFlags(c, target)
	if !opts.debug {
		ldflags += " -X github.com/egoist/mygo.production=1"
	}
	if goos == "windows" {
		// A GUI app, not a console one.
		ldflags += " -H=windowsgui"
	}
	compile := func(arch, out string) error {
		t := con.start("Compiling " + goos + "/" + arch)
		err := func() error {
			if goos == "windows" {
				cleanup, err := windowsResources(c, opts.pkg, arch)
				if err != nil {
					return err
				}
				defer cleanup()
			}
			overlay, err := writeOverlay(filepath.Join(opts.work, "overlay.json"), opts.overlay)
			if err != nil {
				return err
			}
			flags := []string{"-trimpath", "-ldflags", ldflags}
			flags = append(flags, productionTags(opts.debug)...)
			if overlay != "" {
				flags = append(flags, "-overlay", overlay)
			}
			return goBuild(context.Background(), c, t, out, []string{"GOOS=" + goos, "GOARCH=" + arch}, flags...)
		}()
		return t.end(err, "Compiled "+goos+"/"+arch)
	}

	name := c.executableName()
	var artifacts []string
	switch goos {
	case "darwin":
		bin := filepath.Join(stage, name)
		if goarch == "universal" {
			arm, amd := bin+".arm64", bin+".amd64"
			if err := compile("arm64", arm); err != nil {
				return nil, err
			}
			if err := compile("amd64", amd); err != nil {
				return nil, err
			}
			if err := writeUniversal(bin, arm, amd); err != nil {
				return nil, err
			}
			os.Remove(arm)
			os.Remove(amd)
		} else if err := compile(goarch, bin); err != nil {
			return nil, err
		}
		car, err := appCatalog(c)
		if err != nil {
			return nil, err
		}
		var icns []byte
		if car != nil && car.icns != nil {
			icns = car.icns // the .icon's own rendering for older macOS
		} else if icns, err = appIcon(c); err != nil {
			return nil, err
		}
		app, err := writeBundle(c, stage, bin, icns, car, res)
		if err != nil {
			return nil, err
		}
		inProject := func(r resource) bool {
			return r.lipo != "" && strings.HasPrefix(r.src, c.path(resourcesDir)+string(filepath.Separator))
		}
		if slices.ContainsFunc(res, inProject) {
			logf("Made universal binaries of the code in %s and %s", filepath.Join(resourcesDir, "darwin-arm64"), filepath.Join(resourcesDir, "darwin-amd64"))
		}
		var t *task
		if runtime.GOOS == "darwin" {
			t = con.start("Signing " + filepath.Base(app))
		}
		if err := t.end(codesign(c, app, opts.sign, true), "Signed "+filepath.Base(app)); err != nil {
			return nil, err
		}
		artifacts = append(artifacts, app)
		switch {
		case opts.skipDMG:
		case runtime.GOOS != "darwin":
			warnf("Skipping the disk image: it needs macOS")
		default:
			dmg, err := buildDMG(c, app, stage, goarch, opts)
			if err != nil {
				return nil, err
			}
			artifacts = append(artifacts, dmg)
		}
	case "windows":
		exe := filepath.Join(stage, name+".exe")
		if err := compile(goarch, exe); err != nil {
			return nil, err
		}
		if err := copyResources(res, stage); err != nil {
			return nil, err
		}
		if c.Windows.signs() {
			if err := signWindows(c, exe); err != nil {
				return nil, err
			}
			if err := signWindowsResources(c, stage, res); err != nil {
				return nil, err
			}
		}
		artifacts = append(artifacts, exe)
	default:
		exe := filepath.Join(stage, slugify(name))
		if err := compile(goarch, exe); err != nil {
			return nil, err
		}
		files, err := writeLinuxDesktop(c, stage, slugify(name))
		if err != nil {
			return nil, err
		}
		if err := copyResources(res, stage); err != nil {
			return nil, err
		}
		artifacts = append(append(artifacts, exe), files...)
	}

	// The app as installed: the bundle, else everything next to the
	// executable. Packages hold it, and updates replace it.
	var installed []string
	if goos == "darwin" {
		installed = []string{filepath.Base(artifacts[0])}
	} else {
		all, err := os.ReadDir(stage)
		if err != nil {
			return nil, err
		}
		for _, e := range all {
			installed = append(installed, e.Name())
		}
	}
	if goos == "windows" {
		setup, err := writeInstaller(c, stage, opts.work, name+".exe", goarch, installed)
		if err != nil {
			return nil, err
		}
		if setup != "" {
			if c.Windows.signs() {
				if err := signWindows(c, setup); err != nil {
					return nil, err
				}
			}
			artifacts = append(artifacts, setup)
		}
	}
	if _, ok := debArch[goarch]; goos == "linux" && ok {
		deb, err := writeDeb(c, stage, slugify(name), goarch, installed)
		if err != nil {
			return nil, err
		}
		artifacts = append(artifacts, deb)
	}
	files, err := writeArchive(c, stage, target, installed)
	if err != nil {
		return nil, err
	}
	artifacts = append(artifacts, files...)
	if goos == "linux" {
		script, err := writeInstallScript(c, stage)
		if err != nil {
			return nil, err
		}
		artifacts = append(artifacts, script)
	}

	final := filepath.Join(out, target)
	if err := replacePath(stage, final); err != nil {
		return nil, err
	}
	for i, a := range artifacts {
		artifacts[i] = filepath.Join(final, filepath.Base(a))
	}
	if goos != "darwin" {
		top := 0 // entries of the resource directory; the others are inside them
		for _, r := range res {
			if !strings.Contains(r.name, "/") {
				top++
			}
		}
		if top > 0 {
			logf("Copied %d resources next to the executable", top)
		}
	}
	return artifacts, nil
}

// inspectorTag leaves the inspector of native UI out of a build (package
// ui's inspector_off.go), some 250 KB of a binary: production builds have
// their developer tools off.
const inspectorTag = "mygo_noinspector"

// keepInspector reports whether MYGO_INSPECTOR=1 keeps the inspector in
// production builds, as to look into one with DevToolsEnabled.
func keepInspector() bool { return os.Getenv("MYGO_INSPECTOR") == "1" }

// productionTags returns the -tags of a build: those GOFLAGS sets, which a
// -tags flag would override, and the inspector's in a production build
// unless it keeps it.
func productionTags(debug bool) []string {
	if debug || keepInspector() {
		return nil
	}
	var tags []string
	for _, f := range strings.Fields(os.Getenv("GOFLAGS")) {
		if v, ok := strings.CutPrefix(strings.TrimLeft(f, "-"), "tags="); ok && v != "" {
			tags = append(tags, v)
		}
	}
	return []string{"-tags", strings.Join(append(tags, inspectorTag), ",")}
}

// packageFlags are the -ldflags that link what mygo.json says about the app
// into it: Linux executables carry no metadata of their own, and the URL
// schemes tell every platform which launch arguments are deep links.
func packageFlags(c *Config) string {
	var b strings.Builder
	for _, v := range [][2]string{
		{"packageName", c.Name},
		{"packageVersion", c.Version},
		{"packageIdentifier", c.Identifier},
		{"packageURLSchemes", strings.Join(c.URLSchemes, ",")},
		{"packageFileExtensions", strings.Join(c.fileExtensions(), ",")},
	} {
		if v[1] != "" {
			b.WriteString(" -X " + ldflagsQuote("github.com/egoist/mygo."+v[0]+"="+v[1]))
		}
	}
	return b.String()
}

// fileExtensions lists the extensions of all file associations.
func (c *Config) fileExtensions() []string {
	var exts []string
	for _, fa := range c.FileAssociations {
		for _, ext := range fa.Ext {
			exts = append(exts, strings.ToLower(ext))
		}
	}
	return exts
}

// ldflagsQuote quotes an argument in -ldflags, which go build splits at
// spaces outside single or double quotes, without escapes (loadConfig
// rejects values with both kinds of quotes).
func ldflagsQuote(s string) string {
	switch {
	case !strings.ContainsAny(s, " \t'\""):
		return s
	case !strings.Contains(s, "'"):
		return "'" + s + "'"
	}
	return `"` + s + `"`
}

// windowsResources puts the resources of the executable (icon, manifest,
// version) in the main package for one build, unless the app has resources
// of its own. go build -overlay does not apply to .syso files, so the file
// is written into the package and removed by cleanup.
func windowsResources(c *Config, pkg, arch string) (cleanup func(), err error) {
	cleanup = func() {}
	for _, p := range sysoFiles(pkg) {
		if !mygoSyso(filepath.Base(p)) {
			logf("Using the .syso resources of the app")
			return cleanup, nil
		}
	}
	syso, err := winresSyso(c, arch)
	if err != nil {
		return cleanup, err
	}
	path := filepath.Join(pkg, "mygo_windows_"+arch+".syso")
	if err := os.WriteFile(path, syso, 0o644); err != nil {
		return cleanup, err
	}
	return func() { os.Remove(path) }, nil
}

func sysoFiles(pkg string) []string {
	files, _ := filepath.Glob(filepath.Join(pkg, "*.syso"))
	return files
}

// mygoSyso reports whether name is a .syso file that windowsResources
// writes.
func mygoSyso(name string) bool {
	return strings.HasPrefix(name, "mygo_windows_") && strings.HasSuffix(name, ".syso")
}

func sizeOf(path string) string {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return formatSize(total)
}

// formatSize formats a number of bytes for messages.
func formatSize(n int64) string {
	if n < 1<<20 {
		return fmt.Sprintf("%d KB", (n+1<<10-1)>>10)
	}
	return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
}

// writeLinuxDesktop writes a .desktop entry, the icon and the MIME package
// of the file types the app defines, for the binary name in dir, and
// returns the files written. install.sh registers them, and updates
// register them again.
func writeLinuxDesktop(c *Config, dir, name string) ([]string, error) {
	var files []string
	icon := ""
	if c.Icon != "" {
		data, err := os.ReadFile(c.path(c.Icon))
		if err != nil {
			return nil, err
		}
		icon = name
		path := filepath.Join(dir, icon+".png")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return nil, err
		}
		files = append(files, path)
	}
	if xml := mimePackage(c); xml != "" {
		path := filepath.Join(dir, name+".xml")
		if err := os.WriteFile(path, []byte(xml), 0o644); err != nil {
			return nil, err
		}
		files = append(files, path)
	}
	path := filepath.Join(dir, name+".desktop")
	if err := os.WriteFile(path, []byte(linuxDesktopEntry(c, name, icon)), 0o644); err != nil {
		return nil, err
	}
	return append(files, path), nil
}

// linuxDesktopEntry renders the desktop entry of the app, which runs exec
// and shows icon, with its name and comment in the languages its catalogs
// have them in (mygo.appName, mygo.appDescription), which launchers show
// for the system's locale.
func linuxDesktopEntry(c *Config, exec, icon string) string {
	categories := c.Linux.Categories
	if len(categories) == 0 {
		categories = []string{"Utility"}
	}
	langs, _ := c.languages()
	// localized writes the key of each language whose value is its own,
	// once a locale.
	localized := func(b *strings.Builder, key, value string, of func(l appLanguage) string) {
		seen := map[string]bool{}
		for _, l := range langs {
			v, locale := of(l), desktopTag(l.Tag)
			if v == "" || v == value || seen[locale] {
				continue
			}
			seen[locale] = true
			b.WriteString(key + "[" + locale + "]=" + v + "\n")
		}
	}
	var b strings.Builder
	b.WriteString("[Desktop Entry]\nType=Application\nName=" + c.Name + "\n")
	localized(&b, "Name", c.Name, func(l appLanguage) string { return l.Name })
	if c.Linux.Comment != "" {
		b.WriteString("Comment=" + c.Linux.Comment + "\n")
	}
	localized(&b, "Comment", c.Linux.Comment, func(l appLanguage) string { return l.Description })
	var mime []string
	for _, fa := range c.FileAssociations {
		t, _ := c.mimeType(fa)
		mime = append(mime, t)
	}
	for _, s := range c.URLSchemes {
		mime = append(mime, "x-scheme-handler/"+s)
	}
	if len(mime) > 0 {
		// Files and deep links, which reach mygo.App.OnOpenFile and
		// OnOpenURL.
		exec += " %U"
	}
	b.WriteString("Exec=" + exec + "\nIcon=" + icon + "\nCategories=" + strings.Join(categories, ";") + ";\nTerminal=false\n")
	if len(mime) > 0 {
		b.WriteString("MimeType=" + strings.Join(mime, ";") + ";\n")
	}
	return b.String()
}
