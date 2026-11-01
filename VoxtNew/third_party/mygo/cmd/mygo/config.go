package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/egoist/mygo/i18n"
)

// The configuration of a project is in one of these files at its root.
const (
	jsonConfig = "mygo.json"
	tsConfig   = "mygo.config.ts"
)

// Config is read from mygo.json or mygo.config.ts in the project root.
// Every field is optional.
type Config struct {
	// Name is the display name of the app (default: directory name).
	Name string `json:"name"`
	// Identifier is the reverse-DNS bundle identifier.
	Identifier string `json:"identifier"`
	Version    string `json:"version"`
	Copyright  string `json:"copyright"`
	// Icon is a square PNG, ideally 1024x1024 (default:
	// resources/icon.png when it exists).
	Icon string `json:"icon"`
	// Main is the Go package of the app (default ".").
	Main string `json:"main"`
	// Out is where builds are written (default "dist").
	Out string `json:"out"`
	// Bindings is the path of the generated TypeScript client (default:
	// frontend/src/mygo.ts when there is a frontend directory,
	// src/mygo.ts when package.json is at the root, else mygo.ts when the
	// configuration has a frontend, and no client for an app without one,
	// such as an app of native UI).
	Bindings string `json:"bindings"`

	// DevURL is what the app loads during `mygo dev` in place of its built
	// frontend, usually the dev server that DevCommand starts, e.g.
	// "http://localhost:5173". Windows load it with relative URLs such as
	// "/" (see mygo.WindowOptions.URL).
	DevURL string `json:"devUrl"`
	// DevCommand runs in the project directory while `mygo dev` runs, e.g.
	// "bun run --cwd frontend dev". mygo dev launches the app once DevURL
	// answers.
	DevCommand string `json:"devCommand"`
	// BuildCommand builds the frontend before `mygo build` compiles the
	// app, e.g. "bun run --cwd frontend build". It runs in the project
	// directory.
	BuildCommand string `json:"buildCommand"`
	// FrontendDist is the directory of the built frontend. `mygo build`
	// embeds it into the app, which serves it at mygo://localhost/. During
	// `mygo dev` without DevURL, it is served from disk.
	FrontendDist string `json:"frontendDist"`

	// Resources lists extra files and directories to ship with the app of
	// every platform. Each is copied under its base name into the app's
	// resource directory (mygo.PathResources), next to the contents of the
	// project's resources directory, which are always included; its
	// directories named after a platform, such as resources/linux-amd64,
	// hold the files of one platform.
	Resources []string `json:"resources"`

	// URLSchemes are the custom URL schemes (deep links) of the app, e.g.
	// "myapp" for myapp://…, whose URLs reach mygo.App.OnOpenURL. macOS
	// registers them with the app bundle, Linux with the desktop entry
	// `mygo build` writes; apps register them at run time with
	// mygo.App.RegisterURLScheme where there is no installer.
	URLSchemes []string `json:"urlSchemes"`
	// FileAssociations are the types of files the app opens, which reach
	// mygo.App.OnOpenFile.
	FileAssociations []FileAssociation `json:"fileAssociations"`
	// I18n configures the catalogs of the app's messages, which make the
	// languages it ships. Without it, the app ships English only.
	I18n *I18n `json:"i18n"`
	// Updates configures signed updates, which mygo.Updater installs.
	Updates *Updates `json:"updates"`
	MacOS   MacOS    `json:"macos"`
	Windows Windows  `json:"windows"`
	Linux   Linux    `json:"linux"`

	root string
	file string // the configuration file, "" without one
	// frontend is whether the app has a frontend of its own, for the
	// TypeScript module of the catalogs. The defaults make one of a frontend
	// directory, an explicit bindings path or the frontend options; a
	// package.json at the root declares one unless it is a workspace root,
	// which holds packages beside the app, such as its website.
	frontend bool
	// langs are the languages the app ships, which languages reads once.
	langs     []appLanguage
	langsErr  error
	langsRead bool
}

// configName names the configuration file in messages.
func (c *Config) configName() string {
	if c.file == "" {
		return jsonConfig
	}
	return filepath.Base(c.file)
}

// FileAssociation is a type of file the app opens: the system lists the app
// to open them, and opening one starts the app with it.
type FileAssociation struct {
	// Ext lists the file name extensions, without dots, e.g. ["md"].
	Ext []string `json:"ext"`
	// Name describes the files, e.g. "Markdown Document".
	Name string `json:"name"`
	// MimeType of the files. Linux identifies files by it; without one
	// the app defines application/x-<name>-<first extension>.
	MimeType string `json:"mimeType"`
	// Role is "Editor" (default) or "Viewer" (macOS).
	Role string `json:"role"`
}

// mimeType returns the MIME type of an association, and whether the app
// defines it.
func (c *Config) mimeType(fa FileAssociation) (string, bool) {
	if fa.MimeType != "" {
		return fa.MimeType, false
	}
	return "application/x-" + slugify(c.executableName()) + "-" + strings.ToLower(fa.Ext[0]), true
}

// I18n configures the catalogs of the app's messages: a JSON file per
// language in Dir, named after its BCP 47 tag ("de.json", "zh-Hant.json"),
// whose messages `mygo generate` writes typed Go and TypeScript functions
// for. The languages the app ships are those of the files.
type I18n struct {
	// Dir is the directory of the catalogs (default "locales"), a Go
	// package of its own, where the Go functions are written.
	Dir string `json:"dir"`
	// Source is the language the app is written in, whose catalog has
	// every message, and which the others fall back to (default "en").
	Source string `json:"source"`
}

// MacOS configures macOS packaging.
type MacOS struct {
	// Icon is the app's icon on macOS as an asset catalog: an Assets.car
	// compiled with actool, whose app icon is named AppIcon unless
	// infoPlist's CFBundleIconName says otherwise, or an Icon Composer .icon,
	// which builds compile on a Mac with Xcode 26 or later and skip
	// elsewhere. The .icns from icon remains the fallback.
	Icon string `json:"icon"`
	// MinimumSystemVersion is the oldest macOS version supported (default
	// "12.0").
	MinimumSystemVersion string `json:"minimumSystemVersion"`
	// SigningIdentity signs the app and the disk image, e.g.
	// "Developer ID Application: Jane Doe (TEAMID)". The default "-" signs
	// ad hoc: the app runs on the Mac that built it, but Gatekeeper blocks
	// it on others.
	SigningIdentity string `json:"signingIdentity"`
	// Entitlements is a plist of entitlements to sign the app with.
	Entitlements string `json:"entitlements"`
	// HelperEntitlements gives code among the resources entitlements of its
	// own, by its path there: executables, libraries and bundles, e.g.
	//   "bin/server": "server.entitlements.plist"
	// for a helper that needs com.apple.security.cs.allow-jit under the
	// hardened runtime. Other code keeps the entitlements it is signed with.
	HelperEntitlements map[string]string `json:"helperEntitlements"`
	// InfoPlist holds extra keys for the Info.plist of the bundle, which
	// replace MyGo's own, e.g. usage descriptions for apps whose pages use
	// the camera or the microphone:
	//   "NSCameraUsageDescription": "Scan documents with the camera."
	InfoPlist map[string]any `json:"infoPlist"`
	// DMGTitle is the volume name of the disk image (default: name).
	DMGTitle string `json:"dmgTitle"`
	// Notarize submits production disk images to Apple's notary service and
	// staples the ticket. It needs a Developer ID signing identity.
	Notarize *Notarize `json:"notarize"`
}

// Notarize holds notarytool credentials stored in the keychain with
// `xcrun notarytool store-credentials <profile>`.
type Notarize struct {
	KeychainProfile string `json:"keychainProfile"`
	// Keychain is the keychain holding the profile (default: the login
	// keychain).
	Keychain string `json:"keychain"`
}

func loadConfig(root string) (*Config, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if !isDir(abs) {
		return nil, fmt.Errorf("%s is not a directory", root)
	}
	c := &Config{root: abs}
	data, file, err := readConfig(abs)
	c.file = file
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.configName(), err)
	}
	if data != nil {
		if err := json.Unmarshal(data, c); err != nil {
			return nil, fmt.Errorf("%s: %w", c.configName(), err)
		}
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", c.configName(), err)
	}
	return c, nil
}

// readConfig returns the configuration of the project in root as JSON, and
// the file it comes from: mygo.json, or mygo.config.ts, which a JavaScript
// runtime evaluates. Without either it returns nothing.
func readConfig(root string) (data []byte, file string, err error) {
	jsonFile, tsFile := filepath.Join(root, jsonConfig), filepath.Join(root, tsConfig)
	switch hasJSON, hasTS := fileExists(jsonFile), fileExists(tsFile); {
	case hasJSON && hasTS:
		return nil, "", fmt.Errorf("%s configures the project too; keep one of them", jsonConfig)
	case hasTS:
		data, err = evalTSConfig(root, tsFile)
		return data, tsFile, err
	case hasJSON:
		data, err = os.ReadFile(jsonFile)
		return data, jsonFile, err
	}
	return nil, "", nil
}

func (c *Config) validate() error {
	if n := c.MacOS.Notarize; n != nil && n.KeychainProfile == "" {
		return errors.New("macos.notarize needs a keychainProfile (see xcrun notarytool store-credentials)")
	}
	for i, fa := range c.FileAssociations {
		if len(fa.Ext) == 0 {
			return fmt.Errorf("fileAssociations[%d] has no ext", i)
		}
		for _, ext := range fa.Ext {
			if !extRe.MatchString(ext) {
				return fmt.Errorf("fileAssociations[%d]: %q is not a file name extension (without the dot)", i, ext)
			}
		}
		if fa.Role != "" && fa.Role != "Editor" && fa.Role != "Viewer" {
			return fmt.Errorf("fileAssociations[%d].role is Editor or Viewer", i)
		}
	}
	if ext := strings.ToLower(filepath.Ext(c.MacOS.Icon)); c.MacOS.Icon != "" && ext != ".car" && ext != ".icon" {
		return fmt.Errorf("macos.icon %s is an Assets.car or an Icon Composer .icon", c.MacOS.Icon)
	}
	for name, file := range c.MacOS.HelperEntitlements {
		if name == "" || file == "" {
			return fmt.Errorf("macos.helperEntitlements maps the path of code among the resources to a property list of entitlements, not %q to %q", name, file)
		}
	}
	if c.Windows.Certificate != "" && c.Windows.SignCommand != "" {
		return errors.New("windows.certificate and windows.signCommand exclude each other")
	}
	if cmd := c.Linux.Command; cmd != "" && !commandRe.MatchString(cmd) {
		return fmt.Errorf("linux.command %q is not a command name (a letter or digit, then letters, digits, ., _, + or -)", cmd)
	}
	if c.Updates != nil {
		if err := c.Updates.validate(); err != nil {
			return err
		}
	}
	for _, scheme := range c.URLSchemes {
		if !schemeRe.MatchString(scheme) {
			return fmt.Errorf("urlSchemes: %q is not a URL scheme (a letter, then letters, digits, +, - or .)", scheme)
		}
	}
	for _, v := range []string{c.Name, c.Version, c.Identifier} {
		if strings.Contains(v, "'") && strings.Contains(v, `"`) {
			return fmt.Errorf("%q cannot contain both kinds of quotes", v)
		}
	}
	if c.I18n != nil {
		if src := i18n.Normalize(c.I18n.Source); src != c.I18n.Source || i18n.IsPseudo(src) {
			return fmt.Errorf("i18n.source %q is not a BCP 47 tag in its canonical form, as en, de-CH or zh-Hans", c.I18n.Source)
		}
		if filepath.IsAbs(c.I18n.Dir) || filepath.Clean(c.I18n.Dir) == "." || strings.HasPrefix(filepath.Clean(c.I18n.Dir), "..") {
			return fmt.Errorf("i18n.dir %q is not a directory of the project: the catalogs are a Go package of their own, as locales", c.I18n.Dir)
		}
	}
	if c.DevURL != "" {
		if u, err := url.Parse(c.DevURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			return fmt.Errorf("devUrl %q is not an http(s) URL", c.DevURL)
		}
	}
	return nil
}

var (
	nonIdent = regexp.MustCompile(`[^a-zA-Z0-9]+`)
	schemeRe = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9+.-]*$`)
	extRe    = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_+-]*$`)
)

func (c *Config) applyDefaults() {
	if c.Name == "" {
		c.Name = filepath.Base(c.root)
	}
	if c.Identifier == "" {
		slug := strings.ToLower(nonIdent.ReplaceAllString(c.Name, ""))
		if slug == "" {
			slug = "app"
		}
		c.Identifier = "com.mygo." + slug
	}
	if c.Version == "" {
		c.Version = "0.1.0"
	}
	if c.Main == "" {
		c.Main = "."
	}
	if c.Out == "" {
		c.Out = "dist"
	}
	if c.Icon == "" && fileExists(filepath.Join(c.root, resourcesDir, "icon.png")) {
		c.Icon = filepath.Join(resourcesDir, "icon.png")
	}
	if c.MacOS.MinimumSystemVersion == "" {
		c.MacOS.MinimumSystemVersion = "12.0"
	}
	if c.MacOS.SigningIdentity == "" {
		c.MacOS.SigningIdentity = "-"
	}
	if c.MacOS.DMGTitle == "" {
		c.MacOS.DMGTitle = c.Name
	}
	if c.Linux.Maintainer == "" {
		c.Linux.Maintainer = packageAuthor(c.root)
	}
	if c.Linux.Maintainer == "" {
		c.Linux.Maintainer = c.Name
	}
	if c.I18n != nil {
		if c.I18n.Dir == "" {
			c.I18n.Dir = "locales"
		}
		if c.I18n.Source == "" {
			c.I18n.Source = "en"
		}
	}
	frontendOptions := c.DevURL != "" || c.DevCommand != "" || c.BuildCommand != "" || c.FrontendDist != ""
	switch {
	case c.Bindings != "":
		c.frontend = true
	case isDir(filepath.Join(c.root, "frontend")):
		c.Bindings = filepath.Join("frontend", "src", "mygo.ts")
		c.frontend = true
	case fileExists(filepath.Join(c.root, "package.json")):
		c.Bindings = filepath.Join("src", "mygo.ts")
		c.frontend = !workspaceRoot(c.root) || frontendOptions
	case frontendOptions:
		c.Bindings = "mygo.ts"
		c.frontend = true
	}
}

// workspaceRoot reports whether the package.json of root declares
// workspaces, as npm's and Bun's array and Yarn's object do: it holds
// packages, such as the app's website, rather than being a frontend
// itself.
func workspaceRoot(root string) bool {
	data, err := os.ReadFile(filepath.Join(root, "package.json"))
	if err != nil {
		return false
	}
	var pkg struct {
		Workspaces json.RawMessage `json:"workspaces"`
	}
	return json.Unmarshal(data, &pkg) == nil && len(pkg.Workspaces) > 0 && string(pkg.Workspaces) != "null"
}

// packageAuthor returns the author in the package.json of root, which npm
// writes as "Name <email> (url)" or {"name", "email", "url"}, as
// "Name <email>", or "" without one.
func packageAuthor(root string) string {
	data, err := os.ReadFile(filepath.Join(root, "package.json"))
	if err != nil {
		return ""
	}
	var pkg struct {
		Author json.RawMessage `json:"author"`
	}
	if json.Unmarshal(data, &pkg) != nil || pkg.Author == nil {
		return ""
	}
	var name, email string
	var person struct{ Name, Email string }
	if err := json.Unmarshal(pkg.Author, &name); err == nil {
		name, _, _ = strings.Cut(name, "(")
		name, email, _ = strings.Cut(name, "<")
		email, _, _ = strings.Cut(email, ">")
	} else if err := json.Unmarshal(pkg.Author, &person); err == nil {
		name, email = person.Name, person.Email
	}
	name, email = strings.Join(strings.Fields(name), " "), strings.TrimSpace(email)
	switch {
	case name == "":
		return ""
	case email == "":
		return name
	}
	return name + " <" + email + ">"
}

func (c *Config) path(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(c.root, p)
}

// executableName is a file system friendly version of Name.
func (c *Config) executableName() string {
	if name := fsName(c.Name); name != "" {
		return name
	}
	return "app"
}

// fsName makes s usable as a file name on every platform.
func fsName(s string) string {
	return strings.Map(func(r rune) rune {
		if r < ' ' || strings.ContainsRune(`/\:*?"<>|`, r) {
			return '-'
		}
		return r
	}, strings.TrimSpace(s))
}

func isDir(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}
