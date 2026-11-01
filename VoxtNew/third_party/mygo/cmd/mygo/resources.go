package main

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
)

// resourcesDir is the project directory whose contents ship with the app:
// resources/data/words.txt is installed as data/words.txt in the app's
// resource directory (mygo.PathResources), which is Contents/Resources in a
// macOS bundle and the executable's directory elsewhere. Its platform
// directories hold what only one platform's apps ship, laid out the same
// way: resources/linux-amd64/bin/server is installed as bin/server in the
// linux/amd64 app.
const resourcesDir = "resources"

// bundleIcon is the file name of the icon in Contents/Resources.
const bundleIcon = "AppIcon.icns"

// bundleIconName is the name of the app icon in the asset catalog.
const bundleIconName = "AppIcon"

// bundleCatalog is the file name of the asset catalog in Contents/Resources.
const bundleCatalog = "Assets.car"

// Platform directories are named after a system MyGo builds apps for,
// alone or with an architecture: darwin, linux-arm64, windows-amd64.
var (
	platformOS   = []string{"darwin", "linux", "windows"}
	platformArch = []string{"386", "amd64", "arm", "arm64", "loong64", "mips", "mips64", "mips64le", "mipsle", "ppc64", "ppc64le", "riscv64", "s390x"}
)

// platformDir reports whether name, an entry of the resources directory,
// is a platform directory, and for which system and architecture: goarch
// is "" for every architecture, and "universal" for darwin-universal,
// which only universal macOS apps ship.
func platformDir(name string) (goos, goarch string, ok bool) {
	goos, goarch, hasArch := strings.Cut(name, "-")
	switch {
	case !slices.Contains(platformOS, goos):
	case !hasArch:
		return goos, "", true
	case slices.Contains(platformArch, goarch), name == "darwin-universal":
		return goos, goarch, true
	}
	return "", "", false
}

// What other tools call systems and architectures, to catch platform
// directories named like theirs, which would ship to every platform.
var (
	osAliases   = map[string]string{"darwin": "darwin", "macos": "darwin", "mac": "darwin", "osx": "darwin", "linux": "linux", "windows": "windows", "win": "windows", "win32": "windows", "win64": "windows"}
	archAliases = map[string]string{"x64": "amd64", "x86_64": "amd64", "aarch64": "arm64", "x86": "386", "ia32": "386", "i386": "386", "i686": "386"}
)

// platformDirTypo returns the platform directory that name, an entry of the
// resources directory that is not one, looks meant to be, or "".
func platformDirTypo(name string) string {
	lower := strings.ToLower(name)
	if _, _, ok := platformDir(lower); ok {
		return lower // Linux, Darwin-ARM64
	}
	switch lower {
	case "macos", "osx", "win32", "win64":
		return osAliases[lower]
	}
	sys, arch, _ := strings.Cut(lower, "-")
	if alias, ok := archAliases[arch]; ok {
		arch = alias
	}
	want := osAliases[sys] + "-" + arch
	if _, _, ok := platformDir(want); ok {
		return want
	}
	return ""
}

// resource is a file, directory or link installed at name in the resource
// directory.
type resource struct {
	name string // the path there, with slashes
	// src is copied to name; "" makes a directory where directories of
	// several sources merge, whose contents are resources of their own.
	src string
	// nested is set when src is inside a directory of resources: a link
	// is then copied as a link, while the entries of the resources
	// directory, and listed resources, are followed.
	nested bool
	// lipo is the x86_64 half of a universal binary whose arm64 half is
	// src.
	lipo string
}

// resources lists what the app for goos/goarch ships with: the entries of
// the project's resources directory and of its platform directories named
// goos and goos-goarch, whose directories merge, then the extra resources
// of the configuration under their base names. For darwin/universal, the
// contents of darwin-arm64 and darwin-amd64 also merge, into one tree for
// both architectures (see merger.combine). Names starting with a dot are
// left out of directories. Installed paths must be unique, ignoring case
// as macOS and Windows do, and top-level names must differ from reserved,
// the files the packaging adds itself.
func (c *Config) resources(goos, goarch string, reserved ...string) ([]resource, error) {
	return c.resourcesWith(goos, goarch, nil, reserved...)
}

// resourcesWith is resources with the native libraries of the app's
// packages (see natives), installed at the top like listed resources.
func (c *Config) resourcesWith(goos, goarch string, natives []source, reserved ...string) ([]resource, error) {
	m := &merger{c: c, reserved: map[string]string{}}
	for _, name := range reserved {
		m.reserved[strings.ToLower(name)] = name
	}

	dir := c.path(resourcesDir)
	var entries []os.DirEntry
	if info, err := os.Stat(dir); err == nil && !info.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", dir)
	} else if err == nil {
		if entries, err = os.ReadDir(dir); err != nil {
			return nil, err
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	var top []source
	platforms := map[string]bool{}
	for _, e := range entries {
		name, path := e.Name(), filepath.Join(dir, e.Name())
		if hiddenName(name) {
			continue
		}
		if _, _, ok := platformDir(name); ok {
			if !isDir(path) {
				return nil, fmt.Errorf("%s is not a directory, but named like the resources of a platform", m.rel(path))
			}
			platforms[name] = true
			continue
		}
		if want := platformDirTypo(name); want != "" {
			return nil, fmt.Errorf("%s would ship to every platform: name the resources of one platform after its GOOS and GOARCH, as in %s",
				m.rel(path), m.rel(filepath.Join(dir, want)))
		}
		top = append(top, source{name: name, path: path})
	}
	for _, name := range []string{goos, goos + "-" + goarch} {
		if platforms[name] {
			list, err := m.children(source{path: filepath.Join(dir, name)}, false)
			if err != nil {
				return nil, err
			}
			top = append(top, list...)
		}
	}
	if goos == "darwin" && goarch == "universal" {
		arm, amd := filepath.Join(dir, "darwin-arm64"), filepath.Join(dir, "darwin-amd64")
		switch {
		case platforms["darwin-arm64"] && platforms["darwin-amd64"]:
			list, err := m.children(source{path: arm, amd64: amd}, false)
			if err != nil {
				return nil, err
			}
			top = append(top, list...)
		case platforms["darwin-arm64"] || platforms["darwin-amd64"]:
			if platforms["darwin-amd64"] {
				arm, amd = amd, arm
			}
			return nil, fmt.Errorf("a universal app combines %s with %s, which does not exist (files for both architectures go in %s)",
				m.rel(arm), m.rel(amd), filepath.Join(resourcesDir, "darwin"))
		}
	}
	for _, p := range c.Resources {
		src := c.path(p)
		if filepath.Clean(src) == dir {
			return nil, fmt.Errorf("%s: resources lists %s, whose contents are always included; list only extra files", c.configName(), p)
		}
		if _, err := os.Stat(src); err != nil {
			return nil, fmt.Errorf("%s: resource %s: %w", c.configName(), p, err)
		}
		top = append(top, source{name: filepath.Base(src), path: src, listed: true})
	}
	top = append(top, natives...)
	if err := m.merge(top); err != nil {
		return nil, err
	}
	return m.list, nil
}

// otherArch returns the name of a platform directory for goos and another
// architecture when there is none for goarch: the app for goos/goarch
// likely lacks what the other one holds.
func (c *Config) otherArch(goos, goarch string) string {
	if goarch == "universal" {
		return "" // combines darwin-arm64 and darwin-amd64
	}
	entries, _ := os.ReadDir(c.path(resourcesDir))
	other := ""
	for _, e := range entries {
		switch sys, arch, ok := platformDir(e.Name()); {
		case !ok || sys != goos || arch == "":
		case arch == goarch:
			return ""
		case other == "":
			other = e.Name()
		}
	}
	return other
}

// source is a file, directory or link that a resource comes from: one of
// the project's, or for a universal app, one in darwin-arm64 with its
// counterpart in darwin-amd64.
type source struct {
	name   string // where it is installed
	path   string
	amd64  string // the darwin-amd64 counterpart of path
	nested bool   // inside a directory of the project: links are not followed
	listed bool   // in the resources of the configuration, which never merge
}

func (s source) stat(path string) (fs.FileInfo, error) {
	if s.nested {
		return os.Lstat(path)
	}
	return os.Stat(path)
}

// merger lists resources, merging the directories that several sources
// install at the same place.
type merger struct {
	c        *Config
	reserved map[string]string // the packaging's own files, by lower-case name
	list     []resource
}

// merge adds sources to the list. Sources installed at the same place, or
// at names that differ only in case, must be directories of the same name,
// which merge, and not listed in the configuration.
func (m *merger) merge(sources []source) error {
	var keys []string
	groups := map[string][]source{}
	for _, s := range sources {
		key := strings.ToLower(s.name)
		if _, ok := groups[key]; !ok {
			keys = append(keys, key)
		}
		groups[key] = append(groups[key], s)
	}
	for _, key := range keys {
		group := groups[key]
		if own, ok := m.reserved[key]; ok {
			return fmt.Errorf("resource %s would replace the app's own %s", m.rel(group[0].path), own)
		}
		if err := m.place(group); err != nil {
			return err
		}
	}
	return nil
}

// place adds the sources installed at one place: a source, a universal
// pair, which merges when it is two directories and combines when it is
// not, or directories that merge.
func (m *merger) place(group []source) error {
	first := group[0]
	if len(group) == 1 && first.amd64 == "" {
		m.list = append(m.list, resource{name: first.name, src: first.path, nested: first.nested})
		return nil
	}
	var children []source
	for i, s := range group {
		dir, err := m.isDir(s)
		if err != nil {
			return err
		}
		if !dir && len(group) == 1 {
			return m.combine(s)
		}
		if !dir || s.name != first.name || s.listed || first.listed {
			other := group[1]
			if i > 0 {
				other = s
			}
			return fmt.Errorf("resources %s and %s would both be installed as %s", m.rel(first.path), m.rel(other.path), other.name)
		}
		list, err := m.children(s, true)
		if err != nil {
			return err
		}
		children = append(children, list...)
	}
	m.list = append(m.list, resource{name: first.name})
	return m.merge(children)
}

// isDir reports whether s is a directory: for a universal pair, both
// halves or neither must be.
func (m *merger) isDir(s source) (bool, error) {
	info, err := s.stat(s.path)
	if err != nil {
		return false, err
	}
	if s.amd64 != "" {
		other, err := s.stat(s.amd64)
		if err != nil {
			return false, err
		}
		if info.IsDir() != other.IsDir() {
			return false, fmt.Errorf("%s and %s differ: a universal app holds one of them", m.rel(s.path), m.rel(s.amd64))
		}
	}
	return info.IsDir(), nil
}

// children returns the entries of the directory s, installed under its
// name, and for a universal pair, the pairs of entries of its directories,
// which must have the same names. nested tells whether s is in a
// directory of the project rather than a platform directory.
func (m *merger) children(s source, nested bool) ([]source, error) {
	names, err := entryNames(s.path)
	if err != nil {
		return nil, err
	}
	if s.amd64 != "" {
		amd64, err := entryNames(s.amd64)
		if err != nil {
			return nil, err
		}
		for _, n := range names {
			if _, ok := slices.BinarySearch(amd64, n); !ok {
				return nil, m.unpaired(filepath.Join(s.path, n), s.amd64)
			}
		}
		for _, n := range amd64 {
			if _, ok := slices.BinarySearch(names, n); !ok {
				return nil, m.unpaired(filepath.Join(s.amd64, n), s.path)
			}
		}
	}
	list := make([]source, 0, len(names))
	for _, n := range names {
		child := source{name: n, path: filepath.Join(s.path, n), nested: nested}
		if s.name != "" {
			child.name = s.name + "/" + n
		}
		if s.amd64 != "" {
			child.amd64 = filepath.Join(s.amd64, n)
		}
		list = append(list, child)
	}
	return list, nil
}

// entryNames returns the names in dir, sorted, but those starting with a
// dot.
func entryNames(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !hiddenName(e.Name()) {
			names = append(names, e.Name())
		}
	}
	return names, nil
}

// combine adds the files or links of a universal pair: an arm64 and an
// x86_64 Mach-O file become a universal binary, and identical links and
// files, but code for one architecture, are installed once. Anything else
// cannot be one file for both architectures.
func (m *merger) combine(s source) error {
	arm, err := s.stat(s.path)
	if err != nil {
		return err
	}
	amd, err := s.stat(s.amd64)
	if err != nil {
		return err
	}
	switch {
	case arm.Mode().IsRegular() && amd.Mode().IsRegular():
		armCPU, thin := machOCPU(s.path)
		amdCPU, _ := machOCPU(s.amd64)
		if armCPU == cpuARM64 && amdCPU == cpuAMD64 {
			m.list = append(m.list, resource{name: s.name, src: s.path, nested: s.nested, lipo: s.amd64})
			return nil
		}
		if !thin { // the code of one architecture would not run on the other
			same, err := sameContents(s.path, s.amd64)
			if err != nil {
				return err
			}
			if same {
				m.list = append(m.list, resource{name: s.name, src: s.path, nested: s.nested})
				return nil
			}
		}
		if a, b := machOKind(s.path), machOKind(s.amd64); a != "" || b != "" {
			return fmt.Errorf("cannot make a universal binary of %s (%s) and %s (%s): it takes arm64 and x86_64 code",
				m.rel(s.path), cmp.Or(a, "not code"), m.rel(s.amd64), cmp.Or(b, "not code"))
		}
	case arm.Mode()&fs.ModeSymlink != 0 && amd.Mode()&fs.ModeSymlink != 0:
		armLink, err := os.Readlink(s.path)
		if err != nil {
			return err
		}
		amdLink, err := os.Readlink(s.amd64)
		if err != nil {
			return err
		}
		if armLink == amdLink {
			m.list = append(m.list, resource{name: s.name, src: s.path, nested: s.nested})
			return nil
		}
	}
	return fmt.Errorf("%s and %s differ: a universal app holds one of them (files for both architectures go in %s)",
		m.rel(s.path), m.rel(s.amd64), filepath.Join(resourcesDir, "darwin"))
}

// unpaired is the error for path, in one of darwin-arm64 and darwin-amd64,
// whose counterpart is missing from dir, the other one.
func (m *merger) unpaired(path, dir string) error {
	return fmt.Errorf("%s has no counterpart in %s to make a universal app with (files for both architectures go in %s)",
		m.rel(path), m.rel(dir), filepath.Join(resourcesDir, "darwin"))
}

// rel returns path relative to the project, for messages.
func (m *merger) rel(path string) string {
	rel, err := filepath.Rel(m.c.root, path)
	if err != nil || strings.HasPrefix(rel, "..") {
		return path
	}
	return rel
}

// CPU types of Mach-O files.
const (
	cpuAMD64 = 0x01000007
	cpuARM64 = 0x0100000c
)

// machOCPU returns the CPU type of a 64-bit Mach-O file for one
// architecture, what writeUniversal combines.
func machOCPU(path string) (uint32, bool) {
	f, err := os.Open(path)
	if err != nil {
		return 0, false
	}
	defer f.Close()
	var h [12]byte
	if _, err := io.ReadFull(f, h[:]); err != nil || binary.LittleEndian.Uint32(h[:]) != 0xfeedfacf {
		return 0, false
	}
	return binary.LittleEndian.Uint32(h[4:]), true
}

// machOKind describes the code in a file for messages, or returns "" when
// it is not a Mach-O file.
func machOKind(path string) string {
	switch cpu, thin := machOCPU(path); {
	case cpu == cpuARM64:
		return "arm64 code"
	case cpu == cpuAMD64:
		return "x86_64 code"
	case thin:
		return "code for another architecture"
	}
	if ok, _ := machO(path); ok {
		return "universal or 32-bit code"
	}
	return ""
}

// sameContents reports whether the files a and b hold the same bytes.
func sameContents(a, b string) (bool, error) {
	fa, err := os.Open(a)
	if err != nil {
		return false, err
	}
	defer fa.Close()
	fb, err := os.Open(b)
	if err != nil {
		return false, err
	}
	defer fb.Close()
	ia, err := fa.Stat()
	if err != nil {
		return false, err
	}
	ib, err := fb.Stat()
	if err != nil {
		return false, err
	}
	if ia.Size() != ib.Size() {
		return false, nil
	}
	bufA, bufB := make([]byte, 64<<10), make([]byte, 64<<10)
	for {
		n, err := io.ReadFull(fa, bufA)
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
			return false, err
		}
		if _, err := io.ReadFull(fb, bufB[:n]); err != nil {
			return false, err
		}
		if !bytes.Equal(bufA[:n], bufB[:n]) {
			return false, nil
		}
		if n < len(bufA) {
			return true, nil // the end of both, which have the same size
		}
	}
}

// reservedNames are the files the packaging puts in the resource directory
// of an app for goos, which resources must not replace.
func reservedNames(c *Config, goos string) []string {
	switch goos {
	case "darwin":
		if c.MacOS.Icon != "" {
			return []string{bundleIcon, bundleCatalog}
		}
		return []string{bundleIcon}
	case "windows":
		return []string{c.executableName() + ".exe"}
	}
	name := slugify(c.executableName())
	return []string{name, name + ".desktop", name + ".png", name + ".xml", installScriptName}
}

func hiddenName(name string) bool { return strings.HasPrefix(name, ".") }

// copyResources copies list into dir.
func copyResources(list []resource, dir string) error {
	for _, r := range list {
		dst := filepath.Join(dir, filepath.FromSlash(r.name))
		if r.src == "" {
			if err := os.MkdirAll(dst, 0o755); err != nil {
				return err
			}
			continue
		}
		if within(dir, r.src) {
			return fmt.Errorf("resource %s contains the directory it is copied to", r.src)
		}
		if err := r.copyTo(dst); err != nil {
			return err
		}
	}
	return nil
}

// copyResource copies a file or directory to dst like `cp -RH`: src itself
// is followed when it is a symbolic link, links inside it are copied as
// links. Permissions are kept, so executables stay executable.
func copyResource(src, dst string) error {
	return resource{src: src}.copyTo(dst)
}

// copyTo copies r to dst like copyResource, and combines the halves of a
// universal binary.
func (r resource) copyTo(dst string) error {
	if r.lipo != "" {
		info, err := os.Stat(r.src)
		if err != nil {
			return err
		}
		if err := writeUniversal(dst, r.src, r.lipo); err != nil {
			return err
		}
		return os.Chmod(dst, info.Mode().Perm())
	}
	return r.walk(func(path string, info fs.FileInfo) error {
		rel, err := filepath.Rel(r.src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		switch mode := info.Mode(); {
		case mode&fs.ModeSymlink != 0:
			link, err := os.Readlink(path)
			if err != nil {
				return err
			}
			return os.Symlink(link, target)
		case mode.IsDir():
			return os.MkdirAll(target, 0o755)
		case mode.IsRegular():
			return copyFile(path, target, mode.Perm())
		}
		return fmt.Errorf("resource %s is not a regular file", path)
	})
}

// walkResource calls fn for src and, when it is a directory, everything in
// it, in the order copyResource copies them.
func walkResource(src string, fn func(path string, info fs.FileInfo) error) error {
	return resource{src: src}.walk(fn)
}

// walk calls fn for the source of r and, when it is a directory,
// everything in it, in the order copyTo copies them. Only links inside it
// are not followed, or with nested, the source itself too.
func (r resource) walk(fn func(path string, info fs.FileInfo) error) error {
	stat := os.Stat
	if r.nested {
		stat = os.Lstat
	}
	info, err := stat(r.src)
	if err != nil {
		return err
	}
	var walk func(path string, info fs.FileInfo) error
	walk = func(path string, info fs.FileInfo) error {
		if err := fn(path, info); err != nil || !info.IsDir() {
			return err
		}
		entries, err := readDir(path)
		if err != nil {
			return err
		}
		for _, e := range entries {
			if hiddenName(e.Name()) {
				continue
			}
			child, err := e.Info() // links are not followed
			if err != nil {
				return err
			}
			if err := walk(filepath.Join(path, e.Name()), child); err != nil {
				return err
			}
		}
		return nil
	}
	return walk(r.src, info)
}

// hashResources writes what list is made of to w (names, permissions,
// sizes and modification times), to tell whether it changed.
func hashResources(w io.Writer, list []resource) error {
	for _, r := range list {
		if r.src == "" {
			fmt.Fprintf(w, "%s\x00dir\x00", r.name)
			continue
		}
		err := r.walk(func(path string, info fs.FileInfo) error {
			rel, _ := filepath.Rel(r.src, path)
			fmt.Fprintf(w, "%s\x00%s\x00%v\x00", r.name, rel, info.Mode())
			if !info.IsDir() { // a directory's time changes with hidden files too
				fmt.Fprintf(w, "%d\x00%d\x00", info.Size(), info.ModTime().UnixNano())
			}
			return nil
		})
		if err != nil {
			return err
		}
		if r.lipo != "" {
			info, err := os.Stat(r.lipo)
			if err != nil {
				return err
			}
			fmt.Fprintf(w, "%d\x00%d\x00", info.Size(), info.ModTime().UnixNano())
		}
	}
	return nil
}

// within reports whether path is dir or inside it.
func within(path, dir string) bool {
	resolve := func(p string) string {
		if abs, err := filepath.Abs(p); err == nil {
			p = abs
		}
		if real, err := filepath.EvalSymlinks(p); err == nil {
			p = real
		}
		return p
	}
	rel, err := filepath.Rel(resolve(dir), resolve(path))
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// signNestedCode signs the code among the resources of a bundle, which
// codesign --deep leaves alone: it only signs the bundle's code
// directories, and notarization rejects unsigned code anywhere. Code is
// signed from the inside out: Mach-O files, then the bundles holding them
// (apps, frameworks, plug-ins), whose signatures seal what they contain. A
// real identity signs all of it. Ad hoc signing only signs what has no
// valid signature, so that copies keep their own: Mach-O files without
// one, which Apple silicon does not run, bundles that are not sealed, and
// the bundles around what it signed. Code keeps the entitlements it is
// signed with, unless macos.helperEntitlements gives it others.
func signNestedCode(c *Config, app, identity string, production bool) error {
	dir := filepath.Join(app, "Contents", "Resources")
	type nested struct {
		path, name string // name: the path in dir, with slashes
		bundle     bool
		signed     bool // a Mach-O file with a signature
	}
	var files, bundles []nested
	err := walkResource(dir, func(path string, info fs.FileInfo) error {
		rel, _ := filepath.Rel(dir, path)
		n := nested{path: path, name: filepath.ToSlash(rel), bundle: info.IsDir()}
		switch {
		case info.Mode().IsRegular():
			var ok bool
			if ok, n.signed = machO(path); ok {
				files = append(files, n)
			}
		case info.IsDir() && isBundle(path):
			bundles = append(bundles, n)
		}
		return nil
	})
	if err != nil {
		return err
	}
	inside := func(bundle string, list []nested) bool {
		return slices.ContainsFunc(list, func(n nested) bool { return strings.HasPrefix(n.name, bundle+"/") })
	}
	list := slices.Clone(files)
	for _, b := range bundles {
		// Bundles without code, such as localizations, are resources.
		if inside(b.name, files) {
			list = append(list, b)
		}
	}
	// What is deeper comes first, so a bundle comes after its contents.
	slices.SortStableFunc(list, func(a, b nested) int {
		return strings.Count(b.name, "/") - strings.Count(a.name, "/")
	})

	entitlements := map[string]string{}
	var unknown []string
	for name, file := range c.MacOS.HelperEntitlements {
		name = filepath.ToSlash(filepath.Clean(name))
		entitlements[name] = c.path(file)
		if !slices.ContainsFunc(list, func(n nested) bool { return n.name == name }) {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return fmt.Errorf("%s: macos.helperEntitlements: the resources hold no executable, library or bundle at %s", c.configName(), strings.Join(unknown, ", "))
	}

	var signed []nested
	for _, n := range list {
		ent := entitlements[n.name]
		if identity == "-" && ent == "" {
			switch {
			case !n.bundle && n.signed:
				continue
			case n.bundle && !inside(n.name, signed):
				if exec.Command("codesign", "--verify", n.path).Run() == nil {
					continue
				}
			}
		}
		args := nestedCodesignArgs(identity, ent, production)
		if out, err := exec.Command("codesign", append(args, n.path)...).CombinedOutput(); err != nil {
			return fmt.Errorf("codesign %s: %v\n%s", n.name, err, out)
		}
		signed = append(signed, n)
	}
	return nil
}

// nestedCodesignArgs returns the codesign arguments, before the path, that
// sign code among the resources like the app: with the hardened runtime and
// a secure timestamp for real identities in production. The code keeps the
// entitlements it is signed with, such as the JIT of a JavaScript runtime,
// unless entitlements names a property list of others.
func nestedCodesignArgs(identity, entitlements string, production bool) []string {
	args := []string{"--force", "--sign", identity}
	if production && identity != "-" {
		args = append(args, "--options", "runtime", "--timestamp")
	}
	if entitlements != "" {
		return append(args, "--entitlements", entitlements)
	}
	return append(args, "--preserve-metadata=entitlements")
}

// isBundle reports whether dir is a bundle that can hold code: an app, a
// framework or a plug-in, with an Info.plist where codesign looks for it.
func isBundle(dir string) bool {
	switch strings.ToLower(filepath.Ext(dir)) {
	case ".app", ".appex", ".bundle", ".framework", ".plugin", ".xpc":
	default:
		return false
	}
	for _, plist := range []string{"Contents/Info.plist", "Resources/Info.plist", "Info.plist"} {
		if fileExists(filepath.Join(dir, filepath.FromSlash(plist))) {
			return true
		}
	}
	return false
}

// machO reports whether the file at path is a Mach-O file or a universal
// binary, and whether every architecture in it has a code signature.
func machO(path string) (ok, signed bool) {
	f, err := os.Open(path)
	if err != nil {
		return false, false
	}
	defer f.Close()
	var head [8]byte
	if _, err := io.ReadFull(f, head[:]); err != nil {
		return false, false
	}
	magic := binary.BigEndian.Uint32(head[:])
	switch magic {
	case 0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe:
		return true, machOSigned(f, 0)
	case 0xcafebabe, 0xcafebabf:
	default:
		return false, false
	}
	n := binary.BigEndian.Uint32(head[4:])
	if magic == 0xcafebabe && n >= 20 {
		// Java class files share the magic; their version follows it, a
		// much larger number than the architectures of a universal binary.
		return false, false
	}
	size := int64(20) // fat_arch
	if magic == 0xcafebabf {
		size = 32 // fat_arch_64
	}
	signed = n > 0
	for i := range int64(n) {
		var arch [32]byte
		if _, err := f.ReadAt(arch[:size], 8+i*size); err != nil {
			return true, false
		}
		offset := int64(binary.BigEndian.Uint32(arch[8:]))
		if magic == 0xcafebabf {
			offset = int64(binary.BigEndian.Uint64(arch[8:]))
		}
		signed = signed && machOSigned(f, offset)
	}
	return true, signed
}

// machOSigned reports whether the Mach-O file at offset in r has a code
// signature, an LC_CODE_SIGNATURE load command.
func machOSigned(r io.ReaderAt, offset int64) bool {
	var h [28]byte // mach_header; mach_header_64 adds a reserved field
	if _, err := r.ReadAt(h[:], offset); err != nil {
		return false
	}
	var order binary.ByteOrder = binary.LittleEndian
	magic := order.Uint32(h[:])
	if magic == 0xcefaedfe || magic == 0xcffaedfe {
		order, magic = binary.BigEndian, binary.BigEndian.Uint32(h[:])
	}
	size := int64(28)
	switch magic {
	case 0xfeedface:
	case 0xfeedfacf:
		size = 32
	default:
		return false
	}
	ncmds, sizeofcmds := order.Uint32(h[16:]), order.Uint32(h[20:])
	if sizeofcmds > 1<<24 {
		return false
	}
	cmds := make([]byte, sizeofcmds)
	if _, err := r.ReadAt(cmds, offset+size); err != nil {
		return false
	}
	for range ncmds {
		if len(cmds) < 8 {
			break
		}
		cmd, cmdsize := order.Uint32(cmds), order.Uint32(cmds[4:])
		if cmd == 0x1d { // LC_CODE_SIGNATURE
			return true
		}
		if cmdsize < 8 || uint64(cmdsize) > uint64(len(cmds)) {
			break
		}
		cmds = cmds[cmdsize:]
	}
	return false
}
