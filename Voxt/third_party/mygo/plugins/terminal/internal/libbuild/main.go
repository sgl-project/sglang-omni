// Libbuild builds libghostty-vt for every platform MyGo supports from
// Ghostty's sources at a commit, with Zig (the version Ghostty asks for),
// and writes mygo-plugin.json, which names the files and their SHA-256 for
// the CLI and the terminal package, and ghostty_themes.go, the themes that
// Ghostty ships:
//
//	go generate ./plugins/terminal
//
// With -themes, it only writes the themes, which needs no Zig.
//
// The files go into plugins/terminal/build, to publish as the assets of the
// release libghostty-vt-<commit>, which mygo-plugin.json points at:
//
//	gh release create libghostty-vt-<commit> plugins/terminal/build/* --latest=false ...
//
// macOS's libraries are linked by Apple's linker when Xcode is installed,
// which the build of Ghostty prefers, so build them on a Mac.
package main

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// commit is the commit of Ghostty this package binds.
const commit = "befcdfd2c3a1cb24d9ec886e93c95b2b5daa7028"

var targets = []struct {
	goos, goarch, zig string
	// name is the file in apps; built is where Zig puts it.
	name, built string
}{
	{"darwin", "arm64", "aarch64-macos", "libghostty-vt.dylib", "lib/libghostty-vt.dylib"},
	{"darwin", "amd64", "x86_64-macos", "libghostty-vt.dylib", "lib/libghostty-vt.dylib"},
	// The glibc of Debian 10 and Ubuntu 20.04, and later.
	{"linux", "amd64", "x86_64-linux-gnu.2.28", "libghostty-vt.so", "lib/libghostty-vt.so"},
	{"linux", "arm64", "aarch64-linux-gnu.2.28", "libghostty-vt.so", "lib/libghostty-vt.so"},
	{"windows", "amd64", "x86_64-windows-gnu", "ghostty-vt.dll", "bin/ghostty-vt.dll"},
	{"windows", "arm64", "aarch64-windows-gnu", "ghostty-vt.dll", "bin/ghostty-vt.dll"},
}

func main() {
	src := flag.String("src", "", "a checkout of Ghostty, at the commit (cloned into the user's cache when empty)")
	out := flag.String("out", "build", "the directory of the libraries")
	release := flag.String("release", "https://github.com/egoist/mygo/releases/download/libghostty-vt-"+commit[:12], "the URL of the release hosting the libraries")
	themes := flag.Bool("themes", false, "only write the themes")
	flag.Parse()
	if *themes {
		writeThemes(*src)
		return
	}
	if *src == "" {
		*src = checkout()
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	files := map[string]map[string]string{}
	for _, t := range targets {
		prefix, err := os.MkdirTemp("", "libghostty-vt-")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(prefix)
		log.Printf("building %s/%s", t.goos, t.goarch)
		run(*src, "zig", "build", "-Demit-lib-vt", "-Dtarget="+t.zig, "-Doptimize=ReleaseFast", "-Dstrip=true", "--prefix", prefix)
		ext := filepath.Ext(t.name)
		asset := fmt.Sprintf("libghostty-vt-%s-%s%s", t.goos, t.goarch, ext)
		built := filepath.Join(prefix, t.built)
		if t.goos == "linux" {
			// The build keeps the debug information of ELF libraries,
			// four fifths of them.
			built = stripELF(built)
		}
		sum := copyFile(built, filepath.Join(*out, asset))
		files[t.goos+"-"+t.goarch] = map[string]string{"name": t.name, "url": *release + "/" + asset, "sha256": sum}
	}
	manifest := map[string]any{"libraries": []any{map[string]any{
		"name":   "libghostty-vt",
		"source": "https://github.com/ghostty-org/ghostty/tree/" + commit,
		"files":  files,
	}}}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("mygo-plugin.json", append(data, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote mygo-plugin.json; publish %s as the assets of %s", *out, *release)
	writeThemes(*src)
}

// checkout returns a checkout of Ghostty at the commit in the user's
// cache, cloning or fetching it.
func checkout() string {
	cache, err := os.UserCacheDir()
	if err != nil {
		log.Fatal(err)
	}
	dir := filepath.Join(cache, "mygo", "ghostty-src")
	if _, err := os.Stat(filepath.Join(dir, ".git")); err != nil {
		run("", "git", "clone", "--filter=blob:none", "https://github.com/ghostty-org/ghostty", dir)
	} else {
		run(dir, "git", "fetch", "origin")
	}
	run(dir, "git", "checkout", "--detach", commit)
	return dir
}

func run(dir, name string, args ...string) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("%s %s: %v", name, strings.Join(args, " "), err)
	}
}

// copyFile copies src, following links, to dst and returns its SHA-256.
func copyFile(src, dst string) string {
	in, err := os.Open(src)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()
	f, err := os.Create(dst)
	if err != nil {
		log.Fatal(err)
	}
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(f, h), in); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// stripELF writes a copy of the ELF library at path without the contents
// of its debug sections, as `strip --strip-debug` would, and returns its
// path. What loading uses (the segments, and the sections in them) stays
// in place; the other sections after the segments move down, and the
// debug sections keep their headers, empty, so that no index changes.
func stripELF(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	if f.Class != elf.ELFCLASS64 || f.Data != elf.ELFDATA2LSB {
		log.Fatalf("%s: not a 64-bit little-endian ELF file", path)
	}
	end := uint64(0)
	for _, p := range f.Progs {
		end = max(end, p.Off+p.Filesz)
	}
	out := append([]byte(nil), data[:end]...)
	le := binary.LittleEndian
	shoff := le.Uint64(data[0x28:])
	shentsize, shnum := uint64(le.Uint16(data[0x3a:])), uint64(le.Uint16(data[0x3c:]))
	headers := append([]byte(nil), data[shoff:shoff+shentsize*shnum]...)
	for i, s := range f.Sections {
		h := headers[uint64(i)*shentsize:]
		if s.Type == elf.SHT_NULL || s.Type == elf.SHT_NOBITS || s.Flags&elf.SHF_ALLOC != 0 || s.Offset < end {
			continue
		}
		size := s.FileSize
		if strings.HasPrefix(s.Name, ".debug") || strings.HasPrefix(s.Name, ".zdebug") {
			size = 0
		}
		align := max(s.Addralign, 1)
		for uint64(len(out))%align != 0 {
			out = append(out, 0)
		}
		le.PutUint64(h[0x18:], uint64(len(out))) // sh_offset
		le.PutUint64(h[0x20:], size)             // sh_size
		out = append(out, data[s.Offset:s.Offset+size]...)
	}
	for len(out)%8 != 0 {
		out = append(out, 0)
	}
	le.PutUint64(out[0x28:], uint64(len(out))) // e_shoff
	out = append(out, headers...)
	stripped := path + ".stripped"
	if err := os.WriteFile(stripped, out, 0o755); err != nil {
		log.Fatal(err)
	}
	return stripped
}
