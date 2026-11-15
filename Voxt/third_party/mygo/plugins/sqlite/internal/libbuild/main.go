// Builds SQLite's C amalgamation and the plugin's C shim with Zig 0.16:
//
//	go generate ./plugins/sqlite
//
// The libraries are release assets, as with libghostty-vt. Builds also
// populate the CLI's cache, so mygo dev/build work before publication.
package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/sha3"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const version = "3.53.4"
const sourceURL = "https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip"
const sourceSHA3 = "628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e"

var targets = []struct{ name, zig, file, built string }{
	{"darwin-arm64", "aarch64-macos.12.0", "libmygo-sqlite3.dylib", "lib/libmygo-sqlite3.dylib"},
	{"darwin-amd64", "x86_64-macos.12.0", "libmygo-sqlite3.dylib", "lib/libmygo-sqlite3.dylib"},
	{"linux-amd64", "x86_64-linux-gnu.2.28", "libmygo-sqlite3.so", "lib/libmygo-sqlite3.so"},
	{"linux-arm64", "aarch64-linux-gnu.2.28", "libmygo-sqlite3.so", "lib/libmygo-sqlite3.so"},
	{"windows-amd64", "x86_64-windows-gnu", "mygo-sqlite3.dll", "bin/mygo-sqlite3.dll"},
	{"windows-arm64", "aarch64-windows-gnu", "mygo-sqlite3.dll", "bin/mygo-sqlite3.dll"},
}

type manifest struct {
	Libraries []library `json:"libraries"`
}
type library struct {
	Name   string                `json:"name"`
	Source string                `json:"source"`
	Files  map[string]nativeFile `json:"files"`
}
type nativeFile struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
}

func main() {
	src := flag.String("src", "", "verified amalgamation directory (downloaded to the user's cache when empty)")
	out := flag.String("out", "build", "output directory")
	target := flag.String("target", "", "build only GOOS-GOARCH; 'host' means this machine (default: all six)")
	release := flag.String("release", "https://github.com/egoist/mygo/releases/download/sqlite-"+version+"-1", "release asset base URL")
	flag.Parse()
	if *target == "host" {
		*target = runtime.GOOS + "-" + runtime.GOARCH
	}
	if *target != "" {
		found := false
		for _, t := range targets {
			found = found || t.name == *target
		}
		if !found {
			log.Fatalf("unsupported target %q", *target)
		}
	}
	if *src == "" {
		var err error
		*src, err = source()
		if err != nil {
			log.Fatal(err)
		}
	}
	*src, _ = filepath.Abs(*src)
	*out, _ = filepath.Abs(*out)
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	m := manifest{Libraries: []library{{Name: "mygo-sqlite3", Source: sourceURL, Files: map[string]nativeFile{}}}}
	if *target != "" {
		if data, err := os.ReadFile("mygo-plugin.json"); err == nil {
			var old manifest
			if json.Unmarshal(data, &old) == nil && len(old.Libraries) == 1 && old.Libraries[0].Source == sourceURL {
				m.Libraries[0].Files = old.Libraries[0].Files
			}
		}
	}
	for _, t := range targets {
		if *target != "" && *target != t.name {
			continue
		}
		prefix, err := os.MkdirTemp("", "mygo-sqlite-")
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("building SQLite %s for %s", version, t.name)
		cmd := exec.Command("zig", "build", "-Dsqlite="+*src, "-Dtarget="+t.zig, "-Doptimize=ReleaseFast", "--prefix", prefix)
		cmd.Dir, cmd.Stdout, cmd.Stderr = "native", os.Stderr, os.Stderr
		if err := cmd.Run(); err != nil {
			os.RemoveAll(prefix)
			log.Fatal(err)
		}
		data, err := os.ReadFile(filepath.Join(prefix, t.built))
		os.RemoveAll(prefix)
		if err != nil {
			log.Fatal(err)
		}
		sum := sha256.Sum256(data)
		digest := hex.EncodeToString(sum[:])
		asset := "mygo-sqlite3-" + t.name + filepath.Ext(t.file)
		if err := os.WriteFile(filepath.Join(*out, asset), data, 0o755); err != nil {
			log.Fatal(err)
		}
		cache, err := os.UserCacheDir()
		if err != nil {
			log.Fatal(err)
		}
		dir := filepath.Join(cache, "mygo", "natives", digest)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatal(err)
		}
		cached := filepath.Join(dir, t.file)
		// A running Go program may have this cached DLL loaded, which
		// Windows does not let us overwrite. Identical builds need no write.
		if existing, err := os.ReadFile(cached); err != nil || sha256.Sum256(existing) != sum {
			if err := os.WriteFile(cached, data, 0o755); err != nil {
				log.Fatal(err)
			}
		}
		m.Libraries[0].Files[t.name] = nativeFile{t.file, strings.TrimRight(*release, "/") + "/" + asset, digest}
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("mygo-plugin.json", append(data, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote mygo-plugin.json; release assets are in %s", *out)
}

func source() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(cache, "mygo", "sqlite-src", sourceSHA3)
	archive := filepath.Join(dir, "amalgamation.zip")
	data, _ := os.ReadFile(archive)
	if sum := sha3.Sum256(data); hex.EncodeToString(sum[:]) != sourceSHA3 {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
		if err != nil {
			return "", err
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", err
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			return "", fmt.Errorf("%s: %s", sourceURL, res.Status)
		}
		data, err = io.ReadAll(io.LimitReader(res.Body, 16<<20))
		if err != nil {
			return "", err
		}
		sum := sha3.Sum256(data)
		if hex.EncodeToString(sum[:]) != sourceSHA3 {
			return "", fmt.Errorf("%s: SHA3-256 mismatch", sourceURL)
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(archive, data, 0o644); err != nil {
			return "", err
		}
	}
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", err
	}
	found := map[string]bool{}
	for _, f := range z.File {
		name := filepath.Base(f.Name)
		if name != "sqlite3.c" && name != "sqlite3.h" {
			continue
		}
		r, err := f.Open()
		if err != nil {
			return "", err
		}
		contents, err := io.ReadAll(io.LimitReader(r, 16<<20))
		r.Close()
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(dir, name), contents, 0o644); err != nil {
			return "", err
		}
		found[name] = true
	}
	if len(found) != 2 {
		return "", fmt.Errorf("%s: missing sqlite3.c or sqlite3.h", sourceURL)
	}
	return dir, nil
}
