// Command prefetch downloads into the user's cache what this repository's
// tests otherwise download there while they run: the native libraries that
// the plugins name in their mygo-plugin.json, for this platform, and on
// Windows the NSIS that mygo build uses. CI runs it before `go test` and
// keeps what it downloads in the Actions cache, under a key of the files
// that pin them, so that the tests do not depend on the network:
//
//	go run ./internal/prefetch
//
// It downloads them where the plugins and the CLI look:
// <cache>/mygo/natives/<sha256>/<name> and <cache>/mygo/nsis-<version>.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/egoist/mygo/internal/download"
	"github.com/egoist/mygo/internal/nsis"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("prefetch: ")
	cache, err := os.UserCacheDir()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	manifests, err := filepath.Glob(filepath.Join("plugins", "*", "mygo-plugin.json"))
	if err != nil {
		log.Fatal(err)
	}
	if len(manifests) == 0 {
		log.Fatal("no plugins/*/mygo-plugin.json: run it at the root of the repository")
	}
	target := runtime.GOOS + "-" + runtime.GOARCH
	for _, manifest := range manifests {
		data, err := os.ReadFile(manifest)
		if err != nil {
			log.Fatal(err)
		}
		var n struct {
			Libraries []struct {
				Name  string `json:"name"`
				Files map[string]struct {
					Name   string `json:"name"`
					URL    string `json:"url"`
					SHA256 string `json:"sha256"`
				} `json:"files"`
			} `json:"libraries"`
		}
		if err := json.Unmarshal(data, &n); err != nil {
			log.Fatalf("%s: %v", manifest, err)
		}
		for _, l := range n.Libraries {
			f, ok := l.Files[target]
			if !ok {
				fmt.Printf("%s has no build for %s\n", l.Name, target)
				continue
			}
			path := filepath.Join(cache, "mygo", "natives", f.SHA256, f.Name)
			if err := download.Cached(ctx, download.File{URL: f.URL, SHA256: f.SHA256}, path); err != nil {
				log.Fatalf("%s: %v", l.Name, err)
			}
			fmt.Printf("%s: %s\n", l.Name, path)
		}
	}
	if runtime.GOOS == "windows" {
		makensis, err := nsis.Download(ctx, nsis.Dir(), func(url string) download.File { return download.File{URL: url} })
		if err != nil {
			log.Fatalf("NSIS: %v", err)
		}
		fmt.Printf("NSIS %s: %s\n", nsis.Release.Version, makensis)
	}
}
