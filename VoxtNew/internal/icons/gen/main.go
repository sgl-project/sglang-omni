// Command gen writes the Solar icons VoxtNew uses as Go source
// (../solar_gen.go), from the @iconify-json/solar package at a pinned
// version, checked against its SHA-256. Only the icons listed in
// ../names.txt are written, each in its Linear and Bold variants.
//
//	cd internal/icons && go generate
//
// It downloads the package from the npm registry (through the
// environment's proxy), or reads it with -tarball. The tarball is not
// kept: the generated file is committed instead.
//
// Solar is by 480 Design, under CC BY 4.0
// (https://creativecommons.org/licenses/by/4.0/); see NOTICE.
package main

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// The pinned package.
const (
	version = "1.2.13"
	url     = "https://registry.npmjs.org/@iconify-json/solar/-/solar-" + version + ".tgz"
	sum     = "9efec9045e9188c5dd5cf9de52f8ad968ffabee565026480a6c74bfe63b3bbc2"
)

// iconSet is the part of an IconifyJSON file the generator reads.
type iconSet struct {
	Width   int                              `json:"width"`
	Height  int                              `json:"height"`
	Icons   map[string]struct{ Body string } `json:"icons"`
	Aliases map[string]struct {
		Parent string `json:"parent"`
	} `json:"aliases"`
}

func main() {
	tarball := flag.String("tarball", "", "read the package from this file instead of downloading it")
	names := flag.String("names", "names.txt", "the icons to write, one base name a line")
	out := flag.String("o", "solar_gen.go", "the Go file to write")
	flag.Parse()
	log.SetFlags(0)

	data, err := fetch(*tarball)
	if err != nil {
		log.Fatal(err)
	}
	if got := sha256.Sum256(data); hex.EncodeToString(got[:]) != sum {
		log.Fatalf("gen: the package's SHA-256 is %x, want %s", got, sum)
	}
	raw, err := extract(data, "package/icons.json")
	if err != nil {
		log.Fatal(err)
	}
	var set iconSet
	if err := json.Unmarshal(raw, &set); err != nil {
		log.Fatal(err)
	}
	list, err := readNames(*names)
	if err != nil {
		log.Fatal(err)
	}
	src, err := generate(&set, list)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, src, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("gen: %d icons (Linear and Bold) from @iconify-json/solar %s\n", len(list), version)
}

func fetch(path string) ([]byte, error) {
	if path != "" {
		return os.ReadFile(path)
	}
	cl := &http.Client{Timeout: 2 * time.Minute}
	resp, err := cl.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gen: %s: %s", url, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 64<<20))
}

// extract reads one file of a .tgz.
func extract(tgz []byte, name string) ([]byte, error) {
	zr, err := gzip.NewReader(bytes.NewReader(tgz))
	if err != nil {
		return nil, err
	}
	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil, fmt.Errorf("gen: no %s in the package", name)
		}
		if err != nil {
			return nil, err
		}
		if h.Name == name {
			return io.ReadAll(io.LimitReader(tr, 64<<20))
		}
	}
}

// readNames reads the base names (without -linear or -bold; or a whole
// duotone variant's name, variants); # starts a comment.
func readNames(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	seen := map[string]bool{}
	var out []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line, _, _ := strings.Cut(sc.Text(), "#")
		if line = strings.TrimSpace(line); line != "" && !seen[line] {
			seen[line] = true
			out = append(out, line)
		}
	}
	sort.Strings(out)
	return out, sc.Err()
}

// body finds an icon's SVG body, through an alias without transforms.
func body(set *iconSet, name string) (string, error) {
	if ic, ok := set.Icons[name]; ok {
		return ic.Body, nil
	}
	if al, ok := set.Aliases[name]; ok {
		if ic, ok := set.Icons[al.Parent]; ok {
			return ic.Body, nil
		}
	}
	return "", fmt.Errorf("gen: no icon %q in Solar %s", name, version)
}

// variants are an icon's Linear and Bold bodies: its -linear and -bold
// variants, or, for the name of one variant ("…-line-duotone",
// "…-bold-duotone"), that variant for both.
func variants(set *iconSet, n string) (lin, bold string, err error) {
	if strings.HasSuffix(n, "-line-duotone") || strings.HasSuffix(n, "-bold-duotone") {
		lin, err = body(set, n)
		return lin, lin, err
	}
	if lin, err = body(set, n+"-linear"); err != nil {
		return "", "", err
	}
	bold, err = body(set, n+"-bold")
	return lin, bold, err
}

// goName turns "user-speak-rounded" into "UserSpeakRounded".
func goName(base string) string {
	var b strings.Builder
	for _, part := range strings.Split(base, "-") {
		if part == "" {
			continue
		}
		b.WriteString(strings.ToUpper(part[:1]) + part[1:])
	}
	return b.String()
}

func generate(set *iconSet, names []string) ([]byte, error) {
	var b bytes.Buffer
	fmt.Fprintf(&b, "// Code generated by go run ./gen; DO NOT EDIT.\n\n")
	fmt.Fprintf(&b, "// Solar icons by 480 Design (https://www.figma.com/community/file/1166831539721848736),\n")
	fmt.Fprintf(&b, "// CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), from @iconify-json/solar %s.\n\n", version)
	fmt.Fprintf(&b, "package icons\n\n")
	fmt.Fprintf(&b, "// SolarVersion is the version of @iconify-json/solar the icons come from.\nconst SolarVersion = %q\n\n", version)
	fmt.Fprintf(&b, "var (\n")
	for _, n := range names {
		lin, bold, err := variants(set, n)
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(&b, "\t// %s is Solar's %q.\n", goName(n), n)
		fmt.Fprintf(&b, "\t%s = solar(%q,\n\t\t%q,\n\t\t%q)\n", goName(n), n, lin, bold)
	}
	fmt.Fprintf(&b, ")\n")
	return format.Source(b.Bytes())
}
