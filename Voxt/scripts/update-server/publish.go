package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// publish puts the update files of the build in dir into the bucket in
// bucket as actnow.dev's back end publishes a version
// (docs/release-pipeline.md of actnow-www): a release under <app>/, a beta
// (a version with a pre-release part) under <app>/beta/, the archive and
// the deltas first and the manifest last; the beta channel's manifest is
// the newer of the newest beta and release, so a release's manifest goes
// to <app>/beta/ too unless a newer beta is there. Like the site it
// refuses a version not higher than its channel's. The update tests
// (scripts/mac-update-e2e.sh, scripts/mac-beta-e2e.sh) publish with it,
// after checking the build with scripts/actnow-publish.sh --dry-run.
func publish(dir, bucket, app string, out io.Writer) error {
	const manifestName = "update-darwin-arm64.json"
	var m struct {
		Version string `json:"version"`
	}
	data, err := os.ReadFile(filepath.Join(dir, manifestName))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &m); err != nil || m.Version == "" {
		return fmt.Errorf("%s: no version (%v)", manifestName, err)
	}
	beta := strings.Contains(m.Version, "-")
	channel := filepath.Join(bucket, app)
	if beta {
		channel = filepath.Join(channel, "beta")
	}
	published := func(path string) string {
		var p struct {
			Version string `json:"version"`
		}
		b, err := os.ReadFile(path)
		if err != nil || json.Unmarshal(b, &p) != nil {
			return ""
		}
		return p.Version
	}
	if v := published(filepath.Join(channel, manifestName)); v != "" && compareVersions(m.Version, v) <= 0 {
		return fmt.Errorf("%s is not newer than %s, the channel's version", m.Version, v)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var files []string
	for _, e := range entries {
		if n := e.Name(); strings.HasSuffix(n, ".tar.gz") || strings.HasSuffix(n, ".delta") {
			files = append(files, n)
		}
	}
	slices.Sort(files)
	put := func(name, to string) error {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		if err := os.MkdirAll(to, 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(to, name+".tmp"), b, 0o644); err != nil {
			return err
		}
		rel, _ := filepath.Rel(bucket, filepath.Join(to, name))
		fmt.Fprintf(out, "published %s\n", filepath.ToSlash(rel))
		return os.Rename(filepath.Join(to, name+".tmp"), filepath.Join(to, name))
	}
	for _, f := range append(files, manifestName) {
		if err := put(f, channel); err != nil {
			return err
		}
	}
	if !beta {
		betaDir := filepath.Join(bucket, app, "beta")
		if v := published(filepath.Join(betaDir, manifestName)); v == "" || compareVersions(v, m.Version) < 0 {
			return put(manifestName, betaDir)
		}
	}
	return nil
}

// compareVersions orders versions as MyGo's updater does
// (internal/update.Compare): numbers field by field, a version with a
// pre-release part before the same without, pre-release fields as numbers
// when both are, else as text.
func compareVersions(a, b string) int {
	a, b = strings.TrimPrefix(a, "v"), strings.TrimPrefix(b, "v")
	a, _, _ = strings.Cut(a, "+")
	b, _, _ = strings.Cut(b, "+")
	ar, apre, _ := strings.Cut(a, "-")
	br, bpre, _ := strings.Cut(b, "-")
	if c := compareFields(ar, br); c != 0 {
		return c
	}
	switch {
	case apre == bpre:
		return 0
	case apre == "":
		return 1
	case bpre == "":
		return -1
	}
	return compareFields(apre, bpre)
}

func compareFields(a, b string) int {
	af, bf := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(af) || i < len(bf); i++ {
		var x, y string
		if i < len(af) {
			x = af[i]
		}
		if i < len(bf) {
			y = bf[i]
		}
		xn, xerr := strconv.Atoi(x)
		yn, yerr := strconv.Atoi(y)
		switch {
		case x == y:
			continue
		case x == "":
			return -1
		case y == "":
			return 1
		case xerr == nil && yerr == nil:
			if xn < yn {
				return -1
			}
			return 1
		case xerr == nil:
			return -1
		case yerr == nil:
			return 1
		}
		return strings.Compare(x, y)
	}
	return 0
}
