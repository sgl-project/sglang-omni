package main

import (
	"cmp"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/egoist/mygo/internal/update"
)

// Signed updates (mygo.Updater): `mygo keygen` creates the key pair,
// `mygo build` signs an archive of each platform's app with the private key
// and writes update-<target>.json, which apps built with the public key
// check.

// Updates configures signed updates.
type Updates struct {
	// PublicKey is the contents of mygo-update.pub (mygo keygen). Installed
	// apps only accept updates signed with its private key.
	PublicKey string `json:"publicKey"`
	// GitHub is the public repository ("owner/name") whose releases hold
	// the updates: tagged TagPrefix+version, the latest release serving
	// the manifests. With a TagPrefix other than "v", the repository may
	// hold other releases, such as a CLI's, and the newest release tagged
	// TagPrefix serves them, found through the GitHub API.
	GitHub    string `json:"github"`
	TagPrefix string `json:"tagPrefix"`
	// URL is where the update files are served from instead, e.g.
	// "https://downloads.example.com/my-app".
	URL string `json:"url"`
	// S3 is the bucket that serves URL, which `mygo build -upload` uploads
	// to.
	S3 *S3 `json:"s3"`
	// PrivateKey is the path of mygo-update.key. MYGO_UPDATER_PRIVATE_KEY,
	// holding the key itself, takes precedence.
	PrivateKey string `json:"privateKey"`
	// Changelog is a Markdown file whose "## <version>" section becomes the
	// release notes (default CHANGELOG.md when it exists).
	Changelog string `json:"changelog"`
	// Deltas is how many earlier versions get a delta update: a smaller
	// download holding only what changed (default 3, 0 for none).
	Deltas *int `json:"deltas"`
	// Prereleases puts versions with a pre-release part, such as
	// 1.3.0-beta.1, in a beta channel beside the releases: the beta/
	// directory of URL, whose manifest the server keeps the newest version
	// of both channels, for the users who opt in (Updater.SetPrereleases).
	// Builds make delta updates from the newest version of both channels,
	// so that both get one. -upload does not publish channels: S3 is not
	// for it.
	Prereleases bool `json:"prereleases"`
}

// betaDir is the directory of the beta channel in URL and S3's prefix
// (Prereleases).
const betaDir = "beta"

// deltas returns how many earlier versions get a delta update.
func (u *Updates) deltas() int {
	if u.Deltas == nil {
		return 3
	}
	return *u.Deltas
}

// tagged reports whether the manifests are those of the newest release
// tagged TagPrefix on GitHub rather than of the latest release.
func (u *Updates) tagged() bool {
	return u.GitHub != "" && u.TagPrefix != "v"
}

func (u *Updates) validate() error {
	if u.PublicKey == "" {
		return errors.New("updates needs the publicKey of mygo keygen")
	}
	if _, err := update.ParsePublicKey(u.PublicKey); err != nil {
		return fmt.Errorf("updates.publicKey: %w", err)
	}
	switch {
	case (u.GitHub == "") == (u.URL == ""):
		return errors.New("updates needs either github (owner/name) or url")
	case u.GitHub != "" && strings.Count(u.GitHub, "/") != 1:
		return fmt.Errorf("updates.github %q is not owner/name", u.GitHub)
	case u.URL != "":
		p, err := url.Parse(u.URL)
		if err != nil || p.Scheme != "https" || p.Host == "" {
			return fmt.Errorf("updates.url %q is not an https URL", u.URL)
		}
	}
	if u.S3 != nil {
		if u.URL == "" {
			return errors.New("updates.s3 needs url, the HTTPS URL that serves the files of the bucket, in place of github")
		}
		if err := u.S3.validate(); err != nil {
			return err
		}
	}
	if u.Prereleases && (u.URL == "" || u.S3 != nil) {
		return errors.New("updates.prereleases needs url, without s3: the beta channel is the beta/ directory of url, which the server publishes")
	}
	if u.Deltas != nil && *u.Deltas < 0 {
		return fmt.Errorf("updates.deltas is %d, not a number of versions", *u.Deltas)
	}
	if u.TagPrefix == "" {
		u.TagPrefix = "v"
	}
	return nil
}

// updateFeed returns the URL of the manifest of target, the releases'.
func (c *Config) updateFeed(target string) string {
	return c.updateFile(update.ManifestName(target), true)
}

// betaFeed returns the URL of the manifest of target in the beta channel,
// or "" without one (updates.prereleases).
func (c *Config) betaFeed(target string) string {
	if !c.Updates.Prereleases {
		return ""
	}
	return strings.TrimSuffix(c.Updates.URL, "/") + "/" + betaDir + "/" + url.PathEscape(update.ManifestName(target))
}

// prerelease reports whether this version goes to the beta channel: it has
// a pre-release part, and updates.prereleases is set.
func (c *Config) prerelease() bool {
	return c.Updates.Prereleases && strings.Contains(c.Version, "-")
}

// updateFile returns the URL of a published update file; manifests come
// from the latest release on GitHub, or with a tag prefix from the newest
// release tagged with it (a feed that update.ResolveFeed resolves),
// archives from the release of this version. With url, the files of a
// prerelease are in the beta channel's directory.
func (c *Config) updateFile(name string, latest bool) string {
	u := c.Updates
	if u.GitHub == "" {
		dir := strings.TrimSuffix(u.URL, "/")
		if !latest && c.prerelease() {
			dir += "/" + betaDir
		}
		return dir + "/" + url.PathEscape(name)
	}
	if latest && u.tagged() {
		return update.TaggedFeed(u.GitHub, u.TagPrefix, name)
	}
	if latest {
		return "https://github.com/" + u.GitHub + "/releases/latest/download/" + url.PathEscape(name)
	}
	return "https://github.com/" + u.GitHub + "/releases/download/" + url.PathEscape(u.TagPrefix+c.Version) + "/" + url.PathEscape(name)
}

// updateFlags are the -ldflags that point the build for target at its
// manifest.
func updateFlags(c *Config, target string) string {
	if c.Updates == nil {
		return ""
	}
	flags := " -X " + ldflagsQuote("github.com/egoist/mygo.packageUpdateFeed="+c.updateFeed(target)) +
		" -X github.com/egoist/mygo.packageUpdateKey=" + c.Updates.PublicKey
	if beta := c.betaFeed(target); beta != "" {
		flags += " -X " + ldflagsQuote("github.com/egoist/mygo.packageUpdateBetaFeed="+beta)
	}
	return flags
}

// signingKey returns the private key that signs updates, or nil when none
// is available.
func (c *Config) signingKey() (ed25519.PrivateKey, error) {
	text := os.Getenv("MYGO_UPDATER_PRIVATE_KEY")
	if text == "" && c.Updates.PrivateKey != "" {
		path := c.Updates.PrivateKey
		if rest, ok := strings.CutPrefix(path, "~/"); ok {
			home, _ := os.UserHomeDir()
			path = filepath.Join(home, rest)
		}
		b, err := os.ReadFile(c.path(path))
		if err != nil {
			return nil, err
		}
		text = string(b)
	}
	if text == "" {
		return nil, nil
	}
	key, err := update.ParsePrivateKey(text)
	if err != nil {
		return nil, err
	}
	if base64.StdEncoding.EncodeToString(key.Public().(ed25519.PublicKey)) != strings.TrimSpace(c.Updates.PublicKey) {
		return nil, fmt.Errorf("the update signing key does not match updates.publicKey in %s", c.configName())
	}
	return key, nil
}

// releaseNotes returns the changelog section of this version.
func (c *Config) releaseNotes() (string, error) {
	name := c.Updates.Changelog
	if name == "" {
		name = "CHANGELOG.md"
		if !fileExists(c.path(name)) {
			return "", nil
		}
	}
	data, err := os.ReadFile(c.path(name))
	if err != nil {
		return "", err
	}
	notes := update.ReleaseNotes(string(data), c.Version)
	if notes == "" {
		return "", fmt.Errorf("%s has no notes for %s: add a \"## %s\" section", name, c.Version, c.Version)
	}
	return notes, nil
}

// writeArchive archives the entries of stage, the app of target: updates
// install the archive, and Linux users extract it (install.sh). With
// updates and their signing key, it signs the archive, makes delta updates
// from earlier versions and writes the manifest. It returns the files
// written, none but on Linux without the key.
func writeArchive(c *Config, stage, target string, entries []string) ([]string, error) {
	var key ed25519.PrivateKey
	if c.Updates != nil {
		var err error
		if key, err = c.signingKey(); err != nil {
			return nil, fmt.Errorf("updates: %w", err)
		}
		if key == nil {
			warnf("Not signing an update: set MYGO_UPDATER_PRIVATE_KEY or updates.privateKey")
		}
	}
	if key == nil && !strings.HasPrefix(target, "linux-") {
		return nil, nil
	}
	var notes string
	if key != nil {
		var err error
		if notes, err = c.releaseNotes(); err != nil {
			return nil, err
		}
	}
	name := slugify(c.executableName()) + "-" + c.Version + "-" + target + ".tar.gz"
	archive := filepath.Join(stage, name)
	size, sig, err := writeSigned(archive, key, func(w io.Writer) error {
		return update.WriteArchive(w, stage, entries)
	})
	if err != nil {
		return nil, err
	}
	if key == nil {
		return []string{archive}, nil
	}
	m := update.Manifest{
		Version:   c.Version,
		Notes:     notes,
		Date:      time.Now().UTC().Format(time.RFC3339),
		URL:       c.updateFile(name, false),
		Size:      size,
		Signature: sig,
	}
	files := []string{archive}
	if n := c.Updates.deltas(); n > 0 {
		deltas, err := writeDeltas(c, key, stage, target, entries, &m, n)
		if err != nil {
			return nil, err
		}
		files = append(files, deltas...)
	}
	data, _ := json.MarshalIndent(m, "", "  ")
	manifest := filepath.Join(stage, update.ManifestName(target))
	if err := os.WriteFile(manifest, append(data, '\n'), 0o644); err != nil {
		return nil, err
	}
	return append(files, manifest), nil
}

// writeSigned writes a file with write and returns its size and signature,
// none without a key.
func writeSigned(path string, key ed25519.PrivateKey, write func(io.Writer) error) (int64, string, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, "", err
	}
	sum := sha256.New()
	err = write(io.MultiWriter(f, sum))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return 0, "", err
	}
	info, err := os.Stat(path)
	switch {
	case err != nil:
		return 0, "", err
	case key == nil:
		return info.Size(), "", nil
	}
	return info.Size(), update.Sign(key, sum.Sum(nil)), nil
}

// writeDeltas makes delta updates to the app of target in stage from the
// n versions before it, as Sparkle does: the version of the published
// manifest of target and those it lists as previous, whose archives it
// downloads; with updates.prereleases also the beta channel's, the
// published version of each channel first. It adds them to m, with the
// archives the next version makes deltas from, and returns the files
// written. Deltas only save downloads, so a version without one is
// skipped with a message.
func writeDeltas(c *Config, key ed25519.PrivateKey, stage, target string, entries []string, m *update.Manifest, n int) ([]string, error) {
	var heads, rest []update.Archive
	for _, feed := range []string{c.updateFeed(target), c.betaFeed(target)} {
		if feed == "" {
			continue
		}
		published, err := fetchManifest(feed)
		switch {
		case errors.Is(err, errNotPublished):
			continue
		case err != nil:
			warnf("No delta updates for %s from %s: %v", target, feed, err)
			continue
		}
		heads = append(heads, update.Archive{Version: published.Version, URL: published.URL, Size: published.Size, Signature: published.Signature})
		rest = append(rest, published.Previous...)
	}
	if len(heads) == 0 {
		logf("No delta updates for %s: no version is published yet", target)
		return nil, nil
	}
	slices.SortStableFunc(heads, func(a, b update.Archive) int { return update.Compare(b.Version, a.Version) })
	slices.SortStableFunc(rest, func(a, b update.Archive) int { return update.Compare(b.Version, a.Version) })
	var bases []update.Archive
	for _, a := range append(heads, rest...) {
		if update.Compare(a.Version, c.Version) < 0 && !slices.ContainsFunc(bases, func(b update.Archive) bool { return b.Version == a.Version }) {
			bases = append(bases, a)
		}
	}
	if len(bases) == 0 {
		logf("No delta updates for %s: the published version is %s", target, heads[0].Version)
		return nil, nil
	}
	bases = bases[:min(len(bases), n)]
	m.Previous = bases[:min(len(bases), n-1)]

	work, err := os.MkdirTemp("", "mygo-delta-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(work)
	// The app as the delta has it: inside the bundle on macOS.
	newDir, newEntries := stage, entries
	if strings.HasPrefix(target, "darwin-") {
		newDir, newEntries = filepath.Join(stage, entries[0]), nil
	}
	var files []string
	for i, base := range bases {
		name := slugify(c.executableName()) + "-" + base.Version + "-to-" + c.Version + "-" + target + ".delta"
		path := filepath.Join(stage, name)
		d, err := writeDelta(key, base, filepath.Join(work, strconv.Itoa(i)), path, c.Version, newDir, newEntries)
		switch {
		case err != nil:
			warnf("No delta update from %s for %s: %v", base.Version, target, err)
			continue
		case d.Size >= m.Size:
			os.Remove(path)
			logf("No delta update from %s for %s: it is not smaller than the archive", base.Version, target)
			continue
		}
		d.URL = c.updateFile(name, false)
		m.Deltas = append(m.Deltas, d)
		files = append(files, path)
		logf("Made the delta update from %s for %s: %s, instead of %s", base.Version, target, formatSize(d.Size), formatSize(m.Size))
	}
	return files, nil
}

// writeDelta downloads the archive of base into work, checks that it is
// signed with key, and writes the delta from it to the app of version in
// newDir to path.
func writeDelta(key ed25519.PrivateKey, base update.Archive, work, path, version, newDir string, newEntries []string) (update.Delta, error) {
	if err := os.MkdirAll(work, 0o755); err != nil {
		return update.Delta{}, err
	}
	archive := filepath.Join(work, "archive.tar.gz")
	if err := fetchSigned(base.URL, base.Size, base.Signature, key.Public().(ed25519.PublicKey), archive); err != nil {
		return update.Delta{}, err
	}
	f, err := os.Open(archive)
	if err != nil {
		return update.Delta{}, err
	}
	oldDir := filepath.Join(work, "app")
	err = update.ExtractArchive(f, oldDir)
	f.Close()
	if err != nil {
		return update.Delta{}, err
	}
	if newEntries == nil { // a bundle
		all, _ := os.ReadDir(oldDir)
		if len(all) != 1 || !strings.HasSuffix(all[0].Name(), ".app") {
			return update.Delta{}, errors.New("its archive holds no app bundle")
		}
		oldDir = filepath.Join(oldDir, all[0].Name())
	}
	size, sig, err := writeSigned(path, key, func(w io.Writer) error {
		return update.WriteDelta(w, base.Version, version, oldDir, newDir, newEntries)
	})
	if err != nil {
		os.Remove(path)
		return update.Delta{}, err
	}
	return update.Delta{From: base.Version, Size: size, Signature: sig}, nil
}

var errNotPublished = errors.New("not published")

// fetchManifest downloads a published manifest, from its feed.
func fetchManifest(feed string) (*update.Manifest, error) {
	u, err := update.ResolveFeed(context.Background(), feed, func(_ context.Context, u string) (io.ReadCloser, error) {
		resp, err := updateGet(u, 30*time.Second)
		if err != nil {
			return nil, err
		}
		return resp.Body, nil
	})
	if errors.Is(err, update.ErrNotPublished) {
		return nil, errNotPublished
	}
	if err != nil {
		return nil, err
	}
	resp, err := updateGet(u, 30*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var m update.Manifest
	if err := json.NewDecoder(io.LimitReader(resp.Body, update.MaxManifestSize)).Decode(&m); err != nil {
		return nil, fmt.Errorf("reading %s: %w", u, err)
	}
	return &m, nil
}

// fetchSigned downloads a file of an update to path and checks its size
// and signature.
func fetchSigned(u string, size int64, signature string, pub ed25519.PublicKey, path string) error {
	resp, err := updateGet(u, 10*time.Minute)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	sum := sha256.New()
	n, err := io.Copy(io.MultiWriter(f, sum), io.LimitReader(resp.Body, size+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	switch {
	case err != nil:
		return fmt.Errorf("downloading %s: %w", u, err)
	case n != size:
		return fmt.Errorf("%s has %d bytes, not %d", u, n, size)
	case update.Verify(pub, sum.Sum(nil), signature) != nil:
		return fmt.Errorf("%s is not signed with the key of updates.publicKey", u)
	}
	return nil
}

// updateGet requests a published file of an update, which takes at most
// timeout.
func updateGet(u string, timeout time.Duration) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "mygo/"+version)
	// A token, as CI has one, lifts the GitHub API's limit on requests.
	if token := cmp.Or(os.Getenv("GH_TOKEN"), os.Getenv("GITHUB_TOKEN")); token != "" && req.URL.Host == "api.github.com" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound {
			return nil, errNotPublished
		}
		return nil, fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	return resp, nil
}

func runKeygen(args []string) error {
	config, _ := os.UserConfigDir()
	flags := newFlags("keygen", "[flags]", `Creates the Ed25519 key pair that signs updates: mygo-update.key, the secret,
and mygo-update.pub. Put the contents of mygo-update.pub in updates.publicKey
of mygo.json, and keep mygo-update.key out of the repository, e.g. in a
password manager or as the MYGO_UPDATER_PRIVATE_KEY secret of CI: installed
apps only accept updates signed with it, so losing it strands them.`)
	out := flags.String("o", filepath.Join(config, "mygo", "update-keys"), "directory to write the keys to")
	force := flags.Bool("force", false, "overwrite existing keys")
	if err := flags.Parse(args); err != nil {
		return err
	}
	secret, public := filepath.Join(*out, "mygo-update.key"), filepath.Join(*out, "mygo-update.pub")
	if !*force && (fileExists(secret) || fileExists(public)) {
		return fmt.Errorf("%s already holds keys; pass -force to replace them", *out)
	}
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*out, 0o700); err != nil {
		return err
	}
	if err := os.WriteFile(secret, []byte(base64.StdEncoding.EncodeToString(priv)+"\n"), 0o600); err != nil {
		return err
	}
	pubText := base64.StdEncoding.EncodeToString(pub)
	if err := os.WriteFile(public, []byte(pubText+"\n"), 0o644); err != nil {
		return err
	}
	logf("Wrote %s (secret) and %s", secret, public)
	fmt.Printf("Add to mygo.json, or mygo.config.ts:\n\n  \"updates\": {\n    \"publicKey\": %q,\n    \"github\": \"owner/name\"\n  }\n", pubText)
	return nil
}
