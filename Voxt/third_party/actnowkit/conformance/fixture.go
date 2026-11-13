// Package conformance checks an implementation of the ActNow family
// protocol (spec §4–§11) against the reference behaviour, whether or not
// it uses actnowkit:
//
//   - testdata/files and testdata/cases.json: a file of every kind, with
//     the result a conforming reader must reach (read, read-only, corrupt,
//     members to keep). Language-neutral: any implementation can load them.
//   - Fixture: a temporary $FAMILY and $HUB ($FAMILY/models/hub;
//     ACTNOW_FAMILY_DIR set for the test), snapshots in the Hugging Face
//     layout, and peers — other family apps played by actnowkit itself.
//   - FakeHub: a local Hugging Face endpoint that counts the bytes it
//     serves and can pause a file mid-way.
//   - Run: the automatable scenarios of spec §16.1 (model reuse, deletion
//     protection, concurrent and resumed downloads, settings, shortcuts,
//     uninstalled apps, forward compatibility), driving the app under
//     test through small interfaces (ModelApp, Downloader, SettingsApp,
//     ShortcutApp, OrphanCleaner). Scenarios whose interface the app does
//     not implement are skipped.
//   - Check: problems in a real family directory (cmd/actnow-conformance).
//
// An app's test:
//
//	func TestFamilyConformance(t *testing.T) {
//		conformance.Run(t, func(t *testing.T, fx *conformance.Fixture, bundleID string) any {
//			return newMyAdapter(fx.Family.Root, fx.Hub, bundleID) // wraps the app's own code
//		})
//	}
package conformance

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
)

// Fixture is a temporary family directory and model library.
type Fixture struct {
	T      testing.TB
	Family family.Dir
	Hub    string
	// Legacy is a Hugging Face cache beside it ($HF_HUB_CACHE for the
	// test), where a model library moved from (spec v1.0.9) would be.
	Legacy string
	// Apps is what the peers' discovery says (bundle id → state): peers
	// built by the fixture use it, so a test can mark an app uninstalled.
	Apps map[string]models.AppState
}

// New makes a fixture under t.TempDir() and points ACTNOW_FAMILY_DIR at
// it for the test, so code that resolves the defaults lands there too:
// the model library is the family's default, $FAMILY/models/hub (spec
// v1.0.9). HF_HUB_CACHE points at Legacy, HF_HOME is cleared.
func New(t testing.TB) *Fixture {
	t.Helper()
	root := t.TempDir()
	fam := family.At(filepath.Join(root, "ActNow"))
	fx := &Fixture{T: t, Family: fam, Hub: models.DefaultHubIn(fam), Legacy: filepath.Join(root, "huggingface", "hub"), Apps: map[string]models.AppState{}}
	if s, ok := t.(interface{ Setenv(string, string) }); ok {
		s.Setenv(family.EnvDir, fx.Family.Root)
		s.Setenv("HF_HUB_CACHE", fx.Legacy)
		s.Setenv("HF_HOME", "")
	}
	if err := os.MkdirAll(fx.Legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := fx.Family.Ensure("dev.actnow.conformance"); err != nil {
		t.Fatal(err)
	}
	if err := models.EnsureHub(fx.Family, fx.Hub); err != nil {
		t.Fatal(err)
	}
	return fx
}

// RepoFile is a file of a fixture repo.
type RepoFile struct {
	Path    string
	Content []byte
	// LFS files are named by their sha256, others by their git blob id
	// (as the Hub does).
	LFS bool
}

// BlobID is the name of a file's blob: its sha256 when LFS, else its git
// blob id (sha1 of "blob <size>\0" + content).
func (f RepoFile) BlobID() string {
	if f.LFS {
		s := sha256.Sum256(f.Content)
		return hex.EncodeToString(s[:])
	}
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(f.Content))
	h.Write(f.Content)
	return hex.EncodeToString(h.Sum(nil))
}

// Repo is a fixture repo at one commit.
type Repo struct {
	Ref   models.Ref
	Files []RepoFile
}

// Size is the bytes of its files.
func (r Repo) Size() int64 {
	var n int64
	for _, f := range r.Files {
		n += int64(len(f.Content))
	}
	return n
}

// TestRepo makes a repo with a config, a tokenizer and a weights file of
// weightBytes deterministic bytes (seeded by the repo name).
func TestRepo(repo, revision string, weightBytes int) Repo {
	w := make([]byte, weightBytes)
	seed := sha256.Sum256([]byte(repo + "@" + revision))
	for i := range w {
		if i%32 == 0 {
			seed = sha256.Sum256(seed[:])
		}
		w[i] = seed[i%32]
	}
	return Repo{
		Ref: models.Ref{Repo: repo, Revision: revision},
		Files: []RepoFile{
			{Path: "config.json", Content: []byte(`{"model_type":"conformance","repo":"` + repo + `"}`)},
			{Path: "tokenizer/vocab.json", Content: []byte(`{"a":0,"b":1}`)},
			{Path: "model.safetensors", Content: w, LFS: true},
		},
	}
}

// Install writes a repo into $HUB as a complete download would: blobs,
// relative links in snapshots/<commit>, refs/main.
func (fx *Fixture) Install(r Repo) {
	fx.T.Helper()
	dir := models.RepoDir(fx.Hub, r.Ref.Repo)
	for _, f := range r.Files {
		blob := filepath.Join(dir, "blobs", f.BlobID())
		must(fx.T, os.MkdirAll(filepath.Dir(blob), 0o755))
		must(fx.T, os.WriteFile(blob, f.Content, 0o644))
		link := filepath.Join(dir, "snapshots", r.Ref.Revision, filepath.FromSlash(f.Path))
		must(fx.T, os.MkdirAll(filepath.Dir(link), 0o755))
		os.Remove(link)
		target := strings.Repeat("../", strings.Count(f.Path, "/")+2) + "blobs/" + f.BlobID()
		must(fx.T, os.Symlink(target, link))
	}
	must(fx.T, os.MkdirAll(filepath.Join(dir, "refs"), 0o755))
	must(fx.T, os.WriteFile(filepath.Join(dir, "refs", "main"), []byte(r.Ref.Revision), 0o644))
}

// Installed reports whether ref is complete on disk (spec §7.5).
func (fx *Fixture) Installed(ref models.Ref) bool {
	return models.Check(fx.Hub, models.Claim{Repo: ref.Repo, Revision: ref.Revision}).State == models.Installed
}

// Peer is another family app played by actnowkit (the reference
// implementation), sharing the fixture's directories.
func (fx *Fixture) Peer(bundleID string) *models.Store {
	a, _, _ := family.ParseBundleID(bundleID)
	if _, ok := fx.Apps[bundleID]; !ok {
		fx.Apps[bundleID] = models.AppState{BundleID: bundleID, AppID: a.ID, Name: a.Name, Installed: true}
	}
	return &models.Store{Family: fx.Family, Hub: fx.Hub, BundleID: bundleID, AppID: a.ID, Apps: func() map[string]models.AppState { return fx.Apps }}
}

// Claims reads an app's claims file.
func (fx *Fixture) Claims(bundleID string) models.Claims {
	c, err := models.ReadClaims(fx.Family, bundleID)
	if err != nil {
		fx.T.Fatalf("claims of %s: %v", bundleID, err)
	}
	return c
}

// Claimed reports whether bundleID's claims file names ref.
func (fx *Fixture) Claimed(bundleID string, ref models.Ref) bool {
	c, err := models.ReadClaims(fx.Family, bundleID)
	if err != nil {
		return false
	}
	_, ok := c.Find(ref)
	return ok
}

// Incomplete lists the partial-download files left under a repo.
func (fx *Fixture) Incomplete(repo string) []string {
	var out []string
	filepath.Walk(models.RepoDir(fx.Hub, repo), func(p string, info os.FileInfo, err error) error {
		if err == nil && strings.Contains(filepath.Base(p), ".incomplete") {
			out = append(out, p)
		}
		return nil
	})
	return out
}

func must(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
