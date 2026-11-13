// Package models is the family model library (spec §7): the Hugging Face
// cache ($HUB) shared by every family app, with claims ("I use
// repo@commit, keep it"), flock locks compatible with huggingface_hub's
// .locks/ directory, the deletion rules that keep one app from deleting
// another's models, the family pins table and model licence acceptances.
//
// It does not download: hfhub (or an app's own downloader) does, holding
// the locks of this package and claiming what it downloaded.
//
// Typical use:
//
//	s := &models.Store{Family: family.Default(), Hub: hub, BundleID: "dev.hehehai.dicta", AppID: "dicta", Apps: presence}
//	s.Claim(ctx, models.Claim{Repo: r, Revision: c, Model: "qwen3-asr-0.6b-4bit", Role: "asr", Origin: models.OriginMigrated})
//	d, _ := s.Decide(models.Ref{Repo: r, Revision: c}) // who else uses it
//	res, err := s.Delete(ctx, ref, models.ThisApp)    // only removes files nobody else claims
package models

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/hehehai/actnowkit/family"
)

// Ref is a model's identity: repo@commit (spec §7.2).
type Ref struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
}

func (r Ref) String() string { return r.Repo + "@" + r.Revision }

// ParseRef parses "owner/name@commit".
func ParseRef(s string) (Ref, error) {
	repo, rev, ok := strings.Cut(s, "@")
	if !ok || !ValidRepo(repo) || rev == "" {
		return Ref{}, fmt.Errorf("model ref %q: want owner/name@commit", s)
	}
	return Ref{Repo: repo, Revision: rev}, nil
}

var (
	repoPart = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// ValidRepo reports whether repo is "owner/name" with the characters the
// Hub allows, so that paths built from it stay inside $HUB (spec §13).
func ValidRepo(repo string) bool {
	owner, name, ok := strings.Cut(repo, "/")
	return ok && len(repo) <= 200 && repoPart.MatchString(owner) && repoPart.MatchString(name) &&
		!strings.Contains(repo, "..") && !strings.Contains(name, "/")
}

// IsCommit reports a 40-hex commit id.
func IsCommit(rev string) bool { return commitRE.MatchString(rev) }

// Sources of the hub directory (ResolveHub).
type HubSource string

const (
	FromFamily  HubSource = "family"  // settings.json models.root
	FromApp     HubSource = "app"     // the app's own folder, when it does not follow the family
	FromDefault HubSource = "default" // $FAMILY/models/hub (DefaultHub)
	// FromEnvHub and FromEnvHome name Hugging Face caches found through
	// the environment: since v0.8.0 they are import sources only
	// (LegacyHubs), never $HUB.
	FromEnvHub  HubSource = "HF_HUB_CACHE"
	FromEnvHome HubSource = "HF_HOME"
	// FromHFDefault is ~/.cache/huggingface/hub, huggingface_hub's default
	// and the family's $HUB before v0.8.0 (spec v1.0.8 and older).
	FromHFDefault HubSource = "hf-default"
)

// HubDirName is the default hub's directory under $FAMILY/models.
const HubDirName = "hub"

// DefaultHub is the family's default model library (spec §7.1, v1.0.9):
// $FAMILY/models/hub, i.e. ~/Library/Application Support/ActNow/models/hub
// (it follows ACTNOW_FAMILY_DIR). The layout inside is Hugging Face's
// (models--owner--name/{blobs,snapshots,refs}), so engines and hfhub only
// change their root.
func DefaultHub() string { return DefaultHubIn(family.Default()) }

// DefaultHubIn is the default hub of a family directory.
func DefaultHubIn(d family.Dir) string { return filepath.Join(d.ModelsDir(), HubDirName) }

// ResolveHub finds $HUB (spec §7.1), the first that is set of:
//
//  1. familyRoot, settings.json models.root, when the app follows the family;
//  2. own, the app's private folder, when it does not;
//  3. DefaultHub: $FAMILY/models/hub.
//
// HF_HUB_CACHE and HF_HOME no longer choose it (spec v1.0.9): an app
// started from Finder and one started from a shell would use different
// folders. The Hugging Face caches they name are where Import looks for
// models to bring over (LegacyHubs); a user who wants the family to share
// one of them picks it as the model folder (models.root).
func ResolveHub(follow bool, familyRoot, own string) (string, HubSource) {
	if follow && familyRoot != "" {
		return filepath.Clean(familyRoot), FromFamily
	}
	if !follow && own != "" {
		return filepath.Clean(own), FromApp
	}
	return DefaultHub(), FromDefault
}

// EnvHub is the Hugging Face cache huggingface_hub would use, from the
// environment alone: $HF_HUB_CACHE, else $HF_HOME/hub, else
// ~/.cache/huggingface/hub. It was $HUB's fallback before v0.8.0; now it
// is the first of LegacyHubs.
func EnvHub() (string, HubSource) {
	if d := os.Getenv("HF_HUB_CACHE"); d != "" {
		return d, FromEnvHub
	}
	if d := os.Getenv("HF_HOME"); d != "" {
		return filepath.Join(d, "hub"), FromEnvHome
	}
	return HFDefaultHub(), FromHFDefault
}

// HFDefaultHub is ~/.cache/huggingface/hub.
func HFDefaultHub() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// LegacyHubs are the Hugging Face caches models may be imported from
// (Store.Import): $HF_HUB_CACHE, $HF_HOME/hub and ~/.cache/huggingface/hub,
// each once, those that exist and are not hub itself.
func LegacyHubs(hub string) []string {
	var out []string
	add := func(p string) {
		if p == "" || !filepath.IsAbs(p) || SameHub(p, hub) || !isDir(p) {
			return
		}
		for _, o := range out {
			if SameHub(o, p) {
				return
			}
		}
		out = append(out, filepath.Clean(p))
	}
	add(os.Getenv("HF_HUB_CACHE"))
	if d := os.Getenv("HF_HOME"); d != "" {
		add(filepath.Join(d, "hub"))
	}
	add(HFDefaultHub())
	return out
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// RepoFolder is a repo's directory name: "models--owner--name".
func RepoFolder(repo string) string { return "models--" + strings.ReplaceAll(repo, "/", "--") }

// RepoDir is $HUB/models--owner--name.
func RepoDir(hub, repo string) string { return filepath.Join(hub, RepoFolder(repo)) }

// SnapshotDir is $HUB/models--owner--name/snapshots/<commit>, the path an
// engine is given (spec §7.1: never let it pick a snapshot itself).
func SnapshotDir(hub string, r Ref) string {
	return filepath.Join(RepoDir(hub, r.Repo), "snapshots", r.Revision)
}

// LocksDir is $HUB/.locks/models--owner--name, huggingface_hub's lock
// directory for the repo (spec §7.4).
func LocksDir(hub, repo string) string { return filepath.Join(hub, ".locks", RepoFolder(repo)) }

// RepoLockName is the repo lock's file name.
const RepoLockName = "actnow-repo.lock"

// RepoLockFile is $HUB/.locks/models--owner--name/actnow-repo.lock.
func RepoLockFile(hub, repo string) string { return filepath.Join(LocksDir(hub, repo), RepoLockName) }

// BlobLockFile is $HUB/.locks/models--owner--name/<blob>.lock, the same
// file huggingface_hub locks while it downloads that blob.
func BlobLockFile(hub, repo, blob string) string {
	return filepath.Join(LocksDir(hub, repo), blob+".lock")
}

// EngineCopiesDir is $HUB/mlx-audio, mlx-audio-swift's flat copies.
func EngineCopiesDir(hub string) string { return filepath.Join(hub, "mlx-audio") }

// EngineCopy is $HUB/mlx-audio/<owner>_<name> (spec §7.8).
func EngineCopy(hub, repo string) string {
	return filepath.Join(EngineCopiesDir(hub), strings.ReplaceAll(repo, "/", "_"))
}

// SameHub reports whether two hub paths are the same directory (as
// written in claims: cleaned, symlinks resolved when they exist).
func SameHub(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	ca, cb := canonical(a), canonical(b)
	return ca == cb
}

func canonical(p string) string {
	p = filepath.Clean(p)
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return r
	}
	return p
}
