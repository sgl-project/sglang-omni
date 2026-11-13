package conformance

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/credentials"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/hehehai/actnowkit/settings"
)

// Problem is something Check found.
type Problem struct {
	Level   string // "error", "warning" or "info"
	Path    string
	Message string
}

func (p Problem) String() string { return p.Level + ": " + p.Path + ": " + p.Message }

var secretLike = regexp.MustCompile(`\b(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})`)

// Check inspects a family directory (and, for claims, the model library
// hub) the way the spec says files must look: every file parses with its
// schema, permissions are 0700/0600, manifests and claims are consistent,
// no secret is stored, no stale temporary files. sys answers discovery's
// questions (registry.DefaultSystem on a Mac).
func Check(d family.Dir, hub string, sys registry.System) []Problem {
	var out []Problem
	add := func(level, path, format string, args ...any) {
		rel, err := filepath.Rel(d.Root, path)
		if err != nil || strings.HasPrefix(rel, "..") {
			rel = path
		}
		out = append(out, Problem{Level: level, Path: rel, Message: fmt.Sprintf(format, args...)})
	}
	if _, err := os.Stat(d.Root); err != nil {
		add("error", d.Root, "%v", err)
		return out
	}
	load := func(path, kind string, v any) bool {
		err := family.Load(path, kind, 1, v)
		var ce *family.CorruptError
		switch {
		case err == nil:
			return true
		case family.IsNotExist(err):
			return false
		case errors.Is(err, family.ErrNewerSchema):
			add("info", path, "written by a newer major version: read only")
			return true
		case errors.As(err, &ce):
			add("error", path, "corrupt: %v", ce.Err)
		default:
			add("error", path, "%v", err)
		}
		return false
	}
	var info family.Info
	if load(d.FamilyFile(), family.KindFamily, &info) && info.SpecVersion < 1 {
		add("error", d.FamilyFile(), "specVersion %d", info.SpecVersion)
	}
	var s settings.Settings
	if load(d.SettingsFile(), settings.Kind, &s) {
		for _, k := range settings.Keys {
			if v, ok := s.Get(k); ok {
				if _, err := settings.Normalize(k, v); err != nil {
					add("error", d.SettingsFile(), "%s: %v", k, err)
				}
			}
		}
	}
	var cr credentials.File
	if load(d.CredentialsFile(), credentials.Kind, &cr) {
		for id, e := range cr.Vendors {
			if _, ok := credentials.VendorByID(id); !ok {
				add("warning", d.CredentialsFile(), "unknown vendor %q", id)
			}
			if e.Account != credentials.Account(id) {
				add("error", d.CredentialsFile(), "%s: account %q, want %q", id, e.Account, credentials.Account(id))
			}
			if !regexp.MustCompile(`^sha256:[0-9a-f]{12}$`).MatchString(e.Fingerprint) {
				add("error", d.CredentialsFile(), "%s: fingerprint %q is not sha256:<12 hex>", id, e.Fingerprint)
			}
			if _, tail, ok := strings.Cut(e.Hint, "…"); !ok || len([]rune(tail)) > 4 || len([]rune(e.Hint)) > 12 {
				add("error", d.CredentialsFile(), "%s: hint %q shows more than a prefix and 4 characters", id, e.Hint)
			}
		}
	}
	var rm models.Removed
	load(d.RemovedFile(), models.KindRemoved, &rm)
	var lic models.Licences
	load(d.LicencesFile(), models.KindLicences, &lic)

	// Manifests.
	found, _ := registry.Discover(d, sys)
	installed := registry.States(found)
	entries, _ := os.ReadDir(d.AppsDir())
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		p := filepath.Join(d.AppsDir(), e.Name())
		var m registry.Manifest
		if !load(p, registry.Kind, &m) {
			continue
		}
		if bid := strings.TrimSuffix(e.Name(), ".json"); m.BundleID != bid {
			add("error", p, "bundleId %q does not match the file name", m.BundleID)
		}
		if err := m.Validate(); err != nil {
			add("error", p, "%v", err)
		}
		if _, v, err := family.ParseBundleID(m.BundleID); err == nil && v != m.Variant {
			add("error", p, "variant %q, the bundle id says %q", m.Variant, v)
		}
		if st, ok := installed[m.BundleID]; ok && !st.Installed {
			add("info", p, "the app is uninstalled (its claims are orphans)")
		}
	}
	// Claims.
	entries, _ = os.ReadDir(d.ClaimsDir())
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		p := filepath.Join(d.ClaimsDir(), e.Name())
		var c models.Claims
		if !load(p, models.KindClaims, &c) {
			continue
		}
		if bid := strings.TrimSuffix(e.Name(), ".json"); c.BundleID != bid {
			add("error", p, "bundleId %q does not match the file name", c.BundleID)
		}
		if c.Root == "" {
			add("error", p, "no root")
		}
		for _, cl := range c.Claims {
			switch {
			case !models.ValidRepo(cl.Repo):
				add("error", p, "repo %q is not owner/name", cl.Repo)
			case !models.IsCommit(cl.Revision):
				add("warning", p, "%s: revision %q is not a commit (catalogues must pin one, §7.2)", cl.Repo, cl.Revision)
			case hub != "" && models.SameHub(c.Root, hub) && models.Check(hub, cl).State != models.Installed:
				add("warning", p, "claims %s, which is not installed", cl.Ref())
			}
		}
	}
	// Runtime files of gone processes.
	entries, _ = os.ReadDir(d.RuntimeDir())
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		p := filepath.Join(d.RuntimeDir(), e.Name())
		var f famruntime.File
		if load(p, famruntime.Kind, &f) && !family.ProcessAlive(f.PID) {
			add("info", p, "left by a process that is gone (pid %d)", f.PID)
		}
	}
	// Permissions, stray temporaries, secrets. The model library inside
	// (models/hub, spec v1.0.9) is Hugging Face's layout, not family files.
	filepath.WalkDir(d.Root, func(p string, de fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if de.IsDir() && p == models.DefaultHubIn(d) {
			return filepath.SkipDir
		}
		fi, err := de.Info()
		if err != nil {
			return nil
		}
		if de.IsDir() {
			if fi.Mode().Perm()&0o077 != 0 {
				add("warning", p, "directory mode %04o, want 0700", fi.Mode().Perm())
			}
			return nil
		}
		name := de.Name()
		if fi.Mode().Perm()&0o077 != 0 {
			add("warning", p, "file mode %04o, want 0600", fi.Mode().Perm())
		}
		if strings.HasPrefix(name, ".") && strings.Contains(name, ".tmp-") && time.Since(fi.ModTime()) > time.Hour {
			add("info", p, "temporary file left by a crashed writer")
		}
		if strings.Contains(name, ".corrupt-") {
			add("info", p, "a corrupt file set aside")
		}
		if fi.Size() <= family.MaxFileSize {
			if data, err := os.ReadFile(p); err == nil && secretLike.Match(data) {
				add("error", p, "looks like it holds a secret: secrets belong in the Keychain (§13)")
			}
		}
		return nil
	})
	return out
}
