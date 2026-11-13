package family

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// SpecVersion is the major version of the family spec this package
// implements (family.json "specVersion", manifests' "specVersion").
const SpecVersion = 1

// DirName is the family directory's name under Application Support. It is
// fixed: APFS is case-insensitive by default, so no "actnow" beside it.
const DirName = "ActNow"

// EnvDir overrides the family directory (tests, conformance runs, two
// copies of a dev build that must not touch the real one).
const EnvDir = "ACTNOW_FAMILY_DIR"

// Dir is a family directory ($FAMILY). The zero value is not usable: use
// Default or At.
type Dir struct {
	// Root is the directory's absolute path.
	Root string
}

// Default is the family directory of the current user:
// $ACTNOW_FAMILY_DIR, else ~/Library/Application Support/ActNow (on
// systems other than macOS: the user config directory, for tests).
func Default() Dir { return Dir{Root: DefaultRoot()} }

// At is the family directory at root.
func At(root string) Dir { return Dir{Root: root} }

// DefaultRoot is Default().Root.
func DefaultRoot() string {
	if d := os.Getenv(EnvDir); d != "" {
		return d
	}
	return canonicalRoot()
}

// canonicalRoot is the real family directory, ignoring EnvDir.
func canonicalRoot() string {
	if runtime.GOOS == "darwin" {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, "Library", "Application Support", DirName)
	}
	if d, err := os.UserConfigDir(); err == nil {
		return filepath.Join(d, DirName)
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", DirName)
}

// Paths of the family directory (spec §4).

func (d Dir) FamilyFile() string      { return filepath.Join(d.Root, "family.json") }
func (d Dir) AppsDir() string         { return filepath.Join(d.Root, "apps") }
func (d Dir) SettingsFile() string    { return filepath.Join(d.Root, "settings.json") }
func (d Dir) CredentialsFile() string { return filepath.Join(d.Root, "credentials.json") }
func (d Dir) DictionaryFile() string  { return filepath.Join(d.Root, "dictionary.json") }
func (d Dir) ModelsDir() string       { return filepath.Join(d.Root, "models") }
func (d Dir) ClaimsDir() string       { return filepath.Join(d.Root, "models", "claims") }
func (d Dir) RemovedFile() string     { return filepath.Join(d.Root, "models", "removed.json") }
func (d Dir) LicencesFile() string    { return filepath.Join(d.Root, "models", "licences.json") }
func (d Dir) RuntimeDir() string      { return filepath.Join(d.Root, "runtime") }
func (d Dir) AccountDir() string      { return filepath.Join(d.Root, "account") }
func (d Dir) LocksDir() string        { return filepath.Join(d.Root, "locks") }

// AppFile is apps/<bundle id>.json.
func (d Dir) AppFile(bundleID string) string { return filepath.Join(d.AppsDir(), bundleID+".json") }

// ClaimsFile is models/claims/<bundle id>.json.
func (d Dir) ClaimsFile(bundleID string) string {
	return filepath.Join(d.ClaimsDir(), bundleID+".json")
}

// RuntimeFile is runtime/<bundle id>.json.
func (d Dir) RuntimeFile(bundleID string) string {
	return filepath.Join(d.RuntimeDir(), bundleID+".json")
}

// LockFile is locks/<name>.lock, the lock of a multi-writer file
// ("settings", "credentials", "licences", "removed", "dictionary").
func (d Dir) LockFile(name string) string { return filepath.Join(d.LocksDir(), name+".lock") }

// Names of the multi-writer files' locks.
const (
	LockSettings    = "settings"
	LockCredentials = "credentials"
	LockLicences    = "licences"
	LockRemoved     = "removed"
	LockDictionary  = "dictionary"
)

// isCanonical reports whether d is the real family directory, whose
// notifications use the spec's names (see NotifyName).
func (d Dir) isCanonical() bool {
	a, b := filepath.Clean(d.Root), filepath.Clean(canonicalRoot())
	return a == b
}

// id is a short stable hash of the root, naming the notifications of a
// directory other than the real one.
func (d Dir) id() string {
	sum := sha256.Sum256([]byte(filepath.Clean(d.Root)))
	return hex.EncodeToString(sum[:4])
}

// Info is family.json (spec §4.1).
type Info struct {
	Schema      string    `json:"schema"`
	Created     time.Time `json:"created"`
	CreatedBy   string    `json:"createdBy"`
	SpecVersion int       `json:"specVersion"`
	Extra       Extra     `json:"-"`
}

func (i *Info) UnmarshalJSON(b []byte) error {
	type plain Info
	return UnmarshalKnown(b, (*plain)(i), &i.Extra)
}

func (i Info) MarshalJSON() ([]byte, error) {
	type plain Info
	return MarshalKnown(plain(i), i.Extra)
}

// ReadOnly reports whether the directory belongs to a newer major version
// of the spec: an app then reads what it understands and writes nothing
// (spec §4.1, §5.1).
func (i Info) ReadOnly() bool { return i.SpecVersion > SpecVersion }

// KindFamily is family.json's schema kind.
const KindFamily = "family"

// Ensure creates the family directory and its subdirectories (0700) and,
// when missing, family.json naming bundleID as its creator; it returns
// family.json. Safe to call at every start and from several apps at once.
func (d Dir) Ensure(bundleID string) (Info, error) {
	if d.Root == "" {
		return Info{}, errors.New("family: empty directory")
	}
	for _, dir := range []string{d.Root, d.AppsDir(), d.ClaimsDir(), d.RuntimeDir(), d.AccountDir(), d.LocksDir()} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return Info{}, err
		}
	}
	info := Info{Schema: SchemaOf(KindFamily, 1), Created: Now(), CreatedBy: bundleID, SpecVersion: SpecVersion}
	data, err := MarshalIndent(info)
	if err != nil {
		return Info{}, err
	}
	if err := createExclusive(d.FamilyFile(), data); err != nil && !errors.Is(err, fs.ErrExist) {
		return Info{}, err
	}
	var got Info
	err = Load(d.FamilyFile(), KindFamily, 1, &got)
	var ce *CorruptError
	switch {
	case errors.As(err, &ce):
		// Unreadable: set it aside and write ours.
		if qerr := Quarantine(d.FamilyFile()); qerr != nil {
			return Info{}, qerr
		}
		if err := createExclusive(d.FamilyFile(), data); err != nil && !errors.Is(err, fs.ErrExist) {
			return Info{}, err
		}
		if err := Load(d.FamilyFile(), KindFamily, 1, &got); err != nil && !errors.Is(err, ErrNewerSchema) {
			return Info{}, err
		}
	case err != nil && !errors.Is(err, ErrNewerSchema):
		return Info{}, err
	}
	return got, nil
}

// Read reads family.json; fs.ErrNotExist when the directory was never set up.
func (d Dir) Read() (Info, error) {
	var info Info
	err := Load(d.FamilyFile(), KindFamily, 1, &info)
	if errors.Is(err, ErrNewerSchema) {
		err = nil
	}
	return info, err
}

// createExclusive writes a new file at path atomically, failing with
// fs.ErrExist when there is one: the content goes to a temporary file
// first, which is then hard-linked in place (link(2) never replaces).
func createExclusive(path string, data []byte) error {
	tmp, err := writeTemp(path, data, 0o600)
	if err != nil {
		return err
	}
	defer os.Remove(tmp)
	if err := os.Link(tmp, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("%s: %w", path, fs.ErrExist)
		}
		return err
	}
	syncDir(filepath.Dir(path))
	return nil
}

// Now is the time written in family files: UTC, whole seconds (RFC 3339
// without fractions, as the spec's examples).
func Now() time.Time { return time.Now().UTC().Truncate(time.Second) }
