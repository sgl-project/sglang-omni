// Time Machine: model files are excluded from backups by default (spec
// §7.12, v1.0.9).

package models

import (
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/hehehai/actnowkit/family"
)

// BackupExcludeXattr is the extended attribute Time Machine honours to
// skip an item and everything under it ("sticky" exclusion: it moves with
// the folder). NSURLIsExcludedFromBackupKey and `tmutil addexclusion`
// (without -p) set the same attribute.
const BackupExcludeXattr = "com.apple.metadata:com_apple_backup_excludeItem"

// ErrBackupUnsupported: the system has no Time Machine (not macOS).
var ErrBackupUnsupported = errors.New("backup exclusion is only supported on macOS")

// Hooks of the platform (tests replace them).
var (
	setBackupExcluded = platformSetBackupExcluded
	isBackupExcluded  = platformIsBackupExcluded
)

// SetBackupExcluded marks path (a folder: everything under it) as
// excluded from Time Machine backups, or clears the mark. It changes only
// that item, never its parents. On other systems it returns
// ErrBackupUnsupported.
func SetBackupExcluded(path string, excluded bool) error {
	return setBackupExcluded(path, excluded)
}

// IsBackupExcluded reports whether path itself carries the mark.
func IsBackupExcluded(path string) (bool, error) { return isBackupExcluded(path) }

// ApplyBackup applies the family setting models.backup to the model
// library at hub (spec §7.12): with backup false (the default) the
// models are excluded from Time Machine, with true the marks are cleared.
//
//   - hub is the family's default library (DefaultHubIn(d)): the whole
//     folder is marked, which also covers downloads in progress and the
//     mlx-audio copies;
//   - every hub (also a folder the user chose, such as their own Hugging
//     Face cache): each repo folder (models--owner--name) some family app
//     claims is marked, and nothing else — the user's own models there
//     stay as they are.
//
// Every app calls it at launch, after a download or import, and when
// TopicSettings says models.backup changed; it is idempotent and cheap
// (one getxattr per claimed repo). Errors are per item and joined.
func ApplyBackup(d family.Dir, hub string, backup bool) error {
	exclude := !backup
	var errs []error
	if SameHub(hub, DefaultHubIn(d)) {
		if err := EnsureHub(d, hub); err != nil {
			return err
		}
		if err := ensureBackupMark(hub, exclude); err != nil {
			errs = append(errs, err)
		}
	}
	all, _ := ReadAllClaims(d)
	seen := map[string]bool{}
	for _, cf := range all {
		if cf.Root != "" && !SameHub(cf.Root, hub) {
			continue
		}
		for _, c := range cf.Claims {
			if !ValidRepo(c.Repo) || seen[c.Repo] {
				continue
			}
			seen[c.Repo] = true
			if err := MarkRepoBackup(hub, c.Repo, backup); err != nil {
				errs = append(errs, err)
			}
		}
	}
	if errors.Is(errors.Join(errs...), ErrBackupUnsupported) {
		return nil
	}
	return errors.Join(errs...)
}

// EnsureHub creates the model library folder. The default one is inside
// $FAMILY, whose folders are 0700 (spec §4): they are created first with
// that mode, then hub itself (0755, Hugging Face's usual mode).
func EnsureHub(d family.Dir, hub string) error {
	if SameHub(hub, DefaultHubIn(d)) || strings.HasPrefix(filepath.Clean(hub), filepath.Clean(d.Root)+string(filepath.Separator)) {
		if err := os.MkdirAll(d.ModelsDir(), 0o700); err != nil {
			return err
		}
	}
	return os.MkdirAll(hub, 0o755)
}

// MarkRepoBackup marks (backup false) or unmarks one repo folder; a repo
// that is not on disk is skipped. Downloaders and Import call it for the
// repos they finish, with the family setting.
func MarkRepoBackup(hub, repo string, backup bool) error {
	if !ValidRepo(repo) {
		return errBadRepo
	}
	dir := RepoDir(hub, repo)
	if !isDir(dir) {
		return nil
	}
	err := ensureBackupMark(dir, !backup)
	if errors.Is(err, ErrBackupUnsupported) {
		return nil
	}
	return err
}

func ensureBackupMark(path string, exclude bool) error {
	cur, err := isBackupExcluded(path)
	if err != nil {
		return err
	}
	if cur == exclude {
		return nil
	}
	return setBackupExcluded(path, exclude)
}
