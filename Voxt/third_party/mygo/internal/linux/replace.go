//go:build linux

package linux

import (
	"os"
	"syscall"
)

// ReplaceFile renames replacement over target, after giving it the
// target's permissions and, where the user may, its owner and group, as
// GIO's g_file_replace does.
func (*Backend) ReplaceFile(target, replacement string) error {
	if info, err := os.Stat(target); err == nil {
		if err := os.Chmod(replacement, info.Mode().Perm()|info.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky)); err != nil {
			return err
		}
		if st, ok := info.Sys().(*syscall.Stat_t); ok {
			_ = os.Lchown(replacement, int(st.Uid), int(st.Gid))
		}
	}
	return os.Rename(replacement, target)
}
